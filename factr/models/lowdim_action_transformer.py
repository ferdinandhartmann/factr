import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    return 0.5 * ((logvar_p - logvar_q) + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p) - 1.0).sum(
        dim=-1
    )


def _reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * eps


class _PosteriorTransformer(nn.Module):
    def __init__(
        self,
        token_dim,
        d_z,
        action_dim,
        action_chunk,
        hidden_dim,
        nhead,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.action_chunk = int(action_chunk)
        self.action_embed = nn.Linear(action_dim, token_dim)
        self.action_pos_embed = nn.Embedding(action_chunk, token_dim)
        self.post_cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.normal_(self.post_cls, mean=0.0, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(token_dim)
        self.mu = nn.Linear(token_dim, d_z)
        self.logvar = nn.Linear(token_dim, d_z)
        nn.init.constant_(self.logvar.bias, -3.0)

    def forward(self, context_tokens, target_actions):
        batch_size, chunk_len, _ = target_actions.shape
        if chunk_len != self.action_chunk:
            raise ValueError(f"Expected action_chunk={self.action_chunk}, got {chunk_len}.")

        action_tokens = self.action_embed(target_actions)
        pos_ids = torch.arange(chunk_len, device=target_actions.device).unsqueeze(0).expand(batch_size, -1)
        action_tokens = action_tokens + self.action_pos_embed(pos_ids)

        post_cls = self.post_cls.expand(batch_size, -1, -1)
        src = torch.cat([post_cls, context_tokens, action_tokens], dim=1)
        out = self.encoder(src)
        out = self.encoder_norm(out)

        summary = out[:, 0]
        mu = self.mu(summary)
        logvar = self.logvar(summary)
        return mu, logvar


class LowdimStiffnessCVAEAgent(nn.Module):
    requires_stiffness_label = True

    def __init__(
        self,
        obs_dim=27,
        ac_dim=9,
        ac_chunk=30,
        obs_window=8,
        stiffness_classes=3,
        d_z=32,
        token_dim=256,
        hidden_dim=512,
        beta=1.0,
        free_bits=None,
        z_context_mode="cls_all_obs",
        encoder_layers=2,
        decoder_layers=2,
        posterior_layers=2,
        nhead=8,
        dropout=0.1,
        factr_baseline=False,
    ):
        super().__init__()

        self._obs_dim = int(obs_dim)
        self._ac_dim = int(ac_dim)
        self._ac_chunk = int(ac_chunk)
        self.obs_window = int(obs_window)
        self.stiffness_classes = int(stiffness_classes)
        self.beta = float(beta)
        self.free_bits = free_bits
        self.z_context_mode = z_context_mode
        self.factr_baseline = bool(factr_baseline)

        if self._obs_dim != 27:
            raise ValueError(f"Expected obs_dim=27 for grouped tokens, got {self._obs_dim}.")
        if self._ac_dim != 9:
            raise ValueError(f"Expected ac_dim=9 for pose-only prediction, got {self._ac_dim}.")

        self.state_slices = {
            "pose": slice(0, 9),
            "velocity": slice(9, 15),
            "wrench": slice(15, 21),
            "tracking": slice(21, 27),
        }

        self.positional_tokens = nn.Parameter(torch.zeros(1, 6, token_dim))
        nn.init.normal_(self.positional_tokens, mean=0.0, std=0.02)

        def make_group_encoder(group_dim):
            in_dim = self.obs_window * group_dim
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, token_dim),
                nn.GELU(),
                nn.Linear(token_dim, token_dim),
            )

        self.pose_encoder = make_group_encoder(9)
        self.vel_encoder = make_group_encoder(6)
        self.wrench_encoder = make_group_encoder(6)
        self.track_encoder = make_group_encoder(6)
        self.stiffness_embed = nn.Embedding(self.stiffness_classes, token_dim)

        self.cls_from_obs = nn.Sequential(
            nn.LayerNorm(4 * token_dim),
            nn.Linear(4 * token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.context_norm = nn.LayerNorm(token_dim)

        if self.z_context_mode == "cls_all_obs":
            context_dim = 5 * token_dim
            self.pool_proj = None
        elif self.z_context_mode == "cls_force":
            context_dim = 2 * token_dim
            self.pool_proj = None
        elif self.z_context_mode == "all_tokens":
            context_dim = 6 * token_dim
            self.pool_proj = None
        elif self.z_context_mode == "attn_pool":
            context_dim = 2 * token_dim
            self.pool_proj = nn.Linear(token_dim, 1)
        else:
            raise ValueError(f"Unknown z_context_mode: {self.z_context_mode}")

        self.prior_backbone = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.prior_mu = nn.Linear(hidden_dim, d_z)
        self.prior_logvar = nn.Linear(hidden_dim, d_z)
        nn.init.constant_(self.prior_logvar.bias, -3.0)

        self.posterior = _PosteriorTransformer(
            token_dim=token_dim,
            d_z=d_z,
            action_dim=self._ac_dim,
            action_chunk=self._ac_chunk,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=posterior_layers,
            dropout=dropout,
        )

        self.z_to_token = nn.Linear(d_z, token_dim)
        self.action_queries = nn.Embedding(self._ac_chunk, token_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.action_head = nn.Linear(token_dim, self._ac_dim)

    @property
    def ac_dim(self):
        return self._ac_dim

    @property
    def ac_chunk(self):
        return self._ac_chunk

    def _normalize_labels(self, class_labels, batch_size, device):
        if class_labels is None:
            labels = torch.ones(batch_size, device=device, dtype=torch.long)
        else:
            labels = class_labels.to(device=device).long().view(-1)
            if labels.shape[0] != batch_size:
                raise ValueError(f"Expected {batch_size} class labels, got {labels.shape[0]}.")
        if torch.min(labels) >= 1:
            labels = labels - 1
        labels = labels.clamp(min=0, max=self.stiffness_classes - 1)
        return labels

    def _prepare_obs(self, obs):
        if obs.ndim != 3:
            raise ValueError(f"Expected obs shape (B, W, D), got {tuple(obs.shape)}.")
        batch_size, window, dim = obs.shape
        if window != self.obs_window:
            raise ValueError(f"Expected obs_window={self.obs_window}, got {window}.")
        if dim != self._obs_dim:
            raise ValueError(f"Expected obs_dim={self._obs_dim}, got {dim}.")
        return batch_size

    def _build_context_tokens(self, obs, class_labels):
        batch_size = self._prepare_obs(obs)
        labels = self._normalize_labels(class_labels, batch_size=batch_size, device=obs.device)

        pose = obs[:, :, self.state_slices["pose"]].reshape(batch_size, -1)
        vel = obs[:, :, self.state_slices["velocity"]].reshape(batch_size, -1)
        wrench = obs[:, :, self.state_slices["wrench"]].reshape(batch_size, -1)
        track = obs[:, :, self.state_slices["tracking"]].reshape(batch_size, -1)

        pose_token = self.pose_encoder(pose)
        vel_token = self.vel_encoder(vel)
        wrench_token = self.wrench_encoder(wrench)
        track_token = self.track_encoder(track)
        stiffness_token = self.stiffness_embed(labels)

        cls_input = torch.cat([pose_token, vel_token, wrench_token, track_token], dim=-1)
        cls_token = self.cls_from_obs(cls_input)

        tokens = torch.stack([cls_token, pose_token, vel_token, wrench_token, track_token, stiffness_token], dim=1)
        tokens = tokens + self.positional_tokens
        tokens = self.context_encoder(tokens)
        tokens = self.context_norm(tokens)
        return tokens

    def _build_z_context(self, context_tokens):
        cls_token = context_tokens[:, 0]
        pose_token = context_tokens[:, 1]
        vel_token = context_tokens[:, 2]
        wrench_token = context_tokens[:, 3]
        track_token = context_tokens[:, 4]

        if self.z_context_mode == "cls_all_obs":
            return torch.cat([cls_token, pose_token, vel_token, wrench_token, track_token], dim=-1)

        if self.z_context_mode == "cls_force":
            return torch.cat([cls_token, wrench_token], dim=-1)

        if self.z_context_mode == "all_tokens":
            return context_tokens.reshape(context_tokens.shape[0], -1)

        attn_logits = self.pool_proj(context_tokens).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * context_tokens, dim=1)
        return torch.cat([cls_token, pooled], dim=-1)

    def _prior(self, z_context):
        h = self.prior_backbone(z_context)
        mu = self.prior_mu(h)
        logvar = self.prior_logvar(h)
        return mu, logvar

    def _decode_actions(self, context_tokens, z):
        z_query_bias = self.z_to_token(z).unsqueeze(1)
        target_queries = self.action_queries.weight.unsqueeze(0).expand(context_tokens.shape[0], -1, -1)
        target_queries = target_queries + z_query_bias
        decoded = self.decoder(tgt=target_queries, memory=context_tokens)
        return self.action_head(decoded)

    def _reshape_actions(self, action_tensor):
        if action_tensor.ndim == 2:
            return action_tensor.view(action_tensor.shape[0], self._ac_chunk, self._ac_dim)
        if action_tensor.ndim == 3:
            return action_tensor
        raise ValueError(f"Unsupported action tensor shape: {tuple(action_tensor.shape)}")

    def forward(self, imgs, obs, ac_flat, mask_flat, class_labels=None, **kwargs):
        del imgs, kwargs

        target_actions = self._reshape_actions(ac_flat)
        mask = self._reshape_actions(mask_flat)

        context_tokens = self._build_context_tokens(obs, class_labels=class_labels)
        z_context = self._build_z_context(context_tokens)

        mu_p, logvar_p = self._prior(z_context)
        mu_q, logvar_q = self.posterior(context_tokens.detach(), target_actions)

        z = _reparameterize(mu_q, logvar_q)
        pred_actions = self._decode_actions(context_tokens, z)

        recon = F.l1_loss(pred_actions, target_actions, reduction="none")
        recon = (recon * mask).sum() / torch.clamp(mask.sum(), min=1.0)

        kl = _kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)
        if self.free_bits is not None:
            kl = torch.clamp(kl, min=float(self.free_bits) * mu_q.shape[-1])
        kl = kl.mean()

        prior_std_mean = torch.exp(0.5 * logvar_p).mean()
        posterior_std_mean = torch.exp(0.5 * logvar_q).mean()
        dim = float(mu_q.shape[-1])
        entropy_const = 0.5 * dim * (1.0 + math.log(2.0 * math.pi))
        prior_entropy = entropy_const + 0.5 * logvar_p.sum(dim=-1).mean()
        posterior_entropy = entropy_const + 0.5 * logvar_q.sum(dim=-1).mean()

        total_loss = recon + self.beta * kl
        return {
            "total_loss": total_loss,
            "l1_loss": recon,
            "kl": kl,
            "prior_std_mean": prior_std_mean,
            "posterior_std_mean": posterior_std_mean,
            "prior_entropy": prior_entropy,
            "posterior_entropy": posterior_entropy,
        }

    @torch.no_grad()
    def get_actions_base(self, imgs, obs, class_labels=None, sample=False, **kwargs):
        del imgs, kwargs, sample
        context_tokens = self._build_context_tokens(obs, class_labels=class_labels)
        z_context = self._build_z_context(context_tokens)
        mu_p, _ = self._prior(z_context)
        return self._decode_actions(context_tokens, mu_p)

    @torch.no_grad()
    def get_actions_prior(
        self,
        imgs,
        obs,
        class_labels=None,
        sample=True,
        num_samples=1,
        return_weights=False,
        **kwargs,
    ):
        del imgs
        context_tokens = self._build_context_tokens(obs, class_labels=class_labels)
        z_context = self._build_z_context(context_tokens)
        mu_p, logvar_p = self._prior(z_context)

        batch_size, z_dim = mu_p.shape
        if sample:
            eps = torch.randn(batch_size, num_samples, z_dim, device=mu_p.device)
            z = mu_p.unsqueeze(1) + torch.exp(0.5 * logvar_p).unsqueeze(1) * eps
        else:
            z = mu_p.unsqueeze(1).expand(-1, num_samples, -1)

        context_tokens = context_tokens.repeat_interleave(num_samples, dim=0)
        z = z.reshape(batch_size * num_samples, z_dim)
        action_pred = self._decode_actions(context_tokens, z)
        action_pred = action_pred.view(batch_size, num_samples, self._ac_chunk, self._ac_dim)

        if return_weights:
            return action_pred, None
        return action_pred

    @torch.no_grad()
    def get_actions_pos(self, imgs, obs, target_action, class_labels=None, num_samples=1, sample=True, **kwargs):
        del imgs
        target_action = self._reshape_actions(target_action)
        context_tokens = self._build_context_tokens(obs, class_labels=class_labels)

        mu_q, logvar_q = self.posterior(context_tokens.detach(), target_action)
        batch_size, z_dim = mu_q.shape
        if sample:
            eps = torch.randn(batch_size, num_samples, z_dim, device=mu_q.device)
            z = mu_q.unsqueeze(1) + torch.exp(0.5 * logvar_q).unsqueeze(1) * eps
        else:
            z = mu_q.unsqueeze(1).expand(-1, num_samples, -1)

        context_tokens = context_tokens.repeat_interleave(num_samples, dim=0)
        z = z.reshape(batch_size * num_samples, z_dim)
        action_pred = self._decode_actions(context_tokens, z)
        return action_pred.view(batch_size, num_samples, self._ac_chunk, self._ac_dim)

    @torch.no_grad()
    def get_uncertainty_entropy(
        self,
        imgs,
        obs,
        class_labels=None,
        sample=True,
        num_samples=1,
        unc_step_mode=False,
        unc_target_step=0,
        unc_weighted=False,
        w_start=0.1,
        w_end=0.9,
        **kwargs,
    ):
        del kwargs
        actions = self.get_actions_prior(
            imgs=imgs,
            obs=obs,
            class_labels=class_labels,
            sample=sample,
            num_samples=num_samples,
        )
        final_action = actions[:, 0]

        target = actions[:, :, unc_target_step : unc_target_step + 1] if unc_step_mode else actions
        uncertainty = torch.std(target, dim=1, unbiased=True)

        if (not unc_step_mode) and unc_weighted:
            chunk_len = uncertainty.shape[1]
            weights = torch.linspace(w_start, w_end, steps=chunk_len, device=uncertainty.device).view(1, -1, 1)
            weighted_sum = (uncertainty * weights).sum(dim=1)
            uncertainty = (weighted_sum / weights.sum()).unsqueeze(1)

        return final_action, uncertainty
