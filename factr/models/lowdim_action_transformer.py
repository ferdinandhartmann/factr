import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    return 0.5 * ((logvar_p - logvar_q) + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p) - 1.0).sum(
        dim=-1
    )


def _kl_categorical(logits_q, logits_p):
    log_q = F.log_softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    q = torch.exp(log_q)
    return (q * (log_q - log_p)).sum(dim=(-1, -2))


def _reparameterize(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * eps


def _categorical_entropy(logits):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=(-1, -2))


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
        latent_distribution,
        categorical_num_variables,
        categorical_num_categories,
    ):
        super().__init__()
        self.action_chunk = int(action_chunk)
        self.latent_distribution = str(latent_distribution).lower()
        self.categorical_num_variables = int(categorical_num_variables)
        self.categorical_num_categories = int(categorical_num_categories)

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
        if self.latent_distribution == "gaussian":
            self.mu = nn.Linear(token_dim, d_z)
            self.logvar = nn.Linear(token_dim, d_z)
            nn.init.constant_(self.logvar.bias, -3.0)
            self.logits = None
        elif self.latent_distribution == "categorical":
            out_dim = self.categorical_num_variables * self.categorical_num_categories
            self.mu = None
            self.logvar = None
            self.logits = nn.Linear(token_dim, out_dim)
        else:
            raise ValueError(f"Unsupported latent_distribution={latent_distribution}.")

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
        if self.latent_distribution == "gaussian":
            mu = self.mu(summary)
            logvar = self.logvar(summary)
            return {"mu": mu, "logvar": logvar}

        logits = self.logits(summary).view(batch_size, self.categorical_num_variables, self.categorical_num_categories)
        return {"logits": logits}


class LowdimStiffnessCVAEAgent(nn.Module):
    requires_stiffness_label = True

    def __init__(
        self,
        obs_dim=36,
        ac_dim=9,
        ac_chunk=30,
        obs_window=8,
        include_tracking_error=True,
        use_cls_token=True,
        stiffness_classes=3,
        use_stiffness_conditioning=True,
        d_z=32,
        latent_distribution="gaussian",
        categorical_num_variables=2,
        categorical_num_categories=4,
        categorical_temperature=1.0,
        categorical_straight_through=True,
        fixed_prior=False,
        token_dim=256,
        hidden_dim=512,
        beta=1.0,
        free_bits=None,
        kl_balance_alpha=0.5,
        z_context_mode="cls_all_obs",
        encoder_layers=2,
        decoder_layers=2,
        posterior_layers=2,
        nhead=8,
        dropout=0.1,
    ):
        super().__init__()

        self.include_tracking_error = bool(include_tracking_error)
        self.use_cls_token = bool(use_cls_token)
        base_obs_dim = int(obs_dim)
        if not self.include_tracking_error and base_obs_dim >= 36:
            self._obs_dim = base_obs_dim - 6
        else:
            self._obs_dim = base_obs_dim
        self._ac_dim = int(ac_dim)
        self._ac_chunk = int(ac_chunk)
        self.obs_window = int(obs_window)
        self.stiffness_classes = int(stiffness_classes)
        self.use_stiffness_conditioning = bool(use_stiffness_conditioning)
        self.beta = float(beta)
        self.free_bits = free_bits
        self.kl_balance_alpha = float(kl_balance_alpha)
        self.z_context_mode = z_context_mode
        self.latent_distribution = str(latent_distribution).lower()
        self.fixed_prior = bool(fixed_prior)
        self.categorical_num_variables = int(categorical_num_variables)
        self.categorical_num_categories = int(categorical_num_categories)
        self.categorical_temperature = float(categorical_temperature)
        self.categorical_straight_through = bool(categorical_straight_through)

        print(
            f"Initializing LowdimStiffnessCVAEAgent with obs_dim={self._obs_dim}, ac_dim={self._ac_dim}, ac_chunk={self._ac_chunk}, "
            f"obs_window={self.obs_window}, stiffness_classes={self.stiffness_classes}, "
            f"use_stiffness_conditioning={self.use_stiffness_conditioning}, d_z={d_z}, "
            f"latent_distribution={self.latent_distribution}, "
        )

        if not 0.0 <= self.kl_balance_alpha <= 1.0:
            raise ValueError(f"kl_balance_alpha must be in [0, 1], got {self.kl_balance_alpha}.")
        if self.latent_distribution not in {"gaussian", "categorical"}:
            raise ValueError(
                f"latent_distribution must be 'gaussian' or 'categorical', got {self.latent_distribution}."
            )

        if self.latent_distribution == "gaussian":
            self.d_z = int(d_z)
            if self.d_z < 1:
                raise ValueError(f"d_z must be >=1, got {self.d_z}.")
            self._latent_sample_dim = self.d_z
            self._categorical_flat_dim = None
            self._free_bits_dims = self.d_z
        else:
            if self.categorical_num_variables < 1:
                raise ValueError(f"categorical_num_variables must be >=1, got {self.categorical_num_variables}.")
            if self.categorical_num_categories < 2:
                raise ValueError(f"categorical_num_categories must be >=2, got {self.categorical_num_categories}.")
            if self.categorical_temperature <= 0.0:
                raise ValueError(f"categorical_temperature must be >0, got {self.categorical_temperature}.")
            self.d_z = int(d_z)
            if self.d_z < 1:
                raise ValueError(f"d_z must be >=1, got {self.d_z}.")
            self._categorical_flat_dim = self.categorical_num_variables * self.categorical_num_categories
            self._latent_sample_dim = self._categorical_flat_dim
            self._free_bits_dims = self.categorical_num_variables

        if self.include_tracking_error:
            self.state_slices = {
                "pose": slice(0, 9),
                "velocity": slice(9, 15),
                "wrench": slice(15, 21),
                "tracking": slice(21, 27),
                "cmd": slice(27, 36),
            }
            num_tokens = 5
        else:
            self.state_slices = {
                "pose": slice(0, 9),
                "velocity": slice(9, 15),
                "wrench": slice(15, 21),
                "cmd": slice(21, 30),
            }
            num_tokens = 4
        if self.use_stiffness_conditioning:
            num_tokens += 1
        if self.use_cls_token:
            num_tokens += 1

        self.positional_tokens = nn.Parameter(torch.zeros(1, num_tokens, token_dim))
        nn.init.normal_(self.positional_tokens, mean=0.0, std=0.02)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.cls_token = None
        self._printed_context_encoder_output_shape = False

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
        self.track_encoder = make_group_encoder(6) if self.include_tracking_error else None
        # Stiffness/mode can be fully disabled for unconditioned policy training.
        self.stiffness_encoder = (
            nn.Linear(self.stiffness_classes, token_dim, bias=False)
            if self.use_stiffness_conditioning
            else None
        )
        self.cmd_encoder = make_group_encoder(9)  ###

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

        context_dim = (num_tokens - 1) * token_dim

        if not self.fixed_prior:
            self.prior_backbone = nn.Sequential(
                nn.LayerNorm(context_dim),
                nn.Linear(context_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            if self.latent_distribution == "gaussian":
                self.prior_mu = nn.Linear(hidden_dim, self._latent_sample_dim)
                self.prior_logvar = nn.Linear(hidden_dim, self._latent_sample_dim)
                nn.init.constant_(self.prior_logvar.bias, -3.0)
                self.prior_logits = None
            else:
                out_dim = self.categorical_num_variables * self.categorical_num_categories
                self.prior_logits = nn.Linear(hidden_dim, out_dim)
                self.prior_mu = None
                self.prior_logvar = None
        else:
            self.prior_backbone = None
            self.prior_mu = None
            self.prior_logvar = None
            self.prior_logits = None

        self.posterior = _PosteriorTransformer(
            token_dim=token_dim,
            d_z=self._latent_sample_dim,
            action_dim=self._ac_dim,
            action_chunk=self._ac_chunk,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=posterior_layers,
            dropout=dropout,
            latent_distribution=self.latent_distribution,
            categorical_num_variables=self.categorical_num_variables,
            categorical_num_categories=self.categorical_num_categories,
        )

        if self.latent_distribution == "categorical" and self._latent_sample_dim != self.d_z:
            self.categorical_latent_proj = nn.Linear(
                self._latent_sample_dim, self.d_z
            )  # compress the V*K sampled into dz
        else:
            self.categorical_latent_proj = nn.Identity()

        self.z_to_token = nn.Linear(self.d_z, token_dim)
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

    def _labels_to_one_hot(self, class_labels, batch_size, device, dtype):
        if class_labels is None:
            # Default to stiffness class 1 when older call sites do not pass labels.
            labels = torch.ones(batch_size, device=device, dtype=torch.long)
        else:
            labels = class_labels.to(device=device)
            if labels.ndim == 1 and batch_size == 1 and labels.shape[0] == self.stiffness_classes:
                # Single-sample inference may pass one stiffness vector without a batch dim.
                return labels.unsqueeze(0).to(dtype=dtype)
            if labels.ndim == 2 and labels.shape == (batch_size, self.stiffness_classes):
                # Accept already encoded batched stiffness labels directly, e.g. [[1, 0, 0], ...].
                return labels.to(dtype=dtype)
            labels = labels.long().view(-1)
            if labels.shape[0] != batch_size:
                expected = f"({batch_size},) or ({batch_size}, {self.stiffness_classes})"
                raise ValueError(f"Expected class labels with shape {expected}, got {tuple(class_labels.shape)}.")
        # Scalar stiffness labels are 1-based at the data/API boundary, then shifted for one-hot.
        if torch.any((labels < 1) | (labels > self.stiffness_classes)):
            raise ValueError(f"Expected 1-based stiffness labels in [1, {self.stiffness_classes}].")
        return F.one_hot(labels - 1, num_classes=self.stiffness_classes).to(dtype=dtype)

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

        pose = obs[:, :, self.state_slices["pose"]].reshape(batch_size, -1)
        vel = obs[:, :, self.state_slices["velocity"]].reshape(batch_size, -1)
        wrench = obs[:, :, self.state_slices["wrench"]].reshape(batch_size, -1)
        cmd = obs[:, :, self.state_slices["cmd"]].reshape(batch_size, -1)  ### add command as part of the context tokens

        pose_token = self.pose_encoder(pose)
        vel_token = self.vel_encoder(vel)
        wrench_token = self.wrench_encoder(wrench)
        cmd_token = self.cmd_encoder(cmd)  ### Command token
        token_list = []
        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1).squeeze(1)
            token_list.append(cls_token)
        token_list.extend([pose_token, vel_token, wrench_token])

        if self.include_tracking_error:
            track = obs[:, :, self.state_slices["tracking"]].reshape(batch_size, -1)
            track_token = self.track_encoder(track)
            token_list.append(track_token)
        if self.use_stiffness_conditioning:
            stiffness_one_hot = self._labels_to_one_hot(
                class_labels,
                batch_size=batch_size,
                device=obs.device,
                dtype=obs.dtype,
            )
            stiffness_token = self.stiffness_encoder(stiffness_one_hot)
            token_list.append(stiffness_token)
        token_list.append(cmd_token)
        tokens = torch.stack(token_list, dim=1)
        tokens = tokens + self.positional_tokens
        tokens = self.context_encoder(tokens)
        tokens = self.context_norm(tokens)
        if not self._printed_context_encoder_output_shape:
            print(f"Context encoder output shape: {tuple(tokens.shape)}")
            self._printed_context_encoder_output_shape = True
        return tokens

    def _build_z_context(self, context_tokens):
        """
        Args:
            context_tokens: Tensor of shape (batch_size, num_tokens, token_dim) containing
                           all context tokens except the command token
                           and optionally cls_token at index 0 when use_cls_token=True.
        Returns:
            Tensor of shape (batch_size, context_dim) representing the concatenated context vector.
        """
        return context_tokens.reshape(context_tokens.shape[0], -1)

    def _prior(self, z_context):
        batch_size = z_context.shape[0]
        if self.latent_distribution == "gaussian":
            if self.fixed_prior:
                # Fixed Gaussian prior p(z)=N(0,I), independent of context.
                mu = torch.zeros(batch_size, self._latent_sample_dim, device=z_context.device, dtype=z_context.dtype)
                logvar = torch.zeros_like(mu)
                return {"mu": mu, "logvar": logvar}
            h = self.prior_backbone(z_context)
            mu = self.prior_mu(h)
            logvar = self.prior_logvar(h)
            return {"mu": mu, "logvar": logvar}

        if self.latent_distribution == "categorical":
            if self.fixed_prior:
                # Fixed categorical prior p(z): uniform over categories for each variable.
                logits = torch.zeros(
                    batch_size,
                    self.categorical_num_variables,
                    self.categorical_num_categories,
                    device=z_context.device,
                    dtype=z_context.dtype,
                )
                return {"logits": logits}

        h = self.prior_backbone(z_context)
        logits = self.prior_logits(h).view(batch_size, self.categorical_num_variables, self.categorical_num_categories)
        return {"logits": logits}

    def _apply_free_bits(self, kl_values):
        if self.free_bits is None:
            return kl_values
        kl_floor = float(self.free_bits) * float(self._free_bits_dims)
        return torch.clamp(kl_values, min=kl_floor)

    def _compute_kl(self, posterior_params, prior_params):
        if self.latent_distribution == "gaussian":
            mu_q, logvar_q = posterior_params["mu"], posterior_params["logvar"]
            mu_p, logvar_p = prior_params["mu"], prior_params["logvar"]
            if math.isclose(self.kl_balance_alpha, 0.5):
                kl = _kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)
                return self._apply_free_bits(kl).mean()

            kl_prior = _kl_diag_gaussians(mu_q.detach(), logvar_q.detach(), mu_p, logvar_p)
            kl_post = _kl_diag_gaussians(mu_q, logvar_q, mu_p.detach(), logvar_p.detach())
            kl_prior = self._apply_free_bits(kl_prior)
            kl_post = self._apply_free_bits(kl_post)
            return (self.kl_balance_alpha * kl_prior + (1.0 - self.kl_balance_alpha) * kl_post).mean()

        logits_q, logits_p = posterior_params["logits"], prior_params["logits"]
        if math.isclose(self.kl_balance_alpha, 0.5):
            kl = _kl_categorical(logits_q, logits_p)
            return self._apply_free_bits(kl).mean()

        kl_prior = _kl_categorical(logits_q.detach(), logits_p)
        kl_post = _kl_categorical(logits_q, logits_p.detach())
        kl_prior = self._apply_free_bits(kl_prior)
        kl_post = self._apply_free_bits(kl_post)
        return (self.kl_balance_alpha * kl_prior + (1.0 - self.kl_balance_alpha) * kl_post).mean()

    def _sample_train_latent(self, posterior_params):
        if self.latent_distribution == "gaussian":
            return _reparameterize(posterior_params["mu"], posterior_params["logvar"])

        logits_q = posterior_params["logits"]
        z = F.gumbel_softmax(
            logits_q,
            tau=self.categorical_temperature,
            hard=self.categorical_straight_through,
            dim=-1,
        )
        return z.reshape(logits_q.shape[0], self._latent_sample_dim)

    def _sample_latent_batch(self, latent_params, sample, num_samples):
        if num_samples < 1:
            raise ValueError(f"num_samples must be >=1, got {num_samples}.")

        if self.latent_distribution == "gaussian":
            mu, logvar = latent_params["mu"], latent_params["logvar"]
            batch_size, z_dim = mu.shape
            if sample:
                eps = torch.randn(batch_size, num_samples, z_dim, device=mu.device, dtype=mu.dtype)
                return mu.unsqueeze(1) + torch.exp(0.5 * logvar).unsqueeze(1) * eps
            return mu.unsqueeze(1).expand(-1, num_samples, -1)

        logits = latent_params["logits"]
        batch_size, n_var, n_cat = logits.shape
        probs = torch.softmax(logits, dim=-1)
        if sample:
            flat_probs = probs.reshape(batch_size * n_var, n_cat)
            sampled_idx = torch.multinomial(flat_probs, num_samples=num_samples, replacement=True)
            sampled_idx = sampled_idx.view(batch_size, n_var, num_samples).permute(0, 2, 1)
            # Flatten (num_variables, num_categories) -> latent dim for the decoder.
            sampled_one_hot = F.one_hot(sampled_idx, num_classes=n_cat).to(dtype=probs.dtype)
            return sampled_one_hot.reshape(batch_size, num_samples, self._latent_sample_dim)

        # Keep deterministic inference on the same one-hot support as the training path.
        mode_idx = torch.argmax(logits, dim=-1)
        mode_one_hot = F.one_hot(mode_idx, num_classes=n_cat).to(dtype=probs.dtype)
        expanded = mode_one_hot.unsqueeze(1).expand(batch_size, num_samples, n_var, n_cat)
        return expanded.reshape(batch_size, num_samples, self._latent_sample_dim)

    def _deterministic_latent(self, latent_params):
        if self.latent_distribution == "gaussian":
            return latent_params["mu"]

        logits = latent_params["logits"]
        mode_idx = torch.argmax(logits, dim=-1)
        mode_one_hot = F.one_hot(mode_idx, num_classes=self.categorical_num_categories).to(dtype=logits.dtype)
        return mode_one_hot.reshape(mode_one_hot.shape[0], self._latent_sample_dim)

    def _prepare_decoder_latent(self, z):
        if self.latent_distribution == "gaussian":
            return z

        if z.ndim == 2:
            return self.categorical_latent_proj(z)
        if z.ndim == 3:
            batch_size, num_samples, flat_dim = z.shape
            z = self.categorical_latent_proj(z.reshape(batch_size * num_samples, flat_dim))
            return z.view(batch_size, num_samples, self.d_z)
        raise ValueError(f"Unsupported latent tensor shape: {tuple(z.shape)}")

    def _latent_metrics(self, posterior_params, prior_params):
        if self.latent_distribution == "gaussian":
            logvar_p, logvar_q = prior_params["logvar"], posterior_params["logvar"]
            prior_std_mean = torch.exp(0.5 * logvar_p).mean()
            posterior_std_mean = torch.exp(0.5 * logvar_q).mean()
            dim = float(logvar_q.shape[-1])
            entropy_const = 0.5 * dim * (1.0 + math.log(2.0 * math.pi))
            prior_entropy = entropy_const + 0.5 * logvar_p.sum(dim=-1).mean()
            posterior_entropy = entropy_const + 0.5 * logvar_q.sum(dim=-1).mean()
            return prior_std_mean, posterior_std_mean, prior_entropy, posterior_entropy

        prior_entropy = _categorical_entropy(prior_params["logits"]).mean()
        posterior_entropy = _categorical_entropy(posterior_params["logits"]).mean()
        return None, None, prior_entropy, posterior_entropy

    def _decode_actions(self, context_tokens, z):
        z_token = self.z_to_token(z).unsqueeze(1)
        # Inject z as a dedicated memory token so each action step can attend to it via cross-attention.
        memory = torch.cat([z_token, context_tokens], dim=1)
        target_queries = self.action_queries.weight.unsqueeze(0).expand(context_tokens.shape[0], -1, -1)
        decoded = self.decoder(tgt=target_queries, memory=memory)
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
        context_tokens_withcmd = context_tokens
        # Exclude command token for latent prior/posterior context; command is always appended last.
        context_tokens = context_tokens[:, :-1]
        z_context = self._build_z_context(context_tokens)

        prior_params = self._prior(z_context)
        # posterior_params = self.posterior(context_tokens.detach(), target_actions)
        posterior_params = self.posterior(context_tokens, target_actions)

        z = self._sample_train_latent(posterior_params)
        z = self._prepare_decoder_latent(z)
        pred_actions = self._decode_actions(context_tokens_withcmd, z)

        recon = F.l1_loss(pred_actions, target_actions, reduction="none")
        recon = (recon * mask).sum() / torch.clamp(mask.sum(), min=1.0)

        # Keep KL-balance and free-bits behavior shared across latent families.
        kl = self._compute_kl(posterior_params, prior_params)
        prior_std_mean, posterior_std_mean, prior_entropy, posterior_entropy = self._latent_metrics(
            posterior_params, prior_params
        )

        total_loss = recon + self.beta * kl
        return {
            "total_loss": total_loss,
            "l1_loss": recon,
            # "l2_loss": recon_l2,
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
        context_tokens_no_cmd = context_tokens[:, :-1]
        z_context = self._build_z_context(context_tokens_no_cmd)
        prior_params = self._prior(z_context)
        z = self._deterministic_latent(prior_params)
        z = self._prepare_decoder_latent(z)
        return self._decode_actions(context_tokens, z)

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
        context_tokens_no_cmd = context_tokens[:, :-1]
        z_context = self._build_z_context(context_tokens_no_cmd)
        prior_params = self._prior(z_context)

        z = self._sample_latent_batch(prior_params, sample=sample, num_samples=num_samples)
        z = self._prepare_decoder_latent(z)
        batch_size, _, z_dim = z.shape

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
        context_tokens_no_cmd = context_tokens[:, :-1]

        posterior_params = self.posterior(context_tokens_no_cmd, target_action)
        z = self._sample_latent_batch(posterior_params, sample=sample, num_samples=num_samples)
        z = self._prepare_decoder_latent(z)
        batch_size, _, z_dim = z.shape

        context_tokens = context_tokens.repeat_interleave(num_samples, dim=0)
        z = z.reshape(batch_size * num_samples, z_dim)
        action_pred = self._decode_actions(context_tokens, z)
        return action_pred.view(batch_size, num_samples, self._ac_chunk, self._ac_dim)
