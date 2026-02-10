# Copyright (c) Sudeep Dasari, 2023
# Heavy inspiration taken from ACT by Tony Zhao: https://github.com/tonyzhaozh/act
# and DETR by Meta AI: https://github.com/facebookresearch/detr

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import time

from factr.agent import BaseAgent
from factr.models.classification import ClassificationHead
from factr.models.cvae import CVAEModule
from factr.models.ada_transformer import AdaTransformerDecoder, AdaTransformerDecoderLayer
from factr import misc


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src


class _TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, pos=None, query_pos=None, return_weights=False, nheads=None):  # weight関連が増
        q = k = _with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # tgt2, _ = self.multihead_attn(
        tgt2, cross_w = self.multihead_attn(
            query=_with_pos_embed(tgt, query_pos),
            key=_with_pos_embed(memory, pos),
            value=memory,
            need_weights=True,
            average_attn_weights=False,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        if return_weights:
            return tgt, cross_w
        return tgt


class _TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)
        return output


class _TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.linear2.out_features)

    def forward(
        self, tgt, memory, pos, query_pos, return_intermediate=False, return_weights=False, nheads=None
    ):  # weightだけ返すように変更している
        output = tgt
        intermediate = []
        all_cross_w = []

        for layer in self.layers:
            if return_weights:
                output, cross_w = layer(
                    output, memory, pos=pos, query_pos=query_pos, return_weights=True, nheads=nheads
                )
                all_cross_w.append(cross_w.unsqueeze(0))  # (1, B, H, Tgt, Src)
            else:
                output = layer(output, memory, pos=pos, query_pos=query_pos)

            if return_intermediate:
                intermediate.append(self.norm(output))

        if return_intermediate:
            out = torch.stack(intermediate)
        else:
            out = output

        if return_weights:
            return out, torch.cat(all_cross_w, dim=0)

        return out


class _ACT(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()

        encoder_layer = _TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = _TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = _TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = _TransformerDecoder(decoder_layer, num_decoder_layers)

        self._reset_parameters()
        self.pos_helper = _PositionalEncoding(d_model)
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _get_cls_force_context(self, memory):
        cls_feat = memory[:, 0, :]
        force_feat = memory[:, -1, :]
        combined_feat = torch.stack([cls_feat, force_feat], dim=1)  # (B, 2, 512)
        return combined_feat

    def forward(self, input_tokens, query_enc, z_token=None, target_actions=None, return_weights=False, nhads=None):
        """
        input_tokens: (B, Seq, Dim)
        query_enc: (Ac_Chunk, Dim)
        target_actions: (B, 30, 7) 追加
        """

        input_tokens = input_tokens.transpose(0, 1)
        input_pos = self.pos_helper(input_tokens)
        memory = self.encoder(input_tokens, input_pos)  # (198, B, 512)

        if z_token is not None:
            if z_token.dim() == 2:
                z_token = z_token.unsqueeze(1)  # (B,1,D)
            z_token = z_token.transpose(0, 1)  # (1,B,D)
            z_pos = torch.zeros_like(z_token)  # (1,B,D)
            memory = torch.cat([z_token, memory], dim=0)  # (1+N,B,D)
            input_pos = torch.cat([z_pos, input_pos], dim=0)  # (1+N,B,D)

        # query_enc = query_enc[:, None].repeat((1, B, 1)) #(Ac_Chunk, Dim) → (Ac_Chunk, Batch, Dim)
        query_enc = query_enc[:, None].repeat((1, input_tokens.shape[1], 1))  # (T,B,D)
        tgt = torch.zeros_like(query_enc)

        if return_weights:
            print(return_weights)
            acs_tokens, cross_w = self.decoder(
                tgt, memory, input_pos, query_enc, return_weights=True, nheads=self.nhead
            )
            return acs_tokens.transpose(0, 1), cross_w, memory.transpose(0, 1)

        else:
            acs_tokens = self.decoder(tgt, memory, input_pos, query_enc)  # (T,B,D)
            return acs_tokens.transpose(0, 1)  # (B,T,D)


@dataclass
class CVAEOutputs:
    mu_q: torch.Tensor
    logvar_q: torch.Tensor
    mu_p: torch.Tensor
    logvar_p: torch.Tensor
    z: torch.Tensor


class PriorNet(nn.Module):
    """p(z|c): uses [CLS; FORCE] -> MLP -> (mu, logvar)."""

    def __init__(self, d_model=512, d_z=32, hdim=512, clamp_logvar=False, logvar_min=-10.0, logvar_max=5.0):
        super().__init__()
        self.d_z = d_z
        self.clamp_logvar = clamp_logvar
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        in_dim = 2 * d_model
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.mu = nn.Linear(hdim, d_z)
        self.logvar = nn.Linear(hdim, d_z)
        nn.init.constant_(self.logvar.bias, -3.0)

    def forward(self, cls_tok, force_tok):
        u = torch.cat([cls_tok, force_tok], dim=-1)  # (B, 1024)
        u = self.ln(u)
        h = F.gelu(self.fc1(u))
        h = F.gelu(self.fc2(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        if self.clamp_logvar:
            logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)
        return mu, logvar


class PosteriorNet(nn.Module):
    """q(z|x,c): encodes [c_tokens(detach), x_tokens(pos)] with Transformer encoder."""

    """c_tokens = full or only [CLS, FORCE] (2 tokens)."""

    def __init__(
        self,
        d_model=512,
        d_z=32,
        nhead=8,
        num_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        action_dim=7,
        action_chunk=20,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.ac_embed = nn.Linear(action_dim, d_model)

        self.post_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.post_cls, mean=0.0, std=0.02)

        enc_layer = _TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = _TransformerEncoder(enc_layer, num_layers)
        self.pos_helper = _PositionalEncoding(d_model)
        self.mu = nn.Linear(d_model, d_z)
        self.logvar = nn.Linear(d_model, d_z)
        nn.init.constant_(self.logvar.bias, -3.0)

    def forward(
        self, c_tokens: torch.Tensor, actions: torch.Tensor, gt_only: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
        c_tokens: (B, N, D)  (should already be detached by caller)
        actions:  (B, T, action_dim)
        """
        B, T, _ = actions.shape
        x_tokens = self.ac_embed(actions)  # (B,T,D)
        x_tokens = x_tokens.transpose(0, 1)  # (T,B,D)
        x_pos = self.pos_helper(x_tokens)  # (T,B,D) positional encoding

        # prepend posterior CLS（NCLS,B,D）
        post_cls = self.post_cls.expand(B, -1, -1)  # (B,1,D)
        post_cls = post_cls.transpose(0, 1)  # (1,B,D)
        post_pos = torch.zeros_like(post_cls)

        # src = [POST_CLS, c_tokens, x_tokens]
        if gt_only == True:
            src = torch.cat([post_cls, x_tokens], dim=0)  # (1+N+T,B,D)
            pos = torch.cat([post_pos, x_pos], dim=0)

        else:
            c_tokens_t = c_tokens.transpose(0, 1)  # (N,B,D)
            c_pos = torch.zeros_like(c_tokens_t)

            src = torch.cat([post_cls, c_tokens_t, x_tokens], dim=0)  # (1+N+T,B,D)
            pos = torch.cat([post_pos, c_pos, x_pos], dim=0)  # (1+N+T,B,D)

        out = self.encoder(src, pos)  # (1+N+T,B,D)
        # summary: use POST_CLS output
        h = out[0]  # (B,D)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(mu)
    return mu + torch.exp(0.5 * logvar) * eps


def kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p):
    """KL( N(mu_q, sig_q^2) || N(mu_p, sig_p^2) ) for diagonal Gaussians."""
    # 0.5 * [ log(sig_p^2/sig_q^2) + (sig_q^2 + (mu_q-mu_p)^2)/sig_p^2 - 1 ]
    return 0.5 * ((logvar_p - logvar_q) + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p) - 1.0).sum(
        dim=-1
    )  # (B,)


class TransformerAgent(BaseAgent):
    """TransformerAgent + CVAE (Configuration 1: z_token in decoder memory)."""

    def __init__(
        self,
        features,
        odim,
        n_cams,
        ac_dim,
        ac_chunk,
        d_z=32,
        beta=1.0,
        free_bits: Optional[float] = None,
        cls_index: int = 0,
        force_index: int = -1,
        use_obs="add_token",
        imgs_per_cam=1,
        dropout=0,
        img_dropout=0,
        share_cam_features=False,
        early_fusion=False,
        feat_norm=False,
        token_dim=512,
        transformer_kwargs=dict(),
        posterior_kwargs=dict(),
        curriculum=dict(),
        gt_only=False,
        sanity_check_posterior=False,
        factr_baseline: bool = False,
    ):
        super().__init__(
            odim=odim,
            features=features,
            n_cams=n_cams,
            imgs_per_cam=imgs_per_cam,
            use_obs=use_obs,
            share_cam_features=share_cam_features,
            early_fusion=early_fusion,
            dropout=dropout,
            img_dropout=img_dropout,
            feat_norm=feat_norm,
            token_dim=token_dim,
            curriculum=curriculum,
        )

        self.transformer = _ACT(**transformer_kwargs)
        self.ac_query = nn.Embedding(ac_chunk, self.transformer.d_model)
        self.ac_proj = nn.Linear(self.transformer.d_model, ac_dim)
        self._ac_dim, self._ac_chunk = ac_dim, ac_chunk

        self.prior = PriorNet(d_model=self.transformer.d_model, d_z=d_z)
        self.posterior = PosteriorNet(
            d_model=self.transformer.d_model,
            d_z=d_z,
            action_dim=ac_dim,
            action_chunk=ac_chunk,
            **posterior_kwargs,
        )
        self.z_to_token = nn.Linear(d_z, self.transformer.d_model)

        self.beta = beta
        self.free_bits = free_bits
        self.cls_index = cls_index
        self.force_index = force_index

        self.gt_only = gt_only

        self._last_stats = {}
        self.factr_baseline = factr_baseline

    @property
    def ac_chunk(self):
        return self._ac_chunk

    @property
    def ac_dim(self):
        return self._ac_dim

    def _extract_cls_force(self, c_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_tok = c_tokens[:, self.cls_index]
        force_tok = c_tokens[:, self.force_index]
        return cls_tok, force_tok

    def forward(self, imgs, obs, ac_flat, mask_flat, class_labels=None, **kwargs):
        """Training forward. Returns scalar loss (recon + beta*KL)."""

        if getattr(self, "factr_baseline", False):
            print("train FACTR_Baseline")
            tokens = self.tokenize_obs(imgs, obs)

            action_tokens = self.transformer(tokens, self.ac_query.weight)
            actions_hat = self.ac_proj(action_tokens)

            ac_flat_hat = actions_hat.reshape((actions_hat.shape[0], -1))
            all_l1 = F.l1_loss(ac_flat_hat, ac_flat, reduction="none")
            recon = (all_l1 * mask_flat).mean()

            return {
                "total_loss": recon,
                "l1_loss": recon,
                "kl": torch.tensor(0.0, device=recon.device),  # KLは0として出力
            }

        ####################### OTAKE-SANS ADDED CODE #######################

        c_tokens = self.tokenize_obs(imgs, obs)  # (B,N,D)
        cls_tok, force_tok = self._extract_cls_force(c_tokens)  # added
        mu_p, logvar_p = self.prior(cls_tok, force_tok)

        # actions -> (B,T,ac_dim)
        B = ac_flat.shape[0]
        actions = ac_flat.view(B, self.ac_chunk, self.ac_dim)

        # posterior uses detached c
        c_for_posterior = torch.stack([cls_tok, force_tok], dim=1)

        # c_small = torch.stack([cls_tok, force_tok], dim=1)  # (B,2,D)
        mu_q, logvar_q = self.posterior(c_for_posterior.detach(), actions, gt_only=self.gt_only)

        z = reparameterize(mu_q, logvar_q)
        z_token = self.z_to_token(z)  # (B,D)

        action_tokens = self.transformer(c_tokens, self.ac_query.weight, z_token=z_token)
        actions_hat = self.ac_proj(action_tokens)  # (B,T,ac_dim)
        ac_flat_hat = actions_hat.reshape((B, -1))

        all_l1 = F.l1_loss(ac_flat_hat, ac_flat, reduction="none")  # reduction="none"でshape同じものが返る (B,T,A)
        recon = (
            (all_l1 * mask_flat).sum(dim=[-1, -2]).mean()
        )  # (B:mean, T:sum, A:sum) -> (,) 共通しているshapeは平均、その他はsum

        kl = kl_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p)  # (B,)
        if self.free_bits is not None:
            # free_bits in nats per dim -> total free bits = free_bits*d_z
            kl = torch.clamp(kl, min=self.free_bits * mu_q.shape[-1])
        kl = kl.mean()

        loss = recon + self.beta * kl
        self._last_stats = {
            "loss": float(loss.detach().cpu()),
            "recon": float(recon.detach().cpu()),
            "kl": float(kl.detach().cpu()),
        }
        return {
            "total_loss": loss,
            "l1_loss": recon,  # 再構成誤差 (L1)
            "kl": kl,  # KLダイバージェンス
        }

    @torch.no_grad()
    def get_actions_base(self, imgs, obs, sample: bool = True, num_samples: int = 1):
        tokens = self.tokenize_obs(imgs, obs)
        action_tokens = self.transformer(tokens, self.ac_query.weight)

        return self.ac_proj(action_tokens)

    @torch.no_grad()
    def get_actions_prior(self, imgs, obs, sample: bool = True, num_samples: int = 1, return_weights: bool = False):
        """複数サンプリング"""

        c_tokens = self.tokenize_obs(imgs, obs)  # memory(B,198,512)
        cls_tok, force_tok = self._extract_cls_force(c_tokens)
        B = c_tokens.shape[0]

        mu_p, logvar_p = self.prior(cls_tok, force_tok)

        if sample:
            z = self._sample_z_parallel(mu_p, logvar_p, num_samples)
        else:
            z = mu_p.unsqueeze(1).expand(-1, num_samples, -1)

        z_flat = z.reshape(B * num_samples, -1)
        z_token = self.z_to_token(z_flat)
        c_tokens_exp = c_tokens.repeat_interleave(num_samples, dim=0)

        transformer_out = self.transformer(
            c_tokens_exp,
            self.ac_query.weight,
            z_token=z_token,
            return_weights=return_weights,  # ここを追加
        )

        if return_weights:
            action_tokens, cross_w, _ = transformer_out
        else:
            action_tokens = transformer_out
            cross_w = None

        action_tokens = self.transformer(c_tokens_exp, self.ac_query.weight, z_token=z_token)
        actions_flat = self.ac_proj(action_tokens)  # (B*S, Chunk, Dim)
        action_pred = actions_flat.view(B, num_samples, self.ac_chunk, self.ac_dim)

        if return_weights:
            return action_pred, cross_w

        return action_pred  # (B, num_samples, Chunk, Dim)

    def _sample_z_parallel(self, mu, logvar, num_samples):
        """mu, logvar から num_samples 個ずつ並列サンプリングするヘルパー"""
        B, D = mu.shape
        std = torch.exp(0.5 * logvar)

        eps = torch.randn(B, num_samples, D, device=mu.device)

        z = mu.unsqueeze(1) + std.unsqueeze(1) * eps
        return z

    @torch.no_grad()
    def get_actions_pos(self, imgs, obs, target_action, num_samples: int = 1, sample: bool = True):
        """
        Posteriorからzをサンプリングし、アクションを再構成する。
        num_samples > 1 の場合、内部でバッチ次元を拡張して並列計算を行う。

        Returns:
            actions_hat: (B, num_samples, Chunk, Dim)
        """

        c_tokens = self.tokenize_obs(imgs, obs)  # cls_tokはmemoryのcls, force_tokはmemoryのforce
        cls_tok, force_tok = self._extract_cls_force(c_tokens)
        B = target_action.shape[0]  # (B, 210)

        c_for_posterior = torch.stack([cls_tok, force_tok], dim=1)

        mu_q, logvar_q = self.posterior(c_for_posterior.detach(), target_action, gt_only=self.gt_only)

        if sample:
            z = self._sample_z_parallel(mu_q, logvar_q, num_samples)
        else:
            z = mu_q.unsqueeze(1).expand(-1, num_samples, -1)

        z_flat = z.reshape(B * num_samples, -1)
        z_token = self.z_to_token(z_flat)

        c_tokens_exp = c_tokens.repeat_interleave(num_samples, dim=0)  # (B, N, D) -> (B*S, N, D)

        action_tokens = self.transformer(c_tokens_exp, self.ac_query.weight, z_token=z_token)
        actions_flat_hat = self.ac_proj(action_tokens)
        actions_hat = actions_flat_hat.view(
            B, num_samples, self.ac_chunk, self.ac_dim
        )  # 形を戻す (B*S, T, A) -> (B, S, T, A)

        return actions_hat  # (B, S, T, A)

    @torch.no_grad()
    def get_uncertainty_entropy(
        self,
        imgs,
        obs,
        sample: bool = True,
        num_samples: int = 1,
        unc_step_mode: bool = False,
        unc_target_step: int = 0,
        unc_weighted: bool = False,
        w_start: float = 0.1,
        w_end: float = 0.9,
    ):
        c_tokens = self.tokenize_obs(imgs, obs)  # memory(B,198,512)
        cls_tok, force_tok = self._extract_cls_force(c_tokens)
        B = c_tokens.shape[0]

        mu_p, logvar_p = self.prior(cls_tok, force_tok)

        if sample:
            z = self._sample_z_parallel(mu_p, logvar_p, num_samples)
        else:
            z = mu_p.unsqueeze(1).expand(-1, num_samples, -1)

        z_flat = z.reshape(B * num_samples, -1)
        z_token = self.z_to_token(z_flat)
        c_tokens_exp = c_tokens.repeat_interleave(num_samples, dim=0)

        action_tokens = self.transformer(c_tokens_exp, self.ac_query.weight, z_token=z_token)
        actions_flat = self.ac_proj(action_tokens)  # (B*S, Chunk, Dim)
        action_pred = actions_flat.view(B, num_samples, self.ac_chunk, self.ac_dim)

        final_action = action_pred[:, 0, :, :]
        # mean_action = torch.mean(action_pred, dim=1) # (B, Chunk, Dim) #こっちは平均を返す

        if unc_step_mode:
            unc_target_pred = action_pred[:, :, unc_target_step : unc_target_step + 1, :]
        else:
            unc_target_pred = action_pred

        variance = torch.var(unc_target_pred, dim=1, unbiased=True)
        uncertainty = torch.sqrt(variance + 1e-8)

        ##加重平均の処理
        if (not unc_step_mode) and unc_weighted:
            chunk_len = uncertainty.shape[1]
            device = uncertainty.device

            weights = torch.linspace(w_start, w_end, steps=chunk_len, device=device)
            weights = weights.view(1, -1, 1)  # (1, Chunk, 1)

            # 加重平均計算
            weighted_sum = (uncertainty * weights).sum(dim=1)
            sum_of_weights = weights.sum()

            # Wrapperが期待する形状 (B, 1, Dim) に戻す
            uncertainty = (weighted_sum / sum_of_weights).unsqueeze(1)

        return final_action, uncertainty  # entropy #(B, T, D)
