from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, kl_divergence

from .modules import MLP


@dataclass
class RSSMOutput:
    obs_pred: torch.Tensor
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    post_mean: torch.Tensor
    post_std: torch.Tensor
    deter_state: torch.Tensor
    stoch_state: torch.Tensor


class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        stoch_dim: int = 32,
        deter_dim: int = 128,
        hidden_dim: int = 128,
        obs_window: int = 1,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.obs_window = obs_window
        self.min_std = min_std

        self.obs_encoder = MLP(obs_dim * obs_window, hidden_dim, hidden_dim, num_layers=2)
        self.action_encoder = MLP(action_dim, hidden_dim, hidden_dim, num_layers=2)
        self.gru = nn.GRUCell(hidden_dim + stoch_dim, deter_dim)

        self.prior_net = MLP(deter_dim, 2 * stoch_dim, hidden_dim, num_layers=2)
        self.post_net = MLP(deter_dim + hidden_dim, 2 * stoch_dim, hidden_dim, num_layers=2)
        self.decoder = MLP(deter_dim + stoch_dim, obs_dim, hidden_dim, num_layers=3)

    def _stats(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = torch.chunk(params, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        return mean, std

    def forward(self, obs_windowed: torch.Tensor, actions: torch.Tensor, sample: bool = True) -> RSSMOutput:
        batch_size, seq_len, _ = actions.shape
        device = actions.device

        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)

        prior_means = []
        prior_stds = []
        post_means = []
        post_stds = []
        deter_states = []
        stoch_states = []
        obs_preds = []

        for t in range(seq_len):
            act_embed = self.action_encoder(actions[:, t])
            gru_input = torch.cat([act_embed, z], dim=-1)
            h = self.gru(gru_input, h)

            prior_params = self.prior_net(h)
            prior_mean, prior_std = self._stats(prior_params)

            obs_embed = self.obs_encoder(obs_windowed[:, t])
            post_params = self.post_net(torch.cat([h, obs_embed], dim=-1))
            post_mean, post_std = self._stats(post_params)

            if sample:
                eps = torch.randn_like(post_std)
                z = post_mean + post_std * eps
            else:
                # Deterministic latent for low-dim state modeling (reduces recon noise).
                z = post_mean

            dec_in = torch.cat([h, z], dim=-1)
            obs_pred = self.decoder(dec_in)

            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            post_means.append(post_mean)
            post_stds.append(post_std)
            deter_states.append(h)
            stoch_states.append(z)
            obs_preds.append(obs_pred)

        return RSSMOutput(
            obs_pred=torch.stack(obs_preds, dim=1),
            prior_mean=torch.stack(prior_means, dim=1),
            prior_std=torch.stack(prior_stds, dim=1),
            post_mean=torch.stack(post_means, dim=1),
            post_std=torch.stack(post_stds, dim=1),
            deter_state=torch.stack(deter_states, dim=1),
            stoch_state=torch.stack(stoch_states, dim=1),
        )

    @staticmethod
    def kl_loss(
        post_mean: torch.Tensor, post_std: torch.Tensor, prior_mean: torch.Tensor, prior_std: torch.Tensor
    ) -> torch.Tensor:
        post = Normal(post_mean, post_std)
        prior = Normal(prior_mean, prior_std)
        kl = kl_divergence(post, prior).sum(-1)
        return kl

    def rollout_prior(
        self,
        actions: torch.Tensor,
        h0: torch.Tensor | None = None,
        z0: torch.Tensor | None = None,
        sample: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = actions.shape
        device = actions.device
        h = h0 if h0 is not None else torch.zeros(batch_size, self.deter_dim, device=device)
        z = z0 if z0 is not None else torch.zeros(batch_size, self.stoch_dim, device=device)

        obs_preds = []
        for t in range(seq_len):
            act_embed = self.action_encoder(actions[:, t])
            gru_input = torch.cat([act_embed, z], dim=-1)
            h = self.gru(gru_input, h)

            prior_params = self.prior_net(h)
            prior_mean, prior_std = self._stats(prior_params)
            if sample:
                eps = torch.randn_like(prior_std)
                z = prior_mean + prior_std * eps
            else:
                z = prior_mean

            dec_in = torch.cat([h, z], dim=-1)
            obs_pred = self.decoder(dec_in)
            obs_preds.append(obs_pred)

        return torch.stack(obs_preds, dim=1)
