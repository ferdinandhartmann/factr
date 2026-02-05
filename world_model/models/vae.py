from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .modules import MLP


@dataclass
class VAEOutput:
    recon: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    latent: torch.Tensor
    kl: torch.Tensor


class VAE(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        min_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std

        self.encoder = MLP(obs_dim, 2 * latent_dim, hidden_dim, num_layers=2)
        self.decoder = MLP(latent_dim, obs_dim, hidden_dim, num_layers=2)

    def _stats(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = torch.chunk(params, 2, dim=-1)
        std = F.softplus(std) + self.min_std
        return mean, std

    def encode(self, obs: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.encoder(obs)
        mean, std = self._stats(params)
        if sample:
            eps = torch.randn_like(std)
            latent = mean + std * eps
        else:
            latent = mean
        return mean, std, latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, obs: torch.Tensor, sample: bool = True) -> VAEOutput:
        mean, std, latent = self.encode(obs, sample=sample)
        recon = self.decode(latent)
        kl = 0.5 * torch.sum(mean.pow(2) + std.pow(2) - torch.log(std.pow(2) + 1e-8) - 1, dim=-1)
        return VAEOutput(recon=recon, mean=mean, std=std, latent=latent, kl=kl)
