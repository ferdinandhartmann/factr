from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TrainMetrics:
    loss: float
    rssm_recon: float
    rssm_kl: float
    rssm_kl_raw: float | None = None
    rssm_recon_latent: float | None = None
    vae_recon: float | None = None
    vae_kl: float | None = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_obs_window(obs: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return obs
    batch, seq_len, obs_dim = obs.shape
    pad = obs[:, :1].repeat(1, window - 1, 1)
    padded = torch.cat([pad, obs], dim=1)
    windows = padded.unfold(dimension=1, size=window, step=1)
    windows = windows.contiguous().view(batch, seq_len, window * obs_dim)
    return windows


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())
