from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import hydra
import torch
from data.buffer_dataset import BufferSequenceDataset
from models.rssm import RSSM
from models.vae import VAE
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from utils import build_obs_window, mse_loss, set_seed, to_float


def _init_wandb(cfg: DictConfig) -> object | None:
    if not cfg.logging.enabled:
        return None
    import wandb

    run = wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        name=cfg.logging.name,
        tags=list(cfg.logging.tags) if cfg.logging.tags else None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def _log_wandb(run: object | None, step: int, metrics: Dict[str, float]) -> None:
    if run is None:
        return
    run.log(metrics, step=step)


def _timewise_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2, dim=0)


def _plot_error_curves(error: torch.Tensor, title: str, output_path: Path, max_dims: int) -> None:
    import matplotlib.pyplot as plt

    time_steps, dims = error.shape
    dims_to_plot = min(dims, max_dims)
    plt.figure(figsize=(10, 5))
    for d in range(dims_to_plot):
        plt.plot(range(time_steps), error[:, d].cpu().numpy(), label=f"dim_{d}")
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


@hydra.main(version_base=None, config_path="cfg", config_name="eval")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.eval.seed)

    buffer_path = Path(hydra.utils.to_absolute_path(cfg.data.buffer_path))
    stats_path = cfg.data.stats_path
    if stats_path:
        stats_path = str(Path(hydra.utils.to_absolute_path(stats_path)))

    dataset = BufferSequenceDataset(
        buffer_path=buffer_path,
        seq_len=cfg.eval.seq_len,
        obs_keys=cfg.data.obs_keys,
        include_goals=cfg.data.include_goals,
        normalize_obs=cfg.data.normalize_obs,
        normalize_action=cfg.data.normalize_action,
        stats_path=stats_path,
    )

    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim

    vae = None
    rssm_obs_dim = obs_dim
    if cfg.model.use_vae:
        vae = VAE(
            obs_dim=obs_dim,
            latent_dim=cfg.model.vae_latent_dim,
            hidden_dim=cfg.model.vae_hidden_dim,
            min_std=cfg.model.vae_min_std,
        )
        rssm_obs_dim = cfg.model.vae_latent_dim

    model = RSSM(
        obs_dim=rssm_obs_dim,
        action_dim=action_dim,
        stoch_dim=cfg.model.stoch_dim,
        deter_dim=cfg.model.deter_dim,
        hidden_dim=cfg.model.hidden_dim,
        obs_window=cfg.model.obs_window,
        min_std=cfg.model.min_std,
    )

    device = torch.device(cfg.eval.device)
    model.to(device)
    if vae is not None:
        vae.to(device)

    ckpt_path = Path(hydra.utils.to_absolute_path(cfg.eval.checkpoint_path))
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if vae is not None and ckpt.get("vae") is not None:
        vae.load_state_dict(ckpt["vae"])
    model.eval()
    if vae is not None:
        vae.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=True,
        num_workers=cfg.eval.num_workers,
        pin_memory=cfg.eval.pin_memory,
        drop_last=True,
    )

    run = _init_wandb(cfg)
    max_batches = cfg.eval.max_batches
    step = 0
    start_time = time.time()
    plotted = False

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            obs = batch["obs"]
            actions = batch["action"]
            # Dataset returns: obs (B, L+1, D), actions (B, L, A)
            targets = obs[:, 1:]
            if vae is not None:
                obs_flat = obs.view(-1, obs.shape[-1])
                mean, std, latent = vae.encode(obs_flat, sample=False)
                obs_embed = latent.view(obs.shape[0], obs.shape[1], -1)
                obs_windowed = build_obs_window(obs_embed, cfg.model.obs_window)
                posterior_output = model(obs_windowed[:, 1:], actions, sample=False)

                # Posterior recon predicts obs[1:].
                post_obs = vae.decode(posterior_output.obs_pred.reshape(-1, rssm_obs_dim)).view(
                    obs.shape[0], actions.shape[1], obs.shape[-1]
                )

                # Unconditioned prior from zeros (mostly a debugging metric).
                prior_pred_uncond = model.rollout_prior(actions, sample=False)
                prior_obs_uncond = vae.decode(prior_pred_uncond.reshape(-1, rssm_obs_dim)).view(
                    obs.shape[0], actions.shape[1], obs.shape[-1]
                )

                # Open-loop prior conditioned on the first posterior state (more meaningful).
                prior_obs_open = None
                if actions.shape[1] >= 2:
                    h1 = posterior_output.deter_state[:, 0]
                    z1 = posterior_output.stoch_state[:, 0]
                    prior_pred_open = model.rollout_prior(actions[:, 1:], h0=h1, z0=z1, sample=False)
                    prior_obs_open = vae.decode(prior_pred_open.reshape(-1, rssm_obs_dim)).view(
                        obs.shape[0], actions.shape[1] - 1, obs.shape[-1]
                    )

                post_recon = mse_loss(post_obs, targets)
                prior_recon_uncond = mse_loss(prior_obs_uncond, targets)
                prior_recon_open = mse_loss(prior_obs_open, obs[:, 2:]) if prior_obs_open is not None else None

                plot_post = post_obs
                plot_prior = prior_obs_open if prior_obs_open is not None else prior_obs_uncond
                plot_targets = obs[:, 2:] if prior_obs_open is not None else targets
            else:
                obs_windowed = build_obs_window(obs, cfg.model.obs_window)
                posterior_output = model(obs_windowed[:, 1:], actions, sample=False)

                post_pred = posterior_output.obs_pred
                prior_pred_uncond = model.rollout_prior(actions, sample=False)

                prior_pred_open = None
                if actions.shape[1] >= 2:
                    h1 = posterior_output.deter_state[:, 0]
                    z1 = posterior_output.stoch_state[:, 0]
                    prior_pred_open = model.rollout_prior(actions[:, 1:], h0=h1, z0=z1, sample=False)

                post_recon = mse_loss(post_pred, targets)
                prior_recon_uncond = mse_loss(prior_pred_uncond, targets)
                prior_recon_open = mse_loss(prior_pred_open, obs[:, 2:]) if prior_pred_open is not None else None

                plot_post = post_pred
                plot_prior = prior_pred_open if prior_pred_open is not None else prior_pred_uncond
                plot_targets = obs[:, 2:] if prior_pred_open is not None else targets

            metrics = {
                "eval/posterior_mse": to_float(post_recon),
                "eval/prior_mse_uncond": to_float(prior_recon_uncond),
            }
            if prior_recon_open is not None:
                metrics["eval/prior_mse_openloop"] = to_float(prior_recon_open)
            _log_wandb(run, step, metrics)

            if step % cfg.eval.print_every == 0:
                msg = (
                    f"Step {step} | posterior_mse={metrics['eval/posterior_mse']:.6f} "
                    f"| prior_mse_uncond={metrics['eval/prior_mse_uncond']:.6f}"
                )
                if "eval/prior_mse_openloop" in metrics:
                    msg += f" | prior_mse_openloop={metrics['eval/prior_mse_openloop']:.6f}"
                print(msg)

            if cfg.eval.plot.enabled and not plotted:
                # plot_post predicts obs[1:], plot_prior predicts obs[2:] (if open-loop) or obs[1:] (if uncond).
                post_err = _timewise_mse(plot_post, targets)
                prior_err = _timewise_mse(plot_prior, plot_targets)
                output_dir = Path(cfg.eval.plot.output_dir)
                _plot_error_curves(
                    post_err,
                    "Posterior reconstruction error",
                    output_dir / "posterior_error.png",
                    cfg.eval.plot.max_dims,
                )
                _plot_error_curves(
                    prior_err,
                    "Prior rollout error (open-loop if available)",
                    output_dir / "prior_error.png",
                    cfg.eval.plot.max_dims,
                )
                plotted = True

            step += 1
            if max_batches > 0 and step >= max_batches:
                break

    elapsed = time.time() - start_time
    print(f"Eval done in {elapsed:.2f}s for {step} batches.")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
