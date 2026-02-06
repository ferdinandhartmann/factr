from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import torch
from data.buffer_dataset import BufferSequenceDataset
from models.rssm import RSSM
from models.vae import VAE
from torch.utils.data import DataLoader
from utils import build_obs_window, mse_loss, set_seed, to_float


def _init_wandb(
    enabled: bool,
    project: str | None,
    entity: str | None,
    name: str | None,
    tags: list[str] | None,
    config: dict | None,
) -> object | None:
    if not enabled:
        return None
    import wandb

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        tags=list(tags) if tags else None,
        config=config,
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
        plt.plot(range(time_steps), error[:, d].cpu().numpy(), label=f"dim_{d}", alpha=0.6)
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> None:
    # --------- User-editable settings (Hartmannis style) ---------
    seq_len = 500
    batch_size = 64
    max_batches = 25
    print_every = 5
    num_workers = 4
    pin_memory = False
    seed = 7

    plot_enabled = True
    plot_output_dir = Path("plots")
    plot_max_dims = 28

    training_run_name = "rssm_newnorm_train_12"
    ckpt_step = "latest"    # number of ckpt step or "latest"

    checkpoint_path = Path(f"checkpoints/{training_run_name}/ckpt_{ckpt_step}.ckpt")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    buffer_path_override = None
    stats_path_override = None
    wandb_enabled = False
    # ------------------------------------------------------------

    set_seed(seed)
    device = _resolve_device(device_str)

    if not checkpoint_path.exists():
        if ckpt_step == "latest":
            run_dir = Path("checkpoints") / training_run_name
            candidates = sorted(run_dir.glob("rssm_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
            if candidates:
                checkpoint_path = candidates[-1]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    ckpt_cfg = ckpt.get("cfg", None)
    data_cfg = ckpt_cfg.get("data", {}) if isinstance(ckpt_cfg, dict) else {}
    model_cfg = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}
    logging_cfg = ckpt_cfg.get("logging", {}) if isinstance(ckpt_cfg, dict) else {}

    buffer_path = buffer_path_override or data_cfg.get("buffer_path")
    stats_path = stats_path_override or data_cfg.get("stats_path")
    if not buffer_path:
        raise ValueError("buffer_path is required (ensure checkpoint cfg has data.buffer_path).")

    dataset = BufferSequenceDataset(
        buffer_path=Path(buffer_path),
        seq_len=seq_len,
        obs_keys=tuple(data_cfg.get("obs_keys", ("state",))),
        include_goals=bool(data_cfg.get("include_goals", False)),
        normalize_obs=bool(data_cfg.get("normalize_obs", False)),
        normalize_action=bool(data_cfg.get("normalize_action", False)),
        stats_path=Path(stats_path) if stats_path else None,
    )

    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim
    obs_window = int(model_cfg.get("obs_window", 1))

    vae = None
    rssm_obs_dim = obs_dim
    if bool(model_cfg.get("use_vae", True)):
        vae = VAE(
            obs_dim=obs_dim,
            latent_dim=int(model_cfg.get("vae_latent_dim", 16)),
            hidden_dim=int(model_cfg.get("vae_hidden_dim", 128)),
            min_std=float(model_cfg.get("vae_min_std", 0.1)),
        )
        rssm_obs_dim = int(model_cfg.get("vae_latent_dim", 16))

    model = RSSM(
        obs_dim=rssm_obs_dim,
        action_dim=action_dim,
        stoch_dim=int(model_cfg.get("stoch_dim", 32)),
        deter_dim=int(model_cfg.get("deter_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        obs_window=obs_window,
        min_std=float(model_cfg.get("min_std", 0.1)),
    )

    model.to(device)
    if vae is not None:
        vae.to(device)

    model.load_state_dict(ckpt["model"])
    if vae is not None and ckpt.get("vae") is not None:
        vae.load_state_dict(ckpt["vae"])
    model.eval()
    if vae is not None:
        vae.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    run = _init_wandb(
        enabled=wandb_enabled,
        project=logging_cfg.get("project", "world-model-eval"),
        entity=logging_cfg.get("entity", None),
        name=f"{training_run_name}-eval",
        tags=list(logging_cfg.get("tags", [])) if isinstance(logging_cfg.get("tags", None), list) else None,
        config={
            "eval": {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "max_batches": max_batches,
                "print_every": print_every,
                "device": str(device),
            },
            "checkpoint": str(checkpoint_path),
        },
    )

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
                obs_windowed = build_obs_window(obs_embed, obs_window)
                posterior_output = model(obs_windowed[:, 1:], actions, sample=False)

                # Posterior recon predicts obs[1:].
                post_obs = vae.decode(posterior_output.obs_pred_mean.reshape(-1, rssm_obs_dim)).view(
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
                obs_windowed = build_obs_window(obs, obs_window)
                posterior_output = model(obs_windowed[:, 1:], actions, sample=False)

                post_pred = posterior_output.obs_pred_mean
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

            if step % print_every == 0:
                msg = (
                    f"Step {step} | posterior_mse={metrics['eval/posterior_mse']:.6f} "
                    f"| prior_mse_uncond={metrics['eval/prior_mse_uncond']:.6f}"
                )
                if "eval/prior_mse_openloop" in metrics:
                    msg += f" | prior_mse_openloop={metrics['eval/prior_mse_openloop']:.6f}"
                print(msg)

            if plot_enabled and not plotted:
                # plot_post predicts obs[1:], plot_prior predicts obs[2:] (if open-loop) or obs[1:] (if uncond).
                post_err = _timewise_mse(plot_post, targets)
                prior_err = _timewise_mse(plot_prior, plot_targets)
                output_dir = Path(plot_output_dir)
                _plot_error_curves(
                    post_err,
                    "Posterior reconstruction error",
                    output_dir / f"{training_run_name}_{ckpt_step}_posterior_error.png",
                    plot_max_dims,
                )
                _plot_error_curves(
                    prior_err,
                    "Prior rollout error (open-loop if available)",
                    output_dir / f"{training_run_name}_{ckpt_step}_prior_error.png",
                    plot_max_dims,
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
