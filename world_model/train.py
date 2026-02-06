from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import hydra
import torch
from data.buffer_dataset import BufferSequenceDataset, summarize_buffer
from models.rssm import RSSM
from models.vae import VAE
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from utils import TrainMetrics, build_obs_window, mse_loss, set_seed, to_float


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


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def _sanitize_run_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return "run"
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._") or "run"


def _extract_step(path: Path) -> int:
    name = path.stem
    if "ckpt_" not in name:
        return -1
    try:
        return int(name.split("ckpt_")[-1])
    except ValueError:
        return -1


def _prune_checkpoints(ckpt_root: Path, keep_last: int) -> None:
    if keep_last is None or keep_last <= 0:
        return
    ckpts = sorted(ckpt_root.glob("ckpt_*.ckpt"), key=_extract_step)
    if len(ckpts) <= keep_last:
        return
    for old in ckpts[:-keep_last]:
        old.unlink(missing_ok=True)


def _prepare_run_dir(cfg: DictConfig, buffer_path: Path, summary: dict) -> Path:
    """
    Create a run folder under repo-root checkpoints and write run metadata alongside checkpoints.

    Files written:
      - config.yaml: full resolved Hydra config
      - buffer_summary.yaml: quick buffer summary + buffer path
      - rollout_config.yaml: copy of cfg.data.stats_path (if provided)
    """
    import shutil

    run_name = _sanitize_run_name(str(getattr(cfg.logging, "name", "") or ""))
    run_dir = Path(hydra.utils.get_original_cwd()) / "checkpoints" / run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    # Full resolved config.
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(resolved, str(run_dir / "config.yaml"))

    # Buffer summary for quick sanity-checking.
    buf_meta = OmegaConf.create({"buffer_path": str(buffer_path), **summary})
    OmegaConf.save(buf_meta, str(run_dir / "buffer_summary.yaml"))

    # Copy buffer rollout/stats config if available.
    stats_path = getattr(cfg.data, "stats_path", None)
    if stats_path:
        src = Path(hydra.utils.to_absolute_path(str(stats_path)))
        if src.exists():
            shutil.copy2(src, run_dir / "rollout_config.yaml")

    return run_dir


def _compute_losses(
    model: RSSM,
    vae: VAE | None,
    batch: Dict[str, torch.Tensor],
    step: int,
    obs_window: int,
    rssm_sample: bool,
    kl_weight: float,
    kl_warmup_steps: int,
    kl_balance: bool,
    kl_balance_scale: float,
    free_nats: float,
    vae_recon_weight: float,
    vae_kl_weight: float,
    vae_kl_warmup_steps: int,
) -> tuple[TrainMetrics, torch.Tensor]:
    obs = batch["obs"]
    actions = batch["action"]

    # Dataset returns Dreamer-style sequences:
    # obs: (B, L+1, D), actions: (B, L, A) so that action[t] predicts obs[t+1].
    vae_recon = None
    vae_kl = None
    rssm_recon_latent = None
    if vae is not None:
        obs_flat = obs.reshape(-1, obs.shape[-1])
        # Train the VAE with stochastic latents, but feed a deterministic embedding (mean)
        # to the RSSM posterior to reduce target/embedding noise.
        vae_out = vae(obs_flat, sample=True)
        _, _, obs_latent_det = vae.encode(obs_flat, sample=False)
        obs_embed = obs_latent_det.view(obs.shape[0], obs.shape[1], -1)
        vae_recon = vae_out.recon.view_as(obs)
        vae_kl = vae_out.kl.mean()
        obs_for_rssm = obs_embed
    else:
        obs_for_rssm = obs

    # Posterior is conditioned on obs[t+1] (not obs[t]) to match Dreamer-style action alignment.
    obs_windowed = build_obs_window(obs_for_rssm, obs_window)
    output = model(obs_windowed[:, 1:], actions, sample=rssm_sample)
    if vae is not None:
        # RSSM predicts VAE latents; compute recon loss in observation space via the VAE decoder
        # (closer to Dreamer: the world model trains on reconstructing the actual observation).
        pred_latent = output.obs_pred_mean
        pred_obs = vae.decode(pred_latent.reshape(-1, pred_latent.shape[-1])).view(
            obs.shape[0], actions.shape[1], obs.shape[-1]
        )
        target_obs = obs[:, 1:]
        rssm_recon = mse_loss(pred_obs, target_obs)

        # Also log latent-space MSE for debugging (should go down if latents are stable).
        rssm_recon_latent = mse_loss(pred_latent, obs_for_rssm[:, 1:])
    else:
        target = obs_for_rssm[:, 1:]
        rssm_recon = mse_loss(output.obs_pred_mean, target)

    if kl_balance:
        # Balanced KL: combine two KL directions with stop-grad to stabilize training.
        alpha = float(kl_balance_scale)
        kl_lhs = model.kl_loss(
            output.post_mean.detach(),
            output.post_std.detach(),
            output.prior_mean,
            output.prior_std,
        )
        kl_rhs = model.kl_loss(
            output.post_mean,
            output.post_std,
            output.prior_mean.detach(),
            output.prior_std.detach(),
        )
        rssm_kl_raw = (alpha * kl_lhs + (1.0 - alpha) * kl_rhs).mean()
        if free_nats > 0:
            kl_lhs = torch.maximum(kl_lhs, kl_lhs.new_full(kl_lhs.shape, free_nats))
            kl_rhs = torch.maximum(kl_rhs, kl_rhs.new_full(kl_rhs.shape, free_nats))
        rssm_kl = (alpha * kl_lhs + (1.0 - alpha) * kl_rhs).mean()
    else:
        rssm_kl_per = model.kl_loss(output.post_mean, output.post_std, output.prior_mean, output.prior_std)
        rssm_kl_raw = rssm_kl_per.mean()
        if free_nats > 0:
            rssm_kl_per = torch.maximum(rssm_kl_per, torch.tensor(free_nats, device=rssm_kl_per.device))
        rssm_kl = rssm_kl_per.mean()

    # KL warmup (linear) so the model learns recon first.
    if kl_warmup_steps and kl_warmup_steps > 0:
        warm_frac = min(1.0, float(step) / float(kl_warmup_steps))
    else:
        warm_frac = 1.0
    kl_weight_eff = kl_weight * warm_frac

    loss = rssm_recon + kl_weight_eff * rssm_kl
    if vae is not None and vae_recon is not None and vae_kl is not None:
        if vae_kl_warmup_steps and vae_kl_warmup_steps > 0:
            vae_warm_frac = min(1.0, float(step) / float(vae_kl_warmup_steps))
        else:
            vae_warm_frac = 1.0
        vae_kl_weight_eff = vae_kl_weight * vae_warm_frac
        loss = loss + vae_recon_weight * mse_loss(vae_recon, obs) + vae_kl_weight_eff * vae_kl

    metrics = TrainMetrics(
        loss=to_float(loss),
        rssm_recon=to_float(rssm_recon),
        rssm_kl=to_float(rssm_kl),
        rssm_kl_raw=to_float(rssm_kl_raw),
        rssm_recon_latent=to_float(rssm_recon_latent) if rssm_recon_latent is not None else None,
        vae_recon=to_float(mse_loss(vae_recon, obs)) if vae_recon is not None else None,
        vae_kl=to_float(vae_kl) if vae_kl is not None else None,
    )
    return metrics, loss


@hydra.main(version_base=None, config_path="cfg", config_name="train_defualt")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.train.seed)

    buffer_path = Path(hydra.utils.to_absolute_path(cfg.data.buffer_path))
    stats_path = cfg.data.stats_path
    if stats_path:
        stats_path = str(Path(hydra.utils.to_absolute_path(stats_path)))

    summary = summarize_buffer(buffer_path)
    print("Buffer summary:", summary)
    ckpt_root = _prepare_run_dir(cfg, buffer_path=buffer_path, summary=summary)

    dataset = BufferSequenceDataset(
        buffer_path=buffer_path,
        seq_len=cfg.data.seq_len,
        obs_keys=cfg.data.obs_keys,
        include_goals=cfg.data.include_goals,
        normalize_obs=cfg.data.normalize_obs,
        normalize_action=cfg.data.normalize_action,
        stats_path=stats_path,
    )

    print(f"Dataset size: {len(dataset)} sequences")

    obs_dim = dataset.obs_dim
    print(f"Dataset obs_dim: {obs_dim}")
    action_dim = dataset.action_dim
    print(f"Dataset action_dim: {action_dim}")

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

    device = _resolve_device(cfg.train.device)
    model.to(device)
    if vae is not None:
        vae.to(device)
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        print(f"Using device: {device}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=True,
    )

    # Important: if using a VAE, optimize it jointly (otherwise its latents remain random).
    params = list(model.parameters())
    if vae is not None:
        params += list(vae.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.train.lr)
    run = _init_wandb(cfg)

    max_steps = int(getattr(cfg.train, "max_steps", 0) or 0)
    if max_steps <= 0:
        raise ValueError("cfg.train.max_steps must be > 0")

    save_every = int(cfg.train.save_every)
    if save_every <= 0:
        raise ValueError("cfg.train.save_every must be > 0 (steps).")
    keep_last = int(getattr(cfg.train, "keep_last", 0) or 0)
    if keep_last < 0:
        raise ValueError("cfg.train.keep_last must be >= 0")
    keep_last = int(getattr(cfg.train, "keep_last", 0) or 0)

    global_step = 0
    epoch = 0

    def _build_ckpt_payload(step_tag: int) -> dict:
        return {
            "model": model.state_dict(),
            "vae": vae.state_dict() if vae is not None else None,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "global_step": step_tag,
        }

    while global_step < max_steps:
        model.train()
        epoch_start = time.time()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            metrics, loss = _compute_losses(
                model,
                vae,
                batch,
                step=global_step,
                obs_window=cfg.model.obs_window,
                rssm_sample=bool(getattr(cfg.train, "rssm_sample", False)),
                kl_weight=cfg.train.kl_weight,
                kl_warmup_steps=int(getattr(cfg.train, "kl_warmup_steps", 0) or 0),
                kl_balance=bool(getattr(cfg.train, "kl_balance", False)),
                kl_balance_scale=float(getattr(cfg.train, "kl_balance_scale", 0.8)),
                free_nats=cfg.train.free_nats,
                vae_recon_weight=cfg.train.vae_recon_weight,
                vae_kl_weight=cfg.train.vae_kl_weight,
                vae_kl_warmup_steps=int(getattr(cfg.train, "vae_kl_warmup_steps", 0) or 0),
            )

            optimizer.zero_grad()
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            optimizer.step()

            if global_step % cfg.logging.log_every == 0:
                log_payload = {
                    "train/loss": metrics.loss,
                    "train/rssm_recon": metrics.rssm_recon,
                    "train/rssm_kl": metrics.rssm_kl,
                }
                if metrics.rssm_kl_raw is not None:
                    log_payload["train/rssm_kl_raw"] = metrics.rssm_kl_raw
                if metrics.rssm_recon_latent is not None:
                    log_payload["train/rssm_recon_latent"] = metrics.rssm_recon_latent
                if metrics.vae_recon is not None:
                    log_payload["train/vae_recon"] = metrics.vae_recon
                if metrics.vae_kl is not None:
                    log_payload["train/vae_kl"] = metrics.vae_kl
                _log_wandb(run, global_step, log_payload)
            if cfg.train.print_updates and global_step % cfg.train.print_every == 0:
                print(
                    f"Step {global_step}: loss={metrics.loss:.6f} "
                    f" rssm_recon={metrics.rssm_recon:.6f}  rssm_kl={metrics.rssm_kl:.6f}"
                )

            if (global_step + 1) % save_every == 0:
                step_tag = global_step + 1
                if step_tag < max_steps:
                    torch.save(_build_ckpt_payload(step_tag), ckpt_root / f"ckpt_{step_tag}.ckpt")
                if keep_last > 0:
                    # Keep only the most recent N checkpoints for this run.
                    ckpts = sorted(ckpt_root.glob("ckpt_*.ckpt"), key=_extract_step)
                    if len(ckpts) > keep_last:
                        for old in ckpts[: len(ckpts) - keep_last]:
                            old.unlink(missing_ok=True)
                _prune_checkpoints(ckpt_root, keep_last)

            global_step += 1
            if global_step >= max_steps:
                break

        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1} | time: {epoch_time:.2f}s | global_step={global_step}/{max_steps}")
        epoch += 1

    # Always save a final checkpoint for easy resume/eval.
    torch.save(_build_ckpt_payload(global_step), ckpt_root / "ckpt_latest.ckpt")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
