from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from data.buffer_dataset import BufferSequenceDataset
from models.rssm import RSSM
from models.vae import VAE
from utils import build_obs_window, mse_loss, set_seed


def _torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sanity checks for dataset/model wiring (shapes, alignment, action influence)."
    )
    p.add_argument("--buffer-path", type=str, required=True)
    p.add_argument("--stats-path", type=str, default="")
    p.add_argument("--checkpoint-path", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=32, help="Number of actions; observations will be seq_len + 1.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--use-vae", action="store_true")
    p.add_argument("--no-vae", dest="use_vae", action="store_false")
    p.set_defaults(use_vae=False)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = BufferSequenceDataset(
        buffer_path=Path(args.buffer_path),
        stats_path=Path(args.stats_path) if args.stats_path else None,
        seq_len=args.seq_len,
        obs_keys=("state",),
        include_goals=False,
        normalize_obs=False,
        normalize_action=False,
    )

    # Grab one batch (no DataLoader to keep dependencies minimal).
    batch = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
    obs = torch.stack([b["obs"] for b in batch], dim=0).to(device)  # (B, L+1, D)
    act = torch.stack([b["action"] for b in batch], dim=0).to(device)  # (B, L, A)

    print(f"obs shape: {tuple(obs.shape)}  action shape: {tuple(act.shape)}")
    assert obs.shape[1] == act.shape[1] + 1, "Expected obs length = action length + 1 (Dreamer-style alignment)."

    obs_dim = obs.shape[-1]
    action_dim = act.shape[-1]

    # Model hyperparams (prefer checkpoint cfg if provided).
    cfg: Dict[str, object] = {}
    if args.checkpoint_path:
        ckpt = _torch_load(Path(args.checkpoint_path), device=device)
        cfg = ckpt.get("cfg", {}) if isinstance(ckpt.get("cfg", {}), dict) else {}
        mcfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
        stoch_dim = int(mcfg.get("stoch_dim", 32))
        deter_dim = int(mcfg.get("deter_dim", 128))
        hidden_dim = int(mcfg.get("hidden_dim", 128))
        obs_window = int(mcfg.get("obs_window", 1))
        min_std = float(mcfg.get("min_std", 0.1))
        use_vae = bool(mcfg.get("use_vae", args.use_vae))
        vae_latent_dim = int(mcfg.get("vae_latent_dim", 16))
        vae_hidden_dim = int(mcfg.get("vae_hidden_dim", 128))
        vae_min_std = float(mcfg.get("vae_min_std", 0.1))
    else:
        stoch_dim, deter_dim, hidden_dim, obs_window, min_std = 32, 128, 128, 1, 0.1
        use_vae = args.use_vae
        vae_latent_dim, vae_hidden_dim, vae_min_std = 16, 128, 0.1

    vae = None
    rssm_obs_dim = obs_dim
    if use_vae:
        vae = VAE(
            obs_dim=obs_dim,
            latent_dim=vae_latent_dim,
            hidden_dim=vae_hidden_dim,
            min_std=vae_min_std,
        ).to(device)
        rssm_obs_dim = vae_latent_dim

    rssm = RSSM(
        obs_dim=rssm_obs_dim,
        action_dim=action_dim,
        stoch_dim=stoch_dim,
        deter_dim=deter_dim,
        hidden_dim=hidden_dim,
        obs_window=obs_window,
        min_std=min_std,
    ).to(device)

    if args.checkpoint_path:
        rssm.load_state_dict(ckpt["model"])
        if vae is not None and ckpt.get("vae") is not None:
            vae.load_state_dict(ckpt["vae"])

    rssm.eval()
    if vae is not None:
        vae.eval()

    with torch.no_grad():
        # Baseline: MSE if predicting zeros (helps interpret recon magnitudes).
        baseline = mse_loss(torch.zeros_like(obs[:, 1:]), obs[:, 1:])
        print(f"baseline mse (predict zeros for obs[1:]): {baseline.item():.6f}")

        # Baseline: copy previous observation (many physical systems are slow, so this can be strong).
        baseline_copy = mse_loss(obs[:, :-1], obs[:, 1:])
        print(f"baseline mse (copy obs[t] -> obs[t+1]): {baseline_copy.item():.6f}")

        act_mean_abs = act.abs().mean(dim=(0, 1))
        act_std = act.std(dim=(0, 1))
        print(f"action mean|a| (per-dim): {[round(x, 4) for x in act_mean_abs.detach().cpu().tolist()]}")
        print(f"action std (per-dim): {[round(x, 4) for x in act_std.detach().cpu().tolist()]}")

        if vae is not None:
            obs_flat = obs.reshape(-1, obs_dim)
            _, _, emb = vae.encode(obs_flat, sample=False)
            emb = emb.view(obs.shape[0], obs.shape[1], -1)
            windowed = build_obs_window(emb, obs_window)
            out = rssm(windowed[:, 1:], act, sample=False)

            pred_obs = vae.decode(out.obs_pred_mean.reshape(-1, rssm_obs_dim)).view(
                obs.shape[0], act.shape[1], obs_dim
            )
            post_mse = mse_loss(pred_obs, obs[:, 1:])
            print(f"posterior recon mse (obs space): {post_mse.item():.6f}")

            # Action influence check: zero actions should change predictions if actions matter.
            out_zero = rssm(windowed[:, 1:], torch.zeros_like(act), sample=False)
            pred_zero = vae.decode(out_zero.obs_pred_mean.reshape(-1, rssm_obs_dim)).view(
                obs.shape[0], act.shape[1], obs_dim
            )
            delta = mse_loss(pred_obs, pred_zero)
            print(f"mse(pred with real actions vs zero actions): {delta.item():.6f}")

            # Prior open-loop check (no observation conditioning).
            prior_lat = rssm.rollout_prior(act, sample=False)
            prior_obs = vae.decode(prior_lat.reshape(-1, rssm_obs_dim)).view(obs.shape[0], act.shape[1], obs_dim)
            prior_mse = mse_loss(prior_obs, obs[:, 1:])
            print(f"prior rollout mse (unconditioned): {prior_mse.item():.6f}")

            prior_lat_zero = rssm.rollout_prior(torch.zeros_like(act), sample=False)
            prior_obs_zero = vae.decode(prior_lat_zero.reshape(-1, rssm_obs_dim)).view(
                obs.shape[0], act.shape[1], obs_dim
            )
            prior_delta = mse_loss(prior_obs, prior_obs_zero)
            print(f"mse(prior with real actions vs zero actions): {prior_delta.item():.6f}")
        else:
            windowed = build_obs_window(obs, obs_window)
            out = rssm(windowed[:, 1:], act, sample=False)
            post_mse = mse_loss(out.obs_pred_mean, obs[:, 1:])
            print(f"posterior recon mse: {post_mse.item():.6f}")

            out_zero = rssm(windowed[:, 1:], torch.zeros_like(act), sample=False)
            delta = mse_loss(out.obs_pred_mean, out_zero.obs_pred_mean)
            print(f"mse(pred with real actions vs zero actions): {delta.item():.6f}")

            prior_pred = rssm.rollout_prior(act, sample=False)
            prior_mse = mse_loss(prior_pred, obs[:, 1:])
            print(f"prior rollout mse (unconditioned): {prior_mse.item():.6f}")

            prior_pred_zero = rssm.rollout_prior(torch.zeros_like(act), sample=False)
            prior_delta = mse_loss(prior_pred, prior_pred_zero)
            print(f"mse(prior with real actions vs zero actions): {prior_delta.item():.6f}")

        kl = RSSM.kl_loss(out.post_mean, out.post_std, out.prior_mean, out.prior_std).mean()
        print(f"mean KL(q||p): {kl.item():.6f}")


if __name__ == "__main__":
    main()
