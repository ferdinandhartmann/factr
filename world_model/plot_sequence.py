from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from data.buffer_dataset import BufferSequenceDataset
from models.rssm import RSSM
from models.vae import VAE
from utils import build_obs_window, set_seed


def _select_dims(total_dims: int, dims: List[int] | None, max_dims: int) -> List[int]:
    if dims:
        return [d for d in dims if 0 <= d < total_dims]
    return list(range(min(total_dims, max_dims)))


def _load_rollout_config(stats_path: Path | None) -> Dict | None:
    if not stats_path or not stats_path.exists():
        return None
    try:
        import yaml
    except ModuleNotFoundError:
        return None
    with stats_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_state_labels(state_dim: int, norm_stats: Dict | None) -> List[str]:
    labels = [f"{i + 1}" for i in range(state_dim)]
    if not norm_stats or norm_stats.get("mode") != "grouped":
        return labels

    name_map = {
        "ee_position": ["ee_pos_x", "ee_pos_y", "ee_pos_z"],
        "ee_orientation": [
            "ee_rot_c1_x",
            "ee_rot_c1_y",
            "ee_rot_c1_z",
            "ee_rot_c2_x",
            "ee_rot_c2_y",
            "ee_rot_c2_z",
        ],
        "ee_velocity": ["ee_vel_x", "ee_vel_y", "ee_vel_z", "ee_vel_rx", "ee_vel_ry", "ee_vel_rz"],
        "tracking_error": ["trk_x", "trk_y", "trk_z", "trk_rx", "trk_ry", "trk_rz"],
        "external_wrench": ["w_fx", "w_fy", "w_fz", "w_tx", "w_ty", "w_tz"],
    }

    for group in norm_stats.get("groups", []):
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = int(indices[0]), int(indices[1])
        group_name = group.get("name", "group")
        names = name_map.get(group_name, [])
        length = stop - start
        if len(names) != length:
            names = [f"{group_name}_{i}" for i in range(length)]
        for i in range(length):
            if 0 <= start + i < state_dim:
                labels[start + i] = names[i]
    return labels


def _build_state_groups(state_dim: int, norm_stats: Dict | None) -> List[Tuple[str, List[int]]]:
    if norm_stats and norm_stats.get("mode") == "grouped":
        title_map = {
            "ee_position": "EE Position",
            "ee_orientation": "EE Orientation",
            "ee_velocity": "EE Velocity",
            "tracking_error": "Tracking Error",
            "external_wrench": "External Wrench",
        }
        groups: List[Tuple[str, List[int]]] = []
        for group in norm_stats.get("groups", []):
            indices = group.get("indices", None)
            if not indices or len(indices) != 2:
                continue
            start, stop = int(indices[0]), int(indices[1])
            dims = [d for d in range(start, min(stop, state_dim))]
            if not dims:
                continue
            name = str(group.get("name", "group"))
            title = title_map.get(name, name.replace("_", " ").title())
            groups.append((title, dims))
        if groups:
            return groups

    # Fallback grouping: 9-6-6-6 if it matches the full state.
    if state_dim >= 27:
        dims = list(range(27))
        return [
            ("EE Pose", dims[0:9]),
            ("EE Velocity", dims[9:15]),
            ("Tracking Error", dims[15:21]),
            ("External Wrench", dims[21:27]),
        ]
    return [("State", list(range(state_dim)))]


def _plot_sequences(
    obs: torch.Tensor,
    post_pred: torch.Tensor,
    prior_pred: torch.Tensor,
    dims: List[int],
    output_path: Path,
    state_labels: List[str],
    state_groups: List[Tuple[str, List[int]]],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    time_steps = obs.shape[0]
    dims_set = set(dims)
    grouped_dims: List[Tuple[str, List[int]]] = []
    for title, group_dims in state_groups:
        selected = [d for d in group_dims if d in dims_set]
        if selected:
            grouped_dims.append((title, selected))
    if not grouped_dims:
        grouped_dims = [("State", dims)]

    total_rows = sum(len(gdims) for _, gdims in grouped_dims) + max(0, len(grouped_dims) - 1)
    height_ratios: List[float] = []
    for idx, (_, gdims) in enumerate(grouped_dims):
        height_ratios.extend([1.0] * len(gdims))
        if idx < len(grouped_dims) - 1:
            height_ratios.append(0.35)

    fig = plt.figure(figsize=(10, 1.8 * (total_rows - (len(grouped_dims) - 1)) + 0.25 * (len(grouped_dims) - 1)))
    gs = GridSpec(total_rows, 1, height_ratios=height_ratios, hspace=0.18)

    axes: List[plt.Axes] = []
    row = 0
    for group_index, (title, group_dims) in enumerate(grouped_dims):
        first_ax = None
        for dim in group_dims:
            ax = fig.add_subplot(gs[row, 0], sharex=axes[0] if axes else None)
            if first_ax is None:
                first_ax = ax
            ax.plot(range(time_steps), obs[:, dim].cpu().numpy(), label="true")
            ax.plot(range(time_steps), post_pred[:, dim].cpu().numpy(), label="posterior")
            ax.plot(range(time_steps), prior_pred[:, dim].cpu().numpy(), label="prior")
            ax.set_ylabel(state_labels[dim].replace("_", " "), fontsize=12)
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)
            axes.append(ax)
            row += 1
        if first_ax is not None:
            first_ax.set_title(title, loc="center", fontsize=13, pad=8)
        if group_dims:
            last_ax = axes[-1]
            last_ax.set_xlabel("Frame index", fontsize=10)
            for ax in axes[-len(group_dims) : -1]:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False, bottom=False)
            last_ax.tick_params(axis="x", labelbottom=True, bottom=True)
        if group_index < len(grouped_dims) - 1:
            spacer = fig.add_subplot(gs[row, 0])
            spacer.axis("off")
            row += 1

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout(pad=0.05)
    fig.subplots_adjust(top=0.985, bottom=0.04)
    fig.suptitle("True vs Predicted State Trajectories", fontsize=14, y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot true vs predicted trajectories for one sequence.")
    parser.add_argument(
        "--buffer-path",
        type=str,
        default="/home/ferdinand/activeinference/factr/process_data/training_data/fourgoals_1_newnorm_train/buf.pkl",
        help="Path to buf.pkl",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default="/home/ferdinand/activeinference/factr/process_data/training_data/fourgoals_1_newnorm_train/rollout_config.yaml",
        help="Path to rollout_config.yaml",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/rssm_newnorm_train_1/rssm_step_7000.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--seq-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-path", type=str, default="plots/sequence_plot.png")
    parser.add_argument("--max-dims", type=int, default=28)
    parser.add_argument("--dims", type=int, nargs="*", default=None)
    parser.add_argument("--include-goals", dest="include_goals", action="store_true")
    parser.add_argument("--no-include-goals", dest="include_goals", action="store_false")
    parser.add_argument("--normalize-obs", dest="normalize_obs", action="store_true")
    parser.add_argument("--no-normalize-obs", dest="normalize_obs", action="store_false")
    parser.add_argument("--normalize-action", dest="normalize_action", action="store_true")
    parser.add_argument("--no-normalize-action", dest="normalize_action", action="store_false")
    parser.add_argument("--use-vae", dest="use_vae", action="store_true")
    parser.add_argument("--no-vae", dest="use_vae", action="store_false")
    parser.add_argument("--vae-latent-dim", type=int, default=16)
    parser.add_argument("--vae-hidden-dim", type=int, default=128)
    parser.add_argument("--vae-min-std", type=float, default=0.1)
    parser.add_argument("--rssm-stoch-dim", type=int, default=32)
    parser.add_argument("--rssm-deter-dim", type=int, default=128)
    parser.add_argument("--rssm-hidden-dim", type=int, default=128)
    parser.add_argument("--rssm-min-std", type=float, default=0.1)
    parser.add_argument("--obs-window", type=int, default=1)
    # Use None defaults so we can pull the exact training settings from the checkpoint cfg.
    parser.set_defaults(include_goals=None, normalize_obs=None, normalize_action=None, use_vae=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    def _torch_load(path: Path) -> dict:
        # Prefer safe loading if supported by this PyTorch version.
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=device)

    ckpt = _torch_load(Path(args.checkpoint_path))
    ckpt_cfg = ckpt.get("cfg", None)

    # Default to the checkpoint config to avoid subtle mismatches (e.g., include_goals changing obs_dim).
    if isinstance(ckpt_cfg, dict):
        if args.include_goals is None:
            args.include_goals = bool(ckpt_cfg.get("data", {}).get("include_goals", False))
        if args.normalize_obs is None:
            args.normalize_obs = bool(ckpt_cfg.get("data", {}).get("normalize_obs", False))
        if args.normalize_action is None:
            args.normalize_action = bool(ckpt_cfg.get("data", {}).get("normalize_action", False))
        if args.use_vae is None:
            args.use_vae = bool(ckpt_cfg.get("model", {}).get("use_vae", False))

        # Model hyperparams must match the checkpoint to load weights.
        mcfg = ckpt_cfg.get("model", {})
        args.vae_latent_dim = int(mcfg.get("vae_latent_dim", args.vae_latent_dim))
        args.vae_hidden_dim = int(mcfg.get("vae_hidden_dim", args.vae_hidden_dim))
        args.vae_min_std = float(mcfg.get("vae_min_std", args.vae_min_std))
        args.rssm_stoch_dim = int(mcfg.get("stoch_dim", args.rssm_stoch_dim))
        args.rssm_deter_dim = int(mcfg.get("deter_dim", args.rssm_deter_dim))
        args.rssm_hidden_dim = int(mcfg.get("hidden_dim", args.rssm_hidden_dim))
        args.rssm_min_std = float(mcfg.get("min_std", args.rssm_min_std))
        args.obs_window = int(mcfg.get("obs_window", args.obs_window))

    # Fall back if still None (e.g., older checkpoint without cfg).
    if args.include_goals is None:
        args.include_goals = False
    if args.normalize_obs is None:
        args.normalize_obs = False
    if args.normalize_action is None:
        args.normalize_action = False
    if args.use_vae is None:
        args.use_vae = True

    obs_keys = ("state",)
    if isinstance(ckpt_cfg, dict):
        ok = ckpt_cfg.get("data", {}).get("obs_keys", None)
        if isinstance(ok, (list, tuple)) and all(isinstance(x, str) for x in ok):
            obs_keys = tuple(ok)

    dataset = BufferSequenceDataset(
        buffer_path=Path(args.buffer_path),
        seq_len=args.seq_len,
        obs_keys=obs_keys,
        include_goals=args.include_goals,
        normalize_obs=args.normalize_obs,
        normalize_action=args.normalize_action,
        stats_path=Path(args.stats_path) if args.stats_path else None,
    )

    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim
    rollout_cfg = _load_rollout_config(Path(args.stats_path) if args.stats_path else None)
    norm_stats = None
    if isinstance(rollout_cfg, dict):
        norm_stats = rollout_cfg.get("norm_stats", {}).get("state", None)
    state_labels = _build_state_labels(obs_dim, norm_stats)
    state_groups = _build_state_groups(obs_dim, norm_stats)
    ckpt_obs_dim = ckpt.get("obs_dim", None)
    ckpt_action_dim = ckpt.get("action_dim", None)
    if ckpt_obs_dim is not None and int(ckpt_obs_dim) != obs_dim:
        raise RuntimeError(
            f"Observation dim mismatch: dataset obs_dim={obs_dim} but checkpoint obs_dim={int(ckpt_obs_dim)}. "
            f"This is usually caused by include_goals/obs_keys differing from training. "
            f"Try rerunning with '--no-include-goals' (or ensure obs_keys match training)."
        )
    if ckpt_action_dim is not None and int(ckpt_action_dim) != action_dim:
        raise RuntimeError(
            f"Action dim mismatch: dataset action_dim={action_dim} but checkpoint action_dim={int(ckpt_action_dim)}."
        )

    vae = None
    rssm_obs_dim = obs_dim
    if args.use_vae:
        vae = VAE(
            obs_dim=obs_dim,
            latent_dim=args.vae_latent_dim,
            hidden_dim=args.vae_hidden_dim,
            min_std=args.vae_min_std,
        ).to(device)
        rssm_obs_dim = args.vae_latent_dim

    model = RSSM(
        obs_dim=rssm_obs_dim,
        action_dim=action_dim,
        stoch_dim=args.rssm_stoch_dim,
        deter_dim=args.rssm_deter_dim,
        hidden_dim=args.rssm_hidden_dim,
        obs_window=args.obs_window,
        min_std=args.rssm_min_std,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    if vae is not None and ckpt.get("vae") is not None:
        vae.load_state_dict(ckpt["vae"])
    model.eval()
    if vae is not None:
        vae.eval()

    sample = dataset[args.seq_index]
    obs = sample["obs"].to(device)
    actions = sample["action"].to(device)
    # Dataset returns: obs (L+1, D), actions (L, A). We predict obs[1:].
    true = obs[1:]
    with torch.no_grad():
        if vae is not None:
            obs_flat = obs.view(-1, obs.shape[-1])
            _, _, latent = vae.encode(obs_flat, sample=False)
            obs_embed = latent.view(1, obs.shape[0], -1)
            obs_windowed = build_obs_window(obs_embed, args.obs_window)
            posterior_out = model(obs_windowed[:, 1:], actions.unsqueeze(0), sample=False)
            post_latent = posterior_out.obs_pred.squeeze(0)
            post = vae.decode(post_latent.reshape(-1, rssm_obs_dim)).view(actions.shape[0], obs.shape[-1])

            # Prior rollout: prefer open-loop conditioned on the first posterior state.
            prior_latent_uncond = model.rollout_prior(actions.unsqueeze(0), sample=False).squeeze(0)
            prior_uncond = vae.decode(prior_latent_uncond.reshape(-1, rssm_obs_dim)).view(
                actions.shape[0], obs.shape[-1]
            )
            if actions.shape[0] >= 2:
                h1 = posterior_out.deter_state[:, 0]
                z1 = posterior_out.stoch_state[:, 0]
                prior_latent_open = model.rollout_prior(actions[1:].unsqueeze(0), h0=h1, z0=z1, sample=False).squeeze(0)
                prior_open = vae.decode(prior_latent_open.reshape(-1, rssm_obs_dim)).view(
                    actions.shape[0] - 1, obs.shape[-1]
                )
                prior = torch.cat([post[:1], prior_open], dim=0)
            else:
                prior = prior_uncond
        else:
            obs_windowed = build_obs_window(obs.unsqueeze(0), args.obs_window)
            posterior_out = model(obs_windowed[:, 1:], actions.unsqueeze(0), sample=False)
            post = posterior_out.obs_pred.squeeze(0)

            prior_uncond = model.rollout_prior(actions.unsqueeze(0), sample=False).squeeze(0)
            if actions.shape[0] >= 2:
                h1 = posterior_out.deter_state[:, 0]
                z1 = posterior_out.stoch_state[:, 0]
                prior_open = model.rollout_prior(actions[1:].unsqueeze(0), h0=h1, z0=z1, sample=False).squeeze(0)
                prior = torch.cat([post[:1], prior_open], dim=0)
            else:
                prior = prior_uncond

    dims = _select_dims(obs_dim, args.dims, args.max_dims)
    output_path = Path(args.output_path)
    _plot_sequences(true, post, prior, dims, output_path, state_labels, state_groups)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
