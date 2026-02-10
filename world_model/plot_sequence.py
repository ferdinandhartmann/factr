from __future__ import annotations

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

    fig = plt.figure(figsize=(15, 1.8 * (total_rows - (len(grouped_dims) - 1)) + 0.25 * (len(grouped_dims) - 1)))
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
    seq_len = 500
    episode_index = 0
    start_index = 10

    max_dims = 28
    dims = None
    seed = 7

    training_run_name = "rssm_newnorm_train_13"
    ckpt_step = "latest"  # number of ckpt step or "latest"

    checkpoint_path = Path(f"checkpoints/{training_run_name}/ckpt_{ckpt_step}.ckpt")

    output_path = Path(f"plots/{training_run_name}_{ckpt_step}_sequence.png")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ###############################################################

    set_seed(seed)
    device = torch.device(device)

    def _torch_load(path: Path) -> dict:
        # Prefer safe loading if supported by this PyTorch version.
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=device)

    if not checkpoint_path.exists():
        if ckpt_step is None:
            run_dir = Path("checkpoints") / training_run_name
            candidates = sorted(run_dir.glob("rssm_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
            if candidates:
                checkpoint_path = candidates[-1]
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = _torch_load(checkpoint_path)
    ckpt_cfg = ckpt.get("cfg", None)

    # Default to the checkpoint config to avoid subtle mismatches (e.g., include_goals changing obs_dim).
    data_cfg = ckpt_cfg.get("data", {}) if isinstance(ckpt_cfg, dict) else {}
    model_cfg = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}

    buffer_path = data_cfg.get("buffer_path")
    stats_path = data_cfg.get("stats_path")
    if not buffer_path:
        raise ValueError("buffer_path is required (ensure checkpoint cfg has data.buffer_path).")

    include_goals = bool(data_cfg.get("include_goals", False))
    normalize_obs = bool(data_cfg.get("normalize_obs", False))
    normalize_action = bool(data_cfg.get("normalize_action", False))
    use_vae = bool(model_cfg.get("use_vae", True))

    # Model hyperparams must match the checkpoint to load weights.
    vae_latent_dim = int(model_cfg.get("vae_latent_dim", 16))
    vae_hidden_dim = int(model_cfg.get("vae_hidden_dim", 128))
    vae_min_std = float(model_cfg.get("vae_min_std", 0.1))
    rssm_stoch_dim = int(model_cfg.get("stoch_dim", 32))
    rssm_deter_dim = int(model_cfg.get("deter_dim", 128))
    rssm_hidden_dim = int(model_cfg.get("hidden_dim", 128))
    rssm_min_std = float(model_cfg.get("min_std", 0.1))
    obs_window = int(model_cfg.get("obs_window", 1))

    obs_keys = ("state",)
    ok = data_cfg.get("obs_keys", None)
    if isinstance(ok, (list, tuple)) and all(isinstance(x, str) for x in ok):
        obs_keys = tuple(ok)

    dataset = BufferSequenceDataset(
        buffer_path=Path(buffer_path),
        seq_len=seq_len,
        obs_keys=obs_keys,
        include_goals=include_goals,
        normalize_obs=normalize_obs,
        normalize_action=normalize_action,
        stats_path=Path(stats_path) if stats_path else None,
    )

    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim
    rollout_cfg = _load_rollout_config(Path(stats_path) if stats_path else None)
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
            f"Ensure obs_keys/include_goals match training."
        )
    if ckpt_action_dim is not None and int(ckpt_action_dim) != action_dim:
        raise RuntimeError(
            f"Action dim mismatch: dataset action_dim={action_dim} but checkpoint action_dim={int(ckpt_action_dim)}."
        )

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

    model = RSSM(
        obs_dim=rssm_obs_dim,
        action_dim=action_dim,
        stoch_dim=rssm_stoch_dim,
        deter_dim=rssm_deter_dim,
        hidden_dim=rssm_hidden_dim,
        obs_window=obs_window,
        min_std=rssm_min_std,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    if vae is not None and ckpt.get("vae") is not None:
        vae.load_state_dict(ckpt["vae"])
    model.eval()
    if vae is not None:
        vae.eval()

    target_index = None
    if hasattr(dataset, "_index"):
        for i, (ep_idx, start) in enumerate(dataset._index):
            if ep_idx == episode_index and start == start_index:
                target_index = i
                break
    if target_index is None:
        raise IndexError(
            f"(episode_index={episode_index}, start_index={start_index}) not found for seq_len={seq_len}."
        )

    sample = dataset[target_index]
    obs = sample["obs"].to(device)
    actions = sample["action"].to(device)
    # Dataset returns: obs (L+1, D), actions (L, A). We predict obs[1:].
    true = obs[1:]
    with torch.no_grad():
        if vae is not None:
            obs_flat = obs.view(-1, obs.shape[-1])
            _, _, latent = vae.encode(obs_flat, sample=False)
            obs_embed = latent.view(1, obs.shape[0], -1)
            obs_windowed = build_obs_window(obs_embed, obs_window)
            posterior_out = model(obs_windowed[:, 1:], actions.unsqueeze(0), sample=False)
            post_latent = posterior_out.obs_pred_mean.squeeze(0)
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
            obs_windowed = build_obs_window(obs.unsqueeze(0), obs_window)
            posterior_out = model(obs_windowed[:, 1:], actions.unsqueeze(0), sample=False)
            post = posterior_out.obs_pred_mean.squeeze(0)

            prior_uncond = model.rollout_prior(actions.unsqueeze(0), sample=False).squeeze(0)
            if actions.shape[0] >= 2:
                h1 = posterior_out.deter_state[:, 0]
                z1 = posterior_out.stoch_state[:, 0]
                prior_open = model.rollout_prior(actions[1:].unsqueeze(0), h0=h1, z0=z1, sample=False).squeeze(0)
                prior = torch.cat([post[:1], prior_open], dim=0)
            else:
                prior = prior_uncond

    dims = _select_dims(obs_dim, dims, max_dims)
    _plot_sequences(true, post, prior, dims, output_path, state_labels, state_groups)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
