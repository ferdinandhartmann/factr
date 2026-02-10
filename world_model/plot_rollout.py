from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from models.rssm import RSSM
from models.vae import VAE
from utils import build_obs_window, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from factr.process_data.utils_data_process import downsample_data, lowpass_filter, medianfilter, sync_data_slowest


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


def _apply_group_norm_inplace(vec: np.ndarray, stats: Dict, offset: int = 0) -> None:
    groups = stats.get("groups", []) if isinstance(stats, dict) else []
    if not isinstance(groups, list):
        return
    for group in groups:
        if not isinstance(group, dict):
            continue
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = int(indices[0]) + offset, int(indices[1]) + offset
        sl = slice(start, stop)
        gtype = group.get("type", "identity")
        clip_val = group.get("clip", None)

        if gtype == "identity":
            continue
        if gtype == "min_max":
            mins = np.asarray(group.get("min", []), dtype=np.float32)
            maxs = np.asarray(group.get("max", []), dtype=np.float32)
            denom = maxs - mins
            denom[denom == 0] = 1e-17
            vec[sl] = (2.0 * (vec[sl] - mins) / denom) - 1.0
        elif gtype in ("gaussian", "zscore_clip"):
            mean = np.asarray(group.get("mean", []), dtype=np.float32)
            std = np.asarray(group.get("std", []), dtype=np.float32)
            std[std == 0] = 1e-17
            vec[sl] = (vec[sl] - mean) / std
        elif gtype == "log1p":
            x = vec[sl]
            vec[sl] = np.sign(x) * np.log1p(np.abs(x))
        elif gtype == "log1p_zscore_clip":
            x = vec[sl]
            x = np.sign(x) * np.log1p(np.abs(x))
            mean = np.asarray(group.get("mean", []), dtype=np.float32)
            std = np.asarray(group.get("std", []), dtype=np.float32)
            std[std == 0] = 1e-17
            vec[sl] = (x - mean) / std
        elif gtype == "fixed_scale_clip":
            scales = np.asarray(group.get("scales", []), dtype=np.float32)
            scales[scales == 0] = 1e-17
            vec[sl] = vec[sl] / scales

        if clip_val is not None:
            try:
                c = float(clip_val)
            except (TypeError, ValueError):
                c = None
            if c is not None and c > 0:
                vec[sl] = np.clip(vec[sl], -c, c)


def _normalize_matrix(data: np.ndarray, stats: Dict | None) -> np.ndarray:
    if data.size == 0 or not isinstance(stats, dict) or not stats:
        return data
    if stats.get("mode") == "grouped":
        out = data.copy()
        for i in range(out.shape[0]):
            _apply_group_norm_inplace(out[i], stats, offset=0)
        return out
    mean = stats.get("mean", None)
    std = stats.get("std", None)
    if mean is None or std is None:
        return data
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std[std == 0] = 1e-17
    if mean.shape[0] != data.shape[1]:
        return data
    return (data - mean) / std


STATE_TOPIC_SPECS = {
    "/cartesian_impedance_controller/ee_velocity": {"keys": ["ee_velocity"], "dim": 6, "fallback": "data"},
    "/cartesian_impedance_controller/pose_command": {"keys": ["ee_pose_commanded"], "dim": 9, "fallback": None},
    "/cartesian_impedance_controller/tracking_error": {"keys": ["tracking_error"], "dim": 6, "fallback": "data"},
    "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame": {
        "keys": ["external_wrench"],
        "dim": 6,
        "fallback": None,
    },
}

GOAL_TOPIC_SPECS = {
    "/goal": {"keys": ["goal"], "dim": 1, "fallback": None},
}

ACTION_TOPIC_SPECS = {
    "/cartesian_impedance_controller/pose_command": {"keys": ["ee_pose_commanded"], "fallback": None},
}


def _extract_fixed_vector(msg, keys, dim, fallback_key=None) -> np.ndarray:
    if isinstance(msg, dict):
        for k in keys:
            if k in msg and msg[k] is not None:
                arr = np.asarray(msg[k], dtype=float).flatten()
                if arr.size == dim:
                    return arr
        if fallback_key and fallback_key in msg and msg[fallback_key] is not None:
            arr = np.asarray(msg[fallback_key], dtype=float).flatten()
            if arr.size == dim:
                return arr
        return np.full((dim,), np.nan, dtype=float)
    arr = np.asarray(msg, dtype=float).flatten()
    if arr.size == dim:
        return arr
    return np.full((dim,), np.nan, dtype=float)


def _extract_topic_array(messages: list, spec: Dict | None = None) -> np.ndarray:
    if spec is not None:
        dim = int(spec.get("dim", 1))
        keys = spec.get("keys", [])
        fallback = spec.get("fallback", None)
        vectors = [_extract_fixed_vector(msg, keys, dim, fallback) for msg in messages]
        return np.stack(vectors, axis=0)

    topic_data = []
    max_len = 0
    for msg in messages:
        if isinstance(msg, dict):
            parts = []
            for key, value in msg.items():
                if value is not None and isinstance(value, (list, tuple, np.ndarray)):
                    parts.append(np.asarray(value, dtype=float).flatten())
            vec = np.concatenate(parts) if parts else np.zeros(1, dtype=float)
        else:
            vec = np.array(msg, dtype=float).flatten()
        topic_data.append(vec)
        max_len = max(max_len, len(vec))

    topic_padded = []
    for vec in topic_data:
        if len(vec) < max_len:
            vec = np.pad(vec, (0, max_len - len(vec)), constant_values=np.nan)
        topic_padded.append(vec)

    return np.stack(topic_padded, axis=0)


def _extract_ep_index(path: Path) -> int:
    name = path.stem
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return -1


def _load_episode_raw(path: Path) -> dict:
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def _sync_and_process(raw: dict, all_topics: List[str], processing: Dict | None) -> Tuple[dict, float]:
    if "timestamps" in raw and "data" in raw:
        traj_data, avg_freq = sync_data_slowest(raw, all_topics)
    else:
        traj_data = raw
        avg_freq = float(processing.get("data_frequency", 0.0)) if isinstance(processing, dict) else 0.0

    if not isinstance(processing, dict):
        return traj_data, avg_freq

    if processing.get("downsample"):
        if avg_freq <= 0:
            avg_freq = float(processing.get("data_frequency", 50.0))
        traj_data, avg_freq = downsample_data(
            traj_data, avg_freq, float(processing.get("target_downsampling_freq", 25.0))
        )

    return traj_data, avg_freq


def _build_state_action(
    traj_data: dict,
    state_topics: List[str],
    action_topics: List[str],
    action_dims: Dict[str, int],
    goal_topics: List[str],
    include_goals: bool,
    norm_stats: Dict | None,
) -> Tuple[np.ndarray, np.ndarray]:
    state_arrays = []
    for topic in state_topics:
        spec = STATE_TOPIC_SPECS.get(topic, None)
        state_arrays.append(_extract_topic_array(traj_data[topic], spec))
    states = np.concatenate(state_arrays, axis=-1)

    actions_arrays = []
    for topic in action_topics:
        spec = ACTION_TOPIC_SPECS.get(topic, None)
        if spec is not None:
            if topic in action_dims and int(action_dims[topic]) > 0:
                spec = dict(spec)
                spec["dim"] = int(action_dims[topic])
            else:
                spec = None
        actions_arrays.append(_extract_topic_array(traj_data[topic], spec))
    actions = np.concatenate(actions_arrays, axis=-1)

    min_len = min(states.shape[0], actions.shape[0])
    if include_goals and goal_topics:
        goal_arrays = []
        for topic in goal_topics:
            spec = GOAL_TOPIC_SPECS.get(topic, {"keys": ["goal"], "dim": 1, "fallback": None})
            goal_arrays.append(_extract_topic_array(traj_data[topic], spec))
        goals = np.concatenate(goal_arrays, axis=-1)
        min_len = min(min_len, goals.shape[0])
        states = states[:min_len]
        actions = actions[:min_len]
        goals = goals[:min_len]
    else:
        goals = None
        states = states[:min_len]
        actions = actions[:min_len]

    state_stats = norm_stats.get("state") if isinstance(norm_stats, dict) else None
    action_stats = norm_stats.get("action") if isinstance(norm_stats, dict) else None

    states = _normalize_matrix(states, state_stats)
    actions = _normalize_matrix(actions, action_stats)

    if goals is not None:
        states = np.concatenate([states, goals], axis=-1)

    return states, actions


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

    if state_dim >= 27:
        dims = list(range(27))
        return [
            ("EE Pose", dims[0:9]),
            ("EE Velocity", dims[9:15]),
            ("Tracking Error", dims[15:21]),
            ("External Wrench", dims[21:27]),
        ]
    return [("State", list(range(state_dim)))]


def _plot_rollout(
    obs: torch.Tensor,
    rollout: torch.Tensor,
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
            ax.plot(range(time_steps), rollout[:, dim].cpu().numpy(), label="rollout")
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
    fig.suptitle("True vs Open-loop Rollout", fontsize=14, y=0.995)
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
    
    data_dir = Path("~/activeinference/factr/process_data/data_to_process/fourgoals_1_test/data").expanduser()

    output_path = Path(f"plots/{training_run_name}_{ckpt_step}_rollout.png")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #########################################

    set_seed(seed)
    device = torch.device(device)

    def _torch_load(path: Path) -> dict:
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=device)

    ckpt = _torch_load(checkpoint_path)
    ckpt_cfg = ckpt.get("cfg", None)

    data_cfg = ckpt_cfg.get("data", {}) if isinstance(ckpt_cfg, dict) else {}
    model_cfg = ckpt_cfg.get("model", {}) if isinstance(ckpt_cfg, dict) else {}

    stats_path = data_cfg.get("stats_path")
    if not stats_path:
        raise ValueError("stats_path is required (ensure checkpoint cfg has data.stats_path).")

    include_goals = bool(data_cfg.get("include_goals", False))
    use_vae = bool(model_cfg.get("use_vae", True))

    vae_latent_dim = int(model_cfg.get("vae_latent_dim", 16))
    vae_hidden_dim = int(model_cfg.get("vae_hidden_dim", 128))
    vae_min_std = float(model_cfg.get("vae_min_std", 0.1))
    rssm_stoch_dim = int(model_cfg.get("stoch_dim", 32))
    rssm_deter_dim = int(model_cfg.get("deter_dim", 128))
    rssm_hidden_dim = int(model_cfg.get("hidden_dim", 128))
    rssm_min_std = float(model_cfg.get("min_std", 0.1))
    obs_window = int(model_cfg.get("obs_window", 1))

    rollout_cfg = _load_rollout_config(Path(stats_path) if stats_path else None)
    norm_stats = None
    if isinstance(rollout_cfg, dict):
        norm_stats = rollout_cfg.get("norm_stats", {}).get("state", None)

    if not isinstance(rollout_cfg, dict):
        raise ValueError("rollout_config.yaml could not be loaded; required for topic definitions and normalization.")

    obs_cfg = rollout_cfg.get("obs_config", {}) if isinstance(rollout_cfg, dict) else {}
    action_cfg = rollout_cfg.get("action_config", {}) if isinstance(rollout_cfg, dict) else {}
    processing_cfg = rollout_cfg.get("processing_config", {}) if isinstance(rollout_cfg, dict) else {}
    state_topics = list(obs_cfg.get("state_topics", []))
    goal_topics = list(obs_cfg.get("goal_topics", []))
    action_topics = list(action_cfg.keys())
    if not state_topics or not action_topics:
        raise ValueError("rollout_config.yaml must include obs_config.state_topics and action_config.")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    episodes = sorted(
        [p for p in data_dir.iterdir() if p.name.startswith("ep_") and p.suffix == ".pkl"], key=_extract_ep_index
    )
    if not episodes:
        raise FileNotFoundError(f"No episode files found in {data_dir}")

    ep_index = int(episode_index)
    if ep_index < 0 or ep_index >= len(episodes):
        raise IndexError(f"episode_index {ep_index} out of range (0..{len(episodes) - 1})")

    raw = _load_episode_raw(episodes[ep_index])
    all_topics = state_topics + action_topics + (goal_topics if include_goals else [])
    traj_data, _ = _sync_and_process(raw, all_topics, processing_cfg)

    states, actions = _build_state_action(
        traj_data,
        state_topics=state_topics,
        action_topics=action_topics,
        action_dims={k: int(v) for k, v in action_cfg.items()},
        goal_topics=goal_topics,
        include_goals=include_goals,
        norm_stats=rollout_cfg.get("norm_stats", {}) if isinstance(rollout_cfg, dict) else None,
    )

    if states.shape[0] < 2 or actions.shape[0] < 1:
        raise ValueError("Episode too short for rollout plotting (need at least 2 states and 1 action).")

    max_seq_len = max(1, states.shape[0] - 1)
    seq_len = min(int(seq_len), max_seq_len)
    start = int(start_index)
    if start < 0 or start + seq_len + 1 > states.shape[0]:
        raise IndexError(f"start_index {start} with seq_len {seq_len} exceeds episode length {states.shape[0]}.")

    obs_slice = states[start : start + seq_len + 1]
    actions_slice = actions[start : start + seq_len]

    obs_dim = obs_slice.shape[1]
    action_dim = actions_slice.shape[1]
    state_labels = _build_state_labels(obs_dim, norm_stats)
    state_groups = _build_state_groups(obs_dim, norm_stats)

    ckpt_obs_dim = ckpt.get("obs_dim", None)
    ckpt_action_dim = ckpt.get("action_dim", None)
    if ckpt_obs_dim is not None and int(ckpt_obs_dim) != obs_dim:
        raise RuntimeError(
            f"Observation dim mismatch: episode obs_dim={obs_dim} but checkpoint obs_dim={int(ckpt_obs_dim)}."
        )
    if ckpt_action_dim is not None and int(ckpt_action_dim) != action_dim:
        raise RuntimeError(
            f"Action dim mismatch: episode action_dim={action_dim} but checkpoint action_dim={int(ckpt_action_dim)}."
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

    obs = torch.from_numpy(obs_slice).float().to(device)
    actions = torch.from_numpy(actions_slice).float().to(device)
    true = obs[1:]
    with torch.no_grad():
        if vae is not None:
            obs_flat = obs.view(-1, obs.shape[-1])
            _, _, latent = vae.encode(obs_flat, sample=False)
            obs_embed = latent.view(1, obs.shape[0], -1)
            obs_windowed = build_obs_window(obs_embed, obs_window)
            posterior_out = model(obs_windowed[:, 1:2], actions[:1].unsqueeze(0), sample=False)
            post_latent = posterior_out.obs_pred_mean.squeeze(0)
            post_obs = vae.decode(post_latent.reshape(-1, rssm_obs_dim)).view(1, obs.shape[-1])

            if actions.shape[0] >= 2:
                h1 = posterior_out.deter_state[:, 0]
                z1 = posterior_out.stoch_state[:, 0]
                prior_latent = model.rollout_prior(actions[1:].unsqueeze(0), h0=h1, z0=z1, sample=False).squeeze(0)
                prior_obs = vae.decode(prior_latent.reshape(-1, rssm_obs_dim)).view(actions.shape[0] - 1, obs.shape[-1])
                rollout = torch.cat([post_obs, prior_obs], dim=0)
            else:
                rollout = post_obs
        else:
            obs_windowed = build_obs_window(obs.unsqueeze(0), obs_window)
            posterior_out = model(obs_windowed[:, 1:2], actions[:1].unsqueeze(0), sample=False)
            post_obs = posterior_out.obs_pred_mean.squeeze(0)
            if actions.shape[0] >= 2:
                h1 = posterior_out.deter_state[:, 0]
                z1 = posterior_out.stoch_state[:, 0]
                prior_obs = model.rollout_prior(actions[1:].unsqueeze(0), h0=h1, z0=z1, sample=False).squeeze(0)
                rollout = torch.cat([post_obs[:1], prior_obs], dim=0)
            else:
                rollout = post_obs[:1]

    dims = _select_dims(obs_dim, dims, max_dims)
    output_path = Path(output_path)
    _plot_rollout(true, rollout, dims, output_path, state_labels, state_groups)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
