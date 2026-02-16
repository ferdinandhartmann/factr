#!/usr/bin/env python3
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# User Config (edit these variables, then run this script directly)
# ---------------------------------------------------------------------------
RUN_DIR = Path.home() / "activeinference" / "factr" / "checkpoints" / "ai_act_13_freebitsmediumsmall" / "rollout"
CHECKPOINT_NAME = "latest_ckpt.ckpt"

# Raw episode source
RAW_EPISODE_DIR = Path.home() / "activeinference" / "factr" / "process_data" / "data_to_process" / "fourgoals_1" / "data"
EPISODE_FILE_NAME = "ep_19_medium.pkl"
EPISODE_INDEX = 0  # index in sorted *.pkl files
USE_EPISODE_LIST = False
EPISODE_LIST = [
    # "ep_03_soft",
    "ep_09_stiff",
    "ep_09_soft",
    "ep_10_soft",
    "ep_14_medium",
    "ep_19_medium",
    "ep_23_stiff",
    "ep_29_medium",
    "ep_29_soft",
    "ep_33_medium",
    # "ep_39_soft",
    # "ep_40_soft",
]
LIST_EPISODES_ONLY = False

BUFFER_PATH_OVERRIDE = Path.home() / "activeinference" / "factr" / "process_data" / "processed_data" / "fourgoals_1_act" / "buf_test.pkl"
ROLLOUT_CONFIG_OVERRIDE = Path.home() / "activeinference" / "factr" / "process_data" / "processed_data" / "fourgoals_1_act" / "rollout_config.yaml"

NUM_SAMPLES = 10
NORMALIZATION_MODE = "apply"  # one of: auto, apply, skip
PREDICTION_STRIDE = 50  # stride for fan plot + 3d plot
SAMPLE_ANCHOR_STEP = -1  # -1 means middle step
VIEW_ELEV = 24
VIEW_AZIM = -60
SHOW_PLOT = False  # shows also 3d plot
ENABLE_TRAIN_BACKGROUND = True
TRAIN_BACKGROUND_MAX_TRAJ = 200
TRAIN_BACKGROUND_ONLY_MEDIUM = True

GLOBAL_AXIS_LIMITS = {
    "x": (0.2, 0.6),
    "y": (-0.4, 0.4),
    "z": (0.0, 0.7),
}

GOAL_FRAMES = [
    {"name": "goal 1", "pose": [0.384, -0.26, 0.181, 0.998, 0.014, 0.065, 0.011, -0.999, 0.05]},
    {"name": "goal 2", "pose": [0.656, -0.155, -0.023, 0.996, -0.054, 0.065, -0.057, -0.997, 0.045]},
    {"name": "goal 3", "pose": [0.51, 0.269, -0.028, -0.015, 1.0, -0.027, 1.0, 0.015, 0.013]},
    {"name": "goal 4", "pose": [0.466, 0.361, 0.48, 0.998, 0.056, 0.032, 0.034, -0.034, -0.999]},
]

OUT_DIR_OVERRIDE = None

if SHOW_PLOT:
    plt.switch_backend("TkAgg")


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("len"):
        OmegaConf.register_new_resolver("len", lambda x: len(x))


def _load_run_cfg(exp_config_path: Path):
    cfg_all = OmegaConf.load(exp_config_path)
    cfg = cfg_all.params if "params" in cfg_all else cfg_all
    _register_resolvers()
    if "hydra" in cfg:
        cfg.pop("hydra", None)
    OmegaConf.resolve(cfg)
    return cfg


def _load_model(cfg, ckpt_path: Path, device: torch.device):
    model = instantiate(cfg.agent)
    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def _load_rollout_config(rollout_config_path: Path):
    if not rollout_config_path.exists():
        raise FileNotFoundError(f"rollout_config.yaml not found: {rollout_config_path}")
    with open(rollout_config_path, "r") as f:
        return yaml.safe_load(f)


def _get_split_label(rollout_cfg: Dict, episode_name: str) -> str:
    split_cfg = rollout_cfg.get("split_config", {}) if isinstance(rollout_cfg, dict) else {}
    train_eps = set(split_cfg.get("train_episodes", []) or [])
    test_eps = set(split_cfg.get("test_episodes", []) or [])
    if episode_name in test_eps:
        return "test"
    if episode_name in train_eps:
        return "train"
    return "unknown"


def _resolve_train_buffer_path(rollout_cfg: Dict, buffer_path: Path) -> Path:
    split_cfg = rollout_cfg.get("split_config", {}) if isinstance(rollout_cfg, dict) else {}
    train_name = split_cfg.get("train_buffer") or split_cfg.get("train_buffer_name") or "buf_train.pkl"
    train_name = str(train_name)
    if not train_name.endswith(".pkl"):
        train_name = f"{train_name}.pkl"
    train_path = Path(train_name)
    if train_path.is_absolute():
        return train_path
    return buffer_path.parent / train_path


def _load_train_buffer_actions(buf_path: Path) -> List[np.ndarray]:
    import pickle

    with open(buf_path, "rb") as f:
        buffer = pickle.load(f)

    actions_list = []
    if isinstance(buffer, (list, tuple)) and buffer and isinstance(buffer[0], (list, tuple)):
        for traj in buffer:
            if not traj or not isinstance(traj[0], tuple):
                continue
            actions = []
            for entry in traj:
                try:
                    _, action, _ = entry
                    actions.append(np.asarray(action, dtype=np.float32))
                except Exception:
                    continue
            if actions:
                actions_list.append(np.stack(actions, axis=0))
    return actions_list


def _extract_ep_index(path: Path) -> Tuple[int, str]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "ep" and parts[1].isdigit():
        return int(parts[1]), stem
    return (10**9, stem)


def _list_episode_files(raw_episode_dir: Path) -> List[Path]:
    files = sorted(raw_episode_dir.glob("*.pkl"), key=_extract_ep_index)
    return files


def _select_episode_file(episode_files: List[Path], episode_file_name, episode_index: int) -> Path:
    if len(episode_files) == 0:
        raise ValueError("No episode .pkl files found in RAW_EPISODE_DIR.")

    if episode_file_name:
        target = [p for p in episode_files if p.name == str(episode_file_name)]
        if len(target) == 0:
            raise ValueError(f"EPISODE_FILE_NAME '{episode_file_name}' not found")
        return target[0]

    if episode_index < 0 or episode_index >= len(episode_files):
        raise ValueError(f"EPISODE_INDEX must be in [0, {len(episode_files) - 1}], got {episode_index}")
    return episode_files[episode_index]


def _extract_fixed_vector(msg, keys, dim, fallback_key=None):
    if isinstance(msg, dict):
        for key in keys:
            if key in msg and msg[key] is not None:
                arr = np.asarray(msg[key], dtype=np.float32).reshape(-1)
                if arr.size == int(dim):
                    return arr
        if fallback_key and fallback_key in msg and msg[fallback_key] is not None:
            arr = np.asarray(msg[fallback_key], dtype=np.float32).reshape(-1)
            if arr.size == int(dim):
                return arr
        return np.full((int(dim),), np.nan, dtype=np.float32)

    arr = np.asarray(msg, dtype=np.float32).reshape(-1)
    if arr.size == int(dim):
        return arr
    return np.full((int(dim),), np.nan, dtype=np.float32)


def _extract_fallback_vector(msg):
    if isinstance(msg, dict):
        parts = []
        for value in msg.values():
            if value is not None and isinstance(value, (list, tuple, np.ndarray)):
                parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
        if len(parts) > 0:
            return np.concatenate(parts, axis=0)
    return np.asarray(msg, dtype=np.float32).reshape(-1)


def _sync_data_slowest(raw_data, topics: List[str]):
    if "data" not in raw_data or "timestamps" not in raw_data:
        raise ValueError("Raw episode file must contain 'data' and 'timestamps'.")

    data = raw_data["data"]
    timestamps = raw_data["timestamps"]
    missing = [t for t in topics if t not in data or t not in timestamps]
    if len(missing) > 0:
        raise ValueError(f"Missing topics in raw episode file: {missing}")

    ts_arrays = {topic: np.asarray(timestamps[topic], dtype=np.int64) for topic in topics}
    lengths = {topic: len(ts_arrays[topic]) for topic in topics}
    min_topic = min(lengths.keys(), key=lambda key: lengths[key])
    target_ts = ts_arrays[min_topic]

    synced = {topic: [] for topic in topics}
    for ts in target_ts:
        for topic in topics:
            topic_ts = ts_arrays[topic]
            idx = int(np.argmin(np.abs(topic_ts - ts)))
            synced[topic].append(data[topic][idx])
    return synced


def _stiffness_vec_to_class(stiffness_vec, thresholds: List[float]) -> int:
    norm = float(np.linalg.norm(np.asarray(stiffness_vec, dtype=np.float32)))
    if not np.isfinite(norm):
        return 1
    for idx, threshold in enumerate(sorted(float(v) for v in thresholds), start=1):
        if norm < float(threshold):
            return idx
    return len(thresholds) + 1


def _load_raw_episode_to_arrays(episode_file: Path, rollout_cfg) -> Dict:
    import pickle

    with open(episode_file, "rb") as f:
        raw_data = pickle.load(f)

    obs_cfg = rollout_cfg.get("obs_config", {})
    action_cfg = rollout_cfg.get("action_config", {})
    state_topics = list(obs_cfg.get("state_topics", []))
    if len(state_topics) == 0:
        raise ValueError("No state_topics found in rollout_config.")
    action_topics = list(action_cfg.keys())
    if len(action_topics) == 0:
        raise ValueError("No action topics found in rollout_config action_config.")
    if len(action_topics) != 1:
        raise ValueError(f"Expected one action topic, got {action_topics}")
    action_topic = action_topics[0]
    action_dim = int(action_cfg[action_topic])

    stiffness_info = obs_cfg.get("stiffness_label", {})
    stiffness_topic = stiffness_info.get("topic", None)
    stiffness_key = stiffness_info.get("key", "stiffness")
    stiffness_thresholds = stiffness_info.get("norm_thresholds", [200.0, 1000.0])

    topics_for_sync = list(state_topics) + [action_topic]
    if stiffness_topic:
        topics_for_sync.append(stiffness_topic)
    synced = _sync_data_slowest(raw_data, topics_for_sync)

    state_specs = {
        "/franka_robot_state_broadcaster/robot_state": {"keys": ["ee_pose"], "dim": 9, "fallback": "data"},
        "/cartesian_impedance_controller/ee_velocity": {"keys": ["ee_velocity"], "dim": 6, "fallback": "data"},
        "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame": {
            "keys": ["external_wrench"],
            "dim": 6,
            "fallback": "data",
        },
        "/cartesian_impedance_controller/tracking_error": {"keys": ["tracking_error"], "dim": 6, "fallback": "data"},
    }
    action_specs = {
        "/cartesian_impedance_controller/pose_command": {
            "keys": ["ee_pose_commanded"],
            "dim": action_dim,
            "fallback": None,
        }
    }

    state_arrays = []
    for topic in state_topics:
        if topic in state_specs:
            spec = state_specs[topic]
            vecs = [_extract_fixed_vector(msg, spec["keys"], spec["dim"], spec.get("fallback", None)) for msg in synced[topic]]
            topic_arr = np.stack(vecs, axis=0)
        else:
            vecs = [_extract_fallback_vector(msg) for msg in synced[topic]]
            max_dim = max(v.shape[0] for v in vecs)
            padded = []
            for v in vecs:
                if v.shape[0] < max_dim:
                    v = np.pad(v, (0, max_dim - v.shape[0]), constant_values=np.nan)
                padded.append(v)
            topic_arr = np.stack(padded, axis=0)
        state_arrays.append(topic_arr)
    states = np.concatenate(state_arrays, axis=-1).astype(np.float32)

    if action_topic in action_specs:
        spec = action_specs[action_topic]
        action_vecs = [_extract_fixed_vector(msg, spec["keys"], spec["dim"], spec.get("fallback", None)) for msg in synced[action_topic]]
        actions = np.stack(action_vecs, axis=0).astype(np.float32)
    else:
        action_vecs = [_extract_fallback_vector(msg) for msg in synced[action_topic]]
        actions = np.stack(action_vecs, axis=0).astype(np.float32)
        if actions.shape[1] != action_dim:
            raise ValueError(f"Action dim mismatch for {action_topic}: expected {action_dim}, got {actions.shape[1]}")

    num_steps = int(min(states.shape[0], actions.shape[0]))
    states = states[:num_steps]
    actions = actions[:num_steps]

    if stiffness_topic:
        raw_stiff = synced[stiffness_topic][:num_steps]
        stiff_vecs = [_extract_fixed_vector(msg, [stiffness_key], 6, None) for msg in raw_stiff]
        stiff_labels = np.asarray([_stiffness_vec_to_class(v, stiffness_thresholds) for v in stiff_vecs], dtype=np.int64)
        episode_label = int(stiff_labels[0]) if len(stiff_labels) > 0 else 1
    else:
        episode_label = 1

    return {"states": states, "actions": actions, "episode_label": int(episode_label), "num_steps": int(num_steps)}


def _build_eval_samples_from_raw_episode(states, actions, episode_label: int, obs_window: int, ac_chunk: int, action_index_offset: int):
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"Expected 2D states/actions, got states={states.shape} actions={actions.shape}")

    max_t = states.shape[0] - int(action_index_offset)
    if max_t <= 0:
        raise ValueError("Episode is too short for given action_index_offset.")

    obs_arr, action_arr, mask_arr, labels_arr, steps_arr = [], [], [], [], []
    for t_idx in range(max_t):
        start = max(0, t_idx - int(obs_window) + 1)
        window_states = [states[i] for i in range(start, t_idx + 1)]
        while len(window_states) < int(obs_window):
            window_states.insert(0, window_states[0])
        obs_window_arr = np.stack(window_states, axis=0).astype(np.float32)

        chunk_actions = []
        chunk_mask = []
        for k in range(int(ac_chunk)):
            idx = t_idx + int(action_index_offset) + k
            if idx < actions.shape[0]:
                chunk_actions.append(actions[idx])
                chunk_mask.append(1.0)
            else:
                chunk_actions.append(chunk_actions[-1] if len(chunk_actions) > 0 else actions[-1])
                chunk_mask.append(0.0)
        chunk_actions = np.stack(chunk_actions, axis=0).astype(np.float32)
        chunk_mask = np.asarray(chunk_mask, dtype=np.float32)
        chunk_mask = np.repeat(chunk_mask[:, None], chunk_actions.shape[-1], axis=1)

        obs_arr.append(obs_window_arr)
        action_arr.append(chunk_actions)
        mask_arr.append(chunk_mask)
        labels_arr.append(int(episode_label))
        steps_arr.append(int(t_idx))

    return {
        "obs": np.stack(obs_arr, axis=0).astype(np.float32),
        "actions": np.stack(action_arr, axis=0).astype(np.float32),
        "mask": np.stack(mask_arr, axis=0).astype(np.float32),
        "labels": np.asarray(labels_arr, dtype=np.int64),
        "steps": np.asarray(steps_arr, dtype=np.int64),
    }


def _safe_denominator(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr[np.abs(arr) < 1e-12] = 1e-12
    return arr


def _forward_group_transform(values: np.ndarray, group: Dict) -> np.ndarray:
    gtype = group.get("type", "identity")
    if gtype in ("identity",):
        return values
    if gtype in ("gaussian", "gaussian_clip", "zscore_clip"):
        mean = np.asarray(group.get("mean", []), dtype=np.float32)
        std = _safe_denominator(group.get("std", []))
        out = (values - mean) / std
        clip = group.get("clip", None)
        if clip is not None:
            c = float(clip)
            out = np.clip(out, -c, c)
        return out
    if gtype in ("min_max",):
        mins = np.asarray(group.get("min", []), dtype=np.float32)
        maxs = np.asarray(group.get("max", []), dtype=np.float32)
        denom = _safe_denominator(maxs - mins)
        out = (2.0 * (values - mins) / denom) - 1.0
        clip = group.get("clip", None)
        if clip is not None:
            c = float(clip)
            out = np.clip(out, -c, c)
        return out
    if gtype in ("fixed_scale", "fixed_scale_clip"):
        scales = _safe_denominator(group.get("scales", []))
        out = values / scales
        clip = group.get("clip", None)
        if clip is not None:
            c = float(clip)
            out = np.clip(out, -c, c)
        return out
    if gtype == "log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    if gtype == "log1p_zscore_clip":
        mean = np.asarray(group.get("mean", []), dtype=np.float32)
        std = _safe_denominator(group.get("std", []))
        out = np.sign(values) * np.log1p(np.abs(values))
        out = (out - mean) / std
        clip = group.get("clip", None)
        if clip is not None:
            c = float(clip)
            out = np.clip(out, -c, c)
        return out
    return values


def _inverse_group_transform(values: np.ndarray, group: Dict) -> np.ndarray:
    gtype = group.get("type", "identity")
    if gtype in ("identity",):
        return values
    if gtype in ("gaussian", "gaussian_clip", "zscore_clip"):
        mean = np.asarray(group.get("mean", []), dtype=np.float32)
        std = _safe_denominator(group.get("std", []))
        return values * std + mean
    if gtype in ("min_max",):
        mins = np.asarray(group.get("min", []), dtype=np.float32)
        maxs = np.asarray(group.get("max", []), dtype=np.float32)
        return (values + 1.0) * 0.5 * (maxs - mins) + mins
    if gtype in ("fixed_scale", "fixed_scale_clip"):
        scales = _safe_denominator(group.get("scales", []))
        return values * scales
    if gtype == "log1p":
        return np.sign(values) * np.expm1(np.abs(values))
    if gtype == "log1p_zscore_clip":
        mean = np.asarray(group.get("mean", []), dtype=np.float32)
        std = _safe_denominator(group.get("std", []))
        out = values * std + mean
        return np.sign(out) * np.expm1(np.abs(out))
    return values


def _apply_grouped_transform(values: np.ndarray, stats: Dict, inverse: bool = False) -> np.ndarray:
    arr = values.copy()
    if not stats:
        return arr

    mode = stats.get("mode", None)
    if mode != "grouped":
        if (not inverse) and "mean" in stats and "std" in stats:
            mean = np.asarray(stats.get("mean", []), dtype=np.float32)
            std = _safe_denominator(stats.get("std", []))
            if mean.size == arr.shape[-1] and std.size == arr.shape[-1]:
                return (arr - mean) / std
        if inverse and "mean" in stats and "std" in stats:
            mean = np.asarray(stats.get("mean", []), dtype=np.float32)
            std = _safe_denominator(stats.get("std", []))
            if mean.size == arr.shape[-1] and std.size == arr.shape[-1]:
                return arr * std + mean
        return arr

    for group in stats.get("groups", []):
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = int(indices[0]), int(indices[1])
        sl = slice(start, stop)
        if inverse:
            arr[..., sl] = _inverse_group_transform(arr[..., sl], group)
        else:
            arr[..., sl] = _forward_group_transform(arr[..., sl], group)
    return arr


def _detect_already_normalized(values: np.ndarray, stats: Dict) -> bool:
    if not stats or stats.get("mode", None) != "grouped":
        return True

    gaussian_groups = []
    for group in stats.get("groups", []):
        if group.get("type", "identity") in ("gaussian", "gaussian_clip", "zscore_clip"):
            indices = group.get("indices", None)
            if indices and len(indices) == 2:
                gaussian_groups.append((int(indices[0]), int(indices[1]), group))
    if len(gaussian_groups) == 0:
        return True

    score_as_is = []
    score_if_norm = []
    arr = values.reshape(-1, values.shape[-1])
    for start, stop, group in gaussian_groups:
        part = arr[:, start:stop]
        if part.size == 0:
            continue
        as_is_mean = np.mean(part, axis=0)
        as_is_std = np.std(part, axis=0) + 1e-8
        score_as_is.append(float(np.mean(np.abs(as_is_mean)) + np.mean(np.abs(as_is_std - 1.0))))

        normed = _forward_group_transform(part, group)
        norm_mean = np.mean(normed, axis=0)
        norm_std = np.std(normed, axis=0) + 1e-8
        score_if_norm.append(float(np.mean(np.abs(norm_mean)) + np.mean(np.abs(norm_std - 1.0))))

    if len(score_as_is) == 0:
        return True
    return float(np.mean(score_as_is)) <= float(np.mean(score_if_norm))


def _ensure_normalized(values: np.ndarray, stats: Dict, mode: str, name: str) -> Tuple[np.ndarray, bool]:
    if mode == "skip":
        print(f"[normalize] {name}: skip")
        return values.copy(), False
    if mode == "apply":
        print(f"[normalize] {name}: apply")
        return _apply_grouped_transform(values, stats, inverse=False), True

    already = _detect_already_normalized(values, stats)
    if already:
        print(f"[normalize] {name}: auto -> already normalized, skip")
        return values.copy(), False
    print(f"[normalize] {name}: auto -> apply normalization from rollout_config")
    return _apply_grouped_transform(values, stats, inverse=False), True


def _make_pose_dim_names(dim: int) -> List[str]:
    default_names = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6"]
    if dim <= len(default_names):
        return default_names[:dim]
    return [f"dim{i + 1}" for i in range(dim)]


def _build_pose_figure_with_measured(true_actions, pred_actions, measured_pose, mask, title):
    valid_rows = mask[:, 0] > 0
    if np.sum(valid_rows) < 2:
        return None

    true_valid = true_actions[valid_rows]
    pred_valid = pred_actions[valid_rows]
    meas_valid = measured_pose[valid_rows]
    time_index = np.arange(true_valid.shape[0])

    pose_dim = true_valid.shape[1]
    dim_names = _make_pose_dim_names(pose_dim)
    n_cols = min(3, pose_dim)
    n_rows = int(np.ceil(pose_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.8 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for dim in range(pose_dim):
        ax = axes[dim]
        ax.plot(time_index, true_valid[:, dim], color="black", linewidth=1.0, label="ground truth" if dim == 0 else None)
        ax.plot(time_index, pred_valid[:, dim], color="#E41A1C", linewidth=1.4, alpha=1.0, label="prediction" if dim == 0 else None)
        ax.plot(time_index, meas_valid[:, dim], color="#1f78b4", linestyle="--", linewidth=1.0, alpha=0.6, label="measured_pose" if dim == 0 else None)
        ax.set_title(dim_names[dim])
        ax.grid(alpha=0.25)

    for ax in axes[pose_dim:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.92))
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.9])
    return fig


def _build_fan_figure_with_measured(
    true_action_chunks: np.ndarray,
    pred_action_chunks: np.ndarray,
    measured_pose: np.ndarray,
    mask_chunks: np.ndarray,
    source_time_index: np.ndarray,
    prediction_stride: int,
    background_actions: Optional[List[np.ndarray]] = None,
):
    anchor_steps = int(
        min(
            true_action_chunks.shape[0],
            pred_action_chunks.shape[0],
            mask_chunks.shape[0],
            measured_pose.shape[0],
            source_time_index.shape[0],
        )
    )
    if anchor_steps < 1:
        return None

    pose_dim = int(true_action_chunks.shape[-1])
    n_samples = int(pred_action_chunks.shape[1])
    anchor_idx = np.arange(0, anchor_steps, max(1, int(prediction_stride)), dtype=np.int64)
    if anchor_idx[-1] != (anchor_steps - 1):
        anchor_idx = np.concatenate([anchor_idx, np.asarray([anchor_steps - 1], dtype=np.int64)])
    anchor_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, max(1, len(anchor_idx))))

    dim_names = _make_pose_dim_names(pose_dim)
    n_cols = min(3, pose_dim)
    n_rows = int(np.ceil(pose_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    episode_x_max = int(source_time_index[anchor_steps - 1] + true_action_chunks.shape[1] - 1)
    if episode_x_max < 1:
        episode_x_max = int(anchor_steps)

    for dim in range(pose_dim):
        ax = axes[dim]

        if background_actions:
            start_t = int(source_time_index[0]) if len(source_time_index) > 0 else 0
            max_len = max(0, episode_x_max - start_t + 1)
            for traj in background_actions:
                if traj.ndim != 2 or traj.shape[1] < pose_dim:
                    continue
                use_len = min(int(traj.shape[0]), int(max_len))
                if use_len < 2:
                    continue
                x_vals = start_t + np.arange(use_len)
                ax.plot(
                    x_vals,
                    traj[:use_len, dim],
                    color="#BDBDBD",
                    linewidth=0.6,
                    alpha=0.2,
                )

        ax.plot(
            source_time_index[:anchor_steps],
            true_action_chunks[:anchor_steps, 0, dim],
            color="black",
            linewidth=1.0,
            label="ground_truth" if dim == 0 else None,
        )
        ax.plot(
            source_time_index[:anchor_steps],
            measured_pose[:anchor_steps, dim],
            color="#1f78b4",
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label="measured_pose" if dim == 0 else None,
        )

        for anchor_pos, t in enumerate(anchor_idx):
            valid_h = mask_chunks[t, :, 0] > 0
            if not np.any(valid_h):
                continue
            c_t = anchor_colors[anchor_pos]
            base_t = int(source_time_index[t])
            horizon_idx = np.where(valid_h)[0]
            x_vals = base_t + horizon_idx
            within = x_vals <= episode_x_max
            if not np.any(within):
                continue
            x_vals = x_vals[within]
            h_idx = horizon_idx[within]
            for s_idx in range(n_samples):
                ax.plot(
                    x_vals,
                    pred_action_chunks[t, s_idx, h_idx, dim],
                    color=c_t,
                    linewidth=0.8,
                    alpha=0.7,
                    label="prior_samples" if (t == anchor_idx[0] and s_idx == 0) else None,
                )

        ax.set_xlim(int(source_time_index[0]), episode_x_max)
        ax.set_title(dim_names[dim])
        ax.grid(alpha=0.25)

    for ax in axes[pose_dim:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.92))
    fig.suptitle(
        f"Sampled Prior Fan + Measured Pose (full episode, stride={max(1, int(prediction_stride))})",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.9])
    return fig


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return vec / norm


def _rot6d_to_matrix(rot6: np.ndarray) -> np.ndarray:
    a1 = np.asarray(rot6[:3], dtype=np.float32)
    a2 = np.asarray(rot6[3:6], dtype=np.float32)
    b1 = _normalize_vec(a1)
    a2_orth = a2 - float(np.dot(b1, a2)) * b1
    b2 = _normalize_vec(a2_orth)
    b3 = np.cross(b1, b2)
    b3 = _normalize_vec(b3)
    return np.stack([b1, b2, b3], axis=1)


def _draw_frame(ax, pose9: np.ndarray, axis_len: float, alpha: float, lw: float):
    pos = np.asarray(pose9[:3], dtype=np.float32)
    rot6 = np.asarray(pose9[3:9], dtype=np.float32)
    rot = _rot6d_to_matrix(rot6)
    colors = ["#e41a1c", "#4daf4a", "#377eb8"]
    for i in range(3):
        end = pos + axis_len * rot[:, i]
        ax.plot(
            [pos[0], end[0]],
            [pos[1], end[1]],
            [pos[2], end[2]],
            color=colors[i],
            alpha=alpha,
            linewidth=lw,
        )


def _draw_frame_dimmed(ax, pose9: np.ndarray, axis_len: float, alpha: float, lw: float, dim: float):
    pos = np.asarray(pose9[:3], dtype=np.float32)
    rot6 = np.asarray(pose9[3:9], dtype=np.float32)
    rot = _rot6d_to_matrix(rot6)
    colors = ["#e41a1c", "#4daf4a", "#377eb8"]
    dim = float(np.clip(dim, 0.0, 1.0))
    for i in range(3):
        end = pos + axis_len * rot[:, i]
        base = np.asarray(matplotlib.colors.to_rgb(colors[i]), dtype=np.float32)
        color = tuple((base * dim).tolist())
        ax.plot(
            [pos[0], end[0]],
            [pos[1], end[1]],
            [pos[2], end[2]],
            color=color,
            alpha=alpha,
            linewidth=lw,
        )


def _set_axes_equal_3d(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)

    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    half = 0.5 * max_range
    ax.set_xlim3d([x_mid - half, x_mid + half])
    ax.set_ylim3d([y_mid - half, y_mid + half])
    ax.set_zlim3d([z_mid - half, z_mid + half])


def _build_3d_pose_figure(
    measured_pose_first: np.ndarray,
    gt_pose_first: np.ndarray,
    pred_pose_first: np.ndarray,
    sampled_pose_chunks: np.ndarray,
    prediction_stride: int,
    SHOW_PLOT: bool,
):
    if not SHOW_PLOT:
        fig = plt.figure(figsize=(10, 8))
    else:
        fig = plt.figure(figsize=(18, 15))
    ax = fig.add_subplot(111, projection="3d")

    meas_xyz = measured_pose_first[:, :3]
    gt_xyz = gt_pose_first[:, :3]
    pred_xyz = pred_pose_first[:, :3]

    ax.plot(
        meas_xyz[:, 0],
        meas_xyz[:, 1],
        meas_xyz[:, 2],
        color="#1f78b4",
        linewidth=0.8,
        linestyle="--",
        alpha=0.5,
        label="measured",
    )
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], color="black", linewidth=1.0, alpha=1.0, label="ground_truth")
    # ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], color="#e31a1c", linewidth=2.0, alpha=0.95, label="prior_mean")

    num_steps, num_samples = sampled_pose_chunks.shape[0], sampled_pose_chunks.shape[1]

    # Draw start points
    ax.scatter(meas_xyz[0, 0], meas_xyz[0, 1], meas_xyz[0, 2], color="#1f78b4", s=30, marker="o", alpha=0.95, label="measured_start")
    ax.scatter(gt_xyz[0, 0], gt_xyz[0, 1], gt_xyz[0, 2], color="black", s=30, marker="o", alpha=0.95, label="ground_truth_start")
    ax.scatter(pred_xyz[0, 0], pred_xyz[0, 1], pred_xyz[0, 2], color="#e31a1c", s=30, marker="o", alpha=0.95, label="prior_mean_start")

    goal_pos_measured = meas_xyz[-1]
    ax.scatter(goal_pos_measured[0], goal_pos_measured[1], goal_pos_measured[2], color="#1f78b4", s=40, marker="s", alpha=0.95, label="goal measured")

    goal_pos_gt = gt_xyz[-1]
    ax.scatter(goal_pos_gt[0], goal_pos_gt[1], goal_pos_gt[2], color="black", s=40, marker="s", alpha=0.95, label="goal ground truth")

    all_xyz = np.concatenate([meas_xyz, gt_xyz, pred_xyz], axis=0)
    if all_xyz.shape[0] > 1:
        extent = np.ptp(all_xyz, axis=0)
        diag = float(np.linalg.norm(extent))
        axis_len = max(0.01, 0.03 * diag)
    else:
        axis_len = 0.02

    anchor_idx = np.arange(0, num_steps, max(1, int(prediction_stride)), dtype=np.int64)
    if anchor_idx[-1] != (num_steps - 1):
        anchor_idx = np.concatenate([anchor_idx, np.asarray([num_steps - 1], dtype=np.int64)])

    first_idx = 0
    last_idx = max(0, num_steps - 1)
    _draw_frame(ax, gt_pose_first[first_idx], axis_len=axis_len, alpha=0.7, lw=1.1)
    _draw_frame(ax, gt_pose_first[last_idx], axis_len=axis_len, alpha=0.7, lw=1.1)
    _draw_frame(ax, pred_pose_first[first_idx], axis_len=axis_len, alpha=0.7, lw=1.1)
    _draw_frame(ax, pred_pose_first[last_idx], axis_len=axis_len, alpha=0.7, lw=1.1)

    anchor_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, max(1, len(anchor_idx))))
    for anchor_pos, t_idx in enumerate(anchor_idx):
        c_t = anchor_colors[anchor_pos]
        _draw_frame_dimmed(ax, gt_pose_first[t_idx], axis_len=axis_len * 1.2, alpha=0.9, lw=1.2, dim=0.5)
        for s_idx in range(num_samples):
            traj = sampled_pose_chunks[t_idx, s_idx, :, :3]
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                color=c_t,
                linewidth=0.9,
                alpha=0.9,
                label="prior_samples" if (anchor_pos == 0 and s_idx == 0) else None,
            )
            _draw_frame(ax, sampled_pose_chunks[t_idx, s_idx, 0], axis_len=axis_len * 0.7, alpha=0.8, lw=0.6)
            _draw_frame(ax, sampled_pose_chunks[t_idx, s_idx, -1], axis_len=axis_len * 0.7, alpha=0.8, lw=0.6)

    x_min, x_max = GLOBAL_AXIS_LIMITS["x"]
    y_min, y_max = GLOBAL_AXIS_LIMITS["y"]
    z_min, z_max = GLOBAL_AXIS_LIMITS["z"]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    span = np.array([x_max - x_min, y_max - y_min, z_max - z_min], dtype=np.float32)
    text_offset = 0.03 * float(np.linalg.norm(span))
    for goal in GOAL_FRAMES:
        pose = np.asarray(goal["pose"], dtype=np.float32)
        _draw_frame(ax, pose, axis_len=axis_len * 3.0, alpha=1.0, lw=1.5)
        ax.text(
            pose[0] + text_offset,
            pose[1] - text_offset,
            pose[2] - text_offset,
            goal["name"],
            fontsize=6,
            color="black",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D EE Pose Frames: measured vs ground truth vs sampled priors")
    ax.view_init(elev=float(VIEW_ELEV), azim=float(VIEW_AZIM))
    _set_axes_equal_3d(ax)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def _summarize_metrics(
    model,
    device: torch.device,
    obs_norm: np.ndarray,
    actions_norm: np.ndarray,
    mask_norm: np.ndarray,
    labels: np.ndarray,
    num_samples: int,
):
    obs_t = torch.from_numpy(obs_norm).float().to(device)
    actions_t = torch.from_numpy(actions_norm).float().to(device)
    mask_t = torch.from_numpy(mask_norm).float().to(device)
    labels_t = torch.from_numpy(labels).long().to(device)

    ac_flat = actions_t.reshape(actions_t.shape[0], -1)
    mask_flat = mask_t.reshape(mask_t.shape[0], -1)

    with torch.no_grad():
        output = model({}, obs_t, ac_flat, mask_flat, class_labels=labels_t)
        prior_det = model.get_actions_prior({}, obs_t, class_labels=labels_t, sample=False, num_samples=1)
        if prior_det.ndim == 4:
            prior_det = prior_det[:, 0]
        prior_samples = model.get_actions_prior({}, obs_t, class_labels=labels_t, sample=True, num_samples=num_samples)

    mask_den = mask_t.sum((1, 2)).clamp(min=1.0)
    prior_l1 = torch.abs(mask_t * (prior_det - actions_t))
    prior_l1 = prior_l1.sum((1, 2)) / mask_den
    action_l2 = torch.square(mask_t * (prior_det - actions_t))
    action_l2 = action_l2.sum((1, 2)) / mask_den
    lsig = torch.logical_or(
        torch.logical_and(actions_t > 0, prior_det <= 0),
        torch.logical_and(actions_t <= 0, prior_det > 0),
    )
    lsig = (lsig.float() * mask_t).sum((1, 2)) / mask_den

    metrics = {
        "posterior_l1": float(output["l1_loss"].item()),
        "prior_l1": float(prior_l1.mean().item()),
        "posterior_kl": float(output["kl"].item()),
        "action_l2": float(action_l2.mean().item()),
        "action_lsig": float(lsig.mean().item()),
        "prior_std_mean": float(output.get("prior_std_mean", torch.tensor(float("nan"), device=device)).item()),
        "posterior_std_mean": float(output.get("posterior_std_mean", torch.tensor(float("nan"), device=device)).item()),
        "prior_entropy": float(output.get("prior_entropy", torch.tensor(float("nan"), device=device)).item()),
        "posterior_entropy": float(output.get("posterior_entropy", torch.tensor(float("nan"), device=device)).item()),
    }

    return metrics, prior_det.detach().cpu().numpy(), prior_samples.detach().cpu().numpy()


def main():
    if NORMALIZATION_MODE not in ("auto", "apply", "skip"):
        raise ValueError(f"NORMALIZATION_MODE must be one of auto/apply/skip, got: {NORMALIZATION_MODE}")

    run_dir = Path(RUN_DIR)
    checkpoint_name = str(CHECKPOINT_NAME)
    list_episodes_only = bool(LIST_EPISODES_ONLY)
    raw_episode_dir = Path(RAW_EPISODE_DIR)
    episode_file_name = EPISODE_FILE_NAME
    episode_index = int(EPISODE_INDEX)
    episode_list = [str(name) for name in EPISODE_LIST if str(name).strip()]

    buffer_path_override = Path(BUFFER_PATH_OVERRIDE) if BUFFER_PATH_OVERRIDE is not None else None
    rollout_config_override = Path(ROLLOUT_CONFIG_OVERRIDE) if ROLLOUT_CONFIG_OVERRIDE is not None else None
    out_dir_override = Path(OUT_DIR_OVERRIDE) if OUT_DIR_OVERRIDE is not None else None

    exp_config_path = run_dir / "exp_config.yaml"
    ckpt_path = run_dir / checkpoint_name
    if not exp_config_path.exists():
        raise FileNotFoundError(f"exp_config.yaml not found: {exp_config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    cfg = _load_run_cfg(exp_config_path)
    buffer_path = buffer_path_override if buffer_path_override is not None else Path(cfg.test_buffer_path)
    rollout_config_path = rollout_config_override if rollout_config_override is not None else (buffer_path.parent / "rollout_config.yaml")
    rollout_cfg = _load_rollout_config(rollout_config_path)
    state_stats = rollout_cfg.get("norm_stats", {}).get("state", None)
    action_stats = rollout_cfg.get("norm_stats", {}).get("action", None)

    if not buffer_path.exists():
        raise FileNotFoundError(f"buffer not found: {buffer_path}")

    # print(f"Using buffer: {buffer_path}")
    # print(f"Using rollout config: {rollout_config_path}")
    # print(f"Using run dir: {run_dir}")

    if not raw_episode_dir.exists():
        raise FileNotFoundError(f"RAW_EPISODE_DIR not found: {raw_episode_dir}")
    episode_files = _list_episode_files(raw_episode_dir)
    if list_episodes_only:
        print(f"Available raw episodes in: {raw_episode_dir}")
        for i, p in enumerate(episode_files):
            print(f"  [{i:03d}] {p.name}")
        return

    if USE_EPISODE_LIST and episode_list:
        selected = []
        for name in episode_list:
            fname = name if name.endswith(".pkl") else f"{name}.pkl"
            selected.append(_select_episode_file(episode_files, fname, episode_index))
    else:
        selected = [_select_episode_file(episode_files, episode_file_name, episode_index)]

    train_background = None
    if ENABLE_TRAIN_BACKGROUND:
        train_buf_path = _resolve_train_buffer_path(rollout_cfg, buffer_path)
        if train_buf_path.exists():
            train_background = _load_train_buffer_actions(train_buf_path)
            if TRAIN_BACKGROUND_MAX_TRAJ > 0:
                train_background = train_background[: int(TRAIN_BACKGROUND_MAX_TRAJ)]
            if action_stats is not None:
                train_background = [_apply_grouped_transform(traj, action_stats, inverse=True) for traj in train_background]
            print(f"Loaded train background trajectories: {len(train_background)} | {train_buf_path}")
        else:
            print(f"Train buffer not found for background: {train_buf_path}")

    for episode_file in selected:
        print(f"Selected episode file: {episode_file}")

        raw_ep = _load_raw_episode_to_arrays(episode_file, rollout_cfg)
        obs_window = int(cfg.obs_window)
        ac_chunk = int(cfg.ac_chunk)
        action_index_offset = int(OmegaConf.select(cfg, "task.test_buffer.action_index_offset", default=1))
        ep_data = _build_eval_samples_from_raw_episode(
            states=raw_ep["states"],
            actions=raw_ep["actions"],
            episode_label=int(raw_ep["episode_label"]),
            obs_window=obs_window,
            ac_chunk=ac_chunk,
            action_index_offset=action_index_offset,
        )

        obs_arr = ep_data["obs"]
        actions_arr = ep_data["actions"]
        mask_arr = ep_data["mask"]
        labels_arr = ep_data["labels"]
        steps_arr = ep_data["steps"]
        stiffness_arr = np.full((len(steps_arr),), int(raw_ep["episode_label"]), dtype=np.int64)

        obs_norm, obs_applied = _ensure_normalized(obs_arr, state_stats, NORMALIZATION_MODE, "state")
        actions_norm, action_applied = _ensure_normalized(actions_arr, action_stats, NORMALIZATION_MODE, "action")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = _load_model(cfg, ckpt_path, device)

        metrics, prior_det_norm, prior_samples_norm = _summarize_metrics(
            model=model,
            device=device,
            obs_norm=obs_norm,
            actions_norm=actions_norm,
            mask_norm=mask_arr,
            labels=labels_arr,
            num_samples=int(NUM_SAMPLES),
        )

        actions_denorm = _apply_grouped_transform(actions_norm, action_stats, inverse=True)
        prior_det_denorm = _apply_grouped_transform(prior_det_norm, action_stats, inverse=True)
        prior_samples_denorm = _apply_grouped_transform(prior_samples_norm, action_stats, inverse=True)
        obs_denorm = _apply_grouped_transform(obs_norm, state_stats, inverse=True)

        ac_dim = actions_denorm.shape[-1]
        pose_dim = min(9, ac_dim)

        true_first = actions_denorm[:, 0, :pose_dim]
        pred_first = prior_det_denorm[:, 0, :pose_dim]
        measured_first = obs_denorm[:, -1, :pose_dim]
        mask_first = mask_arr[:, 0, :pose_dim]

        episode_name = episode_file.stem
        split_label = _get_split_label(rollout_cfg, episode_name)
        out_suffix = "episode_eval_test" if split_label == "test" else "episode_eval_train"
        if out_dir_override is None:
            out_dir = run_dir.parent / out_suffix
        else:
            out_dir = Path(out_dir_override) / out_suffix
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Episode {episode_file.name} | steps={len(steps_arr)} | stiffness={int(stiffness_arr[0])} | raw_episode_length={raw_ep['num_steps']}")
        print(
            "Metrics | "
            f"Posterior L1={metrics['posterior_l1']:.4f} "
            f"Prior L1={metrics['prior_l1']:.4f} "
            f"KL={metrics['posterior_kl']:.4f} "
            f"Action L2={metrics['action_l2']:.4f} "
            f"LSign={metrics['action_lsig']:.4f} "
            f"prior_std={metrics['prior_std_mean']:.4f} "
            f"post_std={metrics['posterior_std_mean']:.4f} "
            f"prior_H={metrics['prior_entropy']:.4f} "
            f"post_H={metrics['posterior_entropy']:.4f}"
        )

        # fig_pose = _build_pose_figure_with_measured(
        #     true_actions=true_first,
        #     pred_actions=pred_first,
        #     measured_pose=measured_first,
        #     mask=mask_first,
        #     title=f"Episode {episode_file.name} | Ground Truth vs Prior Mean vs Measured Pose",
        # )
        # if fig_pose is not None:
        #     pose_path = out_dir / f"{episode_file.stem}_pred_firststeps.png"
        #     fig_pose.savefig(pose_path, dpi=300, bbox_inches="tight")
        #     plt.close(fig_pose)
        #     print(f"Saved: {pose_path}")

        fig_fan = _build_fan_figure_with_measured(
            true_action_chunks=actions_denorm[:, :, :pose_dim],
            pred_action_chunks=prior_samples_denorm[:, :, :, :pose_dim],
            measured_pose=measured_first,
            mask_chunks=mask_arr[:, :, :pose_dim],
            source_time_index=steps_arr,
            prediction_stride=max(1, int(PREDICTION_STRIDE)),
            background_actions=train_background
            if (ENABLE_TRAIN_BACKGROUND and train_background) and (not TRAIN_BACKGROUND_ONLY_MEDIUM or "medium" in episode_name)
            else None,
        )
        if fig_fan is not None:
            fan_path = out_dir / f"{episode_file.stem}_predictions.png"
            fig_fan.savefig(fan_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {fan_path}")

        sample_anchor_step = int(SAMPLE_ANCHOR_STEP)
        if sample_anchor_step < 0:
            sample_anchor_step = len(steps_arr) // 2
        if pose_dim >= 9:
            fig_3d = _build_3d_pose_figure(
                measured_pose_first=measured_first,
                gt_pose_first=true_first,
                pred_pose_first=pred_first,
                sampled_pose_chunks=prior_samples_denorm[:, :, :, :pose_dim],
                prediction_stride=max(1, int(PREDICTION_STRIDE)),
                SHOW_PLOT=SHOW_PLOT,
            )
            plot3d_path = out_dir / f"{episode_file.stem}_predictions_3d.png"
            fig_3d.savefig(plot3d_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {plot3d_path}")
        else:
            print("Skipping 3D frame plot: pose_dim < 9.")

        if SHOW_PLOT:
            if fig_fan is not None:
                fig_fan.show()
            if pose_dim >= 9:
                fig_3d.show()
            plt.show()

        if fig_fan is not None:
            plt.close(fig_fan)
        if pose_dim >= 9:
            plt.close(fig_3d)

    # metrics_path = out_dir / "metrics.json"
    # metrics_payload = {
    #     "episode_file": episode_file.name,
    #     "episode_index": int(episode_files.index(episode_file)),
    #     "buffer_path": str(buffer_path),
    #     "raw_episode_dir": str(raw_episode_dir),
    #     "checkpoint": str(ckpt_path),
    #     "rollout_config": str(rollout_config_path),
    #     "normalization_mode": NORMALIZATION_MODE,
    #     "state_normalization_applied": bool(obs_applied),
    #     "action_normalization_applied": bool(action_applied),
    #     "num_samples": int(NUM_SAMPLES),
    #     "prediction_stride": int(PREDICTION_STRIDE),
    #     "obs_window": int(obs_window),
    #     "ac_chunk": int(ac_chunk),
    #     "action_index_offset": int(action_index_offset),
    #     "num_eval_steps": int(len(steps_arr)),
    #     "raw_episode_num_steps": int(raw_ep["num_steps"]),
    #     "stiffness_label": int(stiffness_arr[0]),
    #     "metrics": metrics,
    # }
    # with open(metrics_path, "w") as f:
    #     json.dump(metrics_payload, f, indent=2)
    # print(f"Saved: {metrics_path}")
    # print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
