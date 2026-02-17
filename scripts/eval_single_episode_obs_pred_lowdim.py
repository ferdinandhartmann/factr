#!/usr/bin/env python3
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

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
RUN_DIR = Path.home() / "activeinference" / "factr" / "checkpoints" / "obs_pred_mlp_gaussian_h30_lowdim" / "rollout"
CHECKPOINT_NAME = "latest_ckpt.ckpt"

# Raw episode source
RAW_EPISODE_DIR = Path.home() / "activeinference" / "factr" / "process_data" / "data_to_process" / "fourgoals_1" / "data"
EPISODE_FILE_NAME = "ep_29_medium.pkl"
EPISODE_INDEX = 0  # index in sorted *.pkl files
USE_EPISODE_LIST = False
EPISODE_LIST = [
    "ep_09_stiff",
    "ep_09_soft",
    "ep_10_soft",
    "ep_14_medium",
    "ep_19_medium",
    "ep_23_stiff",
    "ep_29_medium",
    "ep_29_soft",
    "ep_33_medium",
]
LIST_EPISODES_ONLY = False

BUFFER_PATH_OVERRIDE = Path.home() / "activeinference" / "factr" / "process_data" / "processed_data" / "fourgoals_1_act" / "buf_test.pkl"
ROLLOUT_CONFIG_OVERRIDE = (
    Path.home() / "activeinference" / "factr" / "process_data" / "processed_data" / "fourgoals_1_act" / "rollout_config.yaml"
)

NUM_SAMPLES = 10  # Monte-Carlo samples from predicted Gaussian for eval-only sample MSE
NORMALIZATION_MODE = "apply"  # one of: auto, apply, skip
PREDICTION_STRIDE = 1
MAX_PLOT_DIMS = 9

OUT_DIR_OVERRIDE = None


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("len"):
        OmegaConf.register_new_resolver("len", lambda x: len(x))


def _load_run_cfg(exp_config_path: Path):
    cfg_all = OmegaConf.load(exp_config_path)
    cfg_src = cfg_all.params if "params" in cfg_all else cfg_all
    # Detach from parent container (e.g., top-level {"params": ..., "wandb_id": ...})
    # so relative interpolations like `${goal_classes}` resolve against the actual run config.
    cfg = OmegaConf.create(OmegaConf.to_container(cfg_src, resolve=False))
    _register_resolvers()
    if "hydra" in cfg:
        cfg.pop("hydra", None)

    # Backward/forward compatibility: older run configs may miss some top-level
    # keys that newer agent/task configs reference via interpolation.
    if OmegaConf.select(cfg, "goal_classes", default=None) is None:
        goal_classes = OmegaConf.select(
            cfg,
            "agent.goal_classes",
            default=OmegaConf.select(cfg, "task.test_buffer.goal_classes", default=4),
        )
        cfg.goal_classes = int(goal_classes)
    if OmegaConf.select(cfg, "stiffness_classes", default=None) is None:
        stiffness_classes = OmegaConf.select(
            cfg,
            "agent.stiffness_classes",
            default=OmegaConf.select(cfg, "task.test_buffer.stiffness_classes", default=3),
        )
        cfg.stiffness_classes = int(stiffness_classes)
    if OmegaConf.select(cfg, "pose_action_dim", default=None) is None:
        pose_action_dim = OmegaConf.select(
            cfg,
            "agent.pose_action_dim",
            default=OmegaConf.select(cfg, "task.test_buffer.pose_action_dim", default=9),
        )
        cfg.pose_action_dim = int(pose_action_dim)
    if OmegaConf.select(cfg, "obs_input_dim", default=None) is None:
        obs_input_dim = OmegaConf.select(
            cfg,
            "agent.obs_input_dim",
            default=OmegaConf.select(cfg, "task.test_buffer.input_obs_dim", default=21),
        )
        cfg.obs_input_dim = int(obs_input_dim)
    if OmegaConf.select(cfg, "obs_target_dim", default=None) is None:
        obs_target_dim = OmegaConf.select(
            cfg,
            "agent.predict_obs_dim",
            default=OmegaConf.select(cfg, "task.test_buffer.predict_obs_dim", default=21),
        )
        cfg.obs_target_dim = int(obs_target_dim)
    if OmegaConf.select(cfg, "pred_horizon", default=None) is None:
        pred_horizon = OmegaConf.select(
            cfg,
            "agent.pred_horizon",
            default=OmegaConf.select(cfg, "task.test_buffer.pred_horizon", default=30),
        )
        cfg.pred_horizon = int(pred_horizon)
    if OmegaConf.select(cfg, "raw_obs_dim", default=None) is None:
        raw_obs_dim = OmegaConf.select(
            cfg,
            "task.test_buffer.obs_dim",
            default=OmegaConf.select(cfg, "task.obs_dim", default=27),
        )
        cfg.raw_obs_dim = int(raw_obs_dim)
    if OmegaConf.select(cfg, "action_index_offset", default=None) is None:
        cfg.action_index_offset = int(OmegaConf.select(cfg, "task.test_buffer.action_index_offset", default=0))
    if OmegaConf.select(cfg, "target_index_offset", default=None) is None:
        cfg.target_index_offset = int(OmegaConf.select(cfg, "task.test_buffer.target_index_offset", default=1))

    OmegaConf.resolve(cfg)
    return cfg


def _load_model(cfg, ckpt_path: Path, device: torch.device):
    model = instantiate(cfg.agent)
    model = model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/model shape mismatch. This usually means the checkpoint was trained with different "
            "prediction horizon or model dimensions. Train a new checkpoint with the current 30-step config "
            "(e.g. `train_obs_pred_lowdim.yaml` with `pred_horizon=30`) and update RUN_DIR/CHECKPOINT_NAME."
        ) from exc
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


def _extract_ep_index(path: Path) -> Tuple[int, str]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "ep" and parts[1].isdigit():
        return int(parts[1]), stem
    return (10**9, stem)


def _list_episode_files(raw_episode_dir: Path) -> List[Path]:
    return sorted(raw_episode_dir.glob("*.pkl"), key=_extract_ep_index)


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


def _goal_value_to_class(raw_goal_vec: np.ndarray, goal_classes: int) -> int:
    goal_vec = np.asarray(raw_goal_vec, dtype=np.float32).reshape(-1)
    if goal_vec.size == 0:
        return 1

    if goal_vec.size > 1:
        is_one_hot = np.all(np.logical_or(np.isclose(goal_vec, 0.0), np.isclose(goal_vec, 1.0)))
        if is_one_hot:
            raw_value = int(np.argmax(goal_vec)) + 1
        else:
            raw_value = int(goal_vec[0])
    else:
        raw_value = int(goal_vec[0])

    if raw_value < 1:
        raw_value += 1
    return int(max(1, min(raw_value, int(goal_classes))))


def _load_raw_episode_to_arrays(episode_file: Path, rollout_cfg: Dict, goal_classes: int) -> Dict:
    import pickle

    with open(episode_file, "rb") as f:
        raw_data = pickle.load(f)

    obs_cfg = rollout_cfg.get("obs_config", {})
    action_cfg = rollout_cfg.get("action_config", {})

    state_topics = list(obs_cfg.get("state_topics", []))
    goal_topics = list(obs_cfg.get("goal_topics", []))
    if len(state_topics) == 0:
        raise ValueError("No state_topics found in rollout_config.")

    action_topics = list(action_cfg.keys())
    if len(action_topics) != 1:
        raise ValueError(f"Expected exactly one action topic, got: {action_topics}")
    action_topic = action_topics[0]
    action_dim = int(action_cfg[action_topic])

    stiffness_info = obs_cfg.get("stiffness_label", {})
    stiffness_topic = stiffness_info.get("topic", None)
    stiffness_key = stiffness_info.get("key", "stiffness")
    stiffness_thresholds = stiffness_info.get("norm_thresholds", [200.0, 1000.0])

    topics_for_sync = list(state_topics) + [action_topic] + list(goal_topics)
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
    goal_topic_specs = {
        "/goal": {"keys": ["goal"], "dim": 1, "fallback": None},
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
            for vec in vecs:
                if vec.shape[0] < max_dim:
                    vec = np.pad(vec, (0, max_dim - vec.shape[0]), constant_values=np.nan)
                padded.append(vec)
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

    goals_concat = None
    if len(goal_topics) > 0:
        goals_arrs = []
        for topic in goal_topics:
            spec = goal_topic_specs.get(topic, {"keys": ["goal"], "dim": 1, "fallback": None})
            topic_vecs = [_extract_fixed_vector(msg, spec["keys"], spec["dim"], spec.get("fallback", None)) for msg in synced[topic]]
            goals_arrs.append(np.stack(topic_vecs, axis=0).astype(np.float32))
        if len(goals_arrs) > 0:
            goals_concat = np.concatenate(goals_arrs, axis=-1).astype(np.float32)

    num_steps = int(min(states.shape[0], actions.shape[0]))
    if goals_concat is not None:
        num_steps = int(min(num_steps, goals_concat.shape[0]))

    states = states[:num_steps]
    actions = actions[:num_steps]

    if goals_concat is not None:
        goals_concat = goals_concat[:num_steps]
        goal_labels = np.asarray(
            [_goal_value_to_class(goals_concat[t], goal_classes=goal_classes) for t in range(num_steps)],
            dtype=np.int64,
        )
    else:
        goal_labels = np.ones((num_steps,), dtype=np.int64)

    if stiffness_topic:
        raw_stiff = synced[stiffness_topic][:num_steps]
        stiff_vecs = [_extract_fixed_vector(msg, [stiffness_key], 6, None) for msg in raw_stiff]
        stiffness_labels = np.asarray(
            [_stiffness_vec_to_class(vec, stiffness_thresholds) for vec in stiff_vecs],
            dtype=np.int64,
        )
    else:
        stiffness_labels = np.ones((num_steps,), dtype=np.int64)
    episode_label = int(stiffness_labels[0]) if len(stiffness_labels) > 0 else 1

    return {
        "states": states,
        "actions": actions,
        "goal_labels": goal_labels,
        "stiffness_labels": stiffness_labels,
        "episode_label": episode_label,
        "num_steps": int(num_steps),
    }


def _build_eval_samples_from_raw_episode(
    states: np.ndarray,
    actions: np.ndarray,
    goal_labels: np.ndarray,
    episode_stiffness_label: int,
    obs_window: int,
    action_index_offset: int,
    target_index_offset: int,
    pred_horizon: int,
    pose_action_dim: int,
):
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"Expected 2D states/actions, got states={states.shape} actions={actions.shape}")
    if goal_labels.ndim != 1:
        raise ValueError(f"Expected 1D goal_labels, got {goal_labels.shape}")

    max_t = states.shape[0] - max(int(action_index_offset), int(target_index_offset))
    if max_t <= 0:
        raise ValueError("Episode is too short for the configured action/target offsets.")

    obs_arr, action_arr, target_arr, target_mask_arr, stiff_arr, goal_arr, steps_arr = [], [], [], [], [], [], []
    for t_idx in range(max_t):
        start = max(0, t_idx - int(obs_window) + 1)
        window_states = [states[i] for i in range(start, t_idx + 1)]
        while len(window_states) < int(obs_window):
            window_states.insert(0, window_states[0])
        obs_window_arr = np.stack(window_states, axis=0).astype(np.float32)

        action_start_idx = t_idx + int(action_index_offset)
        target_start_idx = t_idx + int(target_index_offset)

        action_chunk = []
        target_chunk = []
        target_mask = []
        for h_idx in range(int(pred_horizon)):
            action_idx = action_start_idx + h_idx
            target_idx = target_start_idx + h_idx

            if action_idx < actions.shape[0]:
                action_chunk.append(actions[action_idx][: int(pose_action_dim)])
            else:
                fallback_action_idx = min(actions.shape[0] - 1, action_start_idx)
                action_chunk.append(actions[fallback_action_idx][: int(pose_action_dim)])

            if target_idx < states.shape[0]:
                target_chunk.append(states[target_idx])
                target_mask.append(1.0)
            else:
                fallback_target_idx = min(states.shape[0] - 1, target_start_idx)
                target_chunk.append(states[fallback_target_idx])
                target_mask.append(0.0)

        action_chunk = np.stack(action_chunk, axis=0).astype(np.float32)
        target_chunk = np.stack(target_chunk, axis=0).astype(np.float32)
        target_mask = np.asarray(target_mask, dtype=np.float32)
        goal_label = int(goal_labels[target_start_idx])

        obs_arr.append(obs_window_arr)
        action_arr.append(action_chunk)
        target_arr.append(target_chunk)
        target_mask_arr.append(target_mask)
        stiff_arr.append(int(episode_stiffness_label))
        goal_arr.append(goal_label)
        steps_arr.append(int(t_idx))

    return {
        "obs_full": np.stack(obs_arr, axis=0).astype(np.float32),
        "actions": np.stack(action_arr, axis=0).astype(np.float32),
        "target_full": np.stack(target_arr, axis=0).astype(np.float32),
        "target_mask": np.stack(target_mask_arr, axis=0).astype(np.float32),
        "stiffness": np.asarray(stiff_arr, dtype=np.int64),
        "goals": np.asarray(goal_arr, dtype=np.int64),
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
            out = np.clip(out, -float(clip), float(clip))
        return out
    if gtype in ("min_max",):
        mins = np.asarray(group.get("min", []), dtype=np.float32)
        maxs = np.asarray(group.get("max", []), dtype=np.float32)
        denom = _safe_denominator(maxs - mins)
        out = (2.0 * (values - mins) / denom) - 1.0
        clip = group.get("clip", None)
        if clip is not None:
            out = np.clip(out, -float(clip), float(clip))
        return out
    if gtype in ("fixed_scale", "fixed_scale_clip"):
        scales = _safe_denominator(group.get("scales", []))
        out = values / scales
        clip = group.get("clip", None)
        if clip is not None:
            out = np.clip(out, -float(clip), float(clip))
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
            out = np.clip(out, -float(clip), float(clip))
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


def _replace_prefix_and_inverse(part_norm: np.ndarray, reference_full_norm: np.ndarray, stats: Dict) -> np.ndarray:
    full = reference_full_norm.copy()
    part_dim = int(part_norm.shape[-1])
    full[..., :part_dim] = part_norm
    denorm = _apply_grouped_transform(full, stats, inverse=True)
    return denorm[..., :part_dim]


def _make_obs_dim_names(dim: int) -> List[str]:
    names = [
        "pose_x",
        "pose_y",
        "pose_z",
        "pose_r1",
        "pose_r2",
        "pose_r3",
        "pose_r4",
        "pose_r5",
        "pose_r6",
        "vel_x",
        "vel_y",
        "vel_z",
        "vel_rx",
        "vel_ry",
        "vel_rz",
        "wrench_fx",
        "wrench_fy",
        "wrench_fz",
        "wrench_tx",
        "wrench_ty",
        "wrench_tz",
    ]
    if dim <= len(names):
        return names[:dim]
    return [f"obs_{idx + 1}" for idx in range(dim)]


def _build_obs_prediction_figure(
    true_obs: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    pred_sample: np.ndarray,
    time_index: np.ndarray,
    max_dims: int,
):
    n_dims = min(int(max_dims), int(true_obs.shape[1]))
    if n_dims <= 0:
        return None

    dim_names = _make_obs_dim_names(true_obs.shape[1])
    n_cols = min(3, n_dims)
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.8 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for dim in range(n_dims):
        ax = axes[dim]
        lower = pred_mean[:, dim] - pred_std[:, dim]
        upper = pred_mean[:, dim] + pred_std[:, dim]
        ax.plot(time_index, true_obs[:, dim], color="black", linewidth=1.3, label="target" if dim == 0 else None)
        ax.plot(time_index, pred_mean[:, dim], color="#E41A1C", linewidth=1.2, label="pred mean" if dim == 0 else None)
        ax.plot(
            time_index,
            pred_sample[:, dim],
            color="#377EB8",
            linewidth=0.9,
            alpha=0.7,
            linestyle="--",
            label="pred sample" if dim == 0 else None,
        )
        ax.fill_between(time_index, lower, upper, color="#FB9A99", alpha=0.25, label="mean ± std" if dim == 0 else None)
        ax.set_title(dim_names[dim])
        ax.grid(alpha=0.25)

    for ax in axes[n_dims:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)
    fig.suptitle("Observation prediction on raw episode", fontsize=12)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    return fig


def _collapse_chunk_predictions(
    true_chunks: np.ndarray,
    pred_mean_chunks: np.ndarray,
    pred_std_chunks: np.ndarray,
    pred_sample_chunks: np.ndarray,
    valid_mask: np.ndarray,
    anchor_steps: np.ndarray,
    stride: int,
):
    selected = np.arange(0, true_chunks.shape[0], max(1, int(stride)), dtype=np.int64)
    if selected[-1] != (true_chunks.shape[0] - 1):
        selected = np.concatenate([selected, np.asarray([true_chunks.shape[0] - 1], dtype=np.int64)])

    obs_dim = int(true_chunks.shape[-1])
    max_time = int(anchor_steps[-1] + true_chunks.shape[1] - 1)
    if max_time < 0:
        return None

    sum_true = np.zeros((max_time + 1, obs_dim), dtype=np.float32)
    sum_pred_mean = np.zeros((max_time + 1, obs_dim), dtype=np.float32)
    sum_pred_std = np.zeros((max_time + 1, obs_dim), dtype=np.float32)
    sum_pred_sample = np.zeros((max_time + 1, obs_dim), dtype=np.float32)
    counts = np.zeros((max_time + 1,), dtype=np.float32)

    for sample_idx in selected:
        base_t = int(anchor_steps[sample_idx])
        for h_idx in range(true_chunks.shape[1]):
            if valid_mask[sample_idx, h_idx] <= 0:
                continue
            t_abs = base_t + h_idx
            sum_true[t_abs] += true_chunks[sample_idx, h_idx]
            sum_pred_mean[t_abs] += pred_mean_chunks[sample_idx, h_idx]
            sum_pred_std[t_abs] += pred_std_chunks[sample_idx, h_idx]
            sum_pred_sample[t_abs] += pred_sample_chunks[sample_idx, h_idx]
            counts[t_abs] += 1.0

    valid_time = counts > 0
    if not np.any(valid_time):
        return None
    denom = np.clip(counts[valid_time][:, None], 1e-6, None)
    return {
        "time_index": np.where(valid_time)[0],
        "true": sum_true[valid_time] / denom,
        "pred_mean": sum_pred_mean[valid_time] / denom,
        "pred_std": sum_pred_std[valid_time] / denom,
        "pred_sample": sum_pred_sample[valid_time] / denom,
    }


def _build_tracking_error_figure(tracking_error: np.ndarray, per_dim_mse: np.ndarray):
    if tracking_error.size == 0:
        return None

    l2 = np.linalg.norm(tracking_error, axis=-1)
    pose_dim = int(tracking_error.shape[-1])
    dim_names = _make_obs_dim_names(pose_dim)
    mean_abs = np.mean(np.abs(tracking_error), axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(np.arange(l2.shape[0]), l2, color="#377EB8", linewidth=1.5)
    axes[0].set_title("Tracking error L2 per step")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("L2")
    axes[0].grid(alpha=0.25)

    axes[1].bar(np.arange(pose_dim), mean_abs, color="#FB9A99")
    axes[1].set_title("Mean |tracking error| by pose dim")
    axes[1].set_xticks(np.arange(pose_dim))
    axes[1].set_xticklabels(dim_names, rotation=45, ha="right")
    axes[1].grid(alpha=0.25)

    mse_dim = per_dim_mse[:pose_dim]
    axes[2].bar(np.arange(pose_dim), mse_dim, color="#A6CEE3")
    axes[2].set_title("Prediction MSE by pose dim")
    axes[2].set_xticks(np.arange(pose_dim))
    axes[2].set_xticklabels(dim_names, rotation=45, ha="right")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    return fig


def _summarize_metrics(
    model,
    device: torch.device,
    obs_model_norm: np.ndarray,
    action_model_norm: np.ndarray,
    target_model_norm: np.ndarray,
    target_mask: np.ndarray,
    stiffness_labels: np.ndarray,
    goal_labels: np.ndarray,
    num_samples: int,
):
    obs_t = torch.from_numpy(obs_model_norm).float().to(device)
    action_t = torch.from_numpy(action_model_norm).float().to(device)
    target_t = torch.from_numpy(target_model_norm).float().to(device)
    mask_t = torch.from_numpy(target_mask).float().to(device)
    stiffness_t = torch.from_numpy(stiffness_labels).long().to(device)
    goal_t = torch.from_numpy(goal_labels).long().to(device)

    with torch.no_grad():
        output = model(
            obs_window=obs_t,
            action_chunk=action_t,
            stiffness_labels=stiffness_t,
            goal_labels=goal_t,
            target_obs=target_t,
            target_mask=mask_t,
        )
        mean = output["mean"]
        std = output["std"]
        sample = output["sample"]

        # Evaluate stochastic objective with multiple Monte-Carlo draws.
        mc_count = max(1, int(num_samples))
        eps = torch.randn(mean.shape[0], mc_count, mean.shape[1], mean.shape[2], device=device)
        samples_mc = mean.unsqueeze(1) + std.unsqueeze(1) * eps
        target_mc = target_t.unsqueeze(1)
        mc_sq = (samples_mc - target_mc) ** 2
        mc_sq = torch.mean(mc_sq, dim=1)
        sample_mse_mc = (mc_sq * mask_t.unsqueeze(-1)).sum() / torch.clamp(mask_t.sum() * mean.shape[-1], min=1.0)

        tracking_error_mean = model.compute_tracking_error(mean, action_t)  # (B, H, pose_dim)
        tracking_error_sample = model.compute_tracking_error(sample, action_t)
        track_l2_mean = torch.linalg.norm(tracking_error_mean, dim=-1)  # (B, H)
        track_l2_sample = torch.linalg.norm(tracking_error_sample, dim=-1)
        track_l2_mean = (track_l2_mean * mask_t).sum() / torch.clamp(mask_t.sum(), min=1.0)
        track_l2_sample = (track_l2_sample * mask_t).sum() / torch.clamp(mask_t.sum(), min=1.0)

    metrics = {
        "sample_mse_norm": float(output["sample_mse"].item()),
        "mean_mse_norm": float(output["mean_mse"].item()),
        "sample_mse_mc_norm": float(sample_mse_mc.item()),
        "pred_var_mean_norm": float(output["var"].mean().item()),
        "tracking_l2_mean_norm": float(track_l2_mean.item()),
        "tracking_l2_sample_norm": float(track_l2_sample.item()),
    }
    return (
        metrics,
        mean.detach().cpu().numpy(),
        std.detach().cpu().numpy(),
        sample.detach().cpu().numpy(),
        tracking_error_mean.detach().cpu().numpy(),
    )


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

    if not raw_episode_dir.exists():
        raise FileNotFoundError(f"RAW_EPISODE_DIR not found: {raw_episode_dir}")
    episode_files = _list_episode_files(raw_episode_dir)

    if list_episodes_only:
        print(f"Available raw episodes in: {raw_episode_dir}")
        for idx, ep in enumerate(episode_files):
            print(f"  [{idx:03d}] {ep.name}")
        return

    if USE_EPISODE_LIST and episode_list:
        selected = []
        for name in episode_list:
            fname = name if name.endswith(".pkl") else f"{name}.pkl"
            selected.append(_select_episode_file(episode_files, fname, episode_index))
    else:
        selected = [_select_episode_file(episode_files, episode_file_name, episode_index)]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg, ckpt_path, device)

    obs_window = int(OmegaConf.select(cfg, "obs_window", default=8))
    raw_obs_dim = int(OmegaConf.select(cfg, "raw_obs_dim", default=27))
    obs_input_dim = int(OmegaConf.select(cfg, "obs_input_dim", default=OmegaConf.select(cfg, "agent.obs_input_dim", default=21)))
    obs_target_dim = int(
        OmegaConf.select(cfg, "obs_target_dim", default=OmegaConf.select(cfg, "agent.predict_obs_dim", default=21))
    )
    pred_horizon = int(OmegaConf.select(cfg, "pred_horizon", default=OmegaConf.select(cfg, "agent.pred_horizon", default=30)))
    pose_action_dim = int(
        OmegaConf.select(cfg, "pose_action_dim", default=OmegaConf.select(cfg, "agent.pose_action_dim", default=9))
    )
    goal_classes = int(OmegaConf.select(cfg, "goal_classes", default=OmegaConf.select(cfg, "agent.goal_classes", default=4)))
    action_index_offset = int(
        OmegaConf.select(cfg, "action_index_offset", default=OmegaConf.select(cfg, "task.test_buffer.action_index_offset", default=0))
    )
    target_index_offset = int(
        OmegaConf.select(cfg, "target_index_offset", default=OmegaConf.select(cfg, "task.test_buffer.target_index_offset", default=1))
    )

    if obs_input_dim > raw_obs_dim or obs_target_dim > raw_obs_dim:
        raise ValueError(
            f"obs_input_dim/obs_target_dim must be <= raw_obs_dim. "
            f"Got input={obs_input_dim}, target={obs_target_dim}, raw={raw_obs_dim}."
        )

    for episode_file in selected:
        print(f"Selected episode file: {episode_file}")
        raw_ep = _load_raw_episode_to_arrays(episode_file, rollout_cfg, goal_classes=goal_classes)

        ep_data = _build_eval_samples_from_raw_episode(
            states=raw_ep["states"],
            actions=raw_ep["actions"],
            goal_labels=raw_ep["goal_labels"],
            episode_stiffness_label=int(raw_ep["episode_label"]),
            obs_window=obs_window,
            action_index_offset=action_index_offset,
            target_index_offset=target_index_offset,
            pred_horizon=pred_horizon,
            pose_action_dim=pose_action_dim,
        )

        obs_full = ep_data["obs_full"]
        target_full = ep_data["target_full"]
        target_mask = ep_data["target_mask"]
        action_arr = ep_data["actions"]
        stiffness_arr = ep_data["stiffness"]
        goal_arr = ep_data["goals"]
        steps_arr = ep_data["steps"]

        obs_full_norm, obs_applied = _ensure_normalized(obs_full, state_stats, NORMALIZATION_MODE, "state_window")
        target_full_norm, target_applied = _ensure_normalized(target_full, state_stats, NORMALIZATION_MODE, "state_target")
        action_norm, action_applied = _ensure_normalized(action_arr, action_stats, NORMALIZATION_MODE, "action")

        obs_model_norm = obs_full_norm[..., :obs_input_dim]
        target_model_norm = target_full_norm[..., :obs_target_dim]
        action_model_norm = action_norm[..., :pose_action_dim]

        metrics, pred_mean_norm, pred_std_norm, pred_sample_norm, tracking_err_mean_norm = _summarize_metrics(
            model=model,
            device=device,
            obs_model_norm=obs_model_norm,
            action_model_norm=action_model_norm,
            target_model_norm=target_model_norm,
            target_mask=target_mask,
            stiffness_labels=stiffness_arr,
            goal_labels=goal_arr,
            num_samples=int(NUM_SAMPLES),
        )

        target_denorm_full = _apply_grouped_transform(target_full_norm, state_stats, inverse=True)
        action_denorm = _apply_grouped_transform(action_norm, action_stats, inverse=True)

        pred_mean_denorm = _replace_prefix_and_inverse(pred_mean_norm, target_full_norm, state_stats)
        pred_sample_denorm = _replace_prefix_and_inverse(pred_sample_norm, target_full_norm, state_stats)
        pred_upper_denorm = _replace_prefix_and_inverse(pred_mean_norm + pred_std_norm, target_full_norm, state_stats)
        pred_lower_denorm = _replace_prefix_and_inverse(pred_mean_norm - pred_std_norm, target_full_norm, state_stats)
        pred_std_denorm = 0.5 * np.abs(pred_upper_denorm - pred_lower_denorm)
        target_denorm = target_denorm_full[:, :, :obs_target_dim]

        pose_dim = min(pose_action_dim, pred_mean_denorm.shape[-1], action_denorm.shape[-1])
        tracking_err_denorm = action_denorm[:, :, :pose_dim] - pred_mean_denorm[:, :, :pose_dim]
        tracking_l2_denorm = np.linalg.norm(tracking_err_denorm, axis=-1)

        mask_expanded = target_mask[..., None]
        valid_steps = np.clip(np.sum(target_mask), 1e-6, None)
        valid_elements = valid_steps * float(obs_target_dim)

        sq_mean = (pred_mean_denorm - target_denorm) ** 2
        sq_sample = (pred_sample_denorm - target_denorm) ** 2
        per_dim_mse_denorm = np.sum(sq_mean * mask_expanded, axis=(0, 1)) / valid_steps
        metrics["mean_mse_denorm"] = float(np.sum(sq_mean * mask_expanded) / valid_elements)
        metrics["sample_mse_denorm"] = float(np.sum(sq_sample * mask_expanded) / valid_elements)
        metrics["tracking_l2_mean_denorm"] = float(np.sum(tracking_l2_denorm * target_mask) / valid_steps)
        metrics["tracking_l2_mean_norm"] = float(
            np.sum(np.linalg.norm(tracking_err_mean_norm, axis=-1) * target_mask) / valid_steps
        )

        split_label = _get_split_label(rollout_cfg, episode_file.stem)
        if split_label == "test":
            out_suffix = "episode_eval_obs_pred_test"
        elif split_label == "train":
            out_suffix = "episode_eval_obs_pred_train"
        else:
            out_suffix = "episode_eval_obs_pred_unknown"
        out_dir = (run_dir.parent / out_suffix) if out_dir_override is None else (Path(out_dir_override) / out_suffix)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Episode {episode_file.name} | split={split_label} | "
            f"steps={len(steps_arr)} | horizon={pred_horizon} | "
            f"stiffness={int(stiffness_arr[0])} | raw_episode_length={raw_ep['num_steps']}"
        )
        print(
            "Normalization | "
            f"state_window={obs_applied} state_target={target_applied} action={action_applied}"
        )
        print(
            "Metrics | "
            f"sample_mse_norm={metrics['sample_mse_norm']:.5f} "
            f"mean_mse_norm={metrics['mean_mse_norm']:.5f} "
            f"mc_sample_mse_norm={metrics['sample_mse_mc_norm']:.5f} "
            f"pred_var_mean_norm={metrics['pred_var_mean_norm']:.5f} "
            f"tracking_l2_norm={metrics['tracking_l2_mean_norm']:.5f} "
            f"mean_mse_denorm={metrics['mean_mse_denorm']:.5f} "
            f"tracking_l2_denorm={metrics['tracking_l2_mean_denorm']:.5f}"
        )

        collapsed = _collapse_chunk_predictions(
            true_chunks=target_denorm,
            pred_mean_chunks=pred_mean_denorm,
            pred_std_chunks=pred_std_denorm,
            pred_sample_chunks=pred_sample_denorm,
            valid_mask=target_mask,
            anchor_steps=steps_arr,
            stride=int(PREDICTION_STRIDE),
        )

        fig_obs = None
        if collapsed is not None:
            fig_obs = _build_obs_prediction_figure(
                true_obs=collapsed["true"],
                pred_mean=collapsed["pred_mean"],
                pred_std=collapsed["pred_std"],
                pred_sample=collapsed["pred_sample"],
                time_index=collapsed["time_index"],
                max_dims=int(MAX_PLOT_DIMS),
            )
        if fig_obs is not None:
            obs_path = out_dir / f"{episode_file.stem}_obs_prediction.png"
            fig_obs.savefig(obs_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {obs_path}")
            plt.close(fig_obs)

        collapsed_track = _collapse_chunk_predictions(
            true_chunks=tracking_err_denorm,
            pred_mean_chunks=tracking_err_denorm,
            pred_std_chunks=np.zeros_like(tracking_err_denorm),
            pred_sample_chunks=tracking_err_denorm,
            valid_mask=target_mask,
            anchor_steps=steps_arr,
            stride=int(PREDICTION_STRIDE),
        )
        fig_track = None
        if collapsed_track is not None:
            fig_track = _build_tracking_error_figure(
                tracking_error=collapsed_track["true"],
                per_dim_mse=per_dim_mse_denorm,
            )
        if fig_track is not None:
            track_path = out_dir / f"{episode_file.stem}_tracking_error.png"
            fig_track.savefig(track_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {track_path}")
            plt.close(fig_track)

        # metrics_path = out_dir / f"{episode_file.stem}_metrics.json"
        # payload = {
        #     "episode_file": episode_file.name,
        #     "episode_index": int(episode_files.index(episode_file)),
        #     "buffer_path": str(buffer_path),
        #     "raw_episode_dir": str(raw_episode_dir),
        #     "checkpoint": str(ckpt_path),
        #     "rollout_config": str(rollout_config_path),
        #     "normalization_mode": NORMALIZATION_MODE,
        #     "state_window_normalization_applied": bool(obs_applied),
        #     "state_target_normalization_applied": bool(target_applied),
        #     "action_normalization_applied": bool(action_applied),
        #     "num_samples": int(NUM_SAMPLES),
        #     "prediction_stride": int(PREDICTION_STRIDE),
        #     "obs_window": int(obs_window),
        #     "raw_obs_dim": int(raw_obs_dim),
        #     "obs_input_dim": int(obs_input_dim),
        #     "obs_target_dim": int(obs_target_dim),
        #     "pose_action_dim": int(pose_action_dim),
        #     "action_index_offset": int(action_index_offset),
        #     "target_index_offset": int(target_index_offset),
        #     "num_eval_steps": int(len(steps_arr)),
        #     "raw_episode_num_steps": int(raw_ep["num_steps"]),
        #     "stiffness_label": int(stiffness_arr[0]),
        #     "goal_label_counts": {
        #         str(k): int(v) for k, v in zip(*np.unique(goal_arr, return_counts=True))
        #     },
        #     "metrics": metrics,
        # }
        # with open(metrics_path, "w") as f:
        #     json.dump(payload, f, indent=2)
        # print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
