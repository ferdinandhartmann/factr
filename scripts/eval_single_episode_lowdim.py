#!/usr/bin/env python3
import argparse
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
from factr.utils import apply_grouped_transform as _apply_grouped_transform
from factr.utils import ensure_normalized as _ensure_normalized
from factr.utils_plot import (
    RPYPlotConfig,
    build_pose_3d_figure,
    build_pose_comparison_figure,
    build_pose_fan_figure,
    pose_chunks_for_plot,
)
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
DATASET_NAME = "fourgoals_2"
DATASET_PROJECT_PREFIX = "aifact_"
BUFFER_SET_NAME = "fourgoals_2_allgauss_noclip_cmdinput_rel"
RUN_NAME = "categ_b001_v4k4_ct12_klb08_obs4"

CHECKPOINT_NAME = "latest_ckpt.ckpt"

USE_EPISODE_LIST = True
EPISODE_FILE_NAME = "ep_52_stiff.pkl"
EPISODE_INDEX = 0  # index in sorted *.pkl files
EPISODE_LIST = []
if (DATASET_PROJECT_PREFIX + BUFFER_SET_NAME) == "aifact_fourgoals_2":
    EPISODE_LIST = [
        # "ep_03_soft",
        # "ep_13_stiff",
        # "ep_13_soft",
        # "ep_15_stiff",
        # "ep_20_stiff",
        "ep_28_soft",
        "ep_34_stiff",
        "ep_42_stiff",
        # "ep_43_stiff",
        "ep_49_soft",
        # "ep_57_soft",
        # "ep_58_stiff",
    ]
elif (DATASET_PROJECT_PREFIX + BUFFER_SET_NAME) == "aifact_fourgoals_2_stiff2":
    EPISODE_LIST = [
        # "ep_06_stiff",
        # "ep_20_stiff",
        # "ep_21_stiff",
        "ep_25_stiff",
        "ep_29_stiff",
        "ep_50_stiff",
    ]
elif (DATASET_PROJECT_PREFIX + BUFFER_SET_NAME) == "aifact_fourgoals_2_allgauss_noclip_cmdinput":
    EPISODE_LIST = [
        # "ep_03_soft",
        # "ep_13_stiff",
        # "ep_13_soft",
        # "ep_15_stiff",
        # "ep_20_stiff",
        "ep_28_soft",
        # "ep_34_stiff",
        "ep_42_stiff",
        # "ep_43_stiff",
        # "ep_49_soft",
        # "ep_57_soft",
        "ep_58_stiff",
    ]
elif (DATASET_PROJECT_PREFIX + BUFFER_SET_NAME) == "aifact_fourgoals_12_allgauss_noclip_cmdinput":
    EPISODE_LIST = [
        # "ep_103_soft",
        # "ep_109_stiff",
        # "ep_110_medium",
        # "ep_112_medium",
        # "ep_113_medium",
        "ep_113_stiff",
        "ep_114_medium",
        # "ep_115_medium",
        # "ep_116_medium",
        # "ep_123_soft",
        "ep_124_soft",
        # "ep_128_medium",
        # "ep_129_soft",
        # "ep_130_stiff",
        # "ep_139_stiff",
        # "ep_218_stiff",
        "ep_225_soft",
        # "ep_232_stiff",
        # "ep_247_stiff",
        # "ep_250_stiff",
        # "ep_252_stiff",
        # "ep_253_soft",
        "ep_258_stiff",
        # "ep_260_stiff",
    ]
elif (DATASET_PROJECT_PREFIX + BUFFER_SET_NAME) == "aifact_fourgoals_2_allgauss_noclip_cmdinput_rel":
    EPISODE_LIST = [
        # "ep_03_soft",
        # "ep_13_stiff",
        # "ep_13_soft",
        # "ep_15_stiff",
        # "ep_20_stiff",
        "ep_28_soft",
        # "ep_34_stiff",
        # "ep_42_stiff",
        # "ep_43_stiff",
        # "ep_49_soft",
        # "ep_57_soft",
        "ep_58_stiff",
    ]
elif (DATASET_PROJECT_PREFIX + BUFFER_SET_NAME) == "aifact_fourgoals_12_allgauss_noclip_cmdinput_rel":
    EPISODE_LIST = [
        # "ep_103_soft",
        # "ep_109_stiff",
        # "ep_110_medium",
        # "ep_112_medium",
        # "ep_113_medium",
        # "ep_113_stiff",
        # "ep_114_medium",
        # "ep_115_medium",
        # "ep_116_medium",
        # "ep_123_soft",
        # "ep_124_soft",
        # "ep_128_medium",
        # "ep_129_soft",
        # "ep_130_stiff",
        # "ep_139_stiff",
        # "ep_218_stiff",
        "ep_225_soft",
        # "ep_232_stiff",
        # "ep_247_stiff",
        # "ep_250_stiff",
        # "ep_252_stiff",
        # "ep_253_soft",
        "ep_258_stiff",
        # "ep_260_stiff",
    ]


LIST_EPISODES_ONLY = False  # list available episodes and exit, without running evaluation

RUN_DIR = Path.home() / "activeinference" / "factr" / "checkpoints" / (DATASET_PROJECT_PREFIX + BUFFER_SET_NAME) / RUN_NAME / "rollout"
# Raw episode source
RAW_EPISODE_DIR = Path.home() / "activeinference" / "factr" / "process_data" / "data_to_process" / DATASET_NAME / "data"

BUFFER_PATH_OVERRIDE = Path.home() / "activeinference" / "factr" / "process_data" / "processed_data" / BUFFER_SET_NAME / "buf_test.pkl"
ROLLOUT_CONFIG_OVERRIDE = Path.home() / "activeinference" / "factr" / "process_data" / "processed_data" / BUFFER_SET_NAME / "rollout_config.yaml"

NUM_SAMPLES = 30
ACTION_SOURCE = "prior"  # one of: prior, posterior
NORMALIZATION_MODE = "apply"  # one of: auto, apply, skip # for denormalization
PREDICTION_STRIDE = 40  # stride for fan plot + 3d plot
VIEW_ELEV = 24
VIEW_AZIM = -60
SHOW_PLOT = False  # shows also 3d plot
ENABLE_TRAIN_BACKGROUND = True
TRAIN_BACKGROUND_ONLY_MEDIUM = False
RPY_SUBTRACT_PI = True  # If True, subtract pi from one selected RPY axis for plotting.
RPY_SUBTRACT_PI_AXIS = 0  # 0=roll, 1=pitch, 2=yaw
RPY_PLOT_UNIT = "deg"  # "rad" or "deg"
PLOT_GEODESIC_SUBPLOT = False
EVAL_PLOT_POSE_MODE = "absolute"  # one of: absolute, relative

GPU_ID = 2

GLOBAL_AXIS_LIMITS = {
    "x": (0.2, 0.6),
    "y": (-0.4, 0.4),
    "z": (0.0, 0.7),
}

GOAL_FRAMES = [
    # fourgoals_1
    # {"name": "goal 1", "pose": [0.384, -0.26, 0.181, 0.998, 0.014, 0.065, 0.011, -0.999, 0.05]},
    # {"name": "goal 2", "pose": [0.656, -0.155, -0.023, 0.996, -0.054, 0.065, -0.057, -0.997, 0.045]},
    # {"name": "goal 3", "pose": [0.51, 0.269, -0.028, -0.015, 1.0, -0.027, 1.0, 0.015, 0.013]},
    # {"name": "goal 4", "pose": [0.466, 0.361, 0.48, 0.998, 0.056, 0.032, 0.034, -0.034, -0.999]},
    # fourgoals_2
    {"name": "goal 1", "pose": [0.341, 0.240, 0.606, 0.999, -0.007, 0.013, -0.007, -1.000, -0.010]},  # 33
    {"name": "goal 2", "pose": [0.524, 0.226, 0.381, 1.000, 0.013, 0.025, 0.013, -1.000, 0.004]},  # 27
    {"name": "goal 3", "pose": [0.591, -0.336, -0.038, 0.907, -0.421, 0.037, -0.421, -0.907, 0.006]},  # 31
    {"name": "goal 4", "pose": [0.439, -0.239, -0.043, 0.905, -0.425, 0.023, -0.426, -0.904, 0.028]},  # 29
]

OUT_DIR_OVERRIDE = None

if SHOW_PLOT:
    plt.switch_backend("TkAgg")


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("len"):
        OmegaConf.register_new_resolver("len", lambda x: len(x))


def _normalize_action_source(value: str) -> str:
    action_source = str(value).strip().lower()
    if action_source not in {"prior", "posterior"}:
        raise ValueError(f"ACTION_SOURCE must be one of prior/posterior, got: {value}")
    return action_source


def _action_source_title(action_source: str) -> str:
    return "Prior" if action_source == "prior" else "Posterior"


def _normalize_pose_mode(value: str) -> str:
    pose_mode = str(value).strip().lower()
    if pose_mode not in {"absolute", "relative", "relative_timesteps", "relative_chunks"}:
        raise ValueError(f"pose mode must be one of absolute/relative/relative_timesteps/relative_chunks, got {value}")
    return pose_mode


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


def _optional_metric_float(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _infer_action_pose_mode(cfg, rollout_cfg, action_stats) -> str:
    """Infer whether actions are absolute or relative deltas for eval plotting."""
    mode = OmegaConf.select(cfg, "action_pose_mode", default=None)
    if mode is None and isinstance(rollout_cfg, dict):
        mode = rollout_cfg.get("action_pose_mode", None)
    if mode is not None:
        return str(mode).strip().lower()

    if isinstance(action_stats, dict) and action_stats.get("mode", None) == "grouped":
        std_vals = []
        for group in action_stats.get("groups", []):
            std = np.asarray(group.get("std", []), dtype=np.float32).reshape(-1)
            if std.size > 0:
                std_vals.append(std)
        if std_vals:
            mean_std = float(np.mean(np.concatenate(std_vals)))
            if np.isfinite(mean_std) and mean_std < 0.02:
                return "relative"

    return "absolute"


def _extract_state_from_buffer_obs(obs) -> Optional[np.ndarray]:
    if isinstance(obs, dict) and "state" in obs:
        return np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    if hasattr(obs, "state"):
        return np.asarray(getattr(obs, "state"), dtype=np.float32).reshape(-1)
    if hasattr(obs, "obs") and isinstance(obs.obs, dict) and "state" in obs.obs:
        return np.asarray(obs.obs["state"], dtype=np.float32).reshape(-1)
    return None


def _load_train_buffer_background(
    buf_path: Path,
    action_stats: Optional[dict],
    state_stats: Optional[dict],
    action_pose_mode: str,
) -> List[np.ndarray]:
    import pickle

    with open(buf_path, "rb") as f:
        buffer = pickle.load(f)

    background = []
    if isinstance(buffer, (list, tuple)) and buffer and isinstance(buffer[0], (list, tuple)):
        for traj in buffer:
            if not traj or not isinstance(traj[0], tuple):
                continue

            actions = []
            pose0 = None
            for entry in traj:
                try:
                    obs, action, _ = entry
                    actions.append(np.asarray(action, dtype=np.float32))
                    if pose0 is None:
                        state0 = _extract_state_from_buffer_obs(obs)
                        if state0 is not None and state_stats is not None:
                            state0 = _apply_grouped_transform(state0[None, :], state_stats, inverse=True)[0]
                        if state0 is not None and state0.size >= 9:
                            pose0 = state0[:9]
                except Exception:
                    continue

            if not actions:
                continue

            actions_arr = np.stack(actions, axis=0)
            if action_stats is not None:
                actions_arr = _apply_grouped_transform(actions_arr, action_stats, inverse=True)

            action_dim = actions_arr.shape[-1]
            pose_dim = min(9, action_dim)
            pose_seq = actions_arr[:, :pose_dim]
            if action_pose_mode == "relative":
                base = pose0 if pose0 is not None else np.zeros((pose_dim,), dtype=np.float32)
                pose_seq = np.cumsum(pose_seq, axis=0) + base[None, :]

            background.append(pose_seq)

    return background


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


def _stiffness_vec_to_class(stiffness_vec, thresholds: Optional[List[float]]) -> int:
    # Honor rollout-config behavior: empty thresholds means single-class label 1.
    if not thresholds:
        return 1

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
    stiffness_thresholds = stiffness_info.get("norm_thresholds")

    topics_for_sync = list(state_topics) + [action_topic]
    if stiffness_topic:
        topics_for_sync.append(stiffness_topic)
    # Deduplicate topics while preserving order. This avoids double-processing
    # when the action topic is also present in state_topics (e.g. cmd included).
    topics_for_sync = list(dict.fromkeys(topics_for_sync))
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
        "/cartesian_impedance_controller/pose_command": {
            "keys": ["ee_pose_commanded"],
            "dim": action_dim,
            "fallback": None,
        },
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


def _build_pose_figure_with_measured(true_actions, pred_actions, measured_pose, mask, title):
    rpy_cfg = RPYPlotConfig(
        subtract_pi=bool(RPY_SUBTRACT_PI),
        subtract_pi_axis=int(RPY_SUBTRACT_PI_AXIS),
        unit=str(RPY_PLOT_UNIT),
    )
    return build_pose_comparison_figure(
        true_values=true_actions,
        pred_values=pred_actions,
        mask=mask,
        title=title,
        measured_values=measured_pose,
        plot_geodesic_subplot=bool(PLOT_GEODESIC_SUBPLOT),
        rpy_config=rpy_cfg,
    )


def _build_fan_figure_with_measured(
    true_action_chunks: np.ndarray,
    pred_action_chunks: np.ndarray,
    measured_pose: np.ndarray,
    mask_chunks: np.ndarray,
    source_time_index: np.ndarray,
    prediction_stride: int,
    action_source: str,
    background_actions: Optional[List[np.ndarray]] = None,
):
    rpy_cfg = RPYPlotConfig(
        subtract_pi=bool(RPY_SUBTRACT_PI),
        subtract_pi_axis=int(RPY_SUBTRACT_PI_AXIS),
        unit=str(RPY_PLOT_UNIT),
    )
    return build_pose_fan_figure(
        true_action_chunks=true_action_chunks,
        pred_action_chunks=pred_action_chunks,
        mask_chunks=mask_chunks,
        source_time_index=source_time_index,
        prediction_stride=prediction_stride,
        measured_pose=measured_pose,
        background_actions=background_actions,
        max_plot_steps=None,
        title=(
            f"Sampled {_action_source_title(action_source)} Fan + Measured Pose + Ground-Truth Command Pose "
            f"(full episode, stride={max(1, int(prediction_stride))})"
        ),
        plot_ground_truth_reconstructed=True,
        plot_geodesic_subplot=bool(PLOT_GEODESIC_SUBPLOT),
        rpy_config=rpy_cfg,
    )


def _summarize_metrics(
    model,
    device: torch.device,
    obs_norm: np.ndarray,
    actions_norm: np.ndarray,
    mask_norm: np.ndarray,
    labels: np.ndarray,
    num_samples: int,
    action_source: str,
):
    obs_t = torch.from_numpy(obs_norm).float().to(device)
    actions_t = torch.from_numpy(actions_norm).float().to(device)
    mask_t = torch.from_numpy(mask_norm).float().to(device)
    labels_t = torch.from_numpy(labels).long().to(device)

    ac_flat = actions_t.reshape(actions_t.shape[0], -1)
    mask_flat = mask_t.reshape(mask_t.shape[0], -1)

    with torch.no_grad():
        output = model({}, obs_t, ac_flat, mask_flat, class_labels=labels_t)
        if action_source == "prior":
            pred_det = model.get_actions_prior({}, obs_t, class_labels=labels_t, sample=False, num_samples=1)
            pred_samples = model.get_actions_prior({}, obs_t, class_labels=labels_t, sample=True, num_samples=num_samples)
        else:
            pred_det = model.get_actions_pos({}, obs_t, actions_t, class_labels=labels_t, sample=False, num_samples=1)
            pred_samples = model.get_actions_pos({}, obs_t, actions_t, class_labels=labels_t, sample=True, num_samples=num_samples)

        if pred_det.ndim == 4:
            pred_det = pred_det[:, 0]

    mask_den = mask_t.sum((1, 2)).clamp(min=1.0)
    selected_det_l1 = torch.abs(mask_t * (pred_det - actions_t))
    selected_det_l1 = selected_det_l1.sum((1, 2)) / mask_den
    action_l2 = torch.square(mask_t * (pred_det - actions_t))
    action_l2 = action_l2.sum((1, 2)) / mask_den
    lsig = torch.logical_or(
        torch.logical_and(actions_t > 0, pred_det <= 0),
        torch.logical_and(actions_t <= 0, pred_det > 0),
    )
    lsig = (lsig.float() * mask_t).sum((1, 2)) / mask_den

    metrics = {
        "posterior_l1": float(output["l1_loss"].item()),
        "selected_det_l1": float(selected_det_l1.mean().item()),
        "posterior_kl": float(output["kl"].item()),
        "action_l2": float(action_l2.mean().item()),
        "action_lsig": float(lsig.mean().item()),
        "prior_std_mean": _optional_metric_float(output.get("prior_std_mean")),
        "posterior_std_mean": _optional_metric_float(output.get("posterior_std_mean")),
        "prior_entropy": _optional_metric_float(output.get("prior_entropy")),
        "posterior_entropy": _optional_metric_float(output.get("posterior_entropy")),
    }

    return metrics, pred_det.detach().cpu().numpy(), pred_samples.detach().cpu().numpy()


def main():
    if NORMALIZATION_MODE not in ("auto", "apply", "skip"):
        raise ValueError(f"NORMALIZATION_MODE must be one of auto/apply/skip, got: {NORMALIZATION_MODE}")
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    action_source = _normalize_action_source(ACTION_SOURCE)
    action_source_title = _action_source_title(action_source)

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
    cfg_pose_mode = OmegaConf.select(
        cfg,
        "eval_plot_pose_mode",
        default=OmegaConf.select(cfg, "task.eval_plot_pose_mode", default="absolute"),
    )
    plot_pose_mode = _normalize_pose_mode(cfg_pose_mode)
    action_chunk_mode = str(OmegaConf.select(cfg, "action_chunk_mode", default="absolute")).strip().lower()
    if plot_pose_mode in ("relative", "relative_timesteps") and action_chunk_mode != "relative_timesteps":
        plot_pose_mode = "relative_chunks" if action_chunk_mode == "relative_chunks" else "absolute"
    elif plot_pose_mode == "relative_chunks" and action_chunk_mode != "relative_chunks":
        plot_pose_mode = "relative_timesteps" if action_chunk_mode == "relative_timesteps" else "absolute"
    print(f"Eval config | action_chunk_mode={action_chunk_mode} eval_plot_pose_mode={plot_pose_mode}")
    buffer_path = buffer_path_override if buffer_path_override is not None else Path(cfg.test_buffer_path)
    rollout_config_path = rollout_config_override if rollout_config_override is not None else (buffer_path.parent / "rollout_config.yaml")
    rollout_cfg = _load_rollout_config(rollout_config_path)
    state_stats = rollout_cfg.get("norm_stats", {}).get("state", None)
    action_stats = rollout_cfg.get("norm_stats", {}).get("action", None)
    action_pose_mode = _infer_action_pose_mode(cfg, rollout_cfg, action_stats)

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
            train_background = _load_train_buffer_background(
                train_buf_path,
                action_stats=action_stats,
                state_stats=state_stats,
                action_pose_mode=action_pose_mode,
            )
            print(f"Loaded train background trajectories: {len(train_background)} | {train_buf_path}")
        else:
            print(f"Train buffer not found for background: {train_buf_path}")

    for episode_file in selected:
        print(f"Selected episode file: {episode_file}")

        raw_ep = _load_raw_episode_to_arrays(episode_file, rollout_cfg)
        raw_actions = raw_ep["actions"]
        if action_pose_mode == "relative" and raw_actions.shape[0] > 1:
            rel_actions = np.zeros_like(raw_actions)
            rel_actions[1:] = raw_actions[1:] - raw_actions[:-1]
            raw_actions = rel_actions
        obs_window = int(cfg.obs_window)
        ac_chunk = int(cfg.ac_chunk)
        action_index_offset = int(OmegaConf.select(cfg, "task.test_buffer.action_index_offset", default=1))
        include_tracking_error = bool(
            OmegaConf.select(
                cfg,
                "include_tracking_error",
                default=OmegaConf.select(cfg, "agent.include_tracking_error", default=True),
            )
        )
        states = raw_ep["states"]
        if (not include_tracking_error) and states.shape[-1] >= 36:
            states = np.concatenate([states[:, :21], states[:, 27:]], axis=-1)
        ep_data = _build_eval_samples_from_raw_episode(
            states=states,
            actions=raw_actions,
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

        # Select device based on GPU_ID and CUDA availability
        if torch.cuda.is_available() and GPU_ID is not None:
            device = torch.device(f"cuda:{int(GPU_ID)}")
            try:
                torch.cuda.set_device(int(GPU_ID))
            except Exception:
                pass
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        model = _load_model(cfg, ckpt_path, device)

        metrics, pred_det_norm, pred_samples_norm = _summarize_metrics(
            model=model,
            device=device,
            obs_norm=obs_norm,
            actions_norm=actions_norm,
            mask_norm=mask_arr,
            labels=labels_arr,
            num_samples=int(NUM_SAMPLES),
            action_source=action_source,
        )

        if action_applied:
            actions_denorm = _apply_grouped_transform(actions_norm, action_stats, inverse=True)
            pred_det_denorm = _apply_grouped_transform(pred_det_norm, action_stats, inverse=True)
            pred_samples_denorm = _apply_grouped_transform(pred_samples_norm, action_stats, inverse=True)
        else:
            actions_denorm = actions_norm
            pred_det_denorm = pred_det_norm
            pred_samples_denorm = pred_samples_norm

        if obs_applied:
            obs_denorm = _apply_grouped_transform(obs_norm, state_stats, inverse=True)
        else:
            obs_denorm = obs_norm

        ac_dim = actions_denorm.shape[-1]
        pose_dim = min(9, ac_dim)

        measured_first = obs_denorm[:, -1, :pose_dim]
        actions_plot = pose_chunks_for_plot(actions_denorm[:, :, :pose_dim], measured_first, plot_pose_mode)
        pred_det_plot = pose_chunks_for_plot(pred_det_denorm[:, :, :pose_dim], measured_first, plot_pose_mode)
        pred_samples_plot = pose_chunks_for_plot(pred_samples_denorm[:, :, :, :pose_dim], measured_first[:, None, :], plot_pose_mode)

        true_first = actions_plot[:, 0, :pose_dim]
        pred_first = pred_det_plot[:, 0, :pose_dim]
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
            f"{action_source_title}(det) L1={metrics['selected_det_l1']:.4f} "
            f"KL={metrics['posterior_kl']:.4f} "
            f"Action L2={metrics['action_l2']:.4f} "
            f"LSign={metrics['action_lsig']:.4f} "
            f"prior_std={metrics['prior_std_mean']:.4f} "
            f"post_std={metrics['posterior_std_mean']:.4f} "
            f"prior_H={metrics['prior_entropy']:.4f} "
            f"post_H={metrics['posterior_entropy']:.4f}"
        )

        # First Steps Figure
        # fig_pose = _build_pose_figure_with_measured(
        #     true_actions=true_first,
        #     pred_actions=pred_first,
        #     measured_pose=measured_first,
        #     mask=mask_first,
        #     title=f"Episode {episode_file.name} | Ground Truth vs {action_source_title} Prediction vs Measured Pose",
        # )
        # pose_path = out_dir / f"{episode_file.stem}_pred_firststeps.png"
        # fig_pose.savefig(pose_path, dpi=300, bbox_inches="tight")
        # print(f"Saved: {pose_path}")
        # plt.close(fig_pose)

        fig_fan = _build_fan_figure_with_measured(
            true_action_chunks=actions_plot,
            pred_action_chunks=pred_samples_plot,
            measured_pose=measured_first,
            mask_chunks=mask_arr[:, :, :pose_dim],
            source_time_index=steps_arr,
            prediction_stride=max(1, int(PREDICTION_STRIDE)),
            action_source=action_source,
            background_actions=train_background
            if (ENABLE_TRAIN_BACKGROUND and train_background) and (not TRAIN_BACKGROUND_ONLY_MEDIUM or "medium" in episode_name)
            else None,
        )
        fan_path = out_dir / f"{episode_file.stem}_predictions.png"
        fig_fan.savefig(fan_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {fan_path}")

        if pose_dim >= 9:
            fig_3d = build_pose_3d_figure(
                measured_pose=measured_first,
                true_pose=true_first,
                pred_pose=pred_first,
                sampled_pose_chunks=pred_samples_plot,
                prediction_stride=max(1, int(PREDICTION_STRIDE)),
                action_source=action_source,
                true_action_chunks=actions_plot,
                mask_chunks=mask_arr[:, :, :pose_dim],
                source_time_index=steps_arr,
                plot_ground_truth_reconstructed=True,
                axis_limits=GLOBAL_AXIS_LIMITS,
                goal_frames=GOAL_FRAMES,
                view_elev=float(VIEW_ELEV),
                view_azim=float(VIEW_AZIM),
                show_plot=SHOW_PLOT,
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
