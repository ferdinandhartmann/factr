#!/usr/bin/env python3
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from factr.arrangement import arrangement_id_to_one_hot
from hydra.utils import instantiate
from matplotlib.collections import LineCollection
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*", category=FutureWarning)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------- Configs -----------------------------------
RUN_NAME = "categ_n_b0005_v8k8"
DATASET_NAME = "fourgoals_2"
CHECKPOINT_NAME = "latest_ckpt.ckpt"

EPISODE_FILE_NAME = "ep_52_stiff.pkl"

GPU_ID = 2  # set to None for CPU fallback

# Number of prior samples per timestep.
NUM_SAMPLES = 30

# Use this trajectory horizon for plotting. None means full available horizon.
MAX_STEPS_TO_PLOT = None

# Plot every Nth timestep.
PREDICTION_STRIDE = 40


RUN_DIR = Path.home() / "activeinference" / "factr" / "checkpoints" / DATASET_NAME / RUN_NAME / "rollout"
RAW_EPISODE_DIR = Path.home() / "activeinference" / "factr" / "process_data" / "data_to_process" / DATASET_NAME / "data"

BUFFER_PATH_OVERRIDE = Path.home() / "activeinference" / "factr" / "process_data" / "processed_data" / DATASET_NAME / "buf_test.pkl"
ROLLOUT_CONFIG_OVERRIDE = Path.home() / "activeinference" / "factr" / "process_data" / "processed_data" / DATASET_NAME / "rollout_config.yaml"

OUT_DIR_OVERRIDE = None


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
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def _load_rollout_config(rollout_config_path: Path):
    if not rollout_config_path.exists():
        raise FileNotFoundError(f"rollout_config.yaml not found: {rollout_config_path}")
    with open(rollout_config_path, "r") as f:
        return yaml.safe_load(f)


def _extract_fixed_vector(msg, keys, dim):
    if isinstance(msg, dict):
        for key in keys:
            if key in msg and msg[key] is not None:
                arr = np.asarray(msg[key], dtype=np.float32).reshape(-1)
                if arr.size == int(dim):
                    return arr
        raise ValueError(f"Missing expected keys {keys} or wrong dim={dim} in message")

    arr = np.asarray(msg, dtype=np.float32).reshape(-1)
    if arr.size == int(dim):
        return arr
    raise ValueError(f"Expected flat vector with dim={dim}, got dim={arr.size}")


def _stiffness_vec_to_class(stiffness_vec, thresholds: Optional[List[float]]) -> int:
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
    if "data" not in raw_data:
        raise ValueError("Raw episode file must contain 'data'.")
    data = raw_data["data"]

    obs_cfg = rollout_cfg.get("obs_config", {})
    action_cfg = rollout_cfg.get("action_config", {})
    state_topics = list(obs_cfg.get("state_topics", []))
    action_topics = list(action_cfg.keys())
    if len(state_topics) == 0:
        raise ValueError("No state_topics found in rollout_config")
    if len(action_topics) != 1:
        raise ValueError(f"Expected one action topic, got {action_topics}")

    action_topic = action_topics[0]
    action_dim = int(action_cfg[action_topic])

    stiffness_info = obs_cfg.get("stiffness_label", {})
    stiffness_topic = stiffness_info.get("topic", None)
    stiffness_key = stiffness_info.get("key", "stiffness")
    stiffness_thresholds = stiffness_info.get("norm_thresholds")
    arrangement_topic = obs_cfg.get("arrangement_topic", None)

    required_topics = list(state_topics) + [action_topic]
    if stiffness_topic:
        required_topics.append(stiffness_topic)
    if arrangement_topic:
        required_topics.append(arrangement_topic)
    missing_topics = [topic for topic in required_topics if topic not in data]
    if len(missing_topics) > 0:
        raise ValueError(f"Missing required topics in raw episode file: {missing_topics}")

    seq_lengths = [len(data[topic]) for topic in required_topics]
    if len(seq_lengths) == 0 or min(seq_lengths) <= 0:
        raise ValueError("Raw episode has empty required topic sequences")
    num_steps = int(min(seq_lengths))

    state_specs = {
        "/franka_robot_state_broadcaster/robot_state": {"keys": ["ee_pose"], "dim": 9},
        "/cartesian_impedance_controller/ee_velocity": {"keys": ["ee_velocity"], "dim": 6},
        "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame": {"keys": ["external_wrench"], "dim": 6},
        "/cartesian_impedance_controller/tracking_error": {"keys": ["tracking_error"], "dim": 6},
    }

    action_specs = {"/cartesian_impedance_controller/pose_command": {"keys": ["ee_pose_commanded"], "dim": action_dim}}

    state_arrays = []
    for topic in state_topics:
        if topic not in state_specs:
            raise ValueError(f"Unsupported state topic without explicit spec: {topic}")
        spec = state_specs[topic]
        msgs = data[topic][:num_steps]
        vecs = [_extract_fixed_vector(msg, spec["keys"], spec["dim"]) for msg in msgs]
        topic_arr = np.stack(vecs, axis=0)
        state_arrays.append(topic_arr)
    states = np.concatenate(state_arrays, axis=-1).astype(np.float32)

    if action_topic not in action_specs:
        raise ValueError(f"Unsupported action topic without explicit spec: {action_topic}")
    spec = action_specs[action_topic]
    action_msgs = data[action_topic][:num_steps]
    action_vecs = [_extract_fixed_vector(msg, spec["keys"], spec["dim"]) for msg in action_msgs]
    actions = np.stack(action_vecs, axis=0).astype(np.float32)

    if stiffness_topic:
        raw_stiff = data[stiffness_topic][:num_steps]
        stiff_vecs = [_extract_fixed_vector(msg, [stiffness_key], 6) for msg in raw_stiff]
        stiff_labels = np.asarray([_stiffness_vec_to_class(v, stiffness_thresholds) for v in stiff_vecs], dtype=np.int64)
        episode_label = int(stiff_labels[0]) if len(stiff_labels) > 0 else 1
    else:
        episode_label = 1

    arrangement_vector = None
    if arrangement_topic:
        arrangement_msg = data[arrangement_topic][0]
        arrangement_raw = _extract_fixed_vector(
            arrangement_msg,
            ["arrangement", "arrangement_id"],
            1,
        )
        arrangement_vector = arrangement_id_to_one_hot(arrangement_raw)

    return {
        "states": states,
        "actions": actions,
        "episode_label": int(episode_label),
        "arrangement_vector": arrangement_vector,
        "num_steps": num_steps,
    }


def _build_eval_windows(states: np.ndarray, actions: np.ndarray, episode_label: int, obs_window: int, ac_chunk: int, action_index_offset: int) -> Dict[str, np.ndarray]:
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(f"Expected 2D states/actions, got states={states.shape} actions={actions.shape}")

    max_t = states.shape[0] - int(action_index_offset)
    if max_t <= 0:
        raise ValueError("Episode too short for action_index_offset")

    obs_arr, action_arr, mask_arr, labels_arr = [], [], [], []

    for t_idx in range(max_t):
        start = max(0, t_idx - int(obs_window) + 1)
        win_states = [states[i] for i in range(start, t_idx + 1)]
        while len(win_states) < int(obs_window):
            win_states.insert(0, win_states[0])
        obs_window_arr = np.stack(win_states, axis=0).astype(np.float32)

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

    return {
        "obs": np.stack(obs_arr, axis=0).astype(np.float32),
        "actions": np.stack(action_arr, axis=0).astype(np.float32),
        "mask": np.stack(mask_arr, axis=0).astype(np.float32),
        "labels": np.asarray(labels_arr, dtype=np.int64),
    }


def _safe_denominator(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr[np.abs(arr) < 1e-12] = 1e-12
    return arr


def _forward_group_transform(values: np.ndarray, group: Dict) -> np.ndarray:
    gtype = group.get("type", "identity")
    if gtype == "identity":
        return values
    if gtype in ("gaussian", "gaussian_clip", "zscore_clip"):
        mean = np.asarray(group.get("mean", []), dtype=np.float32)
        std = _safe_denominator(group.get("std", []))
        out = (values - mean) / std
        clip = group.get("clip", None)
        if clip is not None:
            out = np.clip(out, -float(clip), float(clip))
        return out
    return values


def _inverse_group_transform(values: np.ndarray, group: Dict) -> np.ndarray:
    gtype = group.get("type", "identity")
    if gtype == "identity":
        return values
    if gtype in ("gaussian", "gaussian_clip", "zscore_clip"):
        mean = np.asarray(group.get("mean", []), dtype=np.float32)
        std = _safe_denominator(group.get("std", []))
        return values * std + mean
    return values


def _apply_grouped_transform(values: np.ndarray, stats: Dict, inverse: bool = False) -> np.ndarray:
    arr = values.copy()
    if not stats:
        return arr

    mode = stats.get("mode", None)
    if mode != "grouped":
        if "mean" in stats and "std" in stats:
            mean = np.asarray(stats.get("mean", []), dtype=np.float32)
            std = _safe_denominator(stats.get("std", []))
            if mean.size == arr.shape[-1] and std.size == arr.shape[-1]:
                if inverse:
                    return arr * std + mean
                return (arr - mean) / std
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


def _predict_actions(
    model,
    device: torch.device,
    obs_norm: np.ndarray,
    labels: np.ndarray,
    num_samples: int,
    arrangement_vectors: Optional[np.ndarray] = None,
) -> np.ndarray:
    obs_t = torch.from_numpy(obs_norm).float().to(device)
    labels_t = torch.from_numpy(labels).long().to(device)
    arrangement_t = (
        torch.from_numpy(arrangement_vectors).float().to(device)
        if arrangement_vectors is not None
        else None
    )

    ### Make predictions. image is empty, and class_label is stiffness label
    with torch.no_grad():
        pred = model.get_actions_prior(
            {},
            obs_t,
            class_labels=labels_t,
            arrangement_vectors=arrangement_t,
            sample=True,
            num_samples=int(num_samples),
        )

    # keep (B, S, T, D)
    return pred.detach().cpu().numpy()


def _plot_direct_values(true_actions: np.ndarray, measured_pose: np.ndarray, pred_actions: np.ndarray, title: str, out_path: Path, stride: int, step_idx: np.ndarray):
    if true_actions.ndim != 2:
        raise ValueError(f"Expected true_actions shape (T, D), got {true_actions.shape}")
    if measured_pose.ndim != 2:
        raise ValueError(f"Expected measured_pose shape (T, D), got {measured_pose.shape}")
    if pred_actions.ndim != 4:
        raise ValueError(f"Expected pred_actions shape (T, S, K, D), got {pred_actions.shape}")
    if true_actions.shape[0] != measured_pose.shape[0] or true_actions.shape[0] != pred_actions.shape[0]:
        raise ValueError("Time dimension mismatch between true, measured, and pred")
    if true_actions.shape[1] != measured_pose.shape[1] or true_actions.shape[1] != pred_actions.shape[3]:
        raise ValueError("Dim mismatch between true, measured, and pred")

    num_steps, action_dim = true_actions.shape
    num_samples = pred_actions.shape[1]
    chunk_len = pred_actions.shape[2]
    x = np.arange(num_steps + (chunk_len - 1))
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, max(4, action_dim * 1.6)), sharex=True)
    if action_dim == 1:
        axes = [axes]

    for d in range(action_dim):
        ax = axes[d]
        ax.plot(x[:num_steps], measured_pose[:, d], color="gray", linewidth=1.0, linestyle="--", alpha=0.9, label="measured")
        ax.plot(x[:num_steps], true_actions[:, d], color="black", linewidth=1.0, label="ground truth")
        for t in step_idx:
            t_pred = t + np.arange(chunk_len, dtype=np.int64)
            valid = t_pred < x.shape[0]
            if not np.any(valid):
                continue
            t_pred = t_pred[valid]
            if t_pred.shape[0] < 2:
                continue
            # Plot each sampled chunk to match fan-style sampling behavior.
            for s_idx in range(num_samples):
                y = pred_actions[t, s_idx, : t_pred.shape[0], d]
                points = np.column_stack([x[t_pred], y]).reshape(-1, 1, 2)
                segs = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segs, colors="red", linewidths=0.8, alpha=0.35)
                ax.add_collection(lc)
        ax.set_ylabel(f"a[{d}]")
        ax.grid(True, alpha=0.25)
        if d == 0:
            ax.plot([], [], color="red", linewidth=0.95, label="pred samples")
            ax.legend(loc="upper right", ncols=2)
    axes[-1].set_xlabel("time step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=330)
    print(f"Saved: {out_path}")
    return fig


def main():
    run_dir = Path(RUN_DIR)
    exp_config_path = run_dir / "exp_config.yaml"
    ckpt_path = run_dir / str(CHECKPOINT_NAME)
    if not exp_config_path.exists():
        raise FileNotFoundError(f"exp_config.yaml not found: {exp_config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    cfg = _load_run_cfg(exp_config_path)
    buffer_path = Path(BUFFER_PATH_OVERRIDE) if BUFFER_PATH_OVERRIDE is not None else Path(cfg.test_buffer_path)
    rollout_config_path = Path(ROLLOUT_CONFIG_OVERRIDE) if ROLLOUT_CONFIG_OVERRIDE is not None else (buffer_path.parent / "rollout_config.yaml")
    rollout_cfg = _load_rollout_config(rollout_config_path)

    if not Path(RAW_EPISODE_DIR).exists():
        raise FileNotFoundError(f"RAW_EPISODE_DIR not found: {RAW_EPISODE_DIR}")

    episode_file = Path(RAW_EPISODE_DIR) / str(EPISODE_FILE_NAME)
    if not episode_file.exists():
        raise FileNotFoundError(f"EPISODE_FILE_NAME not found: {episode_file}")
    print(f"Selected episode: {episode_file}")

    raw_ep = _load_raw_episode_to_arrays(episode_file, rollout_cfg)

    obs_window = int(cfg.obs_window)
    ac_chunk = int(cfg.ac_chunk)
    action_index_offset = int(OmegaConf.select(cfg, "task.test_buffer.action_index_offset", default=0))
    print(f"Detected from config: obs_window={obs_window} ac_chunk={ac_chunk} action_index_offset={action_index_offset}")

    ep_data = _build_eval_windows(
        states=raw_ep["states"],
        actions=raw_ep["actions"],
        episode_label=int(raw_ep["episode_label"]),
        obs_window=obs_window,
        ac_chunk=ac_chunk,
        action_index_offset=action_index_offset,
    )

    obs_arr = ep_data["obs"]
    actions_arr = ep_data["actions"]
    labels_arr = ep_data["labels"]
    use_arrangement_conditioning = bool(
        OmegaConf.select(cfg, "use_arrangement_conditioning", default=False)
    )
    arrangement_vectors = None
    if use_arrangement_conditioning:
        if raw_ep["arrangement_vector"] is None:
            raise ValueError(
                "This checkpoint enables arrangement conditioning, but the raw episode has no arrangement topic."
            )
        arrangement_vectors = np.repeat(
            raw_ep["arrangement_vector"][None, :],
            obs_arr.shape[0],
            axis=0,
        ).astype(np.float32)

    state_stats = rollout_cfg.get("norm_stats", {}).get("state", None)
    action_stats = rollout_cfg.get("norm_stats", {}).get("action", None)

    # Normalize with rollout_config stats before model inference.
    obs_norm = _apply_grouped_transform(obs_arr, state_stats, inverse=False)
    actions_norm = _apply_grouped_transform(actions_arr, action_stats, inverse=False)

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
    print(f"Using device: {device}")

    model = _load_model(cfg, ckpt_path, device)

    pred_samples_norm = _predict_actions(
        model=model,
        device=device,
        obs_norm=obs_norm,
        labels=labels_arr,
        num_samples=int(NUM_SAMPLES),
        arrangement_vectors=arrangement_vectors,
    )

    # Convert predictions and targets back to original value space.
    pred_samples_denorm = _apply_grouped_transform(pred_samples_norm, action_stats, inverse=True)
    true_denorm = _apply_grouped_transform(actions_norm, action_stats, inverse=True)
    obs_denorm = _apply_grouped_transform(obs_norm, state_stats, inverse=True)

    if pred_samples_denorm.ndim != 4:
        raise ValueError(f"Expected pred_samples to have 4 dims (B, S, T, D), got {pred_samples_denorm.shape}")

    pose_dim = min(9, true_denorm.shape[-1], obs_denorm.shape[-1])
    true_series = true_denorm[:, 0, :pose_dim]
    measured_series = obs_denorm[:, -1, :pose_dim]
    pred_series = pred_samples_denorm[:, :, :, :pose_dim]

    if MAX_STEPS_TO_PLOT is not None:
        max_steps = int(MAX_STEPS_TO_PLOT)
        true_series = true_series[:max_steps]
        measured_series = measured_series[:max_steps]
        pred_series = pred_series[:max_steps]

    step_idx_plot = np.arange(0, true_series.shape[0], PREDICTION_STRIDE, dtype=np.int64)
    if step_idx_plot[-1] != (true_series.shape[0] - 1):
        step_idx_plot = np.concatenate([step_idx_plot, np.asarray([true_series.shape[0] - 1], dtype=np.int64)])

    if OUT_DIR_OVERRIDE is None:
        out_dir = run_dir.parent / "episode_eval_simple"
    else:
        out_dir = Path(OUT_DIR_OVERRIDE)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{episode_file.stem}_direct_chunk_values_prior.png"
    fig = _plot_direct_values(
        true_actions=true_series,
        measured_pose=measured_series,
        pred_actions=pred_series,
        title=(f"{episode_file.name} | source=prior | samples={int(NUM_SAMPLES)} | stride={PREDICTION_STRIDE} | steps={true_series.shape[0]}"),
        out_path=out_path,
        stride=PREDICTION_STRIDE,
        step_idx=step_idx_plot,
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
