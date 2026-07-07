#!/usr/bin/env python3
import os
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from factr.arrangement import arrangement_id_to_one_hot
from factr.utils import apply_grouped_transform as _apply_grouped_transform
from factr.utils import canonical_action_mode as _canonical_action_mode
from factr.utils import ensure_normalized as _ensure_normalized
from factr.utils import state_stats_without_tracking_error as _state_stats_without_tracking_error
from factr.utils_plot import relative_chunk_from_absolute
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

try:
    import factr.misc
except ImportError:
    pass


def register_if_not_exists(name, resolver):
    if not OmegaConf.has_resolver(name):
        try:
            OmegaConf.register_new_resolver(name, resolver)
        except ValueError:
            pass


register_if_not_exists("mult", lambda x, y: x * y)
register_if_not_exists("add", lambda x, y: x + y)
register_if_not_exists("len", lambda x: len(x))
register_if_not_exists("transform", lambda x: x)
register_if_not_exists("hydra", lambda x: None)

try:
    if "factr.misc" in sys.modules:
        import factr.misc

        factr.misc.get_transform_by_name = lambda name: lambda x: x
except Exception:
    pass


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "eval_params.yaml"


def _load_script_config(config_path: Path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    shared = cfg["shared"]
    script_cfg = cfg["eval_z_distr"]
    merged = dict(shared)
    merged.update(script_cfg)
    return merged


def _materialize_globals(config_path: Path):
    cfg = _load_script_config(config_path)
    dataset_name = cfg["dataset_name"]
    dataset_project_prefix = cfg["dataset_project_prefix"]
    buffer_set_name = str(cfg["buffer_set_name"])
    run_name = cfg["run_name"]
    project_name = dataset_name if buffer_set_name.strip().lower() == "auto" else buffer_set_name
    key = f"{dataset_project_prefix}{project_name}"
    episode_list = list(cfg["episode_lists"].get(key, []))

    return {
        "DATASET_NAME": dataset_name,
        "DATASET_PROJECT_PREFIX": dataset_project_prefix,
        "BUFFER_SET_NAME": buffer_set_name,
        "RUN_NAME": run_name,
        "CHECKPOINT_NAME": cfg["checkpoint_name"],
        "USE_EPISODE_LIST": bool(cfg["use_episode_list"]),
        "EPISODE_FILE_NAME": cfg["episode_file_name"],
        "EPISODE_LIST": episode_list,
        "RUN_DIR": Path.home() / "activeinference" / "factr" / "checkpoints" / key / run_name / "rollout",
        "RAW_EPISODE_DIR": Path.home() / "activeinference" / "factr" / "process_data" / "data_to_process" / dataset_name / "data",
        "GPU_ID": int(cfg["gpu_id"]),
        "SAVE_DIR_OVERRIDE": cfg["save_dir_override"],
        "SAVE_STATIC_PLOTS": bool(cfg["save_static_plots"]),
        "SAVE_VIDEO": bool(cfg["save_video"]),
        "VIDEO_FPS": int(cfg["video_fps"]),
        "VIDEO_DPI": int(cfg["video_dpi"]),
        "VIDEO_FRAME_STRIDE": int(cfg["video_frame_stride"]),
        "VIDEO_X_POINTS": int(cfg["video_x_points"]),
        "VIDEO_X_STD_MULT": float(cfg["video_x_std_mult"]),
        "MAX_PARALLEL_EPISODES": int(cfg["max_parallel_episodes"]),
        "NORMALIZATION_MODE": cfg["normalization_mode"],
    }


globals().update(_materialize_globals(DEFAULT_CONFIG_PATH))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def resolve_paths(project_root: Path, model_name: str, checkpoint: str, rollout_config_arg: Optional[str]) -> Tuple[Path, Path, Path]:
    project_name = DATASET_NAME if str(BUFFER_SET_NAME).strip().lower() == "auto" else BUFFER_SET_NAME
    checkpoints_dir = project_root / "checkpoints" / project_name / model_name
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {checkpoints_dir}")

    if checkpoint == "latest":
        ckpt_path = checkpoints_dir / "rollout" / "latest_ckpt.ckpt"
    else:
        ckpt_name = checkpoint if checkpoint.endswith(".ckpt") else f"{checkpoint}.ckpt"
        ckpt_path = checkpoints_dir / ckpt_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if rollout_config_arg is not None:
        rollout_cfg_path = Path(rollout_config_arg)
    else:
        candidates = [ckpt_path.parent / "rollout_config.yaml", checkpoints_dir / "rollout" / "rollout_config.yaml", checkpoints_dir / "rollout_config.yaml"]
        rollout_cfg_path = None
        for candidate in candidates:
            if candidate.exists():
                rollout_cfg_path = candidate
                break
        if rollout_cfg_path is None:
            raise FileNotFoundError("rollout_config.yaml not found. Please pass --rollout_config explicitly.")

    if not rollout_cfg_path.exists():
        raise FileNotFoundError(f"rollout_config not found: {rollout_cfg_path}")

    return checkpoints_dir, ckpt_path, rollout_cfg_path


def load_model(ckpt_path: Path, device: torch.device):
    cfg_candidates = [
        ckpt_path.parent / "exp_config.yaml",
        ckpt_path.parent / ".hydra" / "config.yaml",
        ckpt_path.parent.parent / "exp_config.yaml",
        ckpt_path.parent.parent / ".hydra" / "config.yaml",
    ]

    cfg_path = None
    for candidate in cfg_candidates:
        if candidate.exists():
            cfg_path = candidate
            break
    if cfg_path is None:
        raise FileNotFoundError(f"Config not found near checkpoint: {ckpt_path}")

    cfg_all = OmegaConf.load(cfg_path)
    cfg = cfg_all.params if "params" in cfg_all else cfg_all
    if "hydra" in cfg:
        cfg.pop("hydra", None)
    OmegaConf.resolve(cfg)

    model = instantiate(cfg.agent)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model, cfg


def _extract_vector(msg, keys: List[str], dim: int) -> np.ndarray:
    if isinstance(msg, dict):
        for key in keys:
            if key in msg and msg[key] is not None:
                arr = np.asarray(msg[key], dtype=np.float32).reshape(-1)
                if arr.size == dim:
                    return arr
    arr = np.asarray(msg, dtype=np.float32).reshape(-1)
    if arr.size == dim:
        return arr
    if arr.size > dim:
        return arr[:dim]
    out = np.full((dim,), np.nan, dtype=np.float32)
    out[: arr.size] = arr
    return out


def _extract_vector_flexible(msg, keys: List[str], expected_dim: Optional[int] = None) -> np.ndarray:
    arr = None
    if isinstance(msg, dict):
        for key in keys:
            if key in msg and msg[key] is not None:
                arr = np.asarray(msg[key], dtype=np.float32).reshape(-1)
                break
        if arr is None:
            parts = []
            for value in msg.values():
                if value is not None and isinstance(value, (list, tuple, np.ndarray)):
                    parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
            if parts:
                arr = np.concatenate(parts, axis=0)

    if arr is None:
        arr = np.asarray(msg, dtype=np.float32).reshape(-1)

    if expected_dim is None:
        return arr

    if arr.size == expected_dim:
        return arr
    if arr.size > expected_dim:
        return arr[:expected_dim]

    out = np.full((expected_dim,), np.nan, dtype=np.float32)
    out[: arr.size] = arr
    return out


def _state_candidate_keys(topic: str) -> List[str]:
    if topic == "/franka_robot_state_broadcaster/robot_state":
        return ["ee_pose", "pose", "data"]
    if topic in ("/cartesian_impedance_controller/ee_velocity", "/cartesian_admittance_controller/ee_velocity"):
        return ["ee_velocity", "data"]
    if topic == "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame":
        return ["external_wrench", "wrench", "data"]
    if topic in ("/cartesian_impedance_controller/tracking_error", "/cartesian_admittance_controller/tracking_error"):
        return ["tracking_error", "data"]
    if topic in ("/cartesian_impedance_controller/pose_command", "/cartesian_admittance_controller/pose_command"):
        return ["ee_pose_commanded", "position", "data"]
    return ["data"]


def _action_candidate_keys(topic: str) -> List[str]:
    if topic in ("/cartesian_impedance_controller/pose_command", "/cartesian_admittance_controller/pose_command"):
        return ["ee_pose_commanded", "position", "data"]
    if topic in ["/joint_impedance_dynamic_gain_controller/joint_impedance_command", "/joint_impedance_command_controller/joint_trajectory"]:
        return ["position", "data"]
    return ["position", "data"]


def _infer_action_pose_mode(cfg, rollout_cfg, action_stats) -> str:
    mode = OmegaConf.select(cfg, "action_pose_mode", default=None)
    if mode is None and isinstance(rollout_cfg, dict):
        mode = (rollout_cfg.get("processing_config") or {}).get("action_pose_mode")
    if mode is not None:
        processing = (rollout_cfg.get("processing_config") or {}) if isinstance(rollout_cfg, dict) else None
        return _canonical_action_mode(mode, processing_config=processing)

    if isinstance(action_stats, dict) and action_stats.get("mode", None) == "grouped":
        std_vals = []
        for group in action_stats.get("groups", []):
            std = np.asarray(group.get("std", []), dtype=np.float32).reshape(-1)
            if std.size > 0:
                std_vals.append(std)
        if std_vals:
            mean_std = float(np.mean(np.concatenate(std_vals)))
            if np.isfinite(mean_std) and mean_std < 0.02:
                return "delta"

    return "absolute"


def _extract_stiffness_vector(msg, key: str) -> np.ndarray:
    if isinstance(msg, dict):
        if key in msg and msg[key] is not None:
            return np.asarray(msg[key], dtype=np.float32).reshape(-1)
        if "data" in msg and msg["data"] is not None:
            return np.asarray(msg["data"], dtype=np.float32).reshape(-1)
    return np.asarray(msg, dtype=np.float32).reshape(-1)


def _stiffness_vec_to_class(stiffness_vec: np.ndarray, thresholds: Optional[List[float]], num_classes: Optional[int] = None) -> int:
    norm = float(np.linalg.norm(stiffness_vec))
    if not np.isfinite(norm):
        return 1

    if num_classes is None:
        num_classes = max(1, len(thresholds) + 1) if thresholds is not None else 1
    num_classes = max(1, int(num_classes))
    if num_classes == 1:
        return 1

    thresholds_sorted = [] if thresholds is None else sorted(float(v) for v in thresholds)
    # Keep mapping consistent with configured class count: N classes -> use at most N-1 thresholds.
    thresholds_sorted = thresholds_sorted[: max(0, num_classes - 1)]

    for class_id, threshold in enumerate(thresholds_sorted, start=1):
        if norm < threshold:
            return class_id
    return min(num_classes, len(thresholds_sorted) + 1)


def _sync_topics_by_slowest(entries: Dict, timestamps: Dict, topics: List[str]) -> Dict[str, List]:
    ts_arrays = {topic: np.asarray(timestamps[topic], dtype=np.int64) for topic in topics}
    lengths = {topic: len(ts_arrays[topic]) for topic in topics}
    slowest_topic = min(lengths, key=lengths.get)
    target_ts = ts_arrays[slowest_topic]

    synced = {topic: [] for topic in topics}
    for ts in target_ts:
        for topic in topics:
            topic_ts = ts_arrays[topic]
            idx = int(np.argmin(np.abs(topic_ts - ts)))
            synced[topic].append(entries[topic][idx])
    return synced


def _sync_topics_by_min_index(entries: Dict, topics: List[str]) -> Dict[str, List]:
    min_len = min(len(entries[topic]) for topic in topics)
    return {topic: entries[topic][:min_len] for topic in topics}


CONTROLLER_TOPIC_FALLBACKS = {
    "/cartesian_impedance_controller/ee_velocity": "/cartesian_admittance_controller/ee_velocity",
    "/cartesian_admittance_controller/ee_velocity": "/cartesian_impedance_controller/ee_velocity",
    "/cartesian_impedance_controller/tracking_error": "/cartesian_admittance_controller/tracking_error",
    "/cartesian_admittance_controller/tracking_error": "/cartesian_impedance_controller/tracking_error",
    "/cartesian_impedance_controller/pose_command": "/cartesian_admittance_controller/pose_command",
    "/cartesian_admittance_controller/pose_command": "/cartesian_impedance_controller/pose_command",
}


def _apply_controller_topic_fallbacks(entries: Dict, timestamps: Optional[Dict], topics: List[str]) -> None:
    """Alias matching impedance/admittance topics before synchronization."""
    for expected_topic in topics:
        if expected_topic in entries:
            continue
        fallback_topic = CONTROLLER_TOPIC_FALLBACKS.get(expected_topic)
        if fallback_topic is None or fallback_topic not in entries:
            continue
        entries[expected_topic] = entries[fallback_topic]
        if timestamps is not None and fallback_topic in timestamps:
            timestamps[expected_topic] = timestamps[fallback_topic]
        print(f"Topic fallback: {expected_topic} <- {fallback_topic}")


def load_episode_arrays(episode_path: Path, rollout_cfg: Dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    with open(episode_path, "rb") as f:
        raw = pickle.load(f)

    if "data" not in raw:
        raise ValueError(f"Episode format error (missing 'data'): {episode_path}")

    entries = raw["data"]
    ts = raw.get("timestamps", None)

    obs_cfg = rollout_cfg.get("obs_config", {})
    action_cfg = rollout_cfg.get("action_config", {})

    state_topics = list(obs_cfg.get("state_topics", []))
    if len(state_topics) == 0:
        raise ValueError("rollout_config.obs_config.state_topics is empty")

    action_topics = list(action_cfg.keys())
    if len(action_topics) != 1:
        raise ValueError(f"Expected exactly one action topic, got: {action_topics}")
    action_topic = action_topics[0]
    action_dim = int(action_cfg[action_topic])

    stiffness_cfg = obs_cfg.get("stiffness_label") or {}
    stiffness_topic = stiffness_cfg.get("topic", None)
    stiffness_key = stiffness_cfg.get("key", "stiffness")
    stiffness_thresholds = stiffness_cfg.get("norm_thresholds")
    stiffness_classes_cfg = stiffness_cfg.get("classes")
    arrangement_topic = obs_cfg.get("arrangement_topic")
    mode_topic = obs_cfg.get("mode_topic")

    if isinstance(stiffness_classes_cfg, (list, tuple)) and len(stiffness_classes_cfg) > 0:
        stiffness_num_classes = len(stiffness_classes_cfg)
    elif stiffness_thresholds is not None:
        stiffness_num_classes = max(1, len(stiffness_thresholds) + 1)
    else:
        stiffness_num_classes = 1

    topics = list(state_topics) + [action_topic]
    if stiffness_topic is not None:
        topics.append(stiffness_topic)
    if arrangement_topic is not None:
        topics.append(arrangement_topic)
    if mode_topic is not None:
        topics.append(mode_topic)
    topics = list(dict.fromkeys(topics))

    _apply_controller_topic_fallbacks(entries, ts, topics)

    missing = [topic for topic in topics if topic not in entries]
    if len(missing) > 0:
        raise ValueError(f"Missing topics in {episode_path.name}: {missing}")

    if ts is not None and all(topic in ts for topic in topics):
        synced = _sync_topics_by_slowest(entries, ts, topics)
    else:
        synced = _sync_topics_by_min_index(entries, topics)

    state_arrays = []
    for topic in state_topics:
        keys = _state_candidate_keys(topic)
        vecs = [_extract_vector_flexible(item, keys, expected_dim=None) for item in synced[topic]]
        max_dim = max(v.shape[0] for v in vecs)
        padded = []
        for v in vecs:
            out = np.full((max_dim,), np.nan, dtype=np.float32)
            out[: v.shape[0]] = v
            padded.append(out)
        arr = np.stack(padded, axis=0)
        state_arrays.append(arr)

    if len(state_arrays) == 0:
        raise ValueError(f"No state topics found in episode: {episode_path}")
    min_steps = min(arr.shape[0] for arr in state_arrays)
    state_arrays = [arr[:min_steps] for arr in state_arrays]
    states = np.concatenate(state_arrays, axis=-1).astype(np.float32)

    act_keys = _action_candidate_keys(action_topic)
    actions = np.stack([_extract_vector_flexible(item, act_keys, expected_dim=action_dim) for item in synced[action_topic]], axis=0).astype(np.float32)
    if actions.shape[0] > min_steps:
        actions = actions[:min_steps]

    classes = None
    if mode_topic is not None:
        classes = np.asarray([
            int(_extract_vector(msg, ["mode", "data"], 1)[0]) + 1
            for msg in synced[mode_topic]
        ], dtype=np.int64)
    elif stiffness_topic is not None:
        class_ids = []
        for msg in synced[stiffness_topic]:
            vec = _extract_stiffness_vector(msg, stiffness_key)
            class_ids.append(_stiffness_vec_to_class(vec, stiffness_thresholds, num_classes=stiffness_num_classes))
        classes = np.asarray(class_ids, dtype=np.int64)

    arrangement_vectors = None
    if arrangement_topic is not None:
        arrangement_vectors = np.stack([
            arrangement_id_to_one_hot(_extract_vector(msg, ["arrangement", "arrangement_id"], 1))
            for msg in synced[arrangement_topic]
        ]).astype(np.float32)

    count = min(len(states), len(actions))
    if classes is not None:
        count = min(count, len(classes))
        classes = classes[:count]
    if arrangement_vectors is not None:
        count = min(count, len(arrangement_vectors))
        arrangement_vectors = arrangement_vectors[:count]

    states = states[:count]
    actions = actions[:count]

    if count == 0:
        raise ValueError(f"No synchronized samples in episode: {episode_path}")

    return states, actions, classes, arrangement_vectors


def normalize_episode(states: np.ndarray, actions: np.ndarray, rollout_cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    norm_stats = rollout_cfg.get("norm_stats", {})
    state_cfg = norm_stats.get("state", None)
    action_cfg = norm_stats.get("action", None)
    norm_states, _ = _ensure_normalized(states, state_cfg, NORMALIZATION_MODE, "state")
    norm_actions, _ = _ensure_normalized(actions, action_cfg, NORMALIZATION_MODE, "action")
    return norm_states, norm_actions


def build_windows(states, actions, classes, arrangements, obs_window: int, ac_chunk: int, action_index_offset: int = 1):
    total_steps = len(actions)
    start_t = obs_window - 1
    end_t = total_steps - action_index_offset - ac_chunk

    if end_t < start_t:
        raise ValueError(f"Episode too short: T={total_steps}, requires at least obs_window({obs_window}) + ac_chunk({ac_chunk}) - 1")

    obs_windows = []
    action_chunks = []
    class_list = [] if classes is not None else None
    arrangement_list = [] if arrangements is not None else None

    for t in range(start_t, end_t + 1):
        obs_windows.append(states[t - obs_window + 1 : t + 1])
        action_chunks.append(actions[t + action_index_offset : t + action_index_offset + ac_chunk])
        if class_list is not None:
            class_list.append(int(classes[t]))
        if arrangement_list is not None:
            arrangement_list.append(arrangements[t])

    obs_np = np.asarray(obs_windows, dtype=np.float32)
    act_np = np.asarray(action_chunks, dtype=np.float32)
    cls_np = np.asarray(class_list, dtype=np.int64) if class_list is not None else None
    arrangement_np = np.asarray(arrangement_list, dtype=np.float32) if arrangement_list is not None else None

    return obs_np, act_np, cls_np, arrangement_np


@torch.no_grad()
def extract_z_params(policy, obs_np, act_np, cls_np, arrangement_np, device: torch.device):
    if not hasattr(policy, "_build_context_tokens") or not hasattr(policy, "_prior") or not hasattr(policy, "posterior"):
        raise TypeError("Policy does not expose low-dim CVAE internals (_build_context_tokens/_prior/posterior).")

    obs_tensor = torch.from_numpy(obs_np).to(device)
    act_tensor = torch.from_numpy(act_np).to(device)
    cls_tensor = torch.from_numpy(cls_np).to(device) if cls_np is not None else None
    arrangement_tensor = torch.from_numpy(arrangement_np).to(device) if arrangement_np is not None else None

    context_tokens_with_command = policy._build_context_tokens(
        obs_tensor, class_labels=cls_tensor, arrangement_vectors=arrangement_tensor
    )
    # Match policy.forward(): the command token is decoder-only and is always last.
    context_tokens = context_tokens_with_command[:, :-1]
    z_context = policy._build_z_context(context_tokens)

    latent_distribution = str(getattr(policy, "latent_distribution", "gaussian")).lower()
    prior_params = policy._prior(z_context)
    posterior_params = policy.posterior(context_tokens.detach(), act_tensor)

    if latent_distribution == "categorical":
        if isinstance(prior_params, dict):
            prior_logits = prior_params["logits"]
        else:
            prior_logits = prior_params

        if isinstance(posterior_params, dict):
            posterior_logits = posterior_params["logits"]
        else:
            posterior_logits = posterior_params

        prior_probs = F.softmax(prior_logits, dim=-1)
        posterior_probs = F.softmax(posterior_logits, dim=-1)
        eps = 1e-8
        prior_entropy = -(prior_probs * torch.log(prior_probs + eps)).sum(dim=-1)
        posterior_entropy = -(posterior_probs * torch.log(posterior_probs + eps)).sum(dim=-1)

        return {
            "latent_distribution": "categorical",
            "Prior": (prior_probs.cpu().numpy(), prior_entropy.cpu().numpy(), "blue"),
            "Posterior": (posterior_probs.cpu().numpy(), posterior_entropy.cpu().numpy(), "red"),
        }

    if isinstance(prior_params, dict):
        mu_p, logvar_p = prior_params["mu"], prior_params["logvar"]
    else:
        mu_p, logvar_p = prior_params

    if isinstance(posterior_params, dict):
        mu_q, logvar_q = posterior_params["mu"], posterior_params["logvar"]
    else:
        mu_q, logvar_q = posterior_params

    std_p = torch.exp(0.5 * logvar_p)
    std_q = torch.exp(0.5 * logvar_q)

    return {"latent_distribution": "gaussian", "Prior": (mu_p.cpu().numpy(), std_p.cpu().numpy(), "blue"), "Posterior": (mu_q.cpu().numpy(), std_q.cpu().numpy(), "red")}


def visualize_z_statistics(dists_data: Dict, save_dir: Path, ep_name: str):
    latent_distribution = str(dists_data.get("latent_distribution", "gaussian")).lower()
    if latent_distribution == "categorical":
        visualize_z_statistics_categorical(dists_data, save_dir, ep_name)
        return

    visualize_z_statistics_gaussian(dists_data, save_dir, ep_name)


def visualize_z_statistics_gaussian(dists_data: Dict, save_dir: Path, ep_name: str):
    if not dists_data:
        return

    prior_mu, prior_std, prior_color = dists_data["Prior"]
    post_mu, post_std, post_color = dists_data["Posterior"]

    prior_var = prior_std**2
    post_var = post_std**2

    all_mu = np.concatenate([prior_mu, post_mu])
    mu_min, mu_max = all_mu.min(), all_mu.max()
    mu_margin = (mu_max - mu_min) * 0.1 if mu_max != mu_min else 1.0
    mu_ylim = (mu_min - mu_margin, mu_max + mu_margin)

    all_var = np.concatenate([prior_var, post_var])
    var_min, var_max = all_var.min(), all_var.max()
    var_margin = (var_max - var_min) * 0.1 if var_max != var_min else 0.1
    var_ylim = (max(0, var_min - var_margin), var_max + var_margin)

    time_steps, z_dim = prior_mu.shape
    x = np.arange(time_steps)

    fig_height = max(2 * z_dim, 8)
    fig, axes = plt.subplots(z_dim, 2, figsize=(16, fig_height), sharex=True)
    if z_dim == 1:
        axes = axes.reshape(1, 2)

    plt.subplots_adjust(top=0.96, bottom=0.03, left=0.08, right=0.95, hspace=0.35, wspace=0.2)
    fig.suptitle(f"Z Statistics: {ep_name} (Dims 0-{z_dim - 1})", fontsize=16)

    for dim in range(z_dim):
        ax_mean = axes[dim, 0]
        ax_var = axes[dim, 1]

        ax_mean.plot(x, prior_mu[:, dim], label="Prior", color=prior_color, alpha=0.75)
        ax_mean.plot(x, post_mu[:, dim], label="Posterior", color=post_color, alpha=0.75)
        ax_mean.set_ylabel(f"Dim {dim} Mean", fontsize=11)
        ax_mean.set_ylim(mu_ylim)
        ax_mean.grid(True, linestyle="--", alpha=0.4)
        if dim == 0:
            ax_mean.legend(loc="upper right")

        ax_var.plot(x, prior_var[:, dim], label="Prior", color=prior_color, alpha=0.75)
        ax_var.plot(x, post_var[:, dim], label="Posterior", color=post_color, alpha=0.75)
        ax_var.set_ylabel("Variance", fontsize=11)
        ax_var.set_ylim(var_ylim)
        ax_var.grid(True, linestyle="--", alpha=0.4)

    axes[-1, 0].set_xlabel("Time Index", fontsize=12)
    axes[-1, 1].set_xlabel("Time Index", fontsize=12)

    save_path = save_dir / f"{ep_name}_z_distr.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {save_path}")


def visualize_z_statistics_categorical(dists_data: Dict, save_dir: Path, ep_name: str):
    if not dists_data:
        return

    prior_probs, prior_entropy, _ = dists_data["Prior"]
    post_probs, post_entropy, _ = dists_data["Posterior"]

    time_steps, num_variables, num_categories = prior_probs.shape
    fig_height = max(2.6 * num_variables, 7.0)
    fig, axes = plt.subplots(num_variables, 2, figsize=(16, fig_height), sharex=True)
    if num_variables == 1:
        axes = axes.reshape(1, 2)

    plt.subplots_adjust(top=0.95, bottom=0.07, left=0.08, right=0.95, hspace=0.35, wspace=0.15)
    fig.suptitle(f"Categorical Z Probabilities: {ep_name}", fontsize=15)

    for var_idx in range(num_variables):
        ax_prior = axes[var_idx, 0]
        ax_post = axes[var_idx, 1]

        im_prior = ax_prior.imshow(prior_probs[:, var_idx, :].T, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")
        ax_prior.set_ylabel(f"Var {var_idx} cat", fontsize=10)
        if var_idx == 0:
            ax_prior.set_title("Prior", fontsize=12)

        ax_post.imshow(post_probs[:, var_idx, :].T, aspect="auto", origin="lower", interpolation="nearest", vmin=0.0, vmax=1.0, cmap="viridis")
        if var_idx == 0:
            ax_post.set_title("Posterior", fontsize=12)

    axes[-1, 0].set_xlabel("Time Index", fontsize=11)
    axes[-1, 1].set_xlabel("Time Index", fontsize=11)

    cbar = fig.colorbar(im_prior, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("Probability", fontsize=10)

    entropy_path = save_dir / f"{ep_name}_z_entropy.png"
    save_path = save_dir / f"{ep_name}_z_distr.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    fig_e, ax_e = plt.subplots(1, 1, figsize=(12, 4))
    x = np.arange(time_steps)
    ax_e.plot(x, prior_entropy.mean(axis=1), label="Prior entropy", color="blue", alpha=0.8)
    ax_e.plot(x, post_entropy.mean(axis=1), label="Posterior entropy", color="red", alpha=0.8)
    ax_e.set_xlabel("Time Index", fontsize=11)
    ax_e.set_ylabel("Mean entropy across variables", fontsize=11)
    ax_e.set_ylim(0.0, np.log(float(max(2, num_categories))) * 1.05)
    ax_e.grid(True, linestyle="--", alpha=0.35)
    ax_e.legend(loc="upper right")
    fig_e.suptitle(f"Categorical Z Entropy: {ep_name}", fontsize=13)
    fig_e.tight_layout(rect=[0, 0, 1, 0.95])
    fig_e.savefig(entropy_path, dpi=150)
    plt.close(fig_e)

    print(f"✅ Saved: {save_path}")
    print(f"✅ Saved: {entropy_path}")


def _gaussian_pdf(x: np.ndarray, mu: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-6)
    z = (x - float(mu)) / std
    coef = 1.0 / (np.sqrt(2.0 * np.pi) * std)
    return coef * np.exp(-0.5 * z * z)


def visualize_distributions_video(
    dists_data: Dict, save_path: Path, ep_name: str, fps: int = 15, dpi: int = 80, frame_stride: int = 1, x_points: int = 80, x_std_mult: float = 4.0
):
    latent_distribution = str(dists_data.get("latent_distribution", "gaussian")).lower()
    if latent_distribution == "categorical":
        visualize_distributions_video_categorical(dists_data, save_path, ep_name, fps=fps, dpi=dpi, frame_stride=frame_stride)
        return

    visualize_distributions_video_gaussian(dists_data, save_path, ep_name, fps=fps, dpi=dpi, frame_stride=frame_stride, x_points=x_points, x_std_mult=x_std_mult)


def visualize_distributions_video_gaussian(
    dists_data: Dict, save_path: Path, ep_name: str, fps: int = 15, dpi: int = 80, frame_stride: int = 1, x_points: int = 80, x_std_mult: float = 4.0
):
    if not dists_data:
        return

    prior_mu, prior_std, _ = dists_data["Prior"]
    post_mu, post_std, _ = dists_data["Posterior"]

    time_steps, z_dim = prior_mu.shape
    ncols = int(np.ceil(np.sqrt(z_dim)))
    nrows = int(np.ceil(z_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = np.asarray(axes).reshape(-1)
    plt.subplots_adjust(left=0.05, right=0.97, bottom=0.04, top=0.93, hspace=0.38, wspace=0.28)

    all_stds = np.concatenate([prior_std.reshape(-1), post_std.reshape(-1)], axis=0)
    safe_stds = np.clip(all_stds, 1e-6, None)
    min_std = float(np.min(safe_stds))

    all_mus = np.concatenate([prior_mu.reshape(-1), post_mu.reshape(-1)], axis=0)
    all_low = np.concatenate(
        [(prior_mu - float(x_std_mult) * np.clip(prior_std, 1e-6, None)).reshape(-1), (post_mu - float(x_std_mult) * np.clip(post_std, 1e-6, None)).reshape(-1)], axis=0
    )
    all_high = np.concatenate(
        [(prior_mu + float(x_std_mult) * np.clip(prior_std, 1e-6, None)).reshape(-1), (post_mu + float(x_std_mult) * np.clip(post_std, 1e-6, None)).reshape(-1)], axis=0
    )

    x_min = float(np.nanmin(all_low))
    x_max = float(np.nanmax(all_high))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        x_min, x_max = -5.0, 5.0
    else:
        span = max(x_max - x_min, 1.0)
        pad = 0.05 * span
        x_min -= pad
        x_max += pad

    mu_abs_max = float(np.nanmax(np.abs(all_mus))) if all_mus.size > 0 else 1.0
    hard_limit = max(20.0, mu_abs_max + 20.0)
    x_min = max(x_min, -hard_limit)
    x_max = min(x_max, hard_limit)
    x_vals = np.linspace(x_min, x_max, int(max(20, x_points)))
    y_max = min(5.0, 1.1 / (np.sqrt(2.0 * np.pi) * min_std))

    frame_indices = np.arange(0, time_steps, max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != (time_steps - 1):
        frame_indices = np.concatenate([frame_indices, np.asarray([time_steps - 1], dtype=np.int64)])

    lines = {}
    for dim in range(len(axes)):
        ax = axes[dim]
        if dim >= z_dim:
            ax.axis("off")
            continue
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_max)
        ax.set_title(f"Dim {dim}", fontsize=9)
        (line_prior,) = ax.plot([], [], color="blue", linewidth=1.5, alpha=0.6, label="Prior")
        (line_post,) = ax.plot([], [], color="red", linewidth=2.0, alpha=0.9, label="Posterior")
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8)
        lines[dim] = (line_prior, line_post)

    writer = animation.FFMpegWriter(fps=int(fps), metadata={"artist": "factr"}, bitrate=1800)
    with writer.saving(fig, str(save_path), dpi=int(dpi)):
        for t in tqdm(frame_indices, desc=f"Rendering {ep_name}"):
            fig.suptitle(f"Z Dist: {ep_name} (t={t}/{time_steps})", fontsize=15)
            for dim in range(z_dim):
                line_prior, line_post = lines[dim]
                y_prior = _gaussian_pdf(x_vals, prior_mu[t, dim], prior_std[t, dim])
                y_post = _gaussian_pdf(x_vals, post_mu[t, dim], post_std[t, dim])
                line_prior.set_data(x_vals, y_prior)
                line_post.set_data(x_vals, y_post)
            writer.grab_frame()

    plt.close(fig)
    print(f"✅ Saved: {save_path}")


def visualize_distributions_video_categorical(dists_data: Dict, save_path: Path, ep_name: str, fps: int = 15, dpi: int = 80, frame_stride: int = 1):
    if not dists_data:
        return

    prior_probs, _, _ = dists_data["Prior"]
    post_probs, _, _ = dists_data["Posterior"]

    time_steps, num_variables, num_categories = prior_probs.shape
    ncols = int(np.ceil(np.sqrt(num_variables)))
    nrows = int(np.ceil(num_variables / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.8))
    axes = np.asarray(axes).reshape(-1)
    plt.subplots_adjust(left=0.05, right=0.97, bottom=0.06, top=0.90, hspace=0.45, wspace=0.35)

    x_idx = np.arange(num_categories)
    prior_bars = {}
    post_bars = {}
    for var_idx in range(len(axes)):
        ax = axes[var_idx]
        if var_idx >= num_variables:
            ax.axis("off")
            continue

        ax.set_title(f"Var {var_idx}", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(-0.5, num_categories - 0.5)
        ax.set_xticks(x_idx[:: max(1, num_categories // 8)])
        ax.set_ylabel("P(cat)", fontsize=8)
        bars_prior = ax.bar(x_idx - 0.2, np.zeros_like(x_idx, dtype=np.float32), width=0.4, color="blue", alpha=0.55)
        bars_post = ax.bar(x_idx + 0.2, np.zeros_like(x_idx, dtype=np.float32), width=0.4, color="red", alpha=0.75)
        if var_idx == 0:
            ax.legend([bars_prior[0], bars_post[0]], ["Prior", "Posterior"], fontsize=8, loc="upper right")
        prior_bars[var_idx] = bars_prior
        post_bars[var_idx] = bars_post

    frame_indices = np.arange(0, time_steps, max(1, int(frame_stride)), dtype=np.int64)
    if frame_indices[-1] != (time_steps - 1):
        frame_indices = np.concatenate([frame_indices, np.asarray([time_steps - 1], dtype=np.int64)])

    writer = animation.FFMpegWriter(fps=int(fps), metadata={"artist": "factr"}, bitrate=1800)
    with writer.saving(fig, str(save_path), dpi=int(dpi)):
        for t in tqdm(frame_indices, desc=f"Rendering {ep_name}"):
            fig.suptitle(f"Categorical Z Dist: {ep_name} (t={t}/{time_steps})", fontsize=14)
            for var_idx in range(num_variables):
                current_prior = prior_probs[t, var_idx, :]
                current_post = post_probs[t, var_idx, :]
                for cat_idx in range(num_categories):
                    prior_bars[var_idx][cat_idx].set_height(float(current_prior[cat_idx]))
                    post_bars[var_idx][cat_idx].set_height(float(current_post[cat_idx]))
            writer.grab_frame()

    plt.close(fig)
    print(f"✅ Saved: {save_path}")


def discover_episode_files(data_root: Path, requested: Optional[List[str]]) -> List[Path]:
    if requested is not None and len(requested) > 0:
        files = []
        for name in requested:
            stem = name[:-4] if name.endswith(".pkl") else name
            p = data_root / f"{stem}.pkl"
            if not p.exists():
                raise FileNotFoundError(f"Episode not found: {p}")
            files.append(p)
        return files

    files = sorted(data_root.glob("*.pkl"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .pkl episodes found under: {data_root}")
    return files


def discover_single_episode_file(data_root: Path, episode_file_name: str) -> List[Path]:
    if not episode_file_name:
        raise ValueError("EPISODE_FILE_NAME must be set when USE_EPISODE_LIST is False")

    stem = episode_file_name[:-4] if episode_file_name.endswith(".pkl") else episode_file_name
    p = data_root / f"{stem}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Episode not found: {p}")
    return [p]


def normalize_rollout_episode_id(episode_id: str) -> str:
    episode_id = str(episode_id).strip()
    if episode_id.endswith(".pkl"):
        episode_id = episode_id[:-4]
    return episode_id.strip("/")


def get_required_split_episodes(rollout_cfg: Dict, split_name: str) -> List[str]:
    split_cfg = rollout_cfg.get("split_config", {}) if isinstance(rollout_cfg, dict) else {}
    episodes = split_cfg.get(f"{split_name}_episodes", []) or []
    if not episodes:
        raise ValueError(
            "Auto episode selection requires rollout_config.yaml split_config "
            f"with non-empty {split_name}_episodes."
        )
    return [normalize_rollout_episode_id(ep) for ep in episodes]


def get_split_label(rollout_cfg: Dict, episode_id: str) -> str:
    episode_id = normalize_rollout_episode_id(episode_id)
    episode_stem = episode_id.rsplit("/", 1)[-1]
    split_cfg = rollout_cfg.get("split_config", {}) if isinstance(rollout_cfg, dict) else {}
    train_eps = {normalize_rollout_episode_id(ep) for ep in split_cfg.get("train_episodes", []) or []}
    test_eps = {normalize_rollout_episode_id(ep) for ep in split_cfg.get("test_episodes", []) or []}

    if episode_id in test_eps or episode_stem in test_eps:
        return "test"
    if episode_id in train_eps or episode_stem in train_eps:
        return "train"
    return "unknown"


def build_rollout_raw_dirs(rollout_cfg: Dict, fallback_data_root: Path) -> Dict[str, Path]:
    processing_cfg = rollout_cfg.get("processing_config", {}) if isinstance(rollout_cfg, dict) else {}
    input_paths = processing_cfg.get("input_paths", []) or []
    raw_dirs: Dict[str, Path] = {}

    for raw_path in input_paths:
        data_dir = Path(raw_path).expanduser()
        dataset_dir = data_dir.parent if data_dir.name == "data" else data_dir
        raw_dirs[dataset_dir.name] = data_dir

    if not raw_dirs and fallback_data_root is not None:
        data_dir = Path(fallback_data_root)
        dataset_dir = data_dir.parent if data_dir.name == "data" else data_dir
        raw_dirs[dataset_dir.name] = data_dir

    return raw_dirs


def resolve_rollout_episode(episode_id: str, raw_dirs: Dict[str, Path]) -> Tuple[str, Path]:
    episode_id = normalize_rollout_episode_id(episode_id)
    if "/" in episode_id:
        dataset_name, episode_stem = episode_id.rsplit("/", 1)
        if dataset_name not in raw_dirs:
            known = ", ".join(sorted(raw_dirs.keys()))
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found in rollout input_paths. Known datasets: {known}")
        path = raw_dirs[dataset_name] / f"{episode_stem}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Episode not found: {path}")
        return f"{dataset_name}/{episode_stem}", path

    matches = []
    for dataset_name, data_dir in raw_dirs.items():
        path = data_dir / f"{episode_id}.pkl"
        if path.exists():
            matches.append((f"{dataset_name}/{episode_id}", path))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        found = ", ".join(ep_id for ep_id, _ in matches)
        raise ValueError(f"Episode name '{episode_id}' is ambiguous across rollout datasets: {found}")
    raise FileNotFoundError(f"Episode '{episode_id}' not found in rollout input_paths.")


def resolve_rollout_episodes(episode_ids: List[str], raw_dirs: Dict[str, Path]) -> List[Tuple[str, Path]]:
    return [resolve_rollout_episode(ep, raw_dirs) for ep in episode_ids]


def plot_file_stem(episode_id: str) -> str:
    return normalize_rollout_episode_id(episode_id).replace("/", "_")


def render_episode_outputs(
    dists: Dict,
    save_dir: Path,
    ep_name: str,
    save_static_plots: bool,
    save_video: bool,
    video_fps: int,
    video_dpi: int,
    video_frame_stride: int,
    video_x_points: int,
    video_x_std_mult: float,
) -> Tuple[str, Optional[str]]:
    try:
        if save_static_plots:
            visualize_z_statistics(dists, save_dir, ep_name)

        if save_video:
            video_path = save_dir / f"{ep_name}_z_distr.mp4"
            visualize_distributions_video(
                dists, video_path, ep_name, fps=video_fps, dpi=video_dpi, frame_stride=video_frame_stride, x_points=video_x_points, x_std_mult=video_x_std_mult
            )
        return ep_name, None
    except Exception as e:
        return ep_name, str(e)


def main():
    global DATASET_NAME, DATASET_PROJECT_PREFIX, BUFFER_SET_NAME, RUN_NAME
    global CHECKPOINT_NAME, USE_EPISODE_LIST, EPISODE_FILE_NAME, EPISODE_LIST
    global RUN_DIR, RAW_EPISODE_DIR, GPU_ID, SAVE_DIR_OVERRIDE, SAVE_STATIC_PLOTS
    global SAVE_VIDEO, VIDEO_FPS, VIDEO_DPI, VIDEO_FRAME_STRIDE, VIDEO_X_POINTS
    global VIDEO_X_STD_MULT, MAX_PARALLEL_EPISODES, NORMALIZATION_MODE

    args = parse_args()
    globals().update(_materialize_globals(Path(args.config)))

    run_dir = Path(RUN_DIR)
    checkpoint_name = str(CHECKPOINT_NAME)
    data_root = Path(RAW_EPISODE_DIR)

    device = torch.device(f"cuda:{int(GPU_ID)}" if torch.cuda.is_available() else "cpu")

    if not run_dir.exists():
        raise FileNotFoundError(f"RUN_DIR not found: {run_dir}")

    ckpt_path = run_dir / checkpoint_name
    if not ckpt_path.exists():
        alt = run_dir.parent / checkpoint_name
        if alt.exists():
            ckpt_path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {run_dir / checkpoint_name} or {alt}")

    rollout_cfg_path = run_dir / "rollout_config.yaml"
    if not rollout_cfg_path.exists():
        alt_cfg = run_dir.parent / "rollout" / "rollout_config.yaml"
        if alt_cfg.exists():
            rollout_cfg_path = alt_cfg
        else:
            raise FileNotFoundError(f"rollout_config.yaml not found near RUN_DIR: {run_dir}")

    model_dir = run_dir.parent
    save_dir_override = Path(SAVE_DIR_OVERRIDE) if SAVE_DIR_OVERRIDE is not None else None

    with open(rollout_cfg_path, "r") as f:
        rollout_cfg = yaml.safe_load(f)

    policy, cfg = load_model(ckpt_path, device)
    obs_window = int(getattr(policy, "obs_window", 8))
    ac_chunk = int(getattr(policy, "ac_chunk", 30))

    auto_buffer_set = str(BUFFER_SET_NAME).strip().lower() == "auto"
    raw_dirs = build_rollout_raw_dirs(rollout_cfg, data_root)
    if not raw_dirs:
        raise ValueError("No raw dataset folders found. Expected rollout_config.processing_config.input_paths.")

    if USE_EPISODE_LIST and EPISODE_LIST:
        episode_items = resolve_rollout_episodes([str(x) for x in EPISODE_LIST], raw_dirs)
    elif auto_buffer_set:
        episode_items = resolve_rollout_episodes(get_required_split_episodes(rollout_cfg, "test"), raw_dirs)
    else:
        if not data_root.exists():
            raise FileNotFoundError(f"data_root not found: {data_root}")
        episode_files = discover_single_episode_file(data_root, str(EPISODE_FILE_NAME))
        episode_items = [(normalize_rollout_episode_id(path.stem), path) for path in episode_files]

    print(f"Using run dir: {run_dir}")
    print(f"Using rollout config: {rollout_cfg_path}")
    print(f"Episodes: {len(episode_items)} | obs_window={obs_window} | ac_chunk={ac_chunk}")

    use_parallel = len(episode_items) > 1 and (SAVE_STATIC_PLOTS or SAVE_VIDEO)
    if MAX_PARALLEL_EPISODES == 0:
        max_workers = min(len(episode_items), max(1, (os.cpu_count() or 2) - 1))
    else:
        max_workers = max(1, int(MAX_PARALLEL_EPISODES))
    if not use_parallel:
        max_workers = 1

    if max_workers > 1:
        print(f"Parallel rendering enabled with workers={max_workers}")

    render_jobs = []

    for ep_id, ep_path in tqdm(episode_items, desc="Episodes"):
        ep_name = plot_file_stem(ep_id)
        print(f"\nProcessing {ep_id} -> {ep_path}")

        try:
            states, actions, classes, arrangements = load_episode_arrays(ep_path, rollout_cfg)
            include_tracking_error = bool(OmegaConf.select(cfg, "include_tracking_error", default=OmegaConf.select(cfg, "agent.include_tracking_error", default=True)))
            state_stats = rollout_cfg.get("norm_stats", {}).get("state", None)
            if (not include_tracking_error) and states.shape[-1] >= 36:
                states = np.concatenate([states[:, :21], states[:, 27:]], axis=-1)
                state_stats = _state_stats_without_tracking_error(state_stats)

            action_stats = rollout_cfg.get("norm_stats", {}).get("action", None)
            action_pose_mode = _infer_action_pose_mode(cfg, rollout_cfg, action_stats)
            if action_pose_mode == "delta" and actions.shape[0] > 1:
                rel_actions = np.zeros_like(actions)
                rel_actions[1:] = actions[1:] - actions[:-1]
                actions = rel_actions
            action_chunk_mode = _canonical_action_mode(OmegaConf.select(cfg, "action_chunk_mode", default="absolute"))
            action_index_offset = int(OmegaConf.select(cfg, "task.test_buffer.action_index_offset", default=1))
            obs_np, act_np, cls_np, arrangement_np = build_windows(
                states, actions, classes, arrangements,
                obs_window=obs_window,
                ac_chunk=ac_chunk,
                action_index_offset=action_index_offset,
            )
            obs_raw_np = obs_np
            obs_np, _ = _ensure_normalized(obs_np, state_stats, NORMALIZATION_MODE, "state")
            if action_chunk_mode == "relative":
                cmd_start = 27 if include_tracking_error else 21
                act_np = relative_chunk_from_absolute(
                    act_np, obs_raw_np[:, -1, cmd_start : cmd_start + 9]
                )
                act_np = _apply_grouped_transform(act_np, action_stats, inverse=False)
            else:
                act_np, _ = _ensure_normalized(act_np, action_stats, NORMALIZATION_MODE, "action")

            use_arrangement = bool(OmegaConf.select(
                cfg, "agent.use_arrangement_conditioning",
                default=OmegaConf.select(cfg, "use_arrangement_conditioning", default=False),
            ))
            if use_arrangement and arrangement_np is None:
                raise ValueError("Checkpoint requires arrangement conditioning, but episode has no arrangement topic.")
            if not use_arrangement:
                arrangement_np = None
            dists = extract_z_params(policy, obs_np, act_np, cls_np, arrangement_np, device=device)
        except Exception as e:
            print(f"Skip {ep_name}: {e}")
            continue

        split_label = get_split_label(rollout_cfg, ep_id)
        out_suffix = "eval_z_test" if split_label == "test" else "eval_z_train"
        save_dir = (model_dir / out_suffix) if save_dir_override is None else (save_dir_override / out_suffix)
        save_dir.mkdir(parents=True, exist_ok=True)

        # raw_path = save_dir / f"z_raw_data_{ep_name}.pkl"
        # with open(raw_path, "wb") as f:
        #     pickle.dump(dists, f)
        # print(f"Saved: {raw_path}")

        render_jobs.append((ep_name, dists, save_dir))

    if len(render_jobs) == 0:
        return

    if max_workers <= 1:
        for ep_name, dists, save_dir in render_jobs:
            _, err = render_episode_outputs(dists, save_dir, ep_name, SAVE_STATIC_PLOTS, SAVE_VIDEO, VIDEO_FPS, VIDEO_DPI, VIDEO_FRAME_STRIDE, VIDEO_X_POINTS, VIDEO_X_STD_MULT)
            if err is not None:
                print(f"Output save failed for {ep_name}: {err}")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                render_episode_outputs, dists, save_dir, ep_name, SAVE_STATIC_PLOTS, SAVE_VIDEO, VIDEO_FPS, VIDEO_DPI, VIDEO_FRAME_STRIDE, VIDEO_X_POINTS, VIDEO_X_STD_MULT
            )
            for ep_name, dists, save_dir in render_jobs
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rendering outputs"):
            ep_name, err = future.result()
            if err is not None:
                print(f"Output save failed for {ep_name}: {err}")


if __name__ == "__main__":
    main()
