#!/usr/bin/env python3
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
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

        factr.misc.get_transform_by_name = lambda name: (lambda x: x)
except Exception:
    pass


STATE_TOPIC_SPECS = {
    "/franka_robot_state_broadcaster/robot_state": {"keys": ["ee_pose", "pose", "data"], "dim": 9},
    "/cartesian_impedance_controller/ee_velocity": {"keys": ["ee_velocity", "data"], "dim": 6},
    "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame": {
        "keys": ["external_wrench", "wrench", "data"],
        "dim": 6,
    },
    "/cartesian_impedance_controller/tracking_error": {"keys": ["tracking_error", "data"], "dim": 6},
}

ACTION_TOPIC_SPECS = {
    "/cartesian_impedance_controller/pose_command": {"keys": ["ee_pose_commanded", "position", "data"], "dim": 9},
    "/joint_impedance_dynamic_gain_controller/joint_impedance_command": {"keys": ["position", "data"], "dim": 9},
    "/joint_impedance_command_controller/joint_trajectory": {"keys": ["position", "data"], "dim": 9},
}


# ---------------------------------------------------------------------------
# User Config (edit these variables, then run this script directly)
# ---------------------------------------------------------------------------
RUN_DIR = Path.home() / "activeinference" / "factr" / "checkpoints" / "aiact_1_beta0001" / "rollout"
CHECKPOINT_NAME = "latest_ckpt.ckpt"  # or "ckpt_020000.ckpt"

RAW_EPISODE_DIR = Path("/home/ferdinand/activeinference/factr/process_data/data_to_process/fourgoals_1/data")

USE_EPISODE_LIST = False
EPISODE_FILE_NAME = "ep_19_medium.pkl"
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

GPU_ID = 0
SAVE_DIR_OVERRIDE = None
SAVE_STATIC_PLOTS = True
SAVE_VIDEO = True
VIDEO_FPS = 30 # playback speed
VIDEO_DPI = 70
VIDEO_FRAME_STRIDE = 4 # use every Nth frame for video 
VIDEO_X_POINTS = 80 # number of x points in distribution plots
VIDEO_X_STD_MULT = 4.0 # x range for distribution plots will be [mu - x_std_mult*std, mu + x_std_mult*std]


def parse_args():
    return None


def resolve_paths(project_root: Path, model_name: str, checkpoint: str, rollout_config_arg: Optional[str]) -> Tuple[Path, Path, Path]:
    checkpoints_dir = project_root / "checkpoints" / model_name
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
        candidates = [
            ckpt_path.parent / "rollout_config.yaml",
            checkpoints_dir / "rollout" / "rollout_config.yaml",
            checkpoints_dir / "rollout_config.yaml",
        ]
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


def _extract_stiffness_vector(msg, key: str) -> np.ndarray:
    if isinstance(msg, dict):
        if key in msg and msg[key] is not None:
            return np.asarray(msg[key], dtype=np.float32).reshape(-1)
        if "data" in msg and msg["data"] is not None:
            return np.asarray(msg["data"], dtype=np.float32).reshape(-1)
    return np.asarray(msg, dtype=np.float32).reshape(-1)


def _stiffness_vec_to_class(stiffness_vec: np.ndarray, thresholds: List[float]) -> int:
    norm = float(np.linalg.norm(stiffness_vec))
    if not np.isfinite(norm):
        return 1
    for class_id, threshold in enumerate(sorted(float(v) for v in thresholds), start=1):
        if norm < threshold:
            return class_id
    return len(thresholds) + 1


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


def load_episode_arrays(episode_path: Path, rollout_cfg: Dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
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

    stiffness_cfg = obs_cfg.get("stiffness_label", {})
    stiffness_topic = stiffness_cfg.get("topic", None)
    stiffness_key = stiffness_cfg.get("key", "stiffness")
    stiffness_thresholds = stiffness_cfg.get("norm_thresholds", [200.0, 1000.0])

    topics = list(state_topics) + [action_topic]
    if stiffness_topic is not None:
        topics.append(stiffness_topic)

    missing = [topic for topic in topics if topic not in entries]
    if len(missing) > 0:
        raise ValueError(f"Missing topics in {episode_path.name}: {missing}")

    if ts is not None and all(topic in ts for topic in topics):
        synced = _sync_topics_by_slowest(entries, ts, topics)
    else:
        synced = _sync_topics_by_min_index(entries, topics)

    state_arrays = []
    for topic in state_topics:
        spec = STATE_TOPIC_SPECS.get(topic, {"keys": ["data"], "dim": None})
        dim = int(spec["dim"]) if spec["dim"] is not None else None
        if dim is None:
            vecs = [np.asarray(item, dtype=np.float32).reshape(-1) for item in synced[topic]]
            max_dim = max(v.shape[0] for v in vecs)
            padded = []
            for v in vecs:
                out = np.full((max_dim,), np.nan, dtype=np.float32)
                out[: v.shape[0]] = v
                padded.append(out)
            arr = np.stack(padded, axis=0)
        else:
            arr = np.stack([_extract_vector(item, spec["keys"], dim) for item in synced[topic]], axis=0)
        state_arrays.append(arr)

    states = np.concatenate(state_arrays, axis=-1).astype(np.float32)

    act_spec = ACTION_TOPIC_SPECS.get(action_topic, {"keys": ["position", "data"], "dim": action_dim})
    actions = np.stack([_extract_vector(item, act_spec["keys"], action_dim) for item in synced[action_topic]], axis=0).astype(np.float32)

    classes = None
    if stiffness_topic is not None:
        class_ids = []
        for msg in synced[stiffness_topic]:
            vec = _extract_stiffness_vector(msg, stiffness_key)
            class_ids.append(_stiffness_vec_to_class(vec, stiffness_thresholds))
        classes = np.asarray(class_ids, dtype=np.int64)

    count = min(len(states), len(actions))
    if classes is not None:
        count = min(count, len(classes))
        classes = classes[:count]

    states = states[:count]
    actions = actions[:count]

    if count == 0:
        raise ValueError(f"No synchronized samples in episode: {episode_path}")

    return states, actions, classes


def apply_grouped_norm(x: np.ndarray, grouped_cfg: Optional[Dict]) -> np.ndarray:
    if grouped_cfg is None:
        return x

    out = x.copy()
    eps = 1e-6
    groups = grouped_cfg.get("groups", [])

    for group in groups:
        indices = group.get("indices", None)
        norm_type = group.get("type", "identity")
        if indices is None or len(indices) != 2:
            continue
        start, end = int(indices[0]), int(indices[1])

        if norm_type == "identity":
            continue

        if norm_type.startswith("gaussian"):
            mean = np.asarray(group.get("mean", []), dtype=np.float32)
            std = np.asarray(group.get("std", []), dtype=np.float32)
            if mean.size != (end - start) or std.size != (end - start):
                continue
            out[:, start:end] = (out[:, start:end] - mean) / (std + eps)
            if "clip" in group:
                clip_val = float(group["clip"])
                out[:, start:end] = np.clip(out[:, start:end], -clip_val, clip_val)

    return out


def normalize_episode(states: np.ndarray, actions: np.ndarray, rollout_cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    norm_stats = rollout_cfg.get("norm_stats", {})
    state_cfg = norm_stats.get("state", None)
    action_cfg = norm_stats.get("action", None)
    norm_states = apply_grouped_norm(states, state_cfg)
    norm_actions = apply_grouped_norm(actions, action_cfg)
    return norm_states, norm_actions


def build_windows(
    states: np.ndarray,
    actions: np.ndarray,
    classes: Optional[np.ndarray],
    obs_window: int,
    ac_chunk: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    total_steps = len(actions)
    start_t = obs_window - 1
    end_t = total_steps - ac_chunk

    if end_t < start_t:
        raise ValueError(f"Episode too short: T={total_steps}, requires at least obs_window({obs_window}) + ac_chunk({ac_chunk}) - 1")

    obs_windows = []
    action_chunks = []
    class_list = [] if classes is not None else None

    for t in range(start_t, end_t + 1):
        obs_windows.append(states[t - obs_window + 1 : t + 1])
        action_chunks.append(actions[t : t + ac_chunk])
        if class_list is not None:
            class_list.append(int(classes[t]))

    obs_np = np.asarray(obs_windows, dtype=np.float32)
    act_np = np.asarray(action_chunks, dtype=np.float32)
    cls_np = np.asarray(class_list, dtype=np.int64) if class_list is not None else None

    return obs_np, act_np, cls_np


@torch.no_grad()
def extract_z_params(policy, obs_np: np.ndarray, act_np: np.ndarray, cls_np: Optional[np.ndarray], device: torch.device):
    if not hasattr(policy, "_build_context_tokens") or not hasattr(policy, "_prior") or not hasattr(policy, "posterior"):
        raise TypeError("Policy does not expose low-dim CVAE internals (_build_context_tokens/_prior/posterior).")

    obs_tensor = torch.from_numpy(obs_np).to(device)
    act_tensor = torch.from_numpy(act_np).to(device)
    cls_tensor = torch.from_numpy(cls_np).to(device) if cls_np is not None else None

    context_tokens = policy._build_context_tokens(obs_tensor, class_labels=cls_tensor)
    z_context = policy._build_z_context(context_tokens)

    mu_p, logvar_p = policy._prior(z_context)
    mu_q, logvar_q = policy.posterior(context_tokens.detach(), act_tensor)

    std_p = torch.exp(0.5 * logvar_p)
    std_q = torch.exp(0.5 * logvar_q)

    return {
        "Prior": (mu_p.cpu().numpy(), std_p.cpu().numpy(), "blue"),
        "Posterior": (mu_q.cpu().numpy(), std_q.cpu().numpy(), "red"),
    }


def visualize_z_statistics(dists_data: Dict, save_dir: Path, ep_name: str):
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

    fig_height = max(3 * z_dim, 8)
    fig, axes = plt.subplots(z_dim, 2, figsize=(16, fig_height), sharex=True)
    if z_dim == 1:
        axes = axes.reshape(1, 2)

    plt.subplots_adjust(top=0.98, bottom=0.03, left=0.1, right=0.95, hspace=0.35, wspace=0.2)
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

    save_path = save_dir / f"z_stat_{ep_name}_long.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def _gaussian_pdf(x: np.ndarray, mu: float, std: float) -> np.ndarray:
    std = max(float(std), 1e-6)
    z = (x - float(mu)) / std
    coef = 1.0 / (np.sqrt(2.0 * np.pi) * std)
    return coef * np.exp(-0.5 * z * z)


def visualize_distributions_video(
    dists_data: Dict,
    save_path: Path,
    ep_name: str,
    fps: int = 15,
    dpi: int = 80,
    frame_stride: int = 1,
    x_points: int = 80,
    x_std_mult: float = 4.0,
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
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.93, hspace=0.38, wspace=0.28)

    all_stds = np.concatenate([prior_std.reshape(-1), post_std.reshape(-1)], axis=0)
    safe_stds = np.clip(all_stds, 1e-6, None)
    min_std = float(np.min(safe_stds))

    all_mus = np.concatenate([prior_mu.reshape(-1), post_mu.reshape(-1)], axis=0)
    all_low = np.concatenate(
        [
            (prior_mu - float(x_std_mult) * np.clip(prior_std, 1e-6, None)).reshape(-1),
            (post_mu - float(x_std_mult) * np.clip(post_std, 1e-6, None)).reshape(-1),
        ],
        axis=0,
    )
    all_high = np.concatenate(
        [
            (prior_mu + float(x_std_mult) * np.clip(prior_std, 1e-6, None)).reshape(-1),
            (post_mu + float(x_std_mult) * np.clip(post_std, 1e-6, None)).reshape(-1),
        ],
        axis=0,
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
    print(f"Saved: {save_path}")


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


def main():
    run_dir = Path(RUN_DIR)
    checkpoint_name = str(CHECKPOINT_NAME)
    data_root = Path(RAW_EPISODE_DIR)

    project_root = Path(__file__).resolve().parent.parent
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

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    model_name = run_dir.parent.name
    save_dir = Path(SAVE_DIR_OVERRIDE) if SAVE_DIR_OVERRIDE is not None else (project_root / "result_output" / model_name / "z_stats")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(rollout_cfg_path, "r") as f:
        rollout_cfg = yaml.safe_load(f)

    policy, _ = load_model(ckpt_path, device)
    obs_window = int(getattr(policy, "obs_window", 8))
    ac_chunk = int(getattr(policy, "ac_chunk", 30))

    if USE_EPISODE_LIST:
        requested_eps = [str(x) for x in EPISODE_LIST]
        episode_files = discover_episode_files(data_root, requested_eps)
    else:
        episode_files = discover_single_episode_file(data_root, str(EPISODE_FILE_NAME))

    print(f"Episodes: {len(episode_files)} | obs_window={obs_window} | ac_chunk={ac_chunk}")

    for ep_path in tqdm(episode_files, desc="Episodes"):
        ep_name = ep_path.stem
        print(f"\nProcessing {ep_name}")

        try:
            states, actions, classes = load_episode_arrays(ep_path, rollout_cfg)
            states_norm, actions_norm = normalize_episode(states, actions, rollout_cfg)
            obs_np, act_np, cls_np = build_windows(states_norm, actions_norm, classes, obs_window=obs_window, ac_chunk=ac_chunk)
            dists = extract_z_params(policy, obs_np, act_np, cls_np, device=device)
        except Exception as e:
            print(f"Skip {ep_name}: {e}")
            continue

        # raw_path = save_dir / f"z_raw_data_{ep_name}.pkl"
        # with open(raw_path, "wb") as f:
        #     pickle.dump(dists, f)
        # print(f"Saved: {raw_path}")

        if SAVE_STATIC_PLOTS:
            visualize_z_statistics(dists, save_dir, ep_name)

        if SAVE_VIDEO:
            video_path = save_dir / f"z_dist_{ep_name}.mp4"
            try:
                visualize_distributions_video(
                    dists,
                    video_path,
                    ep_name,
                    fps=VIDEO_FPS,
                    dpi=VIDEO_DPI,
                    frame_stride=VIDEO_FRAME_STRIDE,
                    x_points=VIDEO_X_POINTS,
                    x_std_mult=VIDEO_X_STD_MULT,
                )
            except Exception as e:
                print(f"Video save failed for {ep_name}: {e}")


if __name__ == "__main__":
    main()
