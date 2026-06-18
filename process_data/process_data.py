# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

import pickle
from pathlib import Path

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
from utils_data_process import (
    downsample_data,
    gaussian_norm,
    generate_robobuf,
    sync_data_slowest,
)


def _build_topic_slices(state_obs_topics, state_topic_dims):
    if len(state_obs_topics) != len(state_topic_dims):
        raise ValueError("State topics and dims length mismatch.")
    topic_slices = {}
    offset = 0
    for topic, dim in zip(state_obs_topics, state_topic_dims):
        topic_slices[topic] = slice(offset, offset + dim)
        offset += dim
    return topic_slices, offset


def _resolve_pose_topic(state_obs_topics, topic_slices, cfg):
    pose_topic = cfg.get("pose_topic", None)
    if pose_topic and pose_topic in topic_slices:
        return pose_topic

    preferred = [
        "/cartesian_impedance_controller/pose_command",
        "/cartesian_impedance_controller/ee_pose",
        "/cartesian_impedance_controller/ee_pose_commanded",
        "/franka_robot_state_broadcaster/robot_state",
    ]
    for topic in preferred:
        if topic in topic_slices:
            return topic

    candidates = [
        topic
        for topic in state_obs_topics
        if (topic in topic_slices) and (topic_slices[topic].stop - topic_slices[topic].start == 9) and ("pose" in topic)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"⚠️ Multiple pose topic candidates found: {candidates}. Using {candidates[0]}.")
        return candidates[0]
    return None


def _resolve_first_available_topic(topic_slices, candidates, label):
    for topic in candidates:
        if topic in topic_slices:
            return topic
    print(f"⚠️ Could not find {label} topic. Tried: {candidates}")
    return None


def _parse_workspace_limits(cfg):
    workspace_limits = cfg.get("workspace_limits", None)
    if workspace_limits is None:
        return None

    if isinstance(workspace_limits, (DictConfig, ListConfig)):
        workspace_limits = OmegaConf.to_container(workspace_limits, resolve=True)

    if isinstance(workspace_limits, (list, tuple)) and len(workspace_limits) == 3:
        mins = [float(v[0]) for v in workspace_limits]
        maxs = [float(v[1]) for v in workspace_limits]
        return np.array(mins, dtype=float), np.array(maxs, dtype=float), "list"

    def _axis_limits(axis):
        if axis in workspace_limits:
            val = workspace_limits[axis]
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return float(val[0]), float(val[1])
        min_key, max_key = f"{axis}_min", f"{axis}_max"
        if min_key in workspace_limits and max_key in workspace_limits:
            return float(workspace_limits[min_key]), float(workspace_limits[max_key])
        return None

    limits = [_axis_limits(axis) for axis in ("x", "y", "z")]
    if any(v is None for v in limits):
        return None
    mins = [v[0] for v in limits]
    maxs = [v[1] for v in limits]
    return np.array(mins, dtype=float), np.array(maxs, dtype=float), "mapping"


def _compute_minmax_from_data(list_of_arrays, feature_slice):
    data = np.concatenate([arr[:, feature_slice] for arr in list_of_arrays], axis=0)
    mins = np.nanmin(data, axis=0)
    maxs = np.nanmax(data, axis=0)
    return mins, maxs


def _compute_gaussian_stats(list_of_arrays, feature_slice):
    data = np.concatenate([arr[:, feature_slice] for arr in list_of_arrays], axis=0)
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    std[std == 0] = 1e-17
    return mean, std


def _compute_pose_gaussian_stats(list_of_arrays, feature_slice, shared_std=False):
    mean, std = _compute_gaussian_stats(list_of_arrays, feature_slice)
    if shared_std:
        data = np.concatenate([arr[:, feature_slice] for arr in list_of_arrays], axis=0)
        shared = float(np.nanstd(data))
        if shared == 0:
            shared = 1e-17
        std = np.full_like(std, shared)
    return mean, std


def _apply_minmax(list_of_arrays, feature_slice, mins, maxs):
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    denom = maxs - mins
    denom[denom == 0] = 1e-17
    for array in list_of_arrays:
        array[:, feature_slice] = (2.0 * (array[:, feature_slice] - mins) / denom) - 1.0


def min_max_norm(list_of_arrays, feature_slice, clip_value=None):
    """Normalize features with min-max scaling to [-1, 1].

    Returns (mins, maxs) for logging/rollout_config.
    """
    mins, maxs = _compute_minmax_from_data(list_of_arrays, feature_slice)
    _apply_minmax(list_of_arrays, feature_slice, mins, maxs)
    if clip_value is not None:
        _apply_clip(list_of_arrays, feature_slice, clip_value)
    return mins, maxs


def _apply_gaussian(list_of_arrays, feature_slice, mean, std):
    for array in list_of_arrays:
        array[:, feature_slice] = (array[:, feature_slice] - mean) / std


def _apply_log1p(list_of_arrays, feature_slice):
    for array in list_of_arrays:
        values = array[:, feature_slice]
        array[:, feature_slice] = np.sign(values) * np.log1p(np.abs(values))


def _apply_clip(list_of_arrays, feature_slice, clip_value):
    if clip_value is None:
        return
    c = float(clip_value)
    if c <= 0:
        return
    for array in list_of_arrays:
        array[:, feature_slice] = np.clip(array[:, feature_slice], -c, c)


def _apply_fixed_scale(list_of_arrays, feature_slice, scales, clip_value=None):
    scales = np.asarray(scales, dtype=float).reshape(-1)
    if scales.size != (feature_slice.stop - feature_slice.start):
        raise ValueError("Scale vector length must match slice width.")
    scales[scales == 0] = 1e-17
    for array in list_of_arrays:
        array[:, feature_slice] = array[:, feature_slice] / scales
    if clip_value is not None:
        _apply_clip(list_of_arrays, feature_slice, clip_value)


def _parse_tracking_error_scales(cfg):
    """
    Returns (pos_scale, rot_scale).
    Defaults chosen per user spec: pos=0.1m, rot=0.5rad.
    """
    scales = cfg.get("tracking_error_scales", None)
    if scales is None:
        return 0.1, 0.5
    if isinstance(scales, (DictConfig, ListConfig)):
        scales = OmegaConf.to_container(scales, resolve=True)
    if isinstance(scales, dict):
        pos = float(scales.get("pos", 0.1))
        rot = float(scales.get("rot", 0.5))
        return pos, rot
    if isinstance(scales, (list, tuple)) and len(scales) == 2:
        return float(scales[0]), float(scales[1])
    return 0.1, 0.5


def normalize_states_groupwise(all_states_for_norm, state_obs_topics, state_topic_dims, cfg):
    topic_slices, state_dim = _build_topic_slices(state_obs_topics, state_topic_dims)

    pose_topic = _resolve_pose_topic(state_obs_topics, topic_slices, cfg)
    vel_topic = _resolve_first_available_topic(
        topic_slices,
        [
            "/cartesian_impedance_controller/ee_velocity",
            "/cartesian_admittance_controller/ee_velocity",
        ],
        "EE velocity",
    )
    track_topic = _resolve_first_available_topic(
        topic_slices,
        [
            "/cartesian_impedance_controller/tracking_error",
            "/cartesian_admittance_controller/tracking_error",
        ],
        "tracking error",
    )
    wrench_topic = "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame"
    cmd_topic = _resolve_first_available_topic(
        topic_slices,
        [
            "/cartesian_impedance_controller/pose_command",
            "/cartesian_admittance_controller/pose_command",
        ],
        "commanded pose",
    )

    required_topics = [pose_topic, vel_topic, track_topic, wrench_topic, cmd_topic]
    missing_topics = [t for t in required_topics if t is None or t not in topic_slices]
    if missing_topics:
        print(f"⚠️ Missing topics for grouped normalization: {missing_topics}. Falling back to gaussian norm.")
        return gaussian_norm(all_states_for_norm)

    pose_slice = topic_slices[pose_topic]
    if (pose_slice.stop - pose_slice.start) != 9:
        raise ValueError(f"Pose topic {pose_topic} must be 9-dim, got {pose_slice.stop - pose_slice.start}.")

    vel_slice = topic_slices[vel_topic]
    if (vel_slice.stop - vel_slice.start) != 6:
        raise ValueError(f"EE velocity topic {vel_topic} must be 6-dim.")

    track_slice = topic_slices[track_topic]
    if (track_slice.stop - track_slice.start) != 6:
        raise ValueError(f"Tracking error topic {track_topic} must be 6-dim.")

    wrench_slice = topic_slices[wrench_topic]
    if (wrench_slice.stop - wrench_slice.start) != 6:
        raise ValueError(f"External wrench topic {wrench_topic} must be 6-dim.")

    cmd_slice = topic_slices[cmd_topic]
    if (cmd_slice.stop - cmd_slice.start) != 9:
        raise ValueError(f"Command topic {cmd_topic} must be 9-dim.")

    pos_slice = slice(pose_slice.start, pose_slice.start + 3)
    ori_slice = slice(pose_slice.start + 3, pose_slice.start + 9)
    pose_norm_mode = str(cfg.get("pose_normalization_mode", "per_dim"))
    if pose_norm_mode not in {"per_dim", "group_shared"}:
        raise ValueError(f"pose_normalization_mode must be 'per_dim' or 'group_shared', got {pose_norm_mode}.")
    shared_pose_std = pose_norm_mode == "group_shared"

    stats = {"mode": "grouped", "state_dim": state_dim, "groups": []}

    pos_mean, pos_std = _compute_pose_gaussian_stats(all_states_for_norm, pos_slice, shared_std=shared_pose_std)
    _apply_gaussian(all_states_for_norm, pos_slice, pos_mean, pos_std)
    pose_clip = cfg.get("pose_clip", None)
    pose_clip = None if pose_clip is None else float(pose_clip)
    pose_clip_value = pose_clip if (pose_clip is not None and pose_clip > 0) else None
    if pose_clip_value is not None:
        _apply_clip(all_states_for_norm, pos_slice, pose_clip_value)

    pos_group = {
        "name": "ee_position",
        "type": "gaussian_clip" if pose_clip_value is not None else "gaussian",
        "indices": [pos_slice.start, pos_slice.stop],
        "mean": [float(x) for x in pos_mean],
        "std": [float(x) for x in pos_std],
    }
    if pose_clip_value is not None:
        pos_group["clip"] = float(pose_clip_value)
    stats["groups"].append(pos_group)

    orientation_mean, orientation_std = _compute_pose_gaussian_stats(
        all_states_for_norm, ori_slice, shared_std=shared_pose_std
    )
    _apply_gaussian(all_states_for_norm, ori_slice, orientation_mean, orientation_std)
    stats["groups"].append(
        {
            "name": "ee_orientation",
            # "type": "identity", ###
            "type": "gaussian",
            "mean": [float(x) for x in orientation_mean],
            "std": [float(x) for x in orientation_std],
            "indices": [ori_slice.start, ori_slice.stop],
        }
    )

    ### added command to input
    cmd_pos_slice = slice(cmd_slice.start, cmd_slice.start + 3)
    cmd_ori_slice = slice(cmd_slice.start + 3, cmd_slice.start + 9)
    cmd_pos_mean, cmd_pos_std = _compute_pose_gaussian_stats(
        all_states_for_norm, cmd_pos_slice, shared_std=shared_pose_std
    )
    cmd_ori_mean, cmd_ori_std = _compute_pose_gaussian_stats(
        all_states_for_norm, cmd_ori_slice, shared_std=shared_pose_std
    )
    cmd_mean = np.concatenate([cmd_pos_mean, cmd_ori_mean], axis=0)
    cmd_std = np.concatenate([cmd_pos_std, cmd_ori_std], axis=0)
    _apply_gaussian(all_states_for_norm, cmd_slice, cmd_mean, cmd_std)
    cmd_group = {
        "name": "ee_pose_commanded",
        "type": "gaussian",
        "indices": [cmd_slice.start, cmd_slice.stop],
        "mean": [float(x) for x in cmd_mean],
        "std": [float(x) for x in cmd_std],
    }
    stats["groups"].append(cmd_group)

    vel_mean, vel_std = _compute_gaussian_stats(all_states_for_norm, vel_slice)
    _apply_gaussian(all_states_for_norm, vel_slice, vel_mean, vel_std)
    velocity_clip = cfg.get("velocity_clip", None)
    velocity_clip = None if velocity_clip is None else float(velocity_clip)
    velocity_clip_value = velocity_clip if (velocity_clip is not None and velocity_clip > 0) else None
    if velocity_clip_value is not None:
        _apply_clip(all_states_for_norm, vel_slice, velocity_clip_value)

    vel_group = {
        "name": "ee_velocity",
        "type": "gaussian_clip" if velocity_clip_value is not None else "gaussian",
        "indices": [vel_slice.start, vel_slice.stop],
        "mean": [float(x) for x in vel_mean],
        "std": [float(x) for x in vel_std],
    }
    if velocity_clip_value is not None:
        vel_group["clip"] = float(velocity_clip_value)
    stats["groups"].append(vel_group)

    track_mean, track_std = _compute_gaussian_stats(all_states_for_norm, track_slice)
    _apply_gaussian(all_states_for_norm, track_slice, track_mean, track_std)

    track_clip = cfg.get("tracking_error_clip", None)
    track_clip = None if track_clip is None else float(track_clip)
    track_clip_value = track_clip if (track_clip is not None and track_clip > 0) else None
    if track_clip_value is not None:
        _apply_clip(all_states_for_norm, track_slice, track_clip_value)

    track_group = {
        "name": "tracking_error",
        "type": "gaussian_clip" if track_clip_value is not None else "gaussian",
        "indices": [track_slice.start, track_slice.stop],
        "mean": [float(x) for x in track_mean],
        "std": [float(x) for x in track_std],
    }
    if track_clip_value is not None:
        track_group["clip"] = float(track_clip_value)
    stats["groups"].append(track_group)

    # wrench_mins, wrench_maxs = min_max_norm(all_states_for_norm, wrench_slice, clip_value=None)
    # stats["groups"].append(
    #     {
    #         "name": "external_wrench",
    #         "type": "min_max",
    #         "indices": [wrench_slice.start, wrench_slice.stop],
    #         "min": [float(x) for x in wrench_mins],
    #         "max": [float(x) for x in wrench_maxs],
    #         "formula": "(2*(x-min)/(max-min) - 1)",
    #     }
    # )

    wrench_mean, wrench_std = _compute_gaussian_stats(all_states_for_norm, wrench_slice)
    _apply_gaussian(all_states_for_norm, wrench_slice, wrench_mean, wrench_std)
    # _apply_log1p(all_states_for_norm, wrench_slice)
    wrench_clip = cfg.get("wrench_clip", None)
    wrench_clip = None if wrench_clip is None else float(wrench_clip)
    wrench_clip_value = wrench_clip if (wrench_clip is not None and wrench_clip > 0) else None
    if wrench_clip_value is not None:
        _apply_clip(all_states_for_norm, wrench_slice, wrench_clip_value)

    wrench_group = {
        "name": "external_wrench",
        "type": "gaussian_clip" if wrench_clip_value is not None else "gaussian",
        "indices": [wrench_slice.start, wrench_slice.stop],
        "mean": [float(x) for x in wrench_mean],
        "std": [float(x) for x in wrench_std],
        # "formula": "clip((sign(x)*log1p(abs(x)) - mean)/std, ±clip)",
    }
    if wrench_clip_value is not None:
        wrench_group["clip"] = float(wrench_clip_value)
    stats["groups"].append(wrench_group)

    used = np.zeros(state_dim, dtype=bool)
    for group in stats["groups"]:
        start, stop = group["indices"]
        used[start:stop] = True
    if not np.all(used):
        remaining = np.where(~used)[0]
        rem_slice = slice(int(remaining[0]), int(remaining[-1]) + 1)
        rem_mean, rem_std = _compute_gaussian_stats(all_states_for_norm, rem_slice)
        _apply_gaussian(all_states_for_norm, rem_slice, rem_mean, rem_std)
        stats["groups"].append(
            {
                "name": "residual_features",
                "type": "gaussian",
                "indices": [rem_slice.start, rem_slice.stop],
                "mean": [float(x) for x in rem_mean],
                "std": [float(x) for x in rem_std],
            }
        )

    return stats


def normalize_actions_groupwise(all_actions_for_norm, cfg):
    """
    Normalize actions assuming commanded EE pose actions:
        - position (x,y,z) z-score using dataset mean/std, optional clip via cfg.pose_clip
        - orientation (6 dims) identity (rotation-matrix columns already in [-1, 1])
    Falls back to full-vector gaussian normalization if the action dimensionality is not compatible.
    """
    if not all_actions_for_norm:
        return {"mode": "none", "action_dim": 0, "groups": []}

    action_dim = int(all_actions_for_norm[0].shape[1])
    if any(arr.shape[1] != action_dim for arr in all_actions_for_norm):
        raise ValueError("Action dimensions changed across episodes; cannot build consistent normalization stats.")

    if action_dim < 9:
        print(f"⚠️ Action dim {action_dim} < 9; falling back to gaussian norm.")
        return gaussian_norm(all_actions_for_norm)

    pos_slice = slice(0, 3)
    ori_slice = slice(3, 9)
    pose_norm_mode = str(cfg.get("pose_normalization_mode", "per_dim"))
    if pose_norm_mode not in {"per_dim", "group_shared"}:
        raise ValueError(f"pose_normalization_mode must be 'per_dim' or 'group_shared', got {pose_norm_mode}.")
    shared_pose_std = pose_norm_mode == "group_shared"

    stats = {"mode": "grouped", "action_dim": action_dim, "groups": []}

    pos_mean, pos_std = _compute_pose_gaussian_stats(all_actions_for_norm, pos_slice, shared_std=shared_pose_std)
    _apply_gaussian(all_actions_for_norm, pos_slice, pos_mean, pos_std)
    pose_clip = cfg.get("pose_clip", None)
    pose_clip = None if pose_clip is None else float(pose_clip)
    pose_clip_value = pose_clip if (pose_clip is not None and pose_clip > 0) else None
    if pose_clip_value is not None:
        _apply_clip(all_actions_for_norm, pos_slice, pose_clip_value)

    pos_group = {
        "name": "ee_position",
        "type": "gaussian_clip" if pose_clip_value is not None else "gaussian",
        "indices": [pos_slice.start, pos_slice.stop],
        "mean": [float(x) for x in pos_mean],
        "std": [float(x) for x in pos_std],
    }
    if pose_clip_value is not None:
        pos_group["clip"] = float(pose_clip_value)
    stats["groups"].append(pos_group)

    orientation_mean, orientation_std = _compute_pose_gaussian_stats(
        all_actions_for_norm, ori_slice, shared_std=shared_pose_std
    )
    _apply_gaussian(all_actions_for_norm, ori_slice, orientation_mean, orientation_std)
    stats["groups"].append(
        {
            "name": "ee_orientation",
            # "type": "identity", ###
            "type": "gaussian",
            "mean": [float(x) for x in orientation_mean],
            "std": [float(x) for x in orientation_std],
            "indices": [ori_slice.start, ori_slice.stop],
        }
    )

    used = np.zeros(action_dim, dtype=bool)
    for group in stats["groups"]:
        start, stop = group["indices"]
        used[int(start) : int(stop)] = True

    # Anything else beyond known groups: gaussian
    if not np.all(used):
        remaining = np.where(~used)[0]
        rem_slice = slice(int(remaining[0]), int(remaining[-1]) + 1)
        rem_mean, rem_std = _compute_gaussian_stats(all_actions_for_norm, rem_slice)
        _apply_gaussian(all_actions_for_norm, rem_slice, rem_mean, rem_std)
        stats["groups"].append(
            {
                "name": "residual_features",
                "type": "gaussian",
                "indices": [rem_slice.start, rem_slice.stop],
                "mean": [float(x) for x in rem_mean],
                "std": [float(x) for x in rem_std],
            }
        )

    return stats


def _normalize_buffer_name(name, default_name):
    if name is None:
        name = default_name
    name = str(name).strip()
    if len(name) == 0:
        name = default_name
    if name.endswith(".pkl"):
        name = name[:-4]
    return name


def _parse_split_cfg(cfg):
    split_cfg = cfg.get("train_test_split", {})
    if isinstance(split_cfg, (DictConfig, ListConfig)):
        split_cfg = OmegaConf.to_container(split_cfg, resolve=True)
    if not isinstance(split_cfg, dict):
        split_cfg = {}

    enabled = bool(split_cfg.get("enabled", False))
    train_ratio = float(split_cfg.get("train_ratio", 0.9))
    seed = int(split_cfg.get("seed", 45))
    train_buffer_name = _normalize_buffer_name(split_cfg.get("train_buffer_name", "buf_train"), "buf_train")
    test_buffer_name = _normalize_buffer_name(split_cfg.get("test_buffer_name", "buf_test"), "buf_test")

    return {
        "enabled": enabled,
        "train_ratio": train_ratio,
        "seed": seed,
        "train_buffer_name": train_buffer_name,
        "test_buffer_name": test_buffer_name,
    }


def _resolve_input_folders(cfg):
    """Accept one input folder or a YAML list of folders."""
    input_paths = cfg.get("input_paths", None)
    if input_paths is None:
        input_paths = cfg.input_path

    if isinstance(input_paths, (DictConfig, ListConfig)):
        input_paths = OmegaConf.to_container(input_paths, resolve=True)

    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]

    if not isinstance(input_paths, (list, tuple)) or len(input_paths) == 0:
        raise ValueError("Set input_path to a folder, or input_paths/input_path to a non-empty list of folders.")

    folders = [Path(path) for path in input_paths]
    missing = [str(folder) for folder in folders if not folder.is_dir()]
    if missing:
        raise FileNotFoundError(f"Input folder(s) not found: {missing}")
    return folders


def _as_topic_list(value, field_name):
    """Accept one topic string, a YAML list of topics, or an empty/null value."""
    if value is None:
        return []
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(topic).strip() for topic in value if str(topic).strip()]
    raise ValueError(f"{field_name} must be a topic string or a list of topic strings, got {type(value).__name__}.")


CONTROLLER_TOPIC_FALLBACKS = {
    "/cartesian_impedance_controller/ee_velocity": ["/cartesian_admittance_controller/ee_velocity"],
    "/cartesian_impedance_controller/tracking_error": ["/cartesian_admittance_controller/tracking_error"],
    "/cartesian_impedance_controller/pose_command": ["/cartesian_admittance_controller/pose_command"],
    "/cartesian_admittance_controller/ee_velocity": ["/cartesian_impedance_controller/ee_velocity"],
    "/cartesian_admittance_controller/tracking_error": ["/cartesian_impedance_controller/tracking_error"],
    "/cartesian_admittance_controller/pose_command": ["/cartesian_impedance_controller/pose_command"],
}


def _apply_topic_fallbacks(traj_data, required_topics):
    """Alias equivalent controller topics into the configured names for one episode."""
    data = traj_data.get("data", {})
    timestamps = traj_data.get("timestamps", {})
    used_fallbacks = []

    for expected_topic in required_topics:
        if expected_topic in data and expected_topic in timestamps:
            continue

        for fallback_topic in CONTROLLER_TOPIC_FALLBACKS.get(expected_topic, []):
            if fallback_topic in data and fallback_topic in timestamps:
                # Keep the config topic as the canonical key so feature order stays stable.
                data[expected_topic] = data[fallback_topic]
                timestamps[expected_topic] = timestamps[fallback_topic]
                used_fallbacks.append((expected_topic, fallback_topic))
                break

    return used_fallbacks


def _split_episode_indices(num_episodes, train_ratio, seed):
    if num_episodes < 2:
        raise ValueError("Need at least 2 episodes to create train/test buffers.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(num_episodes)
    n_train = int(round(num_episodes * train_ratio))
    n_train = max(1, min(num_episodes - 1, n_train))

    train_indices = np.sort(permuted[:n_train]).tolist()
    test_indices = np.sort(permuted[n_train:]).tolist()
    return train_indices, test_indices


@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig):
    output_path = cfg.output_path
    downsample = cfg.get("downsample", False)
    data_frequency = cfg.get("data_frequency", 50.0)
    target_downsampling_freq = cfg.get("target_downsampling_freq", 50.0)

    # rgb_obs_topics = list(cfg.cameras_topics)
    state_obs_topics = _as_topic_list(cfg.obs_topics, "obs_topics")
    goal_topics = _as_topic_list(cfg.get("goal_topic", []), "goal_topic")
    arrangement_topic = cfg.get("arrangement_topic", None)
    mode_topic = cfg.get("mode_topic", None)
    label_topics = list(goal_topics)
    for topic in (arrangement_topic, mode_topic):
        if topic and topic not in label_topics:
            label_topics.append(topic)
    action_config = dict(cfg.action_config)
    action_topics = list(action_config.keys())
    action_pose_mode = str(cfg.get("action_pose_mode", "absolute"))
    stiffness_label_topic = cfg.get("stiffness_label_topic", None)
    stiffness_label_key = cfg.get("stiffness_label_key", "stiffness")
    stiffness_norm_thresholds = cfg.get("stiffness_norm_thresholds", [200.0, 1000.0])
    stiffness_norm_thresholds = sorted([float(v) for v in stiffness_norm_thresholds])
    split_cfg = _parse_split_cfg(cfg)

    assert len(state_obs_topics) > 0, "Require low-dim observation topics"
    # assert len(rgb_obs_topics) > 0, "Require visual observation topics"
    assert len(action_topics) > 0, "Require action topics"
    assert target_downsampling_freq > 0, "Require positive target frequency"

    data_folders = _resolve_input_folders(cfg)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # initialize topics
    # all_topics = state_obs_topics + rgb_obs_topics + action_topics
    all_topics = list(dict.fromkeys(state_obs_topics + action_topics + label_topics))
    if stiffness_label_topic and stiffness_label_topic not in all_topics:
        all_topics.append(stiffness_label_topic)

    state_topic_specs = {
        "/cartesian_impedance_controller/ee_velocity": {"keys": ["ee_velocity"], "dim": 6, "fallback": "data"},
        "/cartesian_admittance_controller/ee_velocity": {"keys": ["ee_velocity"], "dim": 6, "fallback": "data"},
        "/cartesian_impedance_controller/pose_command": {"keys": ["ee_pose_commanded"], "dim": 9, "fallback": None},
        "/cartesian_admittance_controller/pose_command": {"keys": ["ee_pose_commanded"], "dim": 9, "fallback": None},
        "/cartesian_impedance_controller/tracking_error": {"keys": ["tracking_error"], "dim": 6, "fallback": "data"},
        "/cartesian_admittance_controller/tracking_error": {"keys": ["tracking_error"], "dim": 6, "fallback": "data"},
        "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame": {
            "keys": ["external_wrench"],
            "dim": 6,
            "fallback": None,
        },
    }

    goal_topic_specs = {
        "/goal": {"keys": ["goal"], "dim": 1, "fallback": None},
        "/arrangement": {"keys": ["arrangement", "arrangement_id"], "dim": 1, "fallback": None},
        "/mode": {"keys": ["mode"], "dim": 1, "fallback": None},
    }

    def extract_fixed_vector(msg, keys, dim, fallback_key=None):
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

    def extract_concat_vector(msg, concat_keys, dim):
        if not isinstance(msg, dict):
            return np.full((dim,), np.nan, dtype=float)
        parts = []
        for key, kdim in concat_keys:
            if key in msg and msg[key] is not None:
                arr = np.asarray(msg[key], dtype=float).flatten()
                if arr.size != int(kdim):
                    arr = np.full((int(kdim),), np.nan, dtype=float)
            else:
                arr = np.full((int(kdim),), np.nan, dtype=float)
            parts.append(arr)
        vec = np.concatenate(parts, axis=0) if parts else np.full((dim,), np.nan, dtype=float)
        if vec.size != dim:
            return np.full((dim,), np.nan, dtype=float)
        return vec

    def extract_ep_index(path):
        name = path.stem  # e.g., "ep_12"
        return int(name.split("_")[1])

    def stiffness_vec_to_class(stiffness_vec):
        norm = float(np.linalg.norm(np.asarray(stiffness_vec, dtype=float)))
        if not np.isfinite(norm):
            return 1
        for idx, threshold in enumerate(stiffness_norm_thresholds, start=1):
            if norm < threshold:
                return idx
        return len(stiffness_norm_thresholds) + 1

    all_episodes = []
    for folder_index, data_folder in enumerate(data_folders):
        folder_episodes = sorted(
            [f for f in data_folder.iterdir() if f.name.startswith("ep_") and f.name.endswith(".pkl")],
            key=extract_ep_index,
        )
        # Keep YAML folder order stable, while preserving episode order inside each folder.
        all_episodes.extend((folder_index, episode_pkl) for episode_pkl in folder_episodes)

    if len(all_episodes) == 0:
        raise FileNotFoundError(f"No ep_*.pkl files found in input folder(s): {[str(p) for p in data_folders]}")

    def episode_label(folder_index, episode_pkl):
        return f"{data_folders[folder_index].parent.name}/{episode_pkl.stem}"

    print(f"Input folders: {[str(p) for p in data_folders]}")
    print(f"These episodes will be processed in this order: {[episode_label(i, p) for i, p in all_episodes]}")

    trajectories = []
    processed_episode_names = []
    all_states = []
    all_states_for_norm = []
    all_actions = []
    pbar = tqdm(all_episodes)
    topic_fallback_counts = {}

    state_topic_dims = None

    for folder_index, episode_pkl in pbar:
        with open(episode_pkl, "rb") as f:
            traj_data = pickle.load(f)
        processed_episode_names.append(episode_label(folder_index, episode_pkl))
        used_fallbacks = _apply_topic_fallbacks(traj_data, all_topics)
        for expected_topic, fallback_topic in used_fallbacks:
            key = (expected_topic, fallback_topic)
            topic_fallback_counts[key] = topic_fallback_counts.get(key, 0) + 1
        traj_data, avg_freq = sync_data_slowest(traj_data, all_topics)
        pbar.set_postfix({"avg_freq": f"{avg_freq:.1f} Hz"})

        # 🕒 Downsample to target rate
        print("Original lengths:", [len(traj_data[key]) for key in traj_data.keys()])
        if downsample:
            traj_data, avg_freq = downsample_data(traj_data, avg_freq, target_downsampling_freq)
            if isinstance(traj_data, dict):
                print("lengths after downsampling:", [len(traj_data[key]) for key in traj_data.keys()])
            else:
                print("Error: traj_data is not a dictionary. Skipping length calculation.")
        else:
            print(f"Not downsampling, data frequency: {avg_freq:.1f} Hz")

        # ------------------------------
        traj = {}

        # traj['states'] = np.concatenate([np.array(traj_data[topic]) for topic in state_obs_topics], axis=-1)
        # Flatten each dict in state topics into numeric arrays
        ##########################
        # Flatten each dict in state topics into numeric arrays (fixed spec)
        state_arrays = []
        episode_dims = []
        for topic in state_obs_topics:
            if topic in state_topic_specs:
                spec = state_topic_specs[topic]
                if spec.get("concat", False):
                    topic_vectors = [
                        extract_concat_vector(msg, spec.get("concat_keys", []), spec["dim"]) for msg in traj_data[topic]
                    ]
                else:
                    topic_vectors = [
                        extract_fixed_vector(msg, spec["keys"], spec["dim"], spec["fallback"])
                        for msg in traj_data[topic]
                    ]
                topic_array = np.stack(topic_vectors, axis=0)  # (num_steps, dim)
            else:
                topic_data = []
                max_len = 0
                for msg in traj_data[topic]:
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

                topic_array = np.stack(topic_padded, axis=0)  # (num_steps, max_len)
            state_arrays.append(topic_array)
            episode_dims.append(topic_array.shape[1])

        if state_topic_dims is None:
            state_topic_dims = episode_dims
        else:
            if episode_dims != state_topic_dims:
                raise ValueError(
                    "State topic dimensions changed across episodes; cannot build consistent normalization stats."
                )

        # Concatenate all topics along feature dimension
        traj["states"] = np.concatenate(state_arrays, axis=-1)

        # Extract goal labels separately; arrangement/mode stay outside obs["goals"].
        goals_arrays = []
        for topic in goal_topics:
            spec = goal_topic_specs.get(topic, {"keys": ["goal"], "dim": 1, "fallback": None})
            if topic not in traj_data:
                raise KeyError(f"Configured goal topic {topic} missing after sync for {episode_pkl.name}")
            topic_vectors = [
                extract_fixed_vector(msg, spec["keys"], spec["dim"], spec["fallback"]) for msg in traj_data[topic]
            ]
            goals_arrays.append(np.stack(topic_vectors, axis=0))
        if goals_arrays:
            traj["goals"] = np.concatenate(goals_arrays, axis=-1)

        if arrangement_topic:
            spec = goal_topic_specs.get(arrangement_topic, {"keys": ["arrangement"], "dim": 1, "fallback": None})
            if arrangement_topic not in traj_data:
                raise KeyError(f"Configured arrangement topic {arrangement_topic} missing after sync for {episode_pkl.name}")
            arrangement_vectors = [
                extract_fixed_vector(msg, spec["keys"], spec["dim"], spec["fallback"])
                for msg in traj_data[arrangement_topic]
            ]
            traj["arrangement"] = np.stack(arrangement_vectors, axis=0)

        if mode_topic:
            spec = goal_topic_specs.get(mode_topic, {"keys": ["mode"], "dim": 1, "fallback": None})
            if mode_topic not in traj_data:
                raise KeyError(f"Configured mode topic {mode_topic} missing after sync for {episode_pkl.name}")
            mode_vectors = [
                extract_fixed_vector(msg, spec["keys"], spec["dim"], spec["fallback"]) for msg in traj_data[mode_topic]
            ]
            traj["mode"] = np.stack(mode_vectors, axis=0)

        if stiffness_label_topic:
            stiffness_vectors = [
                extract_fixed_vector(msg, [stiffness_label_key], dim=6, fallback_key=None)
                for msg in traj_data[stiffness_label_topic]
            ]
            stiffness_vectors = np.stack(stiffness_vectors, axis=0)
            stiffness_labels = np.asarray([stiffness_vec_to_class(v) for v in stiffness_vectors], dtype=np.int64)
            traj["stiffness_label"] = stiffness_labels[:, None]

        action_list = []
        # for topic in action_topics:
        #     actions = np.array(traj_data[topic])
        #     action_list.append(actions)

        # Flatten each action topic into numeric arrays.
        action_topic_specs = {
            "/cartesian_impedance_controller/pose_command": {"keys": ["ee_pose_commanded"], "dim": 9, "fallback": None},
            "/cartesian_admittance_controller/pose_command": {"keys": ["ee_pose_commanded"], "dim": 9, "fallback": None},
        }

        action_list = []
        action_dims = []
        for topic in action_topics:
            if topic in action_topic_specs:
                spec = action_topic_specs[topic]
                if spec.get("concat", False):
                    topic_vectors = [
                        extract_concat_vector(msg, spec.get("concat_keys", []), spec["dim"]) for msg in traj_data[topic]
                    ]
                else:
                    topic_vectors = [
                        extract_fixed_vector(msg, spec["keys"], spec["dim"], spec["fallback"])
                        for msg in traj_data[topic]
                    ]
                topic_array = np.stack(topic_vectors, axis=0)
            else:
                topic_array = np.stack([np.asarray(m, dtype=float).flatten() for m in traj_data[topic]], axis=0)

            if topic in action_topic_specs and action_pose_mode == "relative":
                relative_topic_array = np.zeros_like(topic_array)
                relative_topic_array[1:] = topic_array[1:] - topic_array[:-1]
                topic_array = relative_topic_array

            action_list.append(topic_array)
            action_dims.append(int(topic_array.shape[1]))

        # Optional: verify action dims match the config (helps catch key/order mistakes).
        expected_dims = [int(action_config.get(t, -1)) for t in action_topics]
        if all(d > 0 for d in expected_dims) and expected_dims != action_dims:
            raise ValueError(
                f"Action topic dims mismatch. expected={expected_dims} actual={action_dims} topics={action_topics}"
            )

        traj["actions"] = np.concatenate(action_list, axis=-1)
        lengths = [traj["states"].shape[0], traj["actions"].shape[0]]
        if "goals" in traj:
            lengths.append(traj["goals"].shape[0])
        if "arrangement" in traj:
            lengths.append(traj["arrangement"].shape[0])
        if "mode" in traj:
            lengths.append(traj["mode"].shape[0])
        if "stiffness_label" in traj:
            lengths.append(traj["stiffness_label"].shape[0])
        num_steps = int(np.min(lengths))
        traj["num_steps"] = num_steps
        traj["states"] = traj["states"][:num_steps]
        traj["actions"] = traj["actions"][:num_steps]
        if "goals" in traj:
            traj["goals"] = traj["goals"][:num_steps]
        if "arrangement" in traj:
            traj["arrangement"] = traj["arrangement"][:num_steps]
        if "mode" in traj:
            traj["mode"] = traj["mode"][:num_steps]
        if "stiffness_label" in traj:
            traj["stiffness_label"] = traj["stiffness_label"][:num_steps]

        all_states.append(traj["states"])
        all_states_for_norm.append(traj["states"])
        all_actions.append(traj["actions"])

        # Process and store images
        # for cam_ind, topic in enumerate(rgb_obs_topics):
        #     enc_images = traj_data[topic]
        #     processed_images = [process_image(img_enc) for img_enc in enc_images]
        #     traj[f'enc_cam_{cam_ind}'] = processed_images
        trajectories.append(traj)

    if len(processed_episode_names) != len(trajectories):
        raise RuntimeError(
            f"Episode name alignment error: processed_episode_names={len(processed_episode_names)} "
            f"trajectories={len(trajectories)}"
        )

    if topic_fallback_counts:
        print("Controller topic fallbacks used:")
        for (expected_topic, fallback_topic), count in sorted(topic_fallback_counts.items()):
            print(f"  {expected_topic} <- {fallback_topic}: {count} episode(s)")

    # normalize states and actions
    state_norm_stats = normalize_states_groupwise(all_states_for_norm, state_obs_topics, state_topic_dims, cfg)
    action_norm_stats = normalize_actions_groupwise(all_actions, cfg)
    norm_stats = dict(state=state_norm_stats, action=action_norm_stats)

    split_info = {
        "enabled": split_cfg["enabled"],
        "seed": split_cfg["seed"] if split_cfg["enabled"] else None,
        "train_ratio": split_cfg["train_ratio"] if split_cfg["enabled"] else None,
        "num_episodes_total": len(trajectories),
    }

    # dump data buffer(s)
    if split_cfg["enabled"]:
        train_indices, test_indices = _split_episode_indices(
            num_episodes=len(trajectories),
            train_ratio=split_cfg["train_ratio"],
            seed=split_cfg["seed"],
        )
        train_trajectories = [trajectories[idx] for idx in train_indices]
        test_trajectories = [trajectories[idx] for idx in test_indices]
        train_episode_names = [processed_episode_names[idx] for idx in train_indices]
        test_episode_names = [processed_episode_names[idx] for idx in test_indices]

        train_buffer = generate_robobuf(train_trajectories)
        test_buffer = generate_robobuf(test_trajectories)

        train_file = output_dir / f"{split_cfg['train_buffer_name']}.pkl"
        test_file = output_dir / f"{split_cfg['test_buffer_name']}.pkl"
        with open(train_file, "wb") as f:
            pickle.dump(train_buffer.to_traj_list(), f)
        with open(test_file, "wb") as f:
            pickle.dump(test_buffer.to_traj_list(), f)

        split_info.update(
            {
                "num_episodes_train": len(train_trajectories),
                "num_episodes_test": len(test_trajectories),
                "train_buffer": train_file.name,
                "test_buffer": test_file.name,
                "train_episodes": train_episode_names,
                "test_episodes": test_episode_names,
            }
        )
        print(
            f"Saved split buffers: train={train_file.name} ({len(train_trajectories)} eps), "
            f"test={test_file.name} ({len(test_trajectories)} eps), seed={split_cfg['seed']}"
        )
    else:
        buffer_name = "buf"
        buffer = generate_robobuf(trajectories)
        with open(output_dir / f"{buffer_name}.pkl", "wb") as f:
            pickle.dump(buffer.to_traj_list(), f)
        split_info.update({"all_buffer": f"{buffer_name}.pkl", "episodes": processed_episode_names})

    # dump rollout config
    obs_config = {
        "state_topics": state_obs_topics,
        "goal_topics": goal_topics,
        "goal_feature_names": [goal_topic_specs.get(topic, {"keys": ["goal"], "dim": 1})["keys"][0] for topic in goal_topics],
        # 'camera_topics': rgb_obs_topics,
    }
    if arrangement_topic:
        obs_config["arrangement_topic"] = arrangement_topic
    if mode_topic:
        obs_config["mode_topic"] = mode_topic
    if stiffness_label_topic:
        # number of classes = number of thresholds + 1; if no thresholds -> single class
        classes = list(range(1, len(stiffness_norm_thresholds) + 2))
        obs_config["stiffness_label"] = {
            "topic": stiffness_label_topic,
            "key": stiffness_label_key,
            "classes": classes,
            "norm_thresholds": stiffness_norm_thresholds,
        }
    processing_config = {
        "input_paths": [str(path) for path in data_folders],
        "pose_normalization_mode": str(cfg.get("pose_normalization_mode", "per_dim")),
        "action_pose_mode": action_pose_mode,
        "downsample": downsample,
        "data_frequency": data_frequency,
        "target_downsampling_freq": target_downsampling_freq,
    }
    rollout_config = {
        "obs_config": obs_config,
        "action_config": action_config,
        "norm_stats": norm_stats,
        "processing_config": processing_config,
        "split_config": split_info,
    }

    with open(output_dir / "rollout_config.yaml", "w") as f:
        yaml.dump(rollout_config, f, sort_keys=False)


if __name__ == "__main__":
    main()
