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
    lowpass_filter,
    medianfilter,
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


def _apply_minmax(list_of_arrays, feature_slice, mins, maxs):
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    denom = maxs - mins
    denom[denom == 0] = 1e-17
    for array in list_of_arrays:
        array[:, feature_slice] = (2.0 * (array[:, feature_slice] - mins) / denom) - 1.0


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
    vel_topic = "/cartesian_impedance_controller/ee_velocity"
    track_topic = "/cartesian_impedance_controller/tracking_error"
    wrench_topic = "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame"

    required_topics = [pose_topic, vel_topic, track_topic, wrench_topic]
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

    pos_slice = slice(pose_slice.start, pose_slice.start + 3)
    ori_slice = slice(pose_slice.start + 3, pose_slice.start + 9)

    stats = {"mode": "grouped", "state_dim": state_dim, "groups": []}

    pose_clip = float(cfg.get("pose_clip", 5.0))
    pos_mean, pos_std = _compute_gaussian_stats(all_states_for_norm, pos_slice)
    _apply_gaussian(all_states_for_norm, pos_slice, pos_mean, pos_std)
    _apply_clip(all_states_for_norm, pos_slice, pose_clip)
    stats["groups"].append(
        {
            "name": "ee_position",
            "type": "zscore_clip",
            "indices": [pos_slice.start, pos_slice.stop],
            "mean": [float(x) for x in pos_mean],
            "std": [float(x) for x in pos_std],
            "clip": pose_clip,
        }
    )
    stats["groups"].append(
        {
            "name": "ee_orientation",
            "type": "identity",
            "indices": [ori_slice.start, ori_slice.stop],
        }
    )

    vel_mean, vel_std = _compute_gaussian_stats(all_states_for_norm, vel_slice)
    _apply_gaussian(all_states_for_norm, vel_slice, vel_mean, vel_std)
    stats["groups"].append(
        {
            "name": "ee_velocity",
            "type": "gaussian",
            "indices": [vel_slice.start, vel_slice.stop],
            "mean": [float(x) for x in vel_mean],
            "std": [float(x) for x in vel_std],
        }
    )

    pos_scale, rot_scale = _parse_tracking_error_scales(cfg)
    track_clip = float(cfg.get("tracking_error_clip", 1.0))
    track_scales = np.array([pos_scale, pos_scale, pos_scale, rot_scale, rot_scale, rot_scale], dtype=float)
    _apply_fixed_scale(all_states_for_norm, track_slice, track_scales, clip_value=track_clip)
    stats["groups"].append(
        {
            "name": "tracking_error",
            "type": "fixed_scale_clip",
            "indices": [track_slice.start, track_slice.stop],
            "scales": [float(x) for x in track_scales],
            "clip": track_clip,
        }
    )

    wrench_clip = float(cfg.get("wrench_clip", 5.0))
    _apply_log1p(all_states_for_norm, wrench_slice)
    wrench_mean, wrench_std = _compute_gaussian_stats(all_states_for_norm, wrench_slice)
    _apply_gaussian(all_states_for_norm, wrench_slice, wrench_mean, wrench_std)
    _apply_clip(all_states_for_norm, wrench_slice, wrench_clip)
    stats["groups"].append(
        {
            "name": "external_wrench",
            "type": "log1p_zscore_clip",
            "indices": [wrench_slice.start, wrench_slice.stop],
            "mean": [float(x) for x in wrench_mean],
            "std": [float(x) for x in wrench_std],
            "clip": wrench_clip,
            "formula": "clip((sign(x)*log1p(abs(x)) - mean)/std, ±clip)",
        }
    )

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
      - position (x,y,z) min-max -> [-1, 1] using workspace_limits
      - orientation (6 dims) identity (rotation-matrix columns already in [-1, 1])
    Falls back to gaussian normalization if the action dimensionality is not compatible.
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

    stats = {"mode": "grouped", "action_dim": action_dim, "groups": []}

    pose_clip = float(cfg.get("pose_clip", 5.0))
    pos_mean, pos_std = _compute_gaussian_stats(all_actions_for_norm, pos_slice)
    _apply_gaussian(all_actions_for_norm, pos_slice, pos_mean, pos_std)
    _apply_clip(all_actions_for_norm, pos_slice, pose_clip)
    stats["groups"].append(
        {
            "name": "ee_position",
            "type": "zscore_clip",
            "indices": [pos_slice.start, pos_slice.stop],
            "mean": [float(x) for x in pos_mean],
            "std": [float(x) for x in pos_std],
            "clip": pose_clip,
        }
    )
    stats["groups"].append(
        {
            "name": "ee_orientation",
            "type": "identity",
            "indices": [ori_slice.start, ori_slice.stop],
        }
    )

    if action_dim > 9:
        rem_slice = slice(9, action_dim)
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


@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig):
    input_path = cfg.input_path
    output_path = cfg.output_path
    downsample = cfg.get("downsample", False)
    data_frequency = cfg.get("data_frequency", 50.0)
    target_downsampling_freq = cfg.get("target_downsampling_freq", 50.0)
    filter_torque = cfg.get("filter_torque", False)
    filter_position = cfg.get("filter_position", False)
    cutoff_freq_torque = cfg.get("cutoff_freq_torque", 10.0)
    cutoff_freq_position = cfg.get("cutoff_freq_position", 5.0)
    median_filter_torque = cfg.get("median_filter_torque", False)
    median_filter_position = cfg.get("median_filter_position", False)
    median_filter_kernel_size_torque = cfg.get("median_filter_kernel_size_torque", 3)
    median_filter_kernel_size_position = cfg.get("median_filter_kernel_size_position", 7)

    # rgb_obs_topics = list(cfg.cameras_topics)
    state_obs_topics = list(cfg.obs_topics)
    goal_topics = list(cfg.get("goal_topic", []))
    action_config = dict(cfg.action_config)
    action_topics = list(action_config.keys())

    assert len(state_obs_topics) > 0, "Require low-dim observation topics"
    # assert len(rgb_obs_topics) > 0, "Require visual observation topics"
    assert len(action_topics) > 0, "Require visual observation topics"
    assert target_downsampling_freq > 0, "Require positive target frequency"

    data_folder = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # initialize topics
    # all_topics = state_obs_topics + rgb_obs_topics + action_topics
    all_topics = state_obs_topics + action_topics + goal_topics

    state_topic_specs = {
        "/cartesian_impedance_controller/ee_velocity": {"keys": ["ee_velocity"], "dim": 6, "fallback": "data"},
        "/cartesian_impedance_controller/pose_command": {"keys": ["ee_pose_commanded"], "dim": 9, "fallback": None},
        "/cartesian_impedance_controller/tracking_error": {"keys": ["tracking_error"], "dim": 6, "fallback": "data"},
        "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame": {
            "keys": ["external_wrench"],
            "dim": 6,
            "fallback": None,
        },
    }

    goal_topic_specs = {
        "/goal": {"keys": ["goal"], "dim": 1, "fallback": None},
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

    def extract_ep_index(path):
        name = path.stem  # e.g., "ep_12"
        return int(name.split("_")[1])

    all_episodes = sorted(
        [f for f in data_folder.iterdir() if f.name.startswith("ep_") and f.name.endswith(".pkl")], key=extract_ep_index
    )

    trajectories = []
    all_states = []
    all_states_for_norm = []
    all_actions = []
    pbar = tqdm(all_episodes)

    state_topic_dims = None

    for episode_pkl in pbar:
        with open(episode_pkl, "rb") as f:
            traj_data = pickle.load(f)
        traj_data, avg_freq = sync_data_slowest(traj_data, all_topics)
        pbar.set_postfix({"avg_freq": f"{avg_freq:.1f} Hz"})

        print("\n")

        if median_filter_torque:
            # Apply median filter to torque sensor data
            traj_data["/franka_robot_state_broadcaster/external_joint_torques"] = medianfilter(
                np.array(traj_data["/franka_robot_state_broadcaster/external_joint_torques"]),
                kernel_size=median_filter_kernel_size_torque,
            ).tolist()

        if median_filter_position:
            # Apply median filter to position command data
            traj_data["/joint_impedance_dynamic_gain_controller/joint_impedance_command"] = medianfilter(
                np.array(traj_data["/joint_impedance_dynamic_gain_controller/joint_impedance_command"]),
                kernel_size=median_filter_kernel_size_position,
            ).tolist()

        # Apply low-pass filter to torque sensor data and position
        if filter_torque:
            lowpass_filter(
                traj_data,
                cutoff_freq_torque,
                data_frequency,
                "/franka_robot_state_broadcaster/external_joint_torques",
                key_options=("effort", "data"),
            )

        if filter_position:
            lowpass_filter(
                traj_data,
                cutoff_freq_position,
                data_frequency,
                "/joint_impedance_dynamic_gain_controller/joint_impedance_command",
                key_options=("position", "data"),
            )

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
        # num_steps = len(traj_data[action_topics[0]])
        # Get the length of all data lists used in generate_robobuf
        all_lengths = []
        all_lengths.append(len(traj_data[action_topics[0]]))  # Length of actions
        # all_lengths.append(len(state_arrays[0]))             # Length of states (assuming all state_arrays have the same length after stacking)
        # for topic in rgb_obs_topics:
        #     all_lengths.append(len(traj_data[topic]))        # Length of camera images
        # Use the minimum length to guarantee no index goes out of range
        num_steps = np.min(all_lengths)
        traj["num_steps"] = num_steps

        # traj['states'] = np.concatenate([np.array(traj_data[topic]) for topic in state_obs_topics], axis=-1)
        # Flatten each dict in state topics into numeric arrays
        ##########################
        # Flatten each dict in state topics into numeric arrays (fixed spec)
        state_arrays = []
        episode_dims = []
        for topic in state_obs_topics:
            if topic in state_topic_specs:
                spec = state_topic_specs[topic]
                topic_vectors = [
                    extract_fixed_vector(msg, spec["keys"], spec["dim"], spec["fallback"]) for msg in traj_data[topic]
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

        # Extract goals separately (not part of states)
        goals_arrays = []
        for topic in goal_topics:
            spec = goal_topic_specs.get(topic, {"keys": ["goal"], "dim": 1, "fallback": None})
            topic_vectors = [
                extract_fixed_vector(msg, spec["keys"], spec["dim"], spec["fallback"]) for msg in traj_data[topic]
            ]
            goals_arrays.append(np.stack(topic_vectors, axis=0))
        if goals_arrays:
            traj["goals"] = np.concatenate(goals_arrays, axis=-1)

        action_list = []
        # for topic in action_topics:
        #     actions = np.array(traj_data[topic])
        #     action_list.append(actions)

        # Flatten each action dict into numeric arrays (position + velocity + effort)
        action_list = []
        for topic in action_topics:
            topic_data = []
            for msg in traj_data[topic]:
                if isinstance(msg, dict):
                    parts = []
                    for key in ["ee_pose_commanded"]:
                        if key in msg and len(msg[key]) > 0:
                            parts.append(np.array(msg[key], dtype=float))
                    if parts:
                        vec = np.concatenate(parts)
                    else:
                        vec = np.zeros(1, dtype=float)
                else:
                    vec = np.array(msg, dtype=float).flatten()
                topic_data.append(vec)

            topic_array = np.stack(topic_data, axis=0)  # shape (num_steps, 7)
            action_list.append(topic_array)

        traj["actions"] = np.concatenate(action_list, axis=-1)

        all_states.append(traj["states"])
        all_states_for_norm.append(traj["states"])
        all_actions.append(traj["actions"])

        # Process and store images
        # for cam_ind, topic in enumerate(rgb_obs_topics):
        #     enc_images = traj_data[topic]
        #     processed_images = [process_image(img_enc) for img_enc in enc_images]
        #     traj[f'enc_cam_{cam_ind}'] = processed_images
        trajectories.append(traj)

    # normalize states and actions
    state_norm_stats = normalize_states_groupwise(all_states_for_norm, state_obs_topics, state_topic_dims, cfg)
    action_norm_stats = normalize_actions_groupwise(all_actions, cfg)
    norm_stats = dict(state=state_norm_stats, action=action_norm_stats)

    # dump data buffer
    buffer_name = "buf"
    buffer = generate_robobuf(trajectories)
    with open(output_dir / f"{buffer_name}.pkl", "wb") as f:
        pickle.dump(buffer.to_traj_list(), f)

    # dump rollout config
    obs_config = {
        "state_topics": state_obs_topics,
        "goal_topics": goal_topics,
        # 'camera_topics': rgb_obs_topics,
    }
    processing_config = {
        "median_filter_torque": median_filter_torque,
        "median_filter_position": median_filter_position,
        "median_filter_kernel_size_torque": median_filter_kernel_size_torque,
        "median_filter_kernel_size_position": median_filter_kernel_size_position,
        "filter_torque": filter_torque,
        "cutoff_freq_torque": cutoff_freq_torque,
        "filter_position": filter_position,
        "cutoff_freq_position": cutoff_freq_position,
        "downsample": downsample,
        "data_frequency": data_frequency,
        "target_downsampling_freq": target_downsampling_freq,
    }
    rollout_config = {
        "obs_config": obs_config,
        "action_config": action_config,
        "norm_stats": norm_stats,
        "processing_config": processing_config,
    }

    with open(output_dir / "rollout_config.yaml", "w") as f:
        yaml.dump(rollout_config, f, sort_keys=False)


if __name__ == "__main__":
    main()
