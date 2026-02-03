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
from omegaconf import DictConfig
from tqdm import tqdm
from utils_data_process import (
    downsample_data,
    gaussian_norm,
    generate_robobuf,
    lowpass_filter,
    medianfilter,
    sync_data_slowest,
)


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
    state_norm_stats = gaussian_norm(all_states_for_norm)
    action_norm_stats = gaussian_norm(all_actions)
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
