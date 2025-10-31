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

import yaml
import hydra
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig

from utils_data_process import sync_data_slowest, process_image, gaussian_norm, generate_robobuf

from scipy.signal import butter, filtfilt

@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig):

    input_path = cfg.input_path
    output_path = cfg.output_path
    data_frequency = cfg.get("data_frequency", 50.0)
    target_downsampling_freq = cfg.get("target_downsampling_freq", 50.0)
    filter_torque = cfg.get("filter_torque", False)
    filter_position = cfg.get("filter_position", False)
    cutoff_freq_torque = cfg.get("cutoff_freq_torque", None)
    cutoff_freq_position = cfg.get("cutoff_freq_position", None)

    rgb_obs_topics = list(cfg.cameras_topics)
    state_obs_topics = list(cfg.obs_topics)
    action_config = dict(cfg.action_config)
    action_topics = list(action_config.keys())

    assert len(state_obs_topics) > 0, "Require low-dim observation topics"
    assert len(rgb_obs_topics) > 0, "Require visual observation topics"
    assert len(action_topics) > 0, "Require visual observation topics"
    assert target_downsampling_freq > 0, "Require positive target frequency"
    
    
    data_folder = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # initialize topics
    all_topics = state_obs_topics + rgb_obs_topics + action_topics
    
    all_episodes = sorted([f for f in data_folder.iterdir() if f.name.startswith('ep_') and f.name.endswith('.pkl')])
    
    trajectories = []
    all_states = []
    all_actions = []
    pbar = tqdm(all_episodes)
    for episode_pkl in pbar:
        with open(episode_pkl, 'rb') as f:
            traj_data = pickle.load(f)
        traj_data, avg_freq = sync_data_slowest(traj_data, all_topics)
        pbar.set_postfix({'avg_freq': f'{avg_freq:.1f} Hz'})

        # ------------------------------
        # Apply low-pass filter to torque sensor data and position
        # ------------------------------
        if filter_torque:
            b, a = butter(N=2, Wn=cutoff_freq_torque / (data_frequency / 2), btype="low")

            # Extract numeric torque arrays
            raw_torques = []
            for msg in traj_data["/franka_robot_state_broadcaster/external_joint_torques"]:
                if isinstance(msg, dict):
                    if "effort" in msg:
                        raw_torques.append(np.array(msg["effort"], dtype=float))
                    elif "data" in msg:
                        raw_torques.append(np.array(msg["data"], dtype=float))
                    else:
                        # fallback if the dict doesn’t have those keys
                        raw_torques.append(np.zeros(7, dtype=float))
                else:
                    raw_torques.append(np.array(msg, dtype=float).flatten())

            torque_data = np.stack(raw_torques, axis=0)  # shape (num_steps, 7)
            # Apply low-pass filter along the time axis
            filtered_torques = filtfilt(b, a, torque_data, axis=0)
            traj_data["/franka_robot_state_broadcaster/external_joint_torques"] = filtered_torques.tolist()
            print(f"  🔉 Applied low-pass filter to external_joint_torques with cutoff {cutoff_freq_torque} Hz")


        if filter_position:
            b, a = butter(N=2, Wn=cutoff_freq_position / (data_frequency / 2), btype="low")

            raw_positions = [] 
            for msg in traj_data["/joint_impedance_command_controller/joint_trajectory"]:
                if isinstance(msg, dict):
                    if "position" in msg:
                        raw_positions.append(np.array(msg["position"], dtype=float))
                    elif "data" in msg:
                        raw_positions.append(np.array(msg["data"], dtype=float))
                    else:
                        # fallback if the dict doesn’t have those keys
                        raw_positions.append(np.zeros(7, dtype=float))
                else:
                    raw_positions.append(np.array(msg, dtype=float).flatten())
                
            position_data = np.stack(raw_positions, axis=0)  # shape (num_steps, 7)
            filtered_positions = filtfilt(b, a, position_data, axis=0)
            traj_data["/joint_impedance_command_controller/joint_trajectory"] = filtered_positions.tolist()
            print(f"  🔉 Applied low-pass filter to joint_trajectory positions with cutoff {cutoff_freq_position} Hz")


        # ------------------------------
        # 🕒 Downsample to target rate (25 Hz)
        # ------------------------------#
        if avg_freq > target_downsampling_freq - 0.1:
            step = int(np.floor(avg_freq / target_downsampling_freq))
            for key in traj_data.keys():
                if isinstance(traj_data[key], list):
                    traj_data[key] = traj_data[key][::step]
            avg_freq = avg_freq / step
            print(f"🔻 Downsampled from ~{avg_freq * step:.1f} Hz to ~{avg_freq:.1f} Hz (step={step})")
        else:
            print(f"Not downsampling, data frequency: {avg_freq:.1f} Hz, target frequency: {target_downsampling_freq:.1f} Hz")


        # ------------------------------
        traj = {}
        # num_steps = len(traj_data[action_topics[0]])
        # Get the length of all data lists used in generate_robobuf
        all_lengths = []
        all_lengths.append(len(traj_data[action_topics[0]])) # Length of actions
        # all_lengths.append(len(state_arrays[0]))             # Length of states (assuming all state_arrays have the same length after stacking)
        for topic in rgb_obs_topics:
            all_lengths.append(len(traj_data[topic]))        # Length of camera images
        # Use the minimum length to guarantee no index goes out of range
        num_steps = np.min(all_lengths)
        traj['num_steps'] = num_steps
        
        
        
        # traj['states'] = np.concatenate([np.array(traj_data[topic]) for topic in state_obs_topics], axis=-1)
        # Flatten each dict in state topics into numeric arrays
        ##########################
        # Flatten each dict in state topics into numeric arrays
        # Flatten each dict in state topics into numeric arrays (robust)
        state_arrays = []
        for topic in state_obs_topics:
            topic_data = []
            max_len = 0

            # First pass — convert each message to numeric vector
            for msg in traj_data[topic]:
                if isinstance(msg, dict):
                    parts = []
                    for key in ["effort"]:
                        if key in msg and len(msg[key]) > 0:
                            parts.append(np.array(msg[key], dtype=float))
                    if parts:
                        vec = np.concatenate(parts)
                    else:
                        vec = np.zeros(1, dtype=float)
                else:
                    vec = np.array(msg, dtype=float).flatten()
                topic_data.append(vec)
                max_len = max(max_len, len(vec))

            # Second pass — pad all to same length
            topic_padded = []
            for vec in topic_data:
                if len(vec) < max_len:
                    vec = np.pad(vec, (0, max_len - len(vec)))
                topic_padded.append(vec)

            topic_array = np.stack(topic_padded, axis=0)  # (num_steps, max_len)
            state_arrays.append(topic_array)

        # Concatenate all topics along feature dimension
        traj["states"] = np.concatenate(state_arrays, axis=-1)

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
                    for key in ["position"]:
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

        
        all_states.append(traj['states'])
        all_actions.append(traj["actions"])
        
        # Process and store images
        for cam_ind, topic in enumerate(rgb_obs_topics):
            enc_images = traj_data[topic]
            processed_images = [process_image(img_enc) for img_enc in enc_images]
            traj[f'enc_cam_{cam_ind}'] = processed_images
        trajectories.append(traj)
        
    # normalize states and actions
    state_norm_stats = gaussian_norm(all_states)
    action_norm_stats = gaussian_norm(all_actions)
    norm_stats = dict(state=state_norm_stats, action=action_norm_stats)
    
    # dump data buffer
    buffer_name = "buf"
    buffer = generate_robobuf(trajectories)
    with open(output_dir / f"{buffer_name}.pkl", "wb") as f:
        pickle.dump(buffer.to_traj_list(), f)
    
    # dump rollout config
    obs_config = {
        'state_topics': state_obs_topics,
        'camera_topics': rgb_obs_topics,
    }
    rollout_config = {
        'obs_config': obs_config,
        'action_config': action_config,
        'norm_stats': norm_stats
    }
    with open(output_dir / "rollout_config.yaml", "w") as f:
        yaml.dump(rollout_config, f, sort_keys=False)
        
if __name__ == "__main__":
    main()