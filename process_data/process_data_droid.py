# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li
#
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

from pathlib import Path

import cv2
import hydra
import numpy as np
import tensorflow_datasets as tfds
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from utils_data_process import gaussian_norm, generate_robobuf, process_decoded_image, process_image


def _normalize_topic_path(topic):
    topic = str(topic).strip()
    if topic.startswith("/"):
        topic = topic[1:]
    return topic


def _get_nested_value(data, topic):
    curr = data
    for key in _normalize_topic_path(topic).split("/"):
        if not key:
            continue
        if isinstance(curr, dict) and key in curr:
            curr = curr[key]
        else:
            return None
    return curr


def _as_flat_array(value):
    arr = np.asarray(value, dtype=float).reshape(-1)
    return arr


def _iter_episode_steps(episode_steps):
    if isinstance(episode_steps, dict):
        length = len(next(iter(episode_steps.values())))
        for idx in range(length):
            yield {k: v[idx] for k, v in episode_steps.items()}
        return
    for step in episode_steps:
        yield step


def _encode_camera_frame(frame):
    if isinstance(frame, np.ndarray):
        # Already decoded RGB image array.
        decoded = process_decoded_image(frame)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        _, compressed = cv2.imencode(".jpg", decoded, encode_param)
        return compressed
    return process_image(frame)


def _wrap_to_pi(angles):
    return np.arctan2(np.sin(angles), np.cos(angles))


def _rpy_to_rot6d(rpy):
    rpy = np.asarray(rpy, dtype=float).reshape(-1)
    if rpy.size != 3 or not np.all(np.isfinite(rpy)):
        rpy = np.zeros(3, dtype=float)
    rpy = _wrap_to_pi(rpy)
    roll, pitch, yaw = rpy
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # ZYX: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    return np.array([r00, r10, r20, r01, r11, r21], dtype=float)


def _maybe_convert_rpy(vec, enable):
    if not enable:
        return vec
    if vec.size != 6:
        return vec
    pos = vec[:3]
    rot6d = _rpy_to_rot6d(vec[3:6])
    return np.concatenate([pos, rot6d], axis=0)


def _set_decoder(decoders, topic_path, decoder):
    cursor = decoders
    parts = [p for p in _normalize_topic_path(topic_path).split("/") if p]
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = decoder


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
    if isinstance(split_cfg, DictConfig):
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


@hydra.main(version_base=None, config_path="cfg", config_name="droid")
def main(cfg: DictConfig):
    input_path = cfg.input_path
    output_path = cfg.output_path
    downsample = bool(cfg.get("downsample", False))
    data_frequency = float(cfg.get("data_frequency", 50.0))
    target_downsampling_freq = float(cfg.get("target_downsampling_freq", 50.0))

    obs_topics = list(cfg.get("obs_topics", []))
    cameras_topics = list(cfg.get("cameras_topics") or [])
    action_config = dict(cfg.get("action_config", {}))
    action_topics = list(action_config.keys())
    rpy_topics = set(cfg.get("rpy_topics") or [])
    action_pose_mode = str(cfg.get("action_pose_mode", "absolute"))
    action_pose_topic = cfg.get("action_pose_topic", "action_dict/cartesian_position")
    split_cfg = _parse_split_cfg(cfg)

    if not obs_topics:
        raise ValueError("Require obs_topics for DROID processing.")
    if not action_topics:
        raise ValueError("Require action_config for DROID processing.")
    if target_downsampling_freq <= 0:
        raise ValueError("Require positive target_downsampling_freq.")

    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    step = 1
    if downsample:
        step = max(1, int(round(data_frequency / target_downsampling_freq)))
        print(f"Downsampling enabled: step={step}, {data_frequency:.1f}Hz -> {data_frequency / step:.1f}Hz")

    decoders = {}
    for cam_topic in cameras_topics:
        _set_decoder(decoders, f"steps/{cam_topic}", tfds.decode.SkipDecoding())

    builder = tfds.builder_from_directory(input_path)
    dataset = builder.as_dataset(split="train", decoders=decoders)

    trajectories = []
    all_states = []
    all_actions = []

    processed_episode_indices = []
    for episode_idx, episode in enumerate(tqdm(tfds.as_numpy(dataset), desc="Processing DROID episodes")):
        processed_episode_indices.append(int(episode_idx))
        episode_steps = episode["steps"]
        states = []
        actions = []
        cameras = {idx: [] for idx in range(len(cameras_topics))}
        prev_action_pose = None

        for step_data in _iter_episode_steps(episode_steps):
            obs_parts = []
            for topic in obs_topics:
                value = _get_nested_value(step_data, topic)
                if value is None:
                    raise KeyError(f"Missing obs topic {topic}")
                obs_vec = _as_flat_array(value)
                obs_parts.append(_maybe_convert_rpy(obs_vec, topic in rpy_topics))
            states.append(np.concatenate(obs_parts, axis=0))

            action_parts = []
            for topic in action_topics:
                value = _get_nested_value(step_data, topic)
                if value is None:
                    raise KeyError(f"Missing action topic {topic}")
                act_vec = _as_flat_array(value)
                if action_pose_mode == "relative" and topic == action_pose_topic:
                    if prev_action_pose is None:
                        delta_pos = np.zeros_like(act_vec[:3])
                    else:
                        delta_pos = act_vec[:3] - prev_action_pose[:3]
                    prev_action_pose = act_vec.copy()
                    if act_vec.size >= 6:
                        act_vec = np.concatenate([delta_pos, act_vec[3:]], axis=0)
                    else:
                        act_vec = np.concatenate([delta_pos], axis=0)
                action_parts.append(_maybe_convert_rpy(act_vec, topic in rpy_topics))
            actions.append(np.concatenate(action_parts, axis=0))

            for cam_idx, cam_topic in enumerate(cameras_topics):
                frame = _get_nested_value(step_data, cam_topic)
                if frame is None:
                    raise KeyError(f"Missing camera topic {cam_topic}")
                cameras[cam_idx].append(_encode_camera_frame(frame))

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)

        if downsample and step > 1:
            states = states[::step]
            actions = actions[::step]
            for cam_idx in cameras:
                cameras[cam_idx] = cameras[cam_idx][::step]

        expected_action_dims = [int(action_config.get(t, -1)) for t in action_topics]
        expected_total_dim = sum([d for d in expected_action_dims if d > 0])
        if expected_total_dim > 0 and expected_total_dim != actions.shape[1]:
            raise ValueError(
                f"Action dim mismatch. expected_total={expected_total_dim} actual_total={actions.shape[1]} topics={action_topics}"
            )

        num_steps = min([states.shape[0], actions.shape[0]] + [len(v) for v in cameras.values()])
        traj = {
            "states": states[:num_steps],
            "actions": actions[:num_steps],
            "num_steps": num_steps,
        }
        for cam_idx, cam_frames in cameras.items():
            traj[f"enc_cam_{cam_idx}"] = cam_frames[:num_steps]

        trajectories.append(traj)
        all_states.append(traj["states"])
        all_actions.append(traj["actions"])

    state_norm_stats = gaussian_norm(all_states) if all_states else {"mean": [], "std": []}
    action_norm_stats = gaussian_norm(all_actions) if all_actions else {"mean": [], "std": []}
    norm_stats = {"state": state_norm_stats, "action": action_norm_stats}

    if split_cfg["enabled"]:
        train_indices, test_indices = _split_episode_indices(
            num_episodes=len(trajectories),
            train_ratio=split_cfg["train_ratio"],
            seed=split_cfg["seed"],
        )
        train_trajectories = [trajectories[idx] for idx in train_indices]
        test_trajectories = [trajectories[idx] for idx in test_indices]

        train_buffer = generate_robobuf(train_trajectories)
        test_buffer = generate_robobuf(test_trajectories)

        train_file = output_dir / f"{split_cfg['train_buffer_name']}.pkl"
        test_file = output_dir / f"{split_cfg['test_buffer_name']}.pkl"
        with open(train_file, "wb") as f:
            import pickle

            pickle.dump(train_buffer.to_traj_list(), f)
        with open(test_file, "wb") as f:
            import pickle

            pickle.dump(test_buffer.to_traj_list(), f)

        split_info = {
            "enabled": True,
            "seed": split_cfg["seed"],
            "train_ratio": split_cfg["train_ratio"],
            "num_episodes_total": len(trajectories),
            "num_episodes_train": len(train_trajectories),
            "num_episodes_test": len(test_trajectories),
            "train_buffer": train_file.name,
            "test_buffer": test_file.name,
            "train_indices": train_indices,
            "test_indices": test_indices,
        }
    else:
        buffer = generate_robobuf(trajectories)
        buffer_name = "buf"
        with open(output_dir / f"{buffer_name}.pkl", "wb") as f:
            import pickle

            pickle.dump(buffer.to_traj_list(), f)
        split_info = {
            "enabled": False,
            "num_episodes_total": len(trajectories),
            "all_buffer": f"{buffer_name}.pkl",
            "episode_indices": processed_episode_indices,
        }

    obs_config = {
        "state_topics": obs_topics,
        "camera_topics": cameras_topics,
    }
    processing_config = {
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
