# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle as pkl
import random
import shutil

import cv2  # ★★★ 必須: OpenCVをインポート ★★★
import numpy as np
import torch
import tqdm
from robobuf import ReplayBuffer as RB
from torch.utils.data import Dataset, IterableDataset

from factr.arrangement import arrangement_id_to_one_hot
from factr.utils import (
    apply_grouped_transform,
    canonical_action_mode,
    load_action_pose_mode_from_buffer_path,
    load_norm_stats_from_buffer_path,
    load_processing_config_from_buffer_path,
    state_stats_without_tracking_error,
)


# helper functions
def _img_to_tensor(x):
    # リストなら配列に
    if isinstance(x, list):
        x = np.array(x)

    # 既に Tensor なら処理して返す
    if isinstance(x, torch.Tensor):
        return x.permute((0, 3, 1, 2)).float() / 255.0

    # NumPy配列であることを保証
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # ★★★ 修正: PyTorchに渡す前に、NumPy側で float32 にして正規化も済ませる ★★★
    # これにより "dtypeが推論できない" 系のエラーを全て回避します
    x = x.astype(np.float32) / 255.0

    # メモリ配置を整える (エラー回避のおまじない)
    x = np.ascontiguousarray(x)

    # Tensor化して軸を入れ替える
    return torch.from_numpy(x).permute((0, 3, 1, 2))


def _to_tensor(x):
    return torch.from_numpy(x).float()


# cache loading from the buffer list to half memory overhead
buf_cache = dict()


def _cached_load(path):
    global buf_cache

    if path in buf_cache:
        return buf_cache[path]

    with open(path, "rb") as f:
        buf = RB.load_traj_list(pkl.load(f))
    buf_cache[path] = buf
    return buf


# ★★★★★ ここが修正の核心！ ★★★★★
def _get_imgs(t, cam_idx, past_frames):
    imgs = []
    # 現在のステップ t から past_frames 分だけ過去に遡る
    curr_t = t

    for _ in range(past_frames + 1):
        img_data = curr_t.obs.image(cam_idx)

        # 1. バイト列 (JPEG等) ならデコード
        if isinstance(img_data, bytes):
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                raise ValueError("Failed to decode image from bytes")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        # 2. 既に NumPy配列 ならそのまま
        elif isinstance(img_data, np.ndarray):
            img = img_data

        else:
            raise TypeError(f"Unexpected image type: {type(img_data)}")

        # リストに追加 (1, H, W, C)
        imgs.append(img[None])

        if curr_t.prev is not None:
            curr_t = curr_t.prev

    # 時間方向に結合: (T, H, W, C)
    # ★★★ 修正: 配列を1つだけ返す (カンマ , arr を消す) ★★★
    return np.concatenate(imgs, axis=0)


BUF_SHUFFLE_RNG = 3904767649


class ReplayBuffer(Dataset):
    def __init__(self, buffer_path, transform=None, n_train_demos=200, mode="train", ac_chunk=1):
        assert mode in ("train", "test"), "Mode must be train/test"
        self.buffer_path = buffer_path
        buffer_data = self._load_buffer(buffer_path)
        assert len(buffer_data) >= n_train_demos, "Not enough demos!"

        # shuffle the list with the fixed seed
        rng = random.Random(BUF_SHUFFLE_RNG)
        rng.shuffle(buffer_data)

        # split data according to mode
        buffer_data = buffer_data[:n_train_demos] if mode == "train" else buffer_data[n_train_demos:]

        self.transform = transform
        self.s_a_mask = []
        for traj in tqdm.tqdm(buffer_data):
            imgs, obs, acs = traj["images"], traj["observations"], traj["actions"]
            assert len(obs) == len(acs) and len(acs) == len(imgs), "All time dimensions must match!"

            # pad camera dimension if needed
            if len(imgs.shape) == 4:
                imgs = imgs[:, None]

            for t in range(len(imgs) - ac_chunk):
                i_t = {f"cam{c}": imgs[t, c] for c in range(imgs.shape[1])}
                loss_mask = np.ones((ac_chunk,), dtype=np.float32)
                o_t, a_t = obs[t], acs[t : t + ac_chunk]
                self.s_a_mask.append(((i_t, o_t), a_t, loss_mask))

    def _load_buffer(self, buffer_path):
        print("loading", buffer_path)
        with open(buffer_path, "rb") as f:
            buffer_data = pkl.load(f)
        return buffer_data

    def __len__(self):
        return len(self.s_a_mask)

    def __getitem__(self, idx):
        (i_t, o_t), a_t, loss_mask = self.s_a_mask[idx]

        i_t = {k: _img_to_tensor(v) for k, v in i_t.items()}
        if self.transform is not None:
            i_t = {k: self.transform(v) for k, v in i_t.items()}

        o_t, a_t = _to_tensor(o_t), _to_tensor(a_t)
        loss_mask = _to_tensor(loss_mask)[:, None].repeat((1, a_t.shape[-1]))
        assert loss_mask.shape[0] == a_t.shape[0], "a_t and mask shape must be ac_chunk!"
        return (i_t, o_t), a_t, loss_mask


class IterableWrapper(IterableDataset):
    def __init__(self, wrapped_dataset, max_count=float("inf")):
        self.wrapped = wrapped_dataset
        self.ctr, self.max_count = 0, max_count

    def __iter__(self):
        self.ctr = 0
        return self

    def __next__(self):
        if self.ctr > self.max_count:
            raise StopIteration

        self.ctr += 1
        idx = int(np.random.choice(len(self.wrapped)))
        return self.wrapped[idx]


class RobobufReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_path,
        transform=None,
        n_test_ratio=0.05,
        mode="train",
        ac_chunk=1,
        cam_indexes=[0],
        past_frames=0,
        ac_dim=7,
        shuffle=True,
    ):
        assert mode in ("train", "test"), "Mode must be train/test"
        self.buffer_path = buffer_path
        buf = _cached_load(buffer_path)

        n_test_trans = int(len(buf) * n_test_ratio)

        norm_file = os.path.join(os.path.dirname(buffer_path), "ac_norm.json")
        if os.path.exists(norm_file):
            shutil.copyfile(norm_file, "./ac_norm.json")

        # shuffle the list with the fixed seed
        rng = random.Random(BUF_SHUFFLE_RNG)

        index_list = list(range(len(buf)))
        # split data according to mode
        index_list = index_list[:-n_test_trans] if mode == "train" else index_list[-n_test_trans:]

        # get and shuffle list of buf indices
        if shuffle:
            rng.shuffle(index_list)

        self.transform = transform
        self.s_a_mask = []

        self.cam_indexes = cam_indexes = list(cam_indexes)
        self.past_frames = past_frames
        self.ac_dim = ac_dim  # 保存

        print(f"Building {mode} buffer with cam_indexes={cam_indexes}")

        for idx in tqdm.tqdm(index_list):
            t = buf[idx]

            loop_t, chunked_actions, loss_mask = t, [], []
            for _ in range(ac_chunk):
                chunked_actions.append(loop_t.action[None])
                loss_mask.append(1.0)

                if loop_t.next is None:
                    break
                loop_t = loop_t.next

            if len(chunked_actions) < ac_chunk:
                for _ in range(ac_chunk - len(chunked_actions)):
                    chunked_actions.append(chunked_actions[-1])
                    loss_mask.append(0.0)

            a_t = np.concatenate(chunked_actions, 0).astype(np.float32)

            if a_t.shape[-1] != ac_dim and a_t.shape[-1] == 7:
                use_indices = [0, 1, 2, 3, 4, 5, 6]  # ← 使用したい3次元のインデックス (0始まり)
                a_t = a_t[..., use_indices]

            assert ac_dim == a_t.shape[-1]

            loss_mask = np.array(loss_mask, dtype=np.float32)
            self.s_a_mask.append((t, a_t, loss_mask, loop_t))

    def __getitem__(self, idx):
        step, a_t, loss_mask, goal = self.s_a_mask[idx]

        i_t, o_t = dict(), step.obs.state
        for idx, cam_idx in enumerate(self.cam_indexes):
            # ★ 修正された _get_imgs を使う (これでデコードされる)
            i_c = _get_imgs(step, cam_idx, self.past_frames)

            i_c = _img_to_tensor(i_c)
            if self.transform is not None:
                i_c = self.transform(i_c)

            i_t[f"cam{idx}"] = i_c

        label_idx = 0
        if o_t.shape[-1] == 11:
            raw_label = o_t[7:]  # 後ろ4つ
            o_t = o_t[:7]  # 前7つ (トルク)
            label_idx = np.argmax(raw_label)

        o_t, a_t = _to_tensor(o_t), _to_tensor(a_t)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        loss_mask = _to_tensor(loss_mask)[:, None].repeat((1, a_t.shape[-1]))
        assert loss_mask.shape[0] == a_t.shape[0], "a_t and mask shape must be ac_chunk!"

        # 7次元を使うならスライス不要 (ac_dim=7)
        if self.ac_dim < a_t.shape[-1]:
            a_t = a_t[:, : self.ac_dim]

        return (i_t, o_t), a_t, loss_mask, label_tensor


def _obs_to_dict(obs):
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "to_dict"):
        out = obs.to_dict()
        if isinstance(out, dict):
            return out
    if hasattr(obs, "obs") and isinstance(obs.obs, dict):
        return obs.obs
    raise TypeError(f"Unsupported obs type: {type(obs)}")


def _to_label_index(raw_label, num_classes):
    value = np.asarray(raw_label).reshape(-1)[0]
    value = int(value)
    if value >= 1:
        value -= 1
    value = max(0, min(value, num_classes - 1))
    return value


class RobobufReplayBufferLowdim(ReplayBuffer):
    def __init__(
        self,
        buffer_path,
        n_test_ratio=0.0,
        mode="train",
        use_internal_split=False,
        ac_chunk=30,
        obs_window=8,
        obs_dim=27,
        pose_action_dim=9,
        action_index_offset=0,
        include_tracking_error=True,
        include_goals=False,
        action_chunk_mode="absolute",
        stiffness_classes=3,
        override_stiffness_with_mode=False,
        use_arrangement_conditioning=False,
        shuffle=True,
    ):
        assert mode in ("train", "test"), "Mode must be train/test"
        assert obs_window >= 1, "obs_window must be >= 1"
        assert ac_chunk >= 1, "ac_chunk must be >= 1"

        self.buffer_path = buffer_path

        self.obs_window = int(obs_window)
        self.include_tracking_error = bool(include_tracking_error)
        self.obs_dim = int(obs_dim)
        self.pose_action_dim = int(pose_action_dim)
        self.action_index_offset = int(action_index_offset)
        self.include_goals = bool(include_goals)
        self.action_chunk_mode = canonical_action_mode(action_chunk_mode)
        self.stiffness_classes = int(stiffness_classes)
        self.override_stiffness_with_mode = bool(override_stiffness_with_mode)
        self.use_arrangement_conditioning = bool(use_arrangement_conditioning)
        self._tracking_slice = slice(21, 27)
        if not self.include_tracking_error and self.obs_dim >= 36:
            self.obs_dim -= 6
        self._state_norm_stats, self._action_norm_stats = load_norm_stats_from_buffer_path(buffer_path)
        if not self.include_tracking_error:
            self._state_norm_stats = state_stats_without_tracking_error(self._state_norm_stats)
        if self.action_chunk_mode == "relative" and not self._action_norm_stats:
            raise ValueError(
                "relative mode requires dedicated action normalization metadata in rollout_config.yaml."
            )
        source_action_mode = load_action_pose_mode_from_buffer_path(buffer_path)
        self._processing_config = load_processing_config_from_buffer_path(buffer_path)
        if self.action_chunk_mode == "relative" and source_action_mode != "relative":
            raise ValueError(
                "Training with relative mode requires a precomputed relative dataset; "
                f"dataset reports {source_action_mode!r}."
            )
        if self.action_chunk_mode == "relative":
            stored_chunk = int(self._processing_config.get("ac_chunk", -1))
            stored_offset = int(self._processing_config.get("action_index_offset", -1))
            if stored_chunk != int(ac_chunk) or stored_offset != self.action_index_offset:
                raise ValueError(
                    "Relative-chunk dataset/training mismatch: "
                    f"stored ac_chunk={stored_chunk}, offset={stored_offset}; "
                    f"requested ac_chunk={ac_chunk}, offset={self.action_index_offset}."
                )
            if not bool(self._processing_config.get("relative_chunk_normalized", False)):
                raise ValueError("Expected a dataset with relative_chunk_normalized=true.")
            if self._processing_config.get("relative_chunk_anchor") != "current_command":
                raise ValueError("Expected relative_chunk_anchor=current_command.")
        self.use_internal_split = bool(use_internal_split)
        self.transform = None
        self.s_a_mask = []
        self.sample_metadata = []

        if self.action_index_offset < 0:
            raise ValueError(f"action_index_offset must be >= 0, got {self.action_index_offset}.")

        buf = _cached_load(buffer_path)
        episodes = self._build_episodes(buf)
        if len(episodes) == 0:
            raise ValueError("No episodes found in buffer.")

        episode_labels = self._infer_episode_stiffness_labels(episodes)

        rng = random.Random(BUF_SHUFFLE_RNG)
        episode_indices = list(range(len(episodes)))
        if shuffle:
            rng.shuffle(episode_indices)

        if self.use_internal_split and len(episodes) > 1 and n_test_ratio > 0:
            n_test_eps = max(1, int(len(episodes) * n_test_ratio))
        else:
            n_test_eps = 0

        if mode == "train":
            use_episode_indices = episode_indices[:-n_test_eps] if n_test_eps > 0 else episode_indices
        else:
            use_episode_indices = episode_indices[-n_test_eps:] if n_test_eps > 0 else episode_indices

        print(
            f"Building {mode} lowdim buffer with episodes={len(use_episode_indices)}, "
            f"obs_window={self.obs_window}, ac_chunk={ac_chunk}, pose_action_dim={self.pose_action_dim}"
        )
        print(f"Loaded from file: {buffer_path}")

        for ep_idx in tqdm.tqdm(use_episode_indices):
            episode = episodes[ep_idx]
            label = episode_labels[ep_idx]
            self._append_episode_samples(
                episode,
                label,
                ac_chunk=ac_chunk,
                episode_id=ep_idx,
            )

    @staticmethod
    def _build_episodes(buf):
        episodes = []
        current = []
        for i in range(len(buf)):
            step = buf[i]
            is_first = bool(getattr(step, "first", False) or getattr(step, "is_first", False) or step.prev is None)
            if is_first and len(current) > 0:
                episodes.append(current)
                current = []
            current.append(step)
        if len(current) > 0:
            episodes.append(current)
        return episodes

    def _extract_obs_vector(self, step):
        obs_dict = _obs_to_dict(step.obs)
        state = np.asarray(obs_dict["state"], dtype=np.float32).reshape(-1)
        if not self.include_tracking_error:
            if state.shape[0] == self.obs_dim + 6:
                state = np.concatenate(
                    [state[: self._tracking_slice.start], state[self._tracking_slice.stop :]], axis=0
                )
            elif state.shape[0] != self.obs_dim:
                raise ValueError(f"Expected state dim {self.obs_dim} without tracking error, got {state.shape[0]}.")
        if self.include_goals and "goals" in obs_dict:
            goals = np.asarray(obs_dict["goals"], dtype=np.float32).reshape(-1)
            state = np.concatenate([state, goals], axis=0)
        return state

    def _extract_pose_action(self, step):
        action = np.asarray(step.action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.pose_action_dim:
            raise ValueError(f"Action dim {action.shape[0]} smaller than pose_action_dim={self.pose_action_dim}.")
        return action[: self.pose_action_dim]

    def _extract_precomputed_chunk(self, step, ac_chunk):
        action = np.asarray(step.action, dtype=np.float32)
        expected_shape = (int(ac_chunk), self.pose_action_dim)
        if action.shape != expected_shape:
            raise ValueError(f"Expected stored relative chunk shape {expected_shape}, got {action.shape}.")
        obs_dict = _obs_to_dict(step.obs)
        if "action_mask" not in obs_dict:
            raise ValueError("Precomputed relative chunk is missing obs['action_mask'].")
        mask = np.asarray(obs_dict["action_mask"], dtype=np.float32).reshape(-1)
        if mask.shape != (int(ac_chunk),):
            raise ValueError(f"Expected action mask shape ({ac_chunk},), got {mask.shape}.")
        return action, mask

    def _extract_arrangement_vector(self, step, episode_id, episode_step):
        obs_dict = _obs_to_dict(step.obs)
        if "arrangement" not in obs_dict:
            raise ValueError(
                "use_arrangement_conditioning=True requires obs['arrangement'] in every step; "
                f"missing from episode {episode_id}, step {episode_step}."
            )
        try:
            return arrangement_id_to_one_hot(obs_dict["arrangement"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid arrangement in episode {episode_id}, step {episode_step}: {exc}") from exc

    def _infer_episode_stiffness_labels(self, episodes):
        labels = []
        for ep_idx, episode in enumerate(episodes):
            first_step = episode[0]
            obs_dict = _obs_to_dict(first_step.obs)

            if self.override_stiffness_with_mode:
                if "mode" not in obs_dict:
                    raise ValueError(
                        "override_stiffness_with_mode=True requires obs['mode'] in every episode; "
                        f"missing from episode {ep_idx}."
                    )
                # Dataset mode is binary 0/1; the model-facing stiffness labels stay 1-based.
                mode_value = int(np.asarray(obs_dict["mode"]).reshape(-1)[0])
                if mode_value not in (0, 1):
                    raise ValueError(
                        "override_stiffness_with_mode=True expects binary obs['mode'] values 0 or 1; "
                        f"got {mode_value} in episode {ep_idx}."
                    )
                labels.append(mode_value + 1)
                continue

            raw_label = None
            for key in ("stiffness_label", "stiffness_class", "stiffness"):
                if key in obs_dict:
                    raw_label = obs_dict[key]
                    break

            if raw_label is None:
                labels.append(1)
            else:
                labels.append(_to_label_index(raw_label, self.stiffness_classes) + 1)
        return labels

    def _append_episode_samples(self, episode, label, ac_chunk, episode_id):
        episode_states = [self._extract_obs_vector(step) for step in episode]
        episode_arrangements = None
        if self.use_arrangement_conditioning:
            episode_arrangements = [
                self._extract_arrangement_vector(step, episode_id=episode_id, episode_step=step_idx)
                for step_idx, step in enumerate(episode)
            ]
        state_dim = episode_states[0].shape[0]
        if state_dim != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {state_dim}.")

        max_t = len(episode_states) - self.action_index_offset
        if max_t <= 0:
            return

        for t_idx in range(max_t):
            start = max(0, t_idx - self.obs_window + 1)
            window_states = episode_states[start : t_idx + 1]
            while len(window_states) < self.obs_window:
                window_states.insert(0, window_states[0])
            obs_window = np.stack(window_states, axis=0).astype(np.float32)

            if self.action_chunk_mode == "relative":
                pose_chunk, loss_mask = self._extract_precomputed_chunk(episode[t_idx], ac_chunk)
            else:
                chunk_actions = []
                loss_mask = []
                for k in range(ac_chunk):
                    idx = t_idx + self.action_index_offset + k
                    if idx < len(episode):
                        chunk_actions.append(self._extract_pose_action(episode[idx]))
                        loss_mask.append(1.0)
                    else:
                        chunk_actions.append(
                            self._extract_pose_action(episode[-1]) if not chunk_actions else chunk_actions[-1]
                        )
                        loss_mask.append(0.0)
                pose_chunk = np.stack(chunk_actions, axis=0).astype(np.float32)
                loss_mask = np.asarray(loss_mask, dtype=np.float32)
            arrangement_vector = episode_arrangements[t_idx] if episode_arrangements is not None else None
            self.s_a_mask.append((obs_window, pose_chunk, loss_mask, int(label), arrangement_vector))
            self.sample_metadata.append(
                {
                    "episode_id": int(episode_id),
                    "episode_step": int(t_idx),
                    "episode_length": int(len(episode)),
                    "stiffness_label": int(label),
                }
            )

    def __getitem__(self, idx):
        obs_window, pose_chunk, loss_mask, label, arrangement_vector = self.s_a_mask[idx]

        obs_tensor = _to_tensor(obs_window)
        action_tensor = _to_tensor(pose_chunk)
        mask_tensor = _to_tensor(loss_mask)[:, None].repeat((1, action_tensor.shape[-1]))
        label_tensor = torch.tensor(label, dtype=torch.long)

        sample = (({}, obs_tensor), action_tensor, mask_tensor, label_tensor)
        if not self.use_arrangement_conditioning:
            return sample
        return (*sample, _to_tensor(arrangement_vector))

    def get_sample_metadata(self, idx):
        if idx < 0 or idx >= len(self.sample_metadata):
            raise IndexError(f"Sample index out of range: {idx}")
        return self.sample_metadata[idx]


class RobobufReplayBufferObsPredLowdim(ReplayBuffer):
    def __init__(
        self,
        buffer_path,
        n_test_ratio=0.0,
        mode="train",
        use_internal_split=False,
        obs_window=8,
        obs_dim=27,
        input_obs_dim=21,
        predict_obs_dim=21,
        pred_horizon=30,
        pose_action_dim=9,
        action_index_offset=0,
        target_index_offset=1,
        stiffness_classes=3,
        goal_classes=4,
        shuffle=True,
    ):
        assert mode in ("train", "test"), "Mode must be train/test"
        assert obs_window >= 1, "obs_window must be >= 1"

        self.buffer_path = buffer_path

        self.obs_window = int(obs_window)
        self.obs_dim = int(obs_dim)
        self.input_obs_dim = int(input_obs_dim)
        self.predict_obs_dim = int(predict_obs_dim)
        self.pred_horizon = int(pred_horizon)
        self.pose_action_dim = int(pose_action_dim)
        self.action_index_offset = int(action_index_offset)
        self.target_index_offset = int(target_index_offset)
        self.stiffness_classes = int(stiffness_classes)
        self.goal_classes = int(goal_classes)
        self.use_internal_split = bool(use_internal_split)
        self.transform = None
        self.s_a_mask = []
        self.sample_metadata = []

        if self.action_index_offset < 0:
            raise ValueError(f"action_index_offset must be >= 0, got {self.action_index_offset}.")
        if self.target_index_offset < 1:
            raise ValueError(f"target_index_offset must be >= 1, got {self.target_index_offset}.")
        if self.input_obs_dim > self.obs_dim:
            raise ValueError(f"input_obs_dim={self.input_obs_dim} cannot exceed obs_dim={self.obs_dim}.")
        if self.predict_obs_dim > self.obs_dim:
            raise ValueError(f"predict_obs_dim={self.predict_obs_dim} cannot exceed obs_dim={self.obs_dim}.")
        if self.pred_horizon < 1:
            raise ValueError(f"pred_horizon must be >= 1, got {self.pred_horizon}.")

        buf = _cached_load(buffer_path)
        episodes = RobobufReplayBufferLowdim._build_episodes(buf)
        if len(episodes) == 0:
            raise ValueError("No episodes found in buffer.")

        episode_labels = self._infer_episode_stiffness_labels(episodes)

        rng = random.Random(BUF_SHUFFLE_RNG)
        episode_indices = list(range(len(episodes)))
        if shuffle:
            rng.shuffle(episode_indices)

        if self.use_internal_split and len(episodes) > 1 and n_test_ratio > 0:
            n_test_eps = max(1, int(len(episodes) * n_test_ratio))
        else:
            n_test_eps = 0

        if mode == "train":
            use_episode_indices = episode_indices[:-n_test_eps] if n_test_eps > 0 else episode_indices
        else:
            use_episode_indices = episode_indices[-n_test_eps:] if n_test_eps > 0 else episode_indices

        print(
            f"Building {mode} obs-pred buffer with episodes={len(use_episode_indices)}, "
            f"obs_window={self.obs_window}, input_obs_dim={self.input_obs_dim}, "
            f"target_obs_dim={self.predict_obs_dim}, pred_horizon={self.pred_horizon}"
        )
        print(f"Loaded from file: {buffer_path}")

        for ep_idx in tqdm.tqdm(use_episode_indices):
            episode = episodes[ep_idx]
            stiffness_label = episode_labels[ep_idx]
            self._append_episode_samples(episode=episode, stiffness_label=stiffness_label, episode_id=ep_idx)

    def _extract_state_vector(self, step):
        obs_dict = _obs_to_dict(step.obs)
        state = np.asarray(obs_dict["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {state.shape[0]}.")
        return state

    def _extract_pose_action(self, step):
        action = np.asarray(step.action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.pose_action_dim:
            raise ValueError(f"Action dim {action.shape[0]} smaller than pose_action_dim={self.pose_action_dim}.")
        return action[: self.pose_action_dim]

    def _extract_goal_label(self, step):
        obs_dict = _obs_to_dict(step.obs)
        raw_goal = obs_dict.get("goals", obs_dict.get("goal", None))
        if raw_goal is None:
            return 1

        goal_array = np.asarray(raw_goal).reshape(-1)
        if goal_array.size == 0:
            return 1

        # Some buffers may store goals as one-hot vectors instead of scalar class ids.
        if goal_array.size > 1:
            if np.all(np.logical_or(np.isclose(goal_array, 0.0), np.isclose(goal_array, 1.0))):
                raw_goal_value = int(np.argmax(goal_array)) + 1
            else:
                raw_goal_value = int(goal_array[0])
        else:
            raw_goal_value = int(goal_array[0])

        return _to_label_index(raw_goal_value, self.goal_classes) + 1

    def _infer_episode_stiffness_labels(self, episodes):
        labels = []
        for episode in episodes:
            first_step = episode[0]
            obs_dict = _obs_to_dict(first_step.obs)
            raw_label = None
            for key in ("stiffness_label", "stiffness_class", "stiffness"):
                if key in obs_dict:
                    raw_label = obs_dict[key]
                    break

            if raw_label is None:
                labels.append(1)
            else:
                labels.append(_to_label_index(raw_label, self.stiffness_classes) + 1)
        return labels

    def _append_episode_samples(self, episode, stiffness_label, episode_id):
        episode_states = [self._extract_state_vector(step) for step in episode]
        episode_goals = [self._extract_goal_label(step) for step in episode]
        max_t = len(episode) - max(self.action_index_offset, self.target_index_offset)
        if max_t <= 0:
            return

        # We keep windows inside episode boundaries and pad with the first valid step.
        for t_idx in range(max_t):
            start = max(0, t_idx - self.obs_window + 1)
            window_states = [state[: self.input_obs_dim] for state in episode_states[start : t_idx + 1]]
            while len(window_states) < self.obs_window:
                window_states.insert(0, window_states[0])
            obs_window = np.stack(window_states, axis=0).astype(np.float32)

            action_idx = t_idx + self.action_index_offset
            target_idx = t_idx + self.target_index_offset

            action_chunk = []
            target_chunk = []
            target_mask = []
            for horizon_idx in range(self.pred_horizon):
                a_idx = action_idx + horizon_idx
                y_idx = target_idx + horizon_idx

                if a_idx < len(episode):
                    action_chunk.append(self._extract_pose_action(episode[a_idx]))
                else:
                    fallback_a_idx = min(len(episode) - 1, action_idx)
                    action_chunk.append(self._extract_pose_action(episode[fallback_a_idx]))

                if y_idx < len(episode_states):
                    target_chunk.append(episode_states[y_idx][: self.predict_obs_dim])
                    target_mask.append(1.0)
                else:
                    fallback_y_idx = min(len(episode_states) - 1, target_idx)
                    target_chunk.append(episode_states[fallback_y_idx][: self.predict_obs_dim])
                    target_mask.append(0.0)

            action_chunk = np.stack(action_chunk, axis=0).astype(np.float32)
            target_chunk = np.stack(target_chunk, axis=0).astype(np.float32)
            target_mask = np.asarray(target_mask, dtype=np.float32)
            goal_label = int(episode_goals[target_idx])

            self.s_a_mask.append(
                (obs_window, action_chunk, target_chunk, target_mask, int(stiffness_label), goal_label)
            )
            self.sample_metadata.append(
                {
                    "episode_id": int(episode_id),
                    "episode_step": int(t_idx),
                    "episode_length": int(len(episode)),
                    "target_step": int(target_idx),
                    "pred_horizon": int(self.pred_horizon),
                    "stiffness_label": int(stiffness_label),
                    "goal_label": int(goal_label),
                }
            )

    def __getitem__(self, idx):
        obs_window, action_chunk, target_chunk, target_mask, stiffness_label, goal_label = self.s_a_mask[idx]
        return (
            _to_tensor(obs_window),
            _to_tensor(action_chunk),
            _to_tensor(target_chunk),
            _to_tensor(target_mask),
            torch.tensor(stiffness_label, dtype=torch.long),
            torch.tensor(goal_label, dtype=torch.long),
        )

    def get_sample_metadata(self, idx):
        if idx < 0 or idx >= len(self.sample_metadata):
            raise IndexError(f"Sample index out of range: {idx}")
        return self.sample_metadata[idx]
