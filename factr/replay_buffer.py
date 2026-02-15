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


_to_tensor = lambda x: torch.from_numpy(x).float()


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


# ★★★★★ Core of the fix! ★★★★★
def _get_imgs(t, cam_idx, past_frames):
    imgs = []
    # Traverse back past_frames steps from the current step t
    curr_t = t

    for _ in range(past_frames + 1):
        img_data = curr_t.obs.image(cam_idx)

        # 1. If bytes (JPEG, etc.), decode it
        if isinstance(img_data, bytes):
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                raise ValueError("Failed to decode image from bytes")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # 2. If already a NumPy array, use as-is
        elif isinstance(img_data, np.ndarray):
            img = img_data

        else:
            raise TypeError(f"Unexpected image type: {type(img_data)}")

        # Append to list with new axis: (1, H, W, C)
        imgs.append(img[None])

        if curr_t.prev is not None:
            curr_t = curr_t.prev

    # Concatenate along time axis: (T, H, W, C)
    # ★★★ Fix: Return a single array (remove comma and arr) ★★★
    return np.concatenate(imgs, axis=0)


BUF_SHUFFLE_RNG = 3904767649


class ReplayBuffer(Dataset):
    def __init__(self, buffer_path, transform=None, n_train_demos=200, mode="train", ac_chunk=1):
        assert mode in ("train", "test"), "Mode must be train/test"
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
        print(f"Loaded from file: {buffer_path}")


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
        include_goals=False,
        stiffness_classes=3,
        shuffle=True,
    ):
        assert mode in ("train", "test"), "Mode must be train/test"
        assert obs_window >= 1, "obs_window must be >= 1"
        assert ac_chunk >= 1, "ac_chunk must be >= 1"

        self.obs_window = int(obs_window)
        self.obs_dim = int(obs_dim)
        self.pose_action_dim = int(pose_action_dim)
        self.action_index_offset = int(action_index_offset)
        self.include_goals = bool(include_goals)
        self.stiffness_classes = int(stiffness_classes)
        self.use_internal_split = bool(use_internal_split)
        self.transform = None
        self.s_a_mask = []

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
            self._append_episode_samples(episode, label, ac_chunk=ac_chunk)

    @staticmethod
    def _build_episodes(buf):
        episodes = []
        current = []
        for i in range(len(buf)):
            step = buf[i]
            is_first = bool(getattr(step, "first", False) or step.prev is None)
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
        if self.include_goals and "goals" in obs_dict:
            goals = np.asarray(obs_dict["goals"], dtype=np.float32).reshape(-1)
            state = np.concatenate([state, goals], axis=0)
        return state

    def _extract_pose_action(self, step):
        action = np.asarray(step.action, dtype=np.float32).reshape(-1)
        if action.shape[0] < self.pose_action_dim:
            raise ValueError(f"Action dim {action.shape[0]} smaller than pose_action_dim={self.pose_action_dim}.")
        return action[: self.pose_action_dim]

    def _infer_episode_stiffness_labels(self, episodes):
        explicit_labels = []
        stiffness_signatures = []

        for episode in episodes:
            first_step = episode[0]
            obs_dict = _obs_to_dict(first_step.obs)

            raw_label = None
            for key in ("stiffness_label", "stiffness_class", "stiffness"):
                if key in obs_dict:
                    raw_label = obs_dict[key]
                    break

            if raw_label is not None:
                explicit_labels.append(_to_label_index(raw_label, self.stiffness_classes))
                stiffness_signatures.append(None)
                continue

            action = np.asarray(first_step.action, dtype=np.float32).reshape(-1)
            if action.shape[0] >= 15:
                signature = tuple(np.round(action[9:15], 5).tolist())
            else:
                signature = None
            explicit_labels.append(None)
            stiffness_signatures.append(signature)

        unique_signatures = [sig for sig in stiffness_signatures if sig is not None]
        signature_to_label = {}
        if len(unique_signatures) > 0:
            unique_signatures = sorted(set(unique_signatures), key=lambda sig: float(np.linalg.norm(np.asarray(sig))))
            for idx, sig in enumerate(unique_signatures):
                signature_to_label[sig] = min(idx, self.stiffness_classes - 1)

        labels = []
        for explicit, signature in zip(explicit_labels, stiffness_signatures):
            if explicit is not None:
                label_idx = explicit
            elif signature is not None and signature in signature_to_label:
                label_idx = signature_to_label[signature]
            else:
                label_idx = 0
            labels.append(label_idx + 1)

        return labels

    def _append_episode_samples(self, episode, label, ac_chunk):
        episode_states = [self._extract_obs_vector(step) for step in episode]
        expected_obs_dim = self.obs_dim + (episode_states[0].shape[0] - self.obs_dim if self.include_goals else 0)

        max_t = len(episode_states) - self.action_index_offset
        if max_t <= 0:
            return

        for t_idx in range(max_t):
            state = episode_states[t_idx]
            if state.shape[0] != expected_obs_dim:
                raise ValueError(f"Inconsistent obs dim in episode: expected {expected_obs_dim}, got {state.shape[0]}.")

            start = max(0, t_idx - self.obs_window + 1)
            window_states = episode_states[start : t_idx + 1]
            while len(window_states) < self.obs_window:
                window_states.insert(0, window_states[0])
            obs_window = np.stack(window_states, axis=0).astype(np.float32)

            chunk_actions = []
            loss_mask = []
            for k in range(ac_chunk):
                idx = t_idx + self.action_index_offset + k
                if idx < len(episode):
                    chunk_actions.append(self._extract_pose_action(episode[idx]))
                    loss_mask.append(1.0)
                else:
                    chunk_actions.append(chunk_actions[-1])
                    loss_mask.append(0.0)

            pose_chunk = np.stack(chunk_actions, axis=0).astype(np.float32)
            loss_mask = np.asarray(loss_mask, dtype=np.float32)
            self.s_a_mask.append((obs_window, pose_chunk, loss_mask, int(label)))

    def __getitem__(self, idx):
        obs_window, pose_chunk, loss_mask, label = self.s_a_mask[idx]

        obs_tensor = _to_tensor(obs_window)
        action_tensor = _to_tensor(pose_chunk)
        mask_tensor = _to_tensor(loss_mask)[:, None].repeat((1, action_tensor.shape[-1]))
        label_tensor = torch.tensor(label, dtype=torch.long)

        return ({}, obs_tensor), action_tensor, mask_tensor, label_tensor
