from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


@dataclass
class BufferStats:
    state_stats: Dict[str, Any] | None = None
    action_stats: Dict[str, Any] | None = None
    state_mean: np.ndarray | None = None
    state_std: np.ndarray | None = None
    action_mean: np.ndarray | None = None
    action_std: np.ndarray | None = None


def _load_stats(stats_path: str | Path) -> BufferStats:
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
    with path.open("r") as f:
        stats = yaml.safe_load(f)
    state_stats = stats.get("norm_stats", {}).get("state", {})
    action_stats = stats.get("norm_stats", {}).get("action", {})
    state_mean = np.array(state_stats.get("mean", []), dtype=np.float32)
    state_std = np.array(state_stats.get("std", []), dtype=np.float32)
    action_mean = np.array(action_stats.get("mean", []), dtype=np.float32)
    action_std = np.array(action_stats.get("std", []), dtype=np.float32)
    return BufferStats(
        state_stats=state_stats if isinstance(state_stats, dict) and state_stats else None,
        action_stats=action_stats if isinstance(action_stats, dict) and action_stats else None,
        state_mean=state_mean if state_mean.size else None,
        state_std=state_std if state_std.size else None,
        action_mean=action_mean if action_mean.size else None,
        action_std=action_std if action_std.size else None,
    )


def _apply_group_norm_inplace(vec: np.ndarray, stats: Dict[str, Any], offset: int = 0) -> None:
    groups = stats.get("groups", []) if isinstance(stats, dict) else []
    if not isinstance(groups, list):
        return
    for group in groups:
        if not isinstance(group, dict):
            continue
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = int(indices[0]) + offset, int(indices[1]) + offset
        sl = slice(start, stop)
        gtype = group.get("type", "identity")
        clip_val = group.get("clip", None)

        if gtype == "identity":
            continue
        if gtype == "min_max":
            mins = np.asarray(group.get("min", []), dtype=np.float32)
            maxs = np.asarray(group.get("max", []), dtype=np.float32)
            denom = maxs - mins
            denom[denom == 0] = 1e-17
            vec[sl] = (2.0 * (vec[sl] - mins) / denom) - 1.0
        elif gtype in ("gaussian", "zscore_clip"):
            mean = np.asarray(group.get("mean", []), dtype=np.float32)
            std = np.asarray(group.get("std", []), dtype=np.float32)
            std[std == 0] = 1e-17
            vec[sl] = (vec[sl] - mean) / std
        elif gtype == "log1p":
            x = vec[sl]
            vec[sl] = np.sign(x) * np.log1p(np.abs(x))
        elif gtype == "log1p_zscore_clip":
            x = vec[sl]
            x = np.sign(x) * np.log1p(np.abs(x))
            mean = np.asarray(group.get("mean", []), dtype=np.float32)
            std = np.asarray(group.get("std", []), dtype=np.float32)
            std[std == 0] = 1e-17
            vec[sl] = (x - mean) / std
        elif gtype == "fixed_scale_clip":
            scales = np.asarray(group.get("scales", []), dtype=np.float32)
            scales[scales == 0] = 1e-17
            vec[sl] = vec[sl] / scales

        if clip_val is not None:
            try:
                c = float(clip_val)
            except (TypeError, ValueError):
                c = None
            if c is not None and c > 0:
                vec[sl] = np.clip(vec[sl], -c, c)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=np.float32)


def _flatten(x: Any) -> np.ndarray:
    return _to_numpy(x).astype(np.float32).reshape(-1)


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "data"):
        data = getattr(obs, "data")
        if isinstance(data, dict):
            return data
        return {"state": data}
    if hasattr(obs, "to_dict"):
        return obs.to_dict()
    if hasattr(obs, "__dict__"):
        d = obs.__dict__
        if isinstance(d, dict):
            return d
    if isinstance(obs, (list, tuple, np.ndarray, torch.Tensor)):
        return {"state": obs}
    raise TypeError(f"Unsupported obs type: {type(obs)}")


def _get_transitions(buffer: Any) -> Sequence[Any]:
    for name in ("storage", "_storage", "data", "_data"):
        if hasattr(buffer, name):
            items = getattr(buffer, name)
            if isinstance(items, (list, tuple)):
                return items
    if hasattr(buffer, "__iter__"):
        return list(buffer)
    raise TypeError("Could not access transitions from ReplayBuffer.")


def _looks_like_obs(x: Any) -> bool:
    if isinstance(x, dict):
        return True
    if hasattr(x, "data"):
        return True
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return True
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return False
        if len(x) == 3 and isinstance(x[-1], (bool, np.bool_)):
            return False
        return all(isinstance(v, (int, float, np.number)) for v in x[: min(5, len(x))])
    return False


def _looks_like_action(x: Any) -> bool:
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return True
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return False
        if len(x) == 3 and isinstance(x[-1], (bool, np.bool_)):
            return False
        return all(isinstance(v, (int, float, np.number)) for v in x[: min(5, len(x))])
    return False


def _looks_like_transition(obj: Any) -> bool:
    if isinstance(obj, dict):
        return any(k in obj for k in ("obs", "observation", "state", "action", "reward", "is_first"))
    if isinstance(obj, (list, tuple)):
        if len(obj) < 2:
            return False
        if len(obj) == 3 and isinstance(obj[-1], (bool, np.bool_)):
            return _looks_like_obs(obj[0]) and _looks_like_action(obj[1])
        return _looks_like_obs(obj[0]) and _looks_like_action(obj[1])
    return False


def _normalize_buffer_transitions(buffer: Any) -> Tuple[List[Any], List[bool] | None]:
    transitions = list(_get_transitions(buffer))
    if not transitions:
        return [], None
    first = transitions[0]
    if isinstance(first, (list, tuple)) and first and not _looks_like_transition(first):
        if _looks_like_transition(first[0]):
            flat: List[Any] = []
            is_first: List[bool] = []
            for ep in transitions:
                for idx, tr in enumerate(ep):
                    flat.append(tr)
                    flag = _extract_is_first_from_transition(tr)
                    is_first.append(flag if flag is not None else idx == 0)
            return flat, is_first
    return transitions, None


def _extract_is_first_from_transition(transition: Any) -> bool | None:
    for name in ("is_first", "first", "episode_start"):
        if hasattr(transition, name):
            return bool(getattr(transition, name))
        if isinstance(transition, dict) and name in transition:
            return bool(transition[name])
        if hasattr(transition, "_fields") and name in getattr(transition, "_fields"):
            return bool(getattr(transition, name))
    if isinstance(transition, (list, tuple)):
        if len(transition) >= 3 and isinstance(transition[-1], (bool, np.bool_)):
            return bool(transition[-1])
    return None


def _get_is_first_list(buffer: Any, transitions: Sequence[Any] | int) -> List[bool]:
    if isinstance(transitions, int):
        n = transitions
        transitions_seq: Sequence[Any] | None = None
    else:
        n = len(transitions)
        transitions_seq = transitions
    for name in ("is_firsts", "_is_firsts", "is_first", "_is_first"):
        if hasattr(buffer, name):
            flags = getattr(buffer, name)
            if isinstance(flags, (list, tuple, np.ndarray)) and len(flags) == n:
                return [bool(x) for x in flags]
    if transitions_seq is None:
        flags = [False] * n
        if n > 0:
            flags[0] = True
        return flags
    extracted = []
    any_found = False
    for transition in transitions_seq:
        flag = _extract_is_first_from_transition(transition)
        if flag is None:
            extracted.append(False)
        else:
            extracted.append(flag)
            any_found = True
    if any_found:
        return extracted
    flags = [False] * n
    if n > 0:
        flags[0] = True
    return flags


def _get_transition_field(transition: Any, key: str | Tuple[str, ...]) -> Any:
    keys = (key,) if isinstance(key, str) else key
    for name in keys:
        if hasattr(transition, name):
            return getattr(transition, name)
        if isinstance(transition, dict) and name in transition:
            return transition[name]
        if hasattr(transition, "_fields") and name in getattr(transition, "_fields"):
            return getattr(transition, name)
    if isinstance(transition, (list, tuple)):
        index_map = {
            "obs": 0,
            "observation": 0,
            "observations": 0,
            "state": 0,
            "action": 1,
            "actions": 1,
            "act": 1,
            "reward": 2,
            "rewards": 2,
            "rew": 2,
        }
        for name in keys:
            if name in index_map and len(transition) > index_map[name]:
                return transition[index_map[name]]
    raise KeyError(f"Transition missing field(s): {keys}")


def _maybe_get_transition_field(transition: Any, key: str | Tuple[str, ...]) -> Any | None:
    try:
        return _get_transition_field(transition, key)
    except KeyError:
        return None


def _extract_obs_action_reward(transition: Any) -> Tuple[Dict[str, Any], np.ndarray, float]:
    obs = _get_transition_field(transition, ("obs", "observation", "observations", "state", "states"))
    action = _get_transition_field(transition, ("action", "actions", "act"))
    reward = _maybe_get_transition_field(transition, ("reward", "rewards", "rew"))
    if isinstance(obs, (list, tuple)) and len(obs) == 3 and isinstance(obs[-1], (bool, np.bool_)):
        obs = obs[0]
    if isinstance(action, (list, tuple)) and len(action) == 3 and isinstance(action[-1], (bool, np.bool_)):
        action = action[1]
    obs_dict = _obs_to_dict(obs)
    action_arr = _flatten(action)
    if reward is None:
        reward_val = 0.0
    elif isinstance(reward, (list, tuple)) and len(reward) == 3 and isinstance(reward[-1], (bool, np.bool_)):
        reward_val = 0.0
    else:
        reward_val = float(reward)
    return obs_dict, action_arr, reward_val


class BufferSequenceDataset(Dataset):
    def __init__(
        self,
        buffer_path: str | Path,
        seq_len: int = 32,
        obs_keys: Sequence[str] = ("state",),
        include_goals: bool = False,
        normalize_obs: bool = False,
        normalize_action: bool = False,
        stats_path: str | Path | None = None,
    ) -> None:
        self.buffer_path = Path(buffer_path)
        # Dreamer-style training typically uses sequences of length L actions and L+1 observations
        # so that action[t] predicts obs[t+1]. Here seq_len == number of actions returned.
        self.seq_len = int(seq_len)
        self.obs_keys = tuple(obs_keys)
        self.include_goals = bool(include_goals)
        self.normalize_obs = bool(normalize_obs)
        self.normalize_action = bool(normalize_action)
        self.stats = _load_stats(stats_path) if stats_path else BufferStats()

        buffer = self._load_buffer()
        transitions, is_first = _normalize_buffer_transitions(buffer)
        if is_first is None:
            is_first = _get_is_first_list(buffer, transitions)

        episodes = self._split_episodes(transitions, is_first)
        if not episodes:
            raise ValueError("No episodes found in buffer.")

        self.episodes = episodes
        self.obs_dim = episodes[0]["obs"].shape[-1]
        self.action_dim = episodes[0]["action"].shape[-1]
        self._index = self._build_index(episodes, self.seq_len)

    def _load_buffer(self) -> Any:
        if not self.buffer_path.exists():
            raise FileNotFoundError(f"Buffer not found: {self.buffer_path}")
        with self.buffer_path.open("rb") as f:
            return pickle.load(f)

    def _select_obs_vector(self, obs_dict: Dict[str, Any]) -> Tuple[np.ndarray, slice | None]:
        vectors: List[np.ndarray] = []
        state_slice: slice | None = None
        offset = 0
        for key in self.obs_keys:
            if key not in obs_dict:
                continue
            vec = _flatten(obs_dict[key])
            if key == "state":
                state_slice = slice(offset, offset + vec.shape[0])
            vectors.append(vec)
            offset += vec.shape[0]
        if self.include_goals and "goals" in obs_dict and "goals" not in self.obs_keys:
            vec = _flatten(obs_dict["goals"])
            vectors.append(vec)
            offset += vec.shape[0]
        if not vectors:
            raise KeyError(f"No observation keys found in obs: {list(obs_dict.keys())}")
        return np.concatenate(vectors, axis=0).astype(np.float32), state_slice

    def _apply_normalization(
        self, obs_vec: np.ndarray, action_vec: np.ndarray, state_slice: slice | None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.normalize_obs and state_slice is not None:
            if isinstance(self.stats.state_stats, dict) and self.stats.state_stats.get("mode") == "grouped":
                _apply_group_norm_inplace(obs_vec, self.stats.state_stats, offset=int(state_slice.start))
            elif self.stats.state_mean is not None and self.stats.state_std is not None:
                if self.stats.state_mean.shape[0] == (state_slice.stop - state_slice.start):
                    obs_vec[state_slice] = (obs_vec[state_slice] - self.stats.state_mean) / self.stats.state_std

        if self.normalize_action:
            if isinstance(self.stats.action_stats, dict) and self.stats.action_stats.get("mode") == "grouped":
                _apply_group_norm_inplace(action_vec, self.stats.action_stats, offset=0)
            elif self.stats.action_mean is not None and self.stats.action_std is not None:
                if self.stats.action_mean.shape[0] == action_vec.shape[0]:
                    action_vec = (action_vec - self.stats.action_mean) / self.stats.action_std
        return obs_vec, action_vec

    def _split_episodes(self, transitions: Sequence[Any], is_first: Sequence[bool]) -> List[Dict[str, np.ndarray]]:
        episodes: List[Dict[str, np.ndarray]] = []
        current_obs: List[np.ndarray] = []
        current_action: List[np.ndarray] = []
        current_reward: List[float] = []

        for idx, transition in enumerate(transitions):
            if is_first[idx] and current_obs:
                episodes.append(
                    {
                        "obs": np.stack(current_obs, axis=0),
                        "action": np.stack(current_action, axis=0),
                        "reward": np.asarray(current_reward, dtype=np.float32),
                    }
                )
                current_obs, current_action, current_reward = [], [], []

            obs_dict, action_arr, reward_val = _extract_obs_action_reward(transition)
            obs_vec, state_slice = self._select_obs_vector(obs_dict)
            # obs_vec, action_arr = self._apply_normalization(obs_vec, action_arr, state_slice)

            current_obs.append(obs_vec)
            current_action.append(action_arr)
            current_reward.append(reward_val)

        if current_obs:
            episodes.append(
                {
                    "obs": np.stack(current_obs, axis=0),
                    "action": np.stack(current_action, axis=0),
                    "reward": np.asarray(current_reward, dtype=np.float32),
                }
            )

        return episodes

    @staticmethod
    def _build_index(episodes: Sequence[Dict[str, np.ndarray]], seq_len: int) -> List[Tuple[int, int]]:
        index: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(episodes):
            length = ep["obs"].shape[0]
            # Need L actions and L+1 observations.
            max_start = length - (seq_len + 1)
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                index.append((ep_idx, start))
        return index

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, start = self._index[idx]
        ep = self.episodes[ep_idx]
        # obs: (L+1, D), action/reward: (L, ...)
        end_obs = start + self.seq_len + 1
        end_act = start + self.seq_len
        obs = ep["obs"][start:end_obs]
        action = ep["action"][start:end_act]
        reward = ep["reward"][start:end_act]
        return {
            "obs": torch.from_numpy(obs).float(),
            "action": torch.from_numpy(action).float(),
            "reward": torch.from_numpy(reward).float(),
        }


def summarize_buffer(buffer_path: str | Path) -> Dict[str, Any]:
    buffer = pickle.load(Path(buffer_path).open("rb"))
    transitions, is_first = _normalize_buffer_transitions(buffer)
    if is_first is None:
        is_first = _get_is_first_list(buffer, transitions)
    obs_keys: Dict[str, int] = {}
    action_dim = None
    for transition in transitions[:10]:
        obs_dict, action, _ = _extract_obs_action_reward(transition)
        for k, v in obs_dict.items():
            try:
                obs_keys[k] = _flatten(v).shape[0]
            except Exception:
                continue
        action_dim = _flatten(action).shape[0]
    num_episodes = sum(1 for x in is_first if x)
    return {
        "num_transitions": len(transitions),
        "num_episodes": num_episodes,
        "obs_keys": obs_keys,
        "action_dim": action_dim,
    }
