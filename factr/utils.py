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


import copy
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def build_episode_goal_probabilities(log_likelihood_per_timestep, goal_classes, obs_dim, temperature=5.0):
    """Build per-episode goal probabilities with a uniform start and softened per-step evidence."""
    log_likelihood_per_timestep = np.asarray(log_likelihood_per_timestep, dtype=np.float32)
    if log_likelihood_per_timestep.ndim != 3:
        raise ValueError(
            f"Expected log_likelihood_per_timestep shape (T, G, H), got {tuple(log_likelihood_per_timestep.shape)}."
        )
    if log_likelihood_per_timestep.shape[1] != int(goal_classes):
        raise ValueError(
            f"Expected goal axis size {int(goal_classes)}, got {int(log_likelihood_per_timestep.shape[1])}."
        )

    episode_steps = int(log_likelihood_per_timestep.shape[0])
    probs = np.zeros((episode_steps, int(goal_classes)), dtype=np.float32)
    uniform = np.full((int(goal_classes),), 1.0 / float(goal_classes), dtype=np.float32)
    if episode_steps == 0:
        return probs

    probs[0] = uniform
    if episode_steps == 1:
        return probs

    # Use the immediate 1-step-ahead evidence to avoid double-counting overlapping
    # prediction windows, but accumulate that evidence over the episode so the
    # posterior can move away from the uniform prior when the model is informative.
    step_logits = np.asarray(log_likelihood_per_timestep[:, :, 0], dtype=np.float32)
    denom = max(1.0, float(obs_dim) * float(temperature))
    step_logits = step_logits / denom
    cumulative_logits = np.cumsum(step_logits, axis=0)
    cumulative_logits = cumulative_logits - np.max(cumulative_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(cumulative_logits)
    step_probs = exp_logits / np.clip(np.sum(exp_logits, axis=-1, keepdims=True), 1e-8, None)
    probs[1:] = step_probs[:-1]
    return probs


def gaussian_2d_kernel(kernel_size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for convolution.

    Args:
        kernel_size: integer, the height/width of the kernel (assumed square).
        sigma: standard deviation for the Gaussian.
        device, dtype: optional, to place the kernel on a specific device / dtype.
    Returns:
        kernel: Tensor of shape (kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords -= (kernel_size - 1) / 2.0  # shift to center
    x, y = torch.meshgrid(coords, coords, indexing="xy")
    kernel_2d = torch.exp(-0.5 * (x**2 + y**2) / sigma**2)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def gaussian_2d_smoothing(img: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Apply 2D Gaussian smoothing (blur) to a batch of images.

    Args:
        img: Tensor of shape (..., C, H, W).
        scale: Controls the standard deviation (sigma) of the Gaussian kernel. Larger scale corresponds to more smoothing.

    Returns:
        blurred: Tensor of the same shape as img.
    """
    if scale <= 0:
        return img

    sigma = scale
    kernel_size = max(3, 2 * math.ceil(3 * sigma) + 1)

    kernel_2d = gaussian_2d_kernel(kernel_size, sigma, device=img.device, dtype=img.dtype)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    C = img.shape[-3]
    kernel_2d = kernel_2d.repeat(C, 1, 1, 1)  # shape: (C, 1, kH, kW)

    padding = kernel_size // 2

    original_shape = img.shape
    batch_shape = original_shape[:-3]
    spatial_shape = original_shape[-2:]
    batch_size = int(torch.prod(torch.tensor(batch_shape)))
    img_reshaped = img.view(batch_size, C, *spatial_shape)

    blurred = F.conv2d(img_reshaped, kernel_2d, groups=C, padding=padding)
    return blurred.view(*original_shape)


def gaussian_1d_smoothing(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian 1D smoothing across the last dimension (feature_dim).

    Args:
        x: Tensor of shape (..., feature_dim).
        scale: Controls the standard deviation (sigma) of the Gaussian kernel. Larger scale corresponds to more smoothing.

    Returns:
        smoothed_x: Tensor of the same shape as x, but blurred along the last dimension.
    """
    # Handle edge case: if scale is very small, just return x
    if scale <= 0:
        return x

    # kernel_size = 2 * int(3*sigma) + 1 as a rule-of-thumb.
    sigma = scale
    kernel_size = max(3, 2 * int(3 * sigma) + 1)  # ensure at least 3

    half_size = (kernel_size - 1) // 2
    arange = torch.arange(-half_size, half_size + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-0.5 * (arange / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_1d = kernel_1d.view(1, 1, -1)

    padding = half_size

    original_shape = x.shape
    feature_dim = x.shape[-1]
    batch_size = int(torch.prod(torch.tensor(x.shape[:-1])))
    x_reshaped = x.view(batch_size, 1, feature_dim)

    smoothed = F.conv1d(x_reshaped, kernel_1d, padding=padding)
    smoothed_x = smoothed.view(*original_shape)

    return smoothed_x


def downsample_1d(x, scale=2):
    scale = int(np.round(scale))
    if scale <= 1:
        return x
    original_shape = x.shape
    x_down = F.avg_pool1d(x.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)  # (B, K/2)
    x_up = F.interpolate(x_down.unsqueeze(1), size=original_shape[-1], mode="nearest").squeeze(1)  # or 'linear'
    return x_up


def downsample_2d(img, scale=2):
    scale = int(np.round(scale))
    if scale <= 1:
        return img
    original_shape = img.shape
    x_down = F.avg_pool2d(img, kernel_size=scale, stride=scale)  # (B, C, H/2, W/2)
    x_up = F.interpolate(x_down, size=original_shape[-2:], mode="nearest")  # or 'bilinear'
    return x_up


def get_scale(scheduler, start, end, cur_step, max_step, ratio):
    assert start >= end, "Start scale must be larger than end scale"
    assert cur_step <= max_step, "Current step must be less than or equal to max step"
    assert cur_step >= 0 and max_step > 0, "Steps must be non-negative and max_step must be positive"

    if scheduler == "no":
        return 0

    t = cur_step / max_step
    if t <= ratio or scheduler == "const":
        return start

    t_rescaled = (t - ratio) / (1 - ratio)
    if scheduler == "linear":
        scale = start + t_rescaled * (end - start)
    elif scheduler == "cos":
        scale = end + 0.5 * (start - end) * (1 + np.cos(t_rescaled * np.pi))
    elif scheduler == "exp":
        scale = start * np.exp(-5 * t_rescaled)
    elif scheduler == "step":
        steps = 10
        step_index = int(t_rescaled * steps)
        scale = start + step_index * (end - start) / steps
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler}")

    return scale


def safe_denominator(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr[np.abs(arr) < 1e-12] = 1e-12
    return arr


def forward_group_transform(values: np.ndarray, group: dict) -> np.ndarray:
    gtype = group.get("type", "identity")
    if gtype in ("identity",):
        return values
    if gtype in ("gaussian", "gaussian_clip", "zscore_clip"):
        mean = np.asarray(group.get("mean", []), dtype=np.float32)
        std = safe_denominator(group.get("std", []))
        out = (values - mean) / std
        clip = group.get("clip", None)
        if clip is not None:
            out = np.clip(out, -float(clip), float(clip))
        return out
    if gtype in ("min_max",):
        mins = np.asarray(group.get("min", []), dtype=np.float32)
        maxs = np.asarray(group.get("max", []), dtype=np.float32)
        denom = safe_denominator(maxs - mins)
        out = (2.0 * (values - mins) / denom) - 1.0
        clip = group.get("clip", None)
        if clip is not None:
            out = np.clip(out, -float(clip), float(clip))
        return out
    if gtype in ("fixed_scale", "fixed_scale_clip"):
        scales = safe_denominator(group.get("scales", []))
        out = values / scales
        clip = group.get("clip", None)
        if clip is not None:
            out = np.clip(out, -float(clip), float(clip))
        return out
    if gtype == "log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    if gtype == "log1p_zscore_clip":
        mean = np.asarray(group.get("mean", []), dtype=np.float32)
        std = safe_denominator(group.get("std", []))
        out = np.sign(values) * np.log1p(np.abs(values))
        out = (out - mean) / std
        clip = group.get("clip", None)
        if clip is not None:
            out = np.clip(out, -float(clip), float(clip))
        return out
    return values


def inverse_group_transform(values: np.ndarray, group: dict) -> np.ndarray:
    gtype = group.get("type", "identity")
    if gtype in ("identity",):
        return values
    if gtype in ("gaussian", "gaussian_clip", "zscore_clip"):
        mean = np.asarray(group.get("mean", []), dtype=np.float32)
        std = safe_denominator(group.get("std", []))
        return values * std + mean
    if gtype in ("min_max",):
        mins = np.asarray(group.get("min", []), dtype=np.float32)
        maxs = np.asarray(group.get("max", []), dtype=np.float32)
        return (values + 1.0) * 0.5 * (maxs - mins) + mins
    if gtype in ("fixed_scale", "fixed_scale_clip"):
        scales = safe_denominator(group.get("scales", []))
        return values * scales
    if gtype == "log1p":
        return np.sign(values) * np.expm1(np.abs(values))
    if gtype == "log1p_zscore_clip":
        mean = np.asarray(group.get("mean", []), dtype=np.float32)
        std = safe_denominator(group.get("std", []))
        out = values * std + mean
        return np.sign(out) * np.expm1(np.abs(out))
    return values


def apply_grouped_transform(values: np.ndarray, stats: dict, inverse: bool = False) -> np.ndarray:
    arr = values.copy()
    if not stats:
        return arr

    mode = stats.get("mode", None)
    if mode != "grouped":
        if (not inverse) and "mean" in stats and "std" in stats:
            mean = np.asarray(stats.get("mean", []), dtype=np.float32)
            std = safe_denominator(stats.get("std", []))
            if mean.size == arr.shape[-1] and std.size == arr.shape[-1]:
                return (arr - mean) / std
        if inverse and "mean" in stats and "std" in stats:
            mean = np.asarray(stats.get("mean", []), dtype=np.float32)
            std = safe_denominator(stats.get("std", []))
            if mean.size == arr.shape[-1] and std.size == arr.shape[-1]:
                return arr * std + mean
        return arr

    for group in stats.get("groups", []):
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = int(indices[0]), int(indices[1])
        if start >= arr.shape[-1]:  # skip groups where slice length doesnt match stat dimensions
            continue
        sl = slice(start, stop)
        part = arr[..., sl]
        gtype = group.get("type", "identity")
        stat_dim = None
        if gtype in ("gaussian", "gaussian_clip", "zscore_clip", "log1p_zscore_clip"):
            mean = np.asarray(group.get("mean", []), dtype=np.float32)
            std = np.asarray(group.get("std", []), dtype=np.float32)
            stat_dim = int(mean.size or std.size or 0) or None
        elif gtype in ("min_max",):
            mins = np.asarray(group.get("min", []), dtype=np.float32)
            maxs = np.asarray(group.get("max", []), dtype=np.float32)
            stat_dim = int(mins.size or maxs.size or 0) or None
        elif gtype in ("fixed_scale", "fixed_scale_clip"):
            scales = np.asarray(group.get("scales", []), dtype=np.float32)
            stat_dim = int(scales.size or 0) or None

        if stat_dim is not None and part.shape[-1] != stat_dim:
            continue
        if inverse:
            arr[..., sl] = inverse_group_transform(part, group)
        else:
            arr[..., sl] = forward_group_transform(part, group)
    return arr


def detect_already_normalized(values: np.ndarray, stats: dict) -> bool:
    if not stats or stats.get("mode", None) != "grouped":
        return True

    gaussian_groups = []
    for group in stats.get("groups", []):
        if group.get("type", "identity") in ("gaussian", "gaussian_clip", "zscore_clip"):
            indices = group.get("indices", None)
            if indices and len(indices) == 2:
                gaussian_groups.append((int(indices[0]), int(indices[1]), group))
    if len(gaussian_groups) == 0:
        return True

    score_as_is = []
    score_if_norm = []
    arr = values.reshape(-1, values.shape[-1])
    for start, stop, group in gaussian_groups:
        part = arr[:, start:stop]
        if part.size == 0:
            continue
        as_is_mean = np.mean(part, axis=0)
        as_is_std = np.std(part, axis=0) + 1e-8
        score_as_is.append(float(np.mean(np.abs(as_is_mean)) + np.mean(np.abs(as_is_std - 1.0))))

        normed = forward_group_transform(part, group)
        norm_mean = np.mean(normed, axis=0)
        norm_std = np.std(normed, axis=0) + 1e-8
        score_if_norm.append(float(np.mean(np.abs(norm_mean)) + np.mean(np.abs(norm_std - 1.0))))

    if len(score_as_is) == 0:
        return True
    return float(np.mean(score_as_is)) <= float(np.mean(score_if_norm))


def ensure_normalized(values: np.ndarray, stats: dict, mode: str, name: str):
    if mode == "skip":
        print(f"[normalize] {name}: skip")
        return values.copy(), False
    if mode == "apply":
        print(f"[normalize] {name}: apply")
        return apply_grouped_transform(values, stats, inverse=False), True

    already = detect_already_normalized(values, stats)
    if already:
        print(f"[normalize] {name}: auto -> already normalized, skip")
        return values.copy(), False
    print(f"[normalize] {name}: auto -> apply normalization from rollout_config")
    return apply_grouped_transform(values, stats, inverse=False), True


def load_norm_stats_from_buffer_path(buffer_path):
    if not buffer_path:
        return None, None
    rollout_path = os.path.join(os.path.dirname(str(buffer_path)), "rollout_config.yaml")
    if not os.path.exists(rollout_path):
        return None, None
    try:
        with open(rollout_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None, None
    if not isinstance(cfg, dict):
        return None, None
    norm_stats = cfg.get("norm_stats", {}) or {}
    state_stats = norm_stats.get("state", None)
    action_stats = norm_stats.get("action", None)
    return state_stats, action_stats


def load_action_pose_mode_from_buffer_path(buffer_path):
    """Read the action representation recorded beside a processed buffer."""
    if not buffer_path:
        return None
    rollout_path = os.path.join(os.path.dirname(str(buffer_path)), "rollout_config.yaml")
    if not os.path.exists(rollout_path):
        return None
    try:
        with open(rollout_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None
    return (cfg.get("processing_config") or {}).get("action_pose_mode")


def state_stats_without_tracking_error(stats: dict):
    """Shift grouped state-stat indices after removing state[21:27]."""
    if not stats or stats.get("mode") != "grouped":
        return stats
    if int(stats.get("state_dim", 36)) <= 30:
        return stats
    adjusted = copy.deepcopy(stats)
    groups = []
    for group in adjusted.get("groups", []):
        indices = group.get("indices")
        if not indices or len(indices) != 2:
            groups.append(group)
            continue
        start, stop = map(int, indices)
        # This feature no longer exists in the reduced 30D observation.
        if (start, stop) == (21, 27) or group.get("name") == "tracking_error":
            continue
        if start >= 27:
            group["indices"] = [start - 6, stop - 6]
        groups.append(group)
    adjusted["groups"] = groups
    if "state_dim" in adjusted:
        adjusted["state_dim"] = int(adjusted["state_dim"]) - 6
    return adjusted
