# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

import wandb
from factr.goal_label import equal_goal_group_samples, format_real_goal_groups
from factr.replay_buffer import IterableWrapper
from factr.utils import (
    apply_grouped_transform as _apply_grouped_transform,
    canonical_action_mode as _canonical_action_mode,
)
from factr.utils import (
    load_norm_stats_from_buffer_path as _load_norm_stats_from_buffer_path,
    lowdim_command_start as _lowdim_command_start,
    lowdim_state_stats_without_features as _lowdim_state_stats_without_features,
)
from factr.utils_plot import (
    RPYPlotConfig,
    build_pose_3d_figure,
    build_pose_fan_figure,
    pose_chunks_for_plot,
    relative_chunk_to_absolute_torch,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 150


def seed_worker(_worker_id: int) -> None:
    """
    DataLoaderのworkerの固定.

    Dataloaderの乱数固定にはgeneratorの固定も必要らしい
    """
    worker_seed = torch.initial_seed() % 2**32
    pl.seed_everything(worker_seed)


def _decode_action_values(values: np.ndarray, action_stats, pose_mode: str) -> np.ndarray:
    return _apply_grouped_transform(values, action_stats, inverse=True)


def _build_data_loader(buffer, batch_size, num_workers, is_train=False, shuffle=True):
    if is_train and not isinstance(buffer, IterableDataset):
        buffer = IterableWrapper(buffer)

    return DataLoader(
        buffer,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not isinstance(buffer, IterableDataset) and shuffle,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
        # worker_init_fn=lambda _: np.random.seed(),
        worker_init_fn=seed_worker,
    )


def _unpack_bc_batch(batch):
    arrangement_vectors = None
    goal_vectors = None
    if len(batch) == 6:
        (imgs, obs), actions, mask, labels, arrangement_vectors, goal_vectors = batch
        return imgs, obs, actions, mask, labels, arrangement_vectors, goal_vectors
    if len(batch) == 5:
        (imgs, obs), actions, mask, labels, condition_vectors = batch
        if condition_vectors.shape[-1] == 9:
            arrangement_vectors = condition_vectors
        elif condition_vectors.shape[-1] == 3:
            goal_vectors = condition_vectors
        else:
            raise ValueError(f"Unknown conditioning vector width {condition_vectors.shape[-1]}.")
        return imgs, obs, actions, mask, labels, arrangement_vectors, goal_vectors
    (imgs, obs), actions, mask, labels = batch
    return imgs, obs, actions, mask, labels, arrangement_vectors, goal_vectors


def _build_eval_trajectory_fan_figure(
    true_action_chunks,
    pred_action_chunks,
    mask_chunks,
    measured_pose=None,
    measured_time_index=None,
    max_steps=300,
    stiffness_label=None,
    source_time_index=None,
    global_step=None,
    title=None,
    plot_geodesic_subplot=True,
    rpy_config=None,
):
    if source_time_index is None:
        source_time_index = np.arange(true_action_chunks.shape[0])
    rpy_cfg = rpy_config if rpy_config is not None else RPYPlotConfig()
    return build_pose_fan_figure(
        true_action_chunks=true_action_chunks,
        pred_action_chunks=pred_action_chunks,
        mask_chunks=mask_chunks,
        source_time_index=np.asarray(source_time_index, dtype=np.int64),
        prediction_stride=1,
        measured_pose=measured_pose,
        measured_time_index=measured_time_index,
        max_plot_steps=int(max_steps),
        stiffness_label=stiffness_label,
        global_step=global_step,
        title=title,
        plot_ground_truth_reconstructed=True,
        plot_geodesic_subplot=plot_geodesic_subplot,
        rpy_config=rpy_cfg,
    )


def _eval_condition_label(stiffness_label, override_stiffness_with_mode=False, use_stiffness_conditioning=True):
    if not bool(use_stiffness_conditioning):
        return None
    if bool(override_stiffness_with_mode):
        mode_names = {1: "follow", 2: "leading"}
        mode_name = mode_names.get(int(stiffness_label), f"unknown_label_{int(stiffness_label)}")
        return f"mode={mode_name}"
    return f"stiffness={int(stiffness_label)}"


def _eval_condition_key(stiffness_label, override_stiffness_with_mode=False, use_stiffness_conditioning=True):
    if not bool(use_stiffness_conditioning):
        return None
    if bool(override_stiffness_with_mode):
        mode_names = {1: "follow", 2: "leading"}
        mode_name = mode_names.get(int(stiffness_label), f"unknown_label_{int(stiffness_label)}")
        return f"mode_{mode_name}"
    return f"stiffness_{int(stiffness_label)}"


def _build_eval_fan_title(
    stiffness_label=None,
    global_step=None,
    override_stiffness_with_mode=False,
    use_stiffness_conditioning=True,
    arrangement_vectors=None,
    goal_vectors=None,
):
    condition = _eval_condition_label(
        stiffness_label=stiffness_label,
        override_stiffness_with_mode=override_stiffness_with_mode,
        use_stiffness_conditioning=use_stiffness_conditioning,
    )
    title_parts = ["Sampled Prior Trajectories vs Ground Truth"]
    if condition is not None:
        title_parts.append(condition)
    arrangement_text = _format_arrangement_vectors_for_title(arrangement_vectors)
    if arrangement_text is not None:
        title_parts.append(arrangement_text)
    goal_text = format_real_goal_groups(goal_vectors)
    if goal_text is not None:
        title_parts.append(goal_text)
    if global_step is not None:
        title_parts.append(f"step={global_step}")
    return " | ".join(title_parts)


def _format_arrangement_vectors_for_title(arrangement_vectors, max_unique=4):
    """Format the unique 9D one-hot arrangements represented in an eval plot."""
    if arrangement_vectors is None:
        return None
    if isinstance(arrangement_vectors, torch.Tensor):
        vectors = arrangement_vectors.detach().cpu().numpy()
    else:
        vectors = np.asarray(arrangement_vectors)
    vectors = np.asarray(vectors).reshape(-1, 9)
    unique_vectors = np.unique(np.rint(vectors).astype(np.int64), axis=0)

    formatted = ["[" + ",".join(str(int(value)) for value in vector) + "]" for vector in unique_vectors]
    if len(formatted) == 1:
        return f"arrangement={formatted[0]}"
    shown = formatted[: int(max_unique)]
    suffix = f" (+{len(formatted) - len(shown)} more)" if len(shown) < len(formatted) else ""
    return f"arrangements={'; '.join(shown)}{suffix}"


def _build_missing_stiffness_figure(
    stiffness_label,
    available_labels,
    override_stiffness_with_mode=False,
    use_stiffness_conditioning=True,
):
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 2.6))
    ax.axis("off")
    available_str = ", ".join(str(v) for v in available_labels) if len(available_labels) > 0 else "none"
    condition = _eval_condition_label(
        stiffness_label=stiffness_label,
        override_stiffness_with_mode=override_stiffness_with_mode,
        use_stiffness_conditioning=use_stiffness_conditioning,
    )
    condition = "unconditioned policy" if condition is None else condition
    ax.text(0.01, 0.68, f"No sampled eval episodes for {condition}", fontsize=11)
    ax.text(0.01, 0.40, f"Available sampled stiffness labels: {available_str}", fontsize=10)
    ax.text(0.01, 0.14, "Adjust eval_plot_max_steps / eval_plot_prediction_stride to change coverage.", fontsize=9)
    fig.tight_layout()
    return fig


def _figure_to_rgb_array(fig):
    """Render a Matplotlib figure to an RGB image array for composition."""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3].copy()


def _build_side_by_side_figure(left_fig, right_fig, left_title, right_title, title):
    """Place two rendered figures into one logged image."""
    left_img = _figure_to_rgb_array(left_fig)
    right_img = _figure_to_rgb_array(right_fig)
    max_h = max(left_img.shape[0], right_img.shape[0])
    max_w = max(left_img.shape[1], right_img.shape[1])

    def pad_to_canvas(img):
        canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
        canvas[: img.shape[0], : img.shape[1], :] = img
        return canvas

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    for ax, img, subtitle in zip(
        axes,
        [pad_to_canvas(left_img), pad_to_canvas(right_img)],
        [left_title, right_title],
    ):
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=12)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    return fig


def _extract_plot_metadata(dataset, sample_index, sample_label=None):
    meta = {}
    if dataset is not None and hasattr(dataset, "get_sample_metadata"):
        try:
            maybe_meta = dataset.get_sample_metadata(sample_index)
            if isinstance(maybe_meta, dict):
                meta = maybe_meta
        except Exception:
            meta = {}

    episode_id = int(meta.get("episode_id", 0))
    episode_step = int(meta.get("episode_step", sample_index))
    episode_length = int(meta.get("episode_length", max(episode_step + 1, 1)))
    metadata_label = meta.get("stiffness_label")
    if sample_label is None and metadata_label is None:
        raise ValueError(f"Eval sample {sample_index} has no mode/stiffness label for plot selection.")

    stiffness_label = int(sample_label if sample_label is not None else metadata_label)
    if metadata_label is not None and int(metadata_label) != stiffness_label:
        raise ValueError(
            "Eval plot label mismatch: "
            f"dataset sample {sample_index} has metadata label {int(metadata_label)}, "
            f"but its batch label is {stiffness_label}."
        )
    return {
        "episode_id": episode_id,
        "episode_step": episode_step,
        "episode_length": episode_length,
        "stiffness_label": stiffness_label,
    }


def _select_episode_plot_candidates(candidates, max_steps):
    if len(candidates) == 0:
        return []

    episode_to_candidates = {}
    for candidate in candidates:
        ep_id = int(candidate["episode_id"])
        if ep_id not in episode_to_candidates:
            episode_to_candidates[ep_id] = []
        episode_to_candidates[ep_id].append(candidate)

    selected = []
    timeline_cursor = 0
    max_steps = int(max_steps)

    for ep_id in sorted(episode_to_candidates):
        episode_candidates = sorted(
            episode_to_candidates[ep_id],
            key=lambda item: int(item["episode_step"]),
        )
        if len(episode_candidates) == 0:
            continue
        if timeline_cursor >= max_steps:
            break

        episode_length = max(1, max(int(item["episode_length"]) for item in episode_candidates))
        for item in episode_candidates:
            x_idx = timeline_cursor + int(item["episode_step"])
            if x_idx >= max_steps:
                continue
            selected_item = dict(item)
            selected_item["plot_time_index"] = int(x_idx)
            selected.append(selected_item)

        timeline_cursor += episode_length

    return selected


def _stack_plot_candidates(candidates, device):
    if len(candidates) == 0:
        return None

    obs = torch.cat([item["obs"] for item in candidates], dim=0).to(device)
    actions = torch.cat([item["actions"] for item in candidates], dim=0)
    mask = torch.cat([item["mask"] for item in candidates], dim=0)
    labels = torch.cat([item["labels"] for item in candidates], dim=0).to(device)

    img_keys = sorted({k for item in candidates for k in item["imgs"].keys()})
    imgs = {}
    for key in img_keys:
        img_list = [item["imgs"][key] for item in candidates if key in item["imgs"]]
        if len(img_list) == len(candidates):
            imgs[key] = torch.cat(img_list, dim=0).to(device)

    plot_time_index = np.asarray([int(item["plot_time_index"]) for item in candidates], dtype=np.int64)
    bundle = {
        "obs": obs,
        "actions": actions,
        "mask": mask,
        "labels": labels,
        "imgs": imgs,
        "time_index": plot_time_index,
    }
    bundle.update(
        {"arrangement_vectors": torch.cat([item["arrangement_vectors"] for item in candidates], dim=0).to(device)}
        if all("arrangement_vectors" in item for item in candidates)
        else {}
    )
    bundle.update(
        {"goal_vectors": torch.cat([item["goal_vectors"] for item in candidates], dim=0).to(device)}
        if all("goal_vectors" in item for item in candidates)
        else {}
    )
    return bundle


def _stack_measured_plot_candidates(candidates, device):
    if len(candidates) == 0:
        return None

    obs = torch.cat([item["obs"] for item in candidates], dim=0).to(device)
    plot_time_index = np.asarray([int(item["plot_time_index"]) for item in candidates], dtype=np.int64)
    bundle = {
        "obs": obs,
        "time_index": plot_time_index,
    }
    if all("actions" in item and "mask" in item for item in candidates):
        bundle["actions"] = torch.cat([item["actions"] for item in candidates], dim=0)
        bundle["mask"] = torch.cat([item["mask"] for item in candidates], dim=0)
    return bundle


def _dense_ground_truth_pose_line(
    action_chunks: np.ndarray,
    mask_chunks: np.ndarray,
    measured_pose: np.ndarray,
) -> np.ndarray:
    """Build a smooth 3D ground-truth path from dense eval samples."""
    if action_chunks.ndim != 3 or mask_chunks.ndim != 3 or measured_pose.ndim != 2:
        return measured_pose
    if action_chunks.shape[0] == 0 or action_chunks.shape[1] == 0:
        return measured_pose

    valid_first_step = mask_chunks[:, 0, 0] > 0
    if not np.any(valid_first_step):
        return measured_pose

    first_future_pose = action_chunks[valid_first_step, 0, :]
    if measured_pose.shape[0] == 0:
        return first_future_pose
    # The replay buffer uses action_index_offset=1, so prepend the current pose
    # and then use every dense first future action for a continuous path.
    return np.concatenate([measured_pose[:1], first_future_pose], axis=0).astype(np.float32)


def _default_eval_plot_axis_limits(pose_mode: str):
    mode = str(pose_mode).strip().lower()
    if mode == "absolute":
        return {
            "x": (0.2, 0.6),
            "y": (-0.4, 0.4),
            "z": (0.0, 0.7),
        }
    return {
        "x": (0.0, 0.8),
        "y": (-0.4, 0.4),
        "z": (0.0, 0.7),
    }


def _select_eval_plot_axis_limits(axis_limits, pose_mode: str):
    if axis_limits is None:
        return _default_eval_plot_axis_limits(pose_mode)
    if not hasattr(axis_limits, "get"):
        return axis_limits

    mode = str(pose_mode).strip().lower()
    relative_keys = ("delta", "relative", "relative_timestep", "relative_timesteps", "relative_chunks")
    if mode == "absolute" and axis_limits.get("absolute") is not None:
        return axis_limits.get("absolute")
    if mode in relative_keys and axis_limits.get("relative") is not None:
        return axis_limits.get("relative")
    return axis_limits


def _compute_endpoint_diversity(sampled_actions: torch.Tensor, mask: torch.Tensor) -> float:
    """Average pairwise L2 distance across sampled action endpoints.

    Args:
        sampled_actions: Tensor of shape (B, S, T, D)
        mask: Tensor of shape (B, T, D)
    """
    if sampled_actions.ndim != 4 or mask.ndim != 3:
        return float("nan")

    _, num_samples, _, _ = sampled_actions.shape
    if num_samples < 2:
        return 0.0

    # Select endpoint: shape (B, S, D)
    endpoints = sampled_actions[:, :, -1, :]
    endpoint_mask = mask[:, -1, :]  # shape (B, D)

    # Pairwise endpoint differences: shape (B, S, S, D)
    diffs = endpoints.unsqueeze(2) - endpoints.unsqueeze(1)
    sq = diffs.pow(2)

    mask_expanded = endpoint_mask.unsqueeze(1).unsqueeze(1)  # shape (B, 1, 1, D)
    denom = mask_expanded.sum(dim=3).clamp(min=1.0)

    rms = torch.sqrt((sq * mask_expanded).sum(dim=3) / denom)

    tri_i, tri_j = torch.triu_indices(num_samples, num_samples, offset=1, device=sampled_actions.device)

    pairwise_vals = rms[:, tri_i, tri_j]
    return float(pairwise_vals.mean().item())


def _compute_end_direction_diversity(sampled_actions: torch.Tensor, mask: torch.Tensor) -> float:
    """Average pairwise L2 distance across sampled end-direction vectors."""
    _, num_samples, _, _ = sampled_actions.shape
    if num_samples < 2:
        return 0.0

    end_directions = sampled_actions[:, :, -1, :] - sampled_actions[:, :, -2, :]
    end_direction_mask = mask[:, -1, :] * mask[:, -2, :]

    diffs = end_directions.unsqueeze(2) - end_directions.unsqueeze(1)
    sq = diffs.pow(2)

    mask_expanded = end_direction_mask.unsqueeze(1).unsqueeze(1)
    denom = mask_expanded.sum(dim=3).clamp(min=1.0)
    rms = torch.sqrt((sq * mask_expanded).sum(dim=3) / denom)

    tri_i, tri_j = torch.triu_indices(num_samples, num_samples, offset=1, device=sampled_actions.device)
    pairwise_vals = rms[:, tri_i, tri_j]
    return float(pairwise_vals.mean().item())


def _compute_end_direction_diversity(sampled_actions: torch.Tensor, mask: torch.Tensor) -> float:
    """Average pairwise L2 distance across sampled end-direction vectors."""
    _, num_samples, _, _ = sampled_actions.shape
    if num_samples < 2:
        return 0.0

    end_directions = sampled_actions[:, :, -1, :] - sampled_actions[:, :, -2, :]
    end_direction_mask = mask[:, -1, :] * mask[:, -2, :]

    diffs = end_directions.unsqueeze(2) - end_directions.unsqueeze(1)
    sq = diffs.pow(2)

    mask_expanded = end_direction_mask.unsqueeze(1).unsqueeze(1)
    denom = mask_expanded.sum(dim=3).clamp(min=1.0)
    rms = torch.sqrt((sq * mask_expanded).sum(dim=3) / denom)

    tri_i, tri_j = torch.triu_indices(num_samples, num_samples, offset=1, device=sampled_actions.device)
    pairwise_vals = rms[:, tri_i, tri_j]
    return float(pairwise_vals.mean().item())


def _compute_sample_diversity(sampled_actions: torch.Tensor, mask: torch.Tensor) -> float:
    """Average pairwise L2 distance across sampled action chunks.

    Args:
        sampled_actions: Tensor of shape (B, S, T, D)
        mask: Tensor of shape (B, T, D)
    """
    if sampled_actions.ndim != 4 or mask.ndim != 3:
        return float("nan")

    _, num_samples, _, _ = sampled_actions.shape
    if num_samples < 2:
        return 0.0

    # Compute pairwise L2 distances
    diffs = sampled_actions.unsqueeze(2) - sampled_actions.unsqueeze(1)
    sq = diffs.pow(2)
    mask_expanded = mask.unsqueeze(1).unsqueeze(1)
    denom = mask_expanded.sum(dim=(3, 4)).clamp(min=1.0)
    rms = torch.sqrt((sq * mask_expanded).sum(dim=(3, 4)) / denom)

    tri_i, tri_j = torch.triu_indices(num_samples, num_samples, offset=1, device=sampled_actions.device)
    if tri_i.numel() == 0:
        return 0.0

    pairwise_vals = rms[:, tri_i, tri_j]
    return float(pairwise_vals.mean().item())


def _compute_traj_variance(
    sampled_abs_trajs: torch.Tensor,
    mask: torch.Tensor,
    w_start: float = 0.0,
    w_end: float = 1.0,
) -> float:
    """Weighted sample variance over absolute sampled trajectories.

    Args:
        sampled_abs_trajs: Tensor of shape (B, S, T, D)
        mask: Tensor of shape (B, T, D)
        w_start: Weight at the first future step.
        w_end: Weight at the final future step.
    """
    if sampled_abs_trajs.ndim != 4 or mask.ndim != 3:
        return float("nan")
    if sampled_abs_trajs.shape[0] != mask.shape[0] or sampled_abs_trajs.shape[2] != mask.shape[1]:
        return float("nan")

    _, num_samples, horizon, dim = sampled_abs_trajs.shape
    if num_samples < 1 or horizon < 1 or dim < 1:
        return float("nan")

    dim_mask = mask[:, :, :dim].to(device=sampled_abs_trajs.device, dtype=sampled_abs_trajs.dtype)
    if dim_mask.shape[-1] != dim:
        return float("nan")

    # Variance uses the population denominator N from the sampled prior trajectories.
    mean_traj = sampled_abs_trajs.mean(dim=1, keepdim=True)
    sq_dist = ((sampled_abs_trajs - mean_traj).pow(2) * dim_mask.unsqueeze(1)).sum(dim=-1)
    step_variance = sq_dist.mean(dim=1)  # (B, T)

    weights = torch.linspace(
        float(w_start),
        float(w_end),
        steps=horizon,
        device=sampled_abs_trajs.device,
        dtype=sampled_abs_trajs.dtype,
    )
    step_valid = (dim_mask.sum(dim=-1) > 0).to(dtype=sampled_abs_trajs.dtype)
    weighted_mask = step_valid * weights.unsqueeze(0)

    # With a single step and w_start=0, keep the metric defined instead of dividing by zero.
    zero_weight_rows = weighted_mask.sum(dim=1, keepdim=True) <= 0
    if torch.any(zero_weight_rows):
        weighted_mask = torch.where(zero_weight_rows, step_valid, weighted_mask)

    denom = weighted_mask.sum(dim=1).clamp(min=torch.finfo(sampled_abs_trajs.dtype).eps)
    per_anchor_variance = (step_variance * weighted_mask).sum(dim=1) / denom
    valid_anchor = step_valid.sum(dim=1) > 0
    if not torch.any(valid_anchor):
        return float("nan")

    return float(per_anchor_variance[valid_anchor].mean().item())


def _compute_goal_distance_sum(
    sampled_actions: torch.Tensor,
    obs: torch.Tensor,
    goal_frames,
    pose_mode: str,
    include_velocity: bool,
    include_tracking_error: bool,
    sampled_abs_trajs: Optional[torch.Tensor] = None,
) -> float:
    """Sum, over goals, of the minimum endpoint distance across sampled trajectories.

    Args:
        sampled_actions: Tensor of shape (B, S, T, D) containing denormalized pose chunks.
        obs: Tensor of shape (B, W, obs_dim) containing denormalized observations.
        goal_frames: Sequence of dicts with a 9D pose entry. Only XYZ is used here.
        pose_mode: Action pose mode used to convert chunks into absolute pose.
        include_velocity: Whether obs contains the 6D velocity slice.
        include_tracking_error: Whether obs contains the 6D tracking-error slice.
    """
    if sampled_actions.ndim != 4 or obs.ndim != 3 or not goal_frames:
        return float("nan")

    if sampled_actions.shape[-1] < 3 or obs.shape[-1] < 24:
        return float("nan")

    cmd_start = _lowdim_command_start(
        include_velocity=include_velocity,
        include_tracking_error=include_tracking_error,
    )
    cmd_stop = cmd_start + sampled_actions.shape[-1]
    if obs.shape[-1] < cmd_stop:
        return float("nan")

    goal_positions = []
    for goal in goal_frames:
        pose = goal.get("pose", None) if isinstance(goal, dict) else None
        if pose is None:
            continue
        pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose_arr.shape[0] < 3:
            continue
        goal_positions.append(torch.as_tensor(pose_arr[:3], device=sampled_actions.device, dtype=sampled_actions.dtype))

    if len(goal_positions) == 0:
        return float("nan")

    goals = torch.stack(goal_positions, dim=0)  # (G, 3)
    if sampled_abs_trajs is None:
        sampled_np = sampled_actions.detach().cpu().numpy()
        current_cmd_np = obs[:, -1, cmd_start:cmd_stop].detach().cpu().numpy()
        sampled_abs_np = pose_chunks_for_plot(sampled_np, current_cmd_np[:, None, :], pose_mode)
        endpoints = torch.as_tensor(
            sampled_abs_np[:, :, -1, :3], device=sampled_actions.device, dtype=sampled_actions.dtype
        )
    else:
        endpoints = sampled_abs_trajs[:, :, -1, :3]
    distances = torch.linalg.norm(endpoints.unsqueeze(2) - goals.unsqueeze(0).unsqueeze(0), dim=-1)  # (B, S, G)
    min_distances = distances.min(dim=1).values  # (B, G)
    return float(min_distances.sum(dim=-1).mean().item())


def _compute_dist_to_opt_traj(
    sampled_actions: torch.Tensor,
    obs: torch.Tensor,
    goal_frames,
    pose_mode: str,
    include_velocity: bool,
    include_tracking_error: bool,
    sampled_abs_trajs: Optional[torch.Tensor] = None,
) -> float:
    """Endpoint excess distance to the four straight-line goal endpoints.

    Args:
        sampled_actions: Tensor of shape (B, S, T, D) containing denormalized pose chunks.
        obs: Tensor of shape (B, W, obs_dim) containing denormalized observations.
        goal_frames: Sequence of dicts with a 9D pose entry. Only XYZ is used here.
        pose_mode: Action pose mode used to convert chunks into absolute pose.
        include_velocity: Whether obs contains the 6D velocity slice.
        include_tracking_error: Whether obs contains the 6D tracking-error slice.
    """
    if sampled_actions.ndim != 4 or obs.ndim != 3 or not goal_frames:
        return float("nan")
    if sampled_actions.shape[-1] < 3 or obs.shape[-1] < 24:
        return float("nan")

    cmd_start = _lowdim_command_start(
        include_velocity=include_velocity,
        include_tracking_error=include_tracking_error,
    )
    cmd_stop = cmd_start + sampled_actions.shape[-1]
    if obs.shape[-1] < cmd_stop:
        return float("nan")

    goal_positions = []
    for goal in goal_frames:
        pose = goal.get("pose", None) if isinstance(goal, dict) else None
        if pose is None:
            continue
        pose_arr = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose_arr.shape[0] < 3:
            continue
        goal_positions.append(pose_arr[:3])

    if len(goal_positions) == 0:
        return float("nan")

    if sampled_abs_trajs is None:
        sampled_np = sampled_actions.detach().cpu().numpy()
        current_cmd_np = obs[:, -1, cmd_start:cmd_stop].detach().cpu().numpy()
        sampled_abs_np = pose_chunks_for_plot(sampled_np, current_cmd_np[:, None, :], pose_mode)
        endpoints = torch.as_tensor(
            sampled_abs_np[:, :, -1, :3], device=sampled_actions.device, dtype=sampled_actions.dtype
        )
    else:
        endpoints = sampled_abs_trajs[:, :, -1, :3]
    goals = torch.as_tensor(
        np.stack(goal_positions, axis=0),
        device=sampled_actions.device,
        dtype=sampled_actions.dtype,
    )  # (G, 3)

    distances = torch.linalg.norm(endpoints.unsqueeze(2) - goals.unsqueeze(0).unsqueeze(0), dim=-1)  # (B, S, G)
    min_distances = distances.min(dim=1).values  # (B, G)
    return float(min_distances.mean(dim=-1).mean().item())


class DefaultTask:
    def __init__(
        self,
        train_buffer,
        test_buffer,
        cam_indexes,
        n_cams,
        obs_dim,
        ac_dim,
        batch_size,
        num_workers,
        factr_baseline: bool = False,
        eval_plot_max_steps: int = 3000,
        eval_plot_prediction_stride: int = 40,
        eval_plot_num_samples: int = 30,
        eval_plot_action_source: str = "prior",
        eval_plot_view_elev: float = 24.0,
        eval_plot_view_azim: float = -60.0,
        eval_plot_rpy_subtract_pi: bool = True,
        eval_plot_rpy_subtract_pi_axis: int = 0,
        eval_plot_rpy_plot_unit: str = "deg",
        eval_plot_geodesic_subplot: bool = False,
        eval_plot_axis_limits=None,
        eval_plot_goal_frames=None,
        eval_plot_pose_mode: str = "absolute",
        eval_traj_variance_w_start: float = 0.0,
        eval_traj_variance_w_end: float = 1.0,
        include_velocity: Optional[bool] = None,
        include_tracking_error: Optional[bool] = None,
        sweep_target_min_diversity: float = 0.02,
        sweep_target_min_kl: float = 0.5,
        sweep_diversity_penalty: float = 2.0,
        sweep_kl_penalty: float = 0.05,
        stiffness_classes: int = 3,
        use_stiffness_conditioning: bool = True,
        use_arrangement_conditioning: bool = False,
        goal_label: bool = False,
        override_stiffness_with_mode: bool = False,
    ):
        eval_plot_axis_limits = _select_eval_plot_axis_limits(eval_plot_axis_limits, eval_plot_pose_mode)
        eval_plot_goal_frames = [
            # fourgoals_2
            # {"name": "goal 1", "pose": [0.341, 0.240, 0.606, 0.999, -0.007, 0.013, -0.007, -1.000, -0.010]},
            # {"name": "goal 2", "pose": [0.524, 0.226, 0.381, 1.000, 0.013, 0.025, 0.013, -1.000, 0.004]},
            # {"name": "goal 3", "pose": [0.591, -0.336, -0.038, 0.907, -0.421, 0.037, -0.421, -0.907, 0.006]},
            # {"name": "goal 4", "pose": [0.439, -0.239, -0.043, 0.905, -0.425, 0.023, -0.426, -0.904, 0.028]},
            # Boxlift goals 9 (no rotation)
            {
                "name": "goal 1",
                "pose": [0.392, -0.042, 0.043, 0.999, -0.004, -0.018, -0.004, -1.000, -0.002],
            },  # Boxlift goal 1
            {
                "name": "goal 2",
                "pose": [0.401, -0.062, 0.170, 0.999, -0.024, -0.011, -0.024, -0.999, -0.003],
            },  # Boxlift goal 2
            {
                "name": "goal 3",
                "pose": [0.393, -0.059, 0.307, 0.999, -0.031, -0.004, -0.031, -0.999, -0.018],
            },  # Boxlift goal 3
            {
                "name": "goal 4",
                "pose": [0.541, -0.350, 0.063, 0.999, 0.019, -0.018, 0.019, -0.999, 0.004],
            },  # Boxlift goal 4
            {
                "name": "goal 5",
                "pose": [0.552, -0.366, 0.189, 0.999, 0.005, -0.019, 0.005, -0.999, 0.021],
            },  # Boxlift goal 5
            {
                "name": "goal 6",
                "pose": [0.548, -0.362, 0.331, 0.999, 0.003, 0.001, 0.002, -0.999, 0.034],
            },  # Boxlift goal 6
            {
                "name": "goal 7",
                "pose": [0.265, -0.374, 0.046, 0.997, 0.013, -0.053, 0.015, -0.997, 0.053],
            },  # Boxlift goal 7
            {
                "name": "goal 8",
                "pose": [0.262, -0.379, 0.180, 0.999, 0.025, 0.002, 0.025, -0.998, 0.031],
            },  # Boxlift goal 8
            {
                "name": "goal 9",
                "pose": [0.280, -0.344, 0.318, 0.999, -0.001, -0.013, -0.001, -0.999, -0.024],
            },  # Boxlift goal 9
        ]
        self.n_cams, self.obs_dim, self.ac_dim = n_cams, obs_dim, ac_dim
        self.train_loader = _build_data_loader(train_buffer, batch_size, num_workers, is_train=True)
        self.eval_plot_max_steps = int(eval_plot_max_steps)
        self.eval_plot_prediction_stride = max(1, int(eval_plot_prediction_stride))
        self.eval_plot_num_samples = max(1, int(eval_plot_num_samples))
        self.eval_plot_action_source = str(eval_plot_action_source)
        self.eval_plot_view_elev = float(eval_plot_view_elev)
        self.eval_plot_view_azim = float(eval_plot_view_azim)
        self.eval_plot_geodesic_subplot = bool(eval_plot_geodesic_subplot)
        self.eval_plot_axis_limits = eval_plot_axis_limits
        self.eval_plot_goal_frames = eval_plot_goal_frames
        self.eval_plot_pose_mode = _canonical_action_mode(eval_plot_pose_mode)
        self.eval_traj_variance_w_start = float(eval_traj_variance_w_start)
        self.eval_traj_variance_w_end = float(eval_traj_variance_w_end)
        self.include_velocity = bool(include_velocity) if include_velocity is not None else True
        self.include_tracking_error = bool(include_tracking_error) if include_tracking_error is not None else True
        self.eval_plot_rpy_config = RPYPlotConfig(
            subtract_pi=bool(eval_plot_rpy_subtract_pi),
            subtract_pi_axis=int(eval_plot_rpy_subtract_pi_axis),
            unit=str(eval_plot_rpy_plot_unit),
        )
        buffer_path = getattr(test_buffer, "buffer_path", None)
        self._eval_plot_state_stats, self._eval_plot_action_stats = _load_norm_stats_from_buffer_path(buffer_path)
        self._eval_plot_state_stats = _lowdim_state_stats_without_features(
            self._eval_plot_state_stats,
            include_velocity=self.include_velocity,
            include_tracking_error=self.include_tracking_error,
        )
        self.sweep_target_min_diversity = max(0.0, float(sweep_target_min_diversity))
        self.sweep_target_min_kl = max(0.0, float(sweep_target_min_kl))
        self.sweep_diversity_penalty = max(0.0, float(sweep_diversity_penalty))
        self.sweep_kl_penalty = max(0.0, float(sweep_kl_penalty))
        self.stiffness_classes = int(stiffness_classes)
        self.use_stiffness_conditioning = bool(use_stiffness_conditioning)
        self.use_arrangement_conditioning = bool(use_arrangement_conditioning)
        self.goal_label = bool(goal_label)
        self.override_stiffness_with_mode = bool(override_stiffness_with_mode)

        self.weights_history = []
        self.weights_steps = []
        # make sure no randomization and shuffling
        self.test_loader = DataLoader(
            test_buffer,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
        )
        self.factr_baseline = factr_baseline

    def eval(self, trainer, global_step, generate_plots=True):
        losses = []
        for batch in self.test_loader:
            with torch.no_grad():
                loss = trainer.training_step(batch, global_step)
                losses.append(loss.item())

        mean_val_loss = np.mean(losses)
        print(f"Step: {global_step}\tVal Loss: {mean_val_loss:.4f}")
        if wandb.run is not None:
            wandb.log({"eval/task_loss": mean_val_loss}, step=global_step)


class BCTask(DefaultTask):
    @staticmethod
    def _predict_actions(model, imgs, obs, labels, arrangement_vectors=None, goal_vectors=None):
        if getattr(model, "factr_baseline", False):
            try:
                pred_actions = model.get_actions_base(
                    imgs,
                    obs,
                    class_labels=labels,
                    arrangement_vectors=arrangement_vectors,
                    goal_vectors=goal_vectors,
                )
            except TypeError:
                pred_actions = model.get_actions_base(imgs, obs)
        else:
            try:
                pred_actions = model.get_actions_prior(
                    imgs,
                    obs,
                    class_labels=labels,
                    arrangement_vectors=arrangement_vectors,
                    goal_vectors=goal_vectors,
                    sample=False,
                    num_samples=1,
                )
            except TypeError:
                pred_actions = model.get_actions_prior(imgs, obs, sample=False, num_samples=1)

        if pred_actions.ndim == 4:
            pred_actions = pred_actions[:, 0]
        return pred_actions

    @staticmethod
    def _sample_actions_for_plot(
        model, imgs, obs, labels, num_samples, arrangement_vectors=None, goal_vectors=None
    ):
        def sample_once(condition, count):
            try:
                return model.get_actions_prior(
                    imgs,
                    obs,
                    class_labels=labels,
                    arrangement_vectors=arrangement_vectors,
                    goal_vectors=condition,
                    sample=True,
                    num_samples=count,
                )
            except TypeError:
                return model.get_actions_prior(imgs, obs, sample=True, num_samples=count)

        pred_actions = equal_goal_group_samples(sample_once, goal_vectors, num_samples)

        if pred_actions.ndim == 3:
            pred_actions = pred_actions.unsqueeze(1)
        return pred_actions

    def _log_eval_plots_for_stiffness(
        self,
        model,
        device,
        global_step,
        stiffness_label,
        label_candidates,
        label_measured_candidates,
        available_labels,
        log_images=True,
        deterministic_decode=False,
    ):
        condition_key = _eval_condition_key(
            stiffness_label=stiffness_label,
            override_stiffness_with_mode=self.override_stiffness_with_mode,
            use_stiffness_conditioning=self.use_stiffness_conditioning,
        )
        fan_prefix = "deterministic_fan" if deterministic_decode else "prior_fan"
        fan_log_key = f"eval/{fan_prefix}" if condition_key is None else f"eval/{fan_prefix}_{condition_key}"
        selected_candidates = _select_episode_plot_candidates(
            label_candidates,
            max_steps=self.eval_plot_max_steps,
        )
        if len(selected_candidates) == 0:
            fig_missing = _build_missing_stiffness_figure(
                stiffness_label=stiffness_label,
                available_labels=available_labels,
                override_stiffness_with_mode=self.override_stiffness_with_mode,
                use_stiffness_conditioning=self.use_stiffness_conditioning,
            )
            if log_images:
                wandb.log({fan_log_key: wandb.Image(fig_missing)}, step=global_step)
                plt.close(fig_missing)
            return {"fan": fig_missing, "fan3d": None, "condition_key": condition_key}

        bundle = _stack_plot_candidates(selected_candidates, device=device)
        measured_candidates = _select_episode_plot_candidates(
            label_measured_candidates,
            max_steps=self.eval_plot_max_steps,
        )
        measured_bundle = _stack_measured_plot_candidates(measured_candidates, device=device)

        sampled_actions = None
        if not deterministic_decode:
            sampled_actions = self._sample_actions_for_plot(
                model=model,
                imgs=bundle["imgs"],
                obs=bundle["obs"],
                labels=bundle["labels"],
                num_samples=self.eval_plot_num_samples,
                arrangement_vectors=bundle.get("arrangement_vectors"),
                goal_vectors=bundle.get("goal_vectors"),
            )
        pred_actions = self._predict_actions(
            model=model,
            imgs=bundle["imgs"],
            obs=bundle["obs"],
            labels=bundle["labels"],
            arrangement_vectors=bundle.get("arrangement_vectors"),
            goal_vectors=bundle.get("goal_vectors"),
        )

        pose_dim = 9
        assert bundle["actions"].shape[-1] == pose_dim
        assert bundle["mask"].shape[-1] == pose_dim
        if sampled_actions is not None:
            assert sampled_actions.shape[-1] == pose_dim
        assert pred_actions.shape[-1] == pose_dim
        assert bundle["obs"].shape[-1] >= pose_dim

        # Denormalize once, then convert chunks into absolute pose values for plotting.
        actions_np = _decode_action_values(
            bundle["actions"].detach().cpu().numpy(),
            self._eval_plot_action_stats,
            self.eval_plot_pose_mode,
        )
        pred_np = _decode_action_values(
            pred_actions.detach().cpu().numpy(),
            self._eval_plot_action_stats,
            self.eval_plot_pose_mode,
        )
        sampled_np = (
            _decode_action_values(
                sampled_actions.detach().cpu().numpy(),
                self._eval_plot_action_stats,
                self.eval_plot_pose_mode,
            )
            if sampled_actions is not None
            else pred_np[:, None, :, :]
        )
        obs_np = _apply_grouped_transform(
            bundle["obs"].detach().cpu().numpy(),
            self._eval_plot_state_stats,
            inverse=True,
        )
        measured_obs_np = (
            _apply_grouped_transform(
                measured_bundle["obs"].detach().cpu().numpy(),
                self._eval_plot_state_stats,
                inverse=True,
            )
            if measured_bundle is not None
            else None
        )

        measured_pose = obs_np[:, -1, :pose_dim]
        cmd_start = _lowdim_command_start(
            include_velocity=self.include_velocity,
            include_tracking_error=self.include_tracking_error,
        )
        command_pose = obs_np[:, -1, cmd_start : cmd_start + pose_dim]
        measured_pose_dense = measured_obs_np[:, -1, :pose_dim] if measured_obs_np is not None else measured_pose
        measured_time_dense = measured_bundle["time_index"] if measured_bundle is not None else bundle["time_index"]
        plot_anchor = command_pose if self.eval_plot_pose_mode == "relative" else measured_pose
        actions_plot = pose_chunks_for_plot(actions_np[:, :, :pose_dim], plot_anchor, self.eval_plot_pose_mode)
        gt_pose_line_3d = actions_plot[:, 0, :pose_dim]
        if measured_bundle is not None and "actions" in measured_bundle and "mask" in measured_bundle:
            measured_actions_np = _decode_action_values(
                measured_bundle["actions"].detach().cpu().numpy(),
                self._eval_plot_action_stats,
                self.eval_plot_pose_mode,
            )
            measured_mask_np = measured_bundle["mask"].detach().cpu().numpy()[:, :, :pose_dim]
            measured_command_pose = measured_obs_np[:, -1, cmd_start : cmd_start + pose_dim]
            measured_anchor = (
                measured_command_pose if self.eval_plot_pose_mode == "relative" else measured_pose_dense
            )
            measured_actions_plot = pose_chunks_for_plot(
                measured_actions_np[:, :, :pose_dim],
                measured_anchor,
                self.eval_plot_pose_mode,
            )
            gt_pose_line_3d = _dense_ground_truth_pose_line(
                measured_actions_plot,
                measured_mask_np,
                measured_anchor,
            )
        sampled_plot = pose_chunks_for_plot(
            sampled_np[:, :, :, :pose_dim],
            plot_anchor[:, None, :],
            self.eval_plot_pose_mode,
        )
        pred_plot = pose_chunks_for_plot(pred_np[:, :, :pose_dim], plot_anchor, self.eval_plot_pose_mode)
        mask_np = bundle["mask"].detach().cpu().numpy()[:, :, :pose_dim]
        fan_title = _build_eval_fan_title(
            stiffness_label=stiffness_label,
            global_step=global_step,
            override_stiffness_with_mode=self.override_stiffness_with_mode,
            use_stiffness_conditioning=self.use_stiffness_conditioning,
            arrangement_vectors=bundle.get("arrangement_vectors"),
            goal_vectors=bundle.get("goal_vectors"),
        )
        if deterministic_decode:
            fan_title = fan_title.replace("Sampled Prior Trajectories", "Deterministic Decode Trajectories")

        fig_fan = _build_eval_trajectory_fan_figure(
            true_action_chunks=actions_plot,
            pred_action_chunks=sampled_plot,
            mask_chunks=mask_np,
            measured_pose=measured_pose_dense,
            measured_time_index=measured_time_dense,
            max_steps=self.eval_plot_max_steps,
            stiffness_label=int(stiffness_label) if self.use_stiffness_conditioning else None,
            source_time_index=bundle["time_index"],
            global_step=global_step,
            title=fan_title,
            plot_geodesic_subplot=self.eval_plot_geodesic_subplot,
            rpy_config=self.eval_plot_rpy_config,
        )
        if log_images:
            wandb.log({fan_log_key: wandb.Image(fig_fan)}, step=global_step)
            plt.close(fig_fan)

        fig_3d = build_pose_3d_figure(
            measured_pose=measured_pose,
            true_pose=actions_plot[:, 0, :pose_dim],
            pred_pose=pred_plot[:, 0, :pose_dim],
            sampled_pose_chunks=sampled_plot,
            prediction_stride=1,
            action_source=self.eval_plot_action_source,
            true_action_chunks=actions_plot,
            mask_chunks=mask_np,
            source_time_index=bundle["time_index"],
            measured_line_pose=measured_pose_dense,
            ground_truth_pose=gt_pose_line_3d,
            plot_ground_truth_reconstructed=True,
            axis_limits=self.eval_plot_axis_limits,
            view_elev=float(self.eval_plot_view_elev),
            view_azim=float(self.eval_plot_view_azim),
            goal_frames=self.eval_plot_goal_frames,
            show_plot=False,
        )
        assert fig_3d is not None
        arrangement_title = _format_arrangement_vectors_for_title(bundle.get("arrangement_vectors"))
        goal_title = format_real_goal_groups(bundle.get("goal_vectors"))
        extra_titles = [title for title in (arrangement_title, goal_title) if title is not None]
        if extra_titles and len(fig_3d.axes) > 0:
            current_title = fig_3d.axes[0].get_title()
            fig_3d.axes[0].set_title(f"{current_title}\n{' | '.join(extra_titles)}")
        action_key = str(self.eval_plot_action_source).strip().lower() or "prior"
        fan3d_prefix = "deterministic_fan3d" if deterministic_decode else f"{action_key}_fan3d"
        fan3d_log_key = (
            f"eval/{fan3d_prefix}" if condition_key is None else f"eval/{fan3d_prefix}_{condition_key}"
        )
        if log_images:
            wandb.log({fan3d_log_key: wandb.Image(fig_3d)}, step=global_step)
            plt.close(fig_3d)
        return {"fan": fig_fan, "fan3d": fig_3d, "condition_key": condition_key}

    def _log_eval_low_high_plots(
        self,
        model,
        device,
        global_step,
        plot_candidates,
        plot_measured_candidates,
        available_labels,
    ):
        low_label = 1
        high_label = max(1, int(self.stiffness_classes))
        label_pair = [low_label] if high_label == low_label else [low_label, high_label]

        sampled_results = []
        deterministic_results = []
        for label in label_pair:
            label_candidates = [item for item in plot_candidates if int(item["stiffness_label"]) == int(label)]
            label_measured_candidates = [
                item for item in plot_measured_candidates if int(item["stiffness_label"]) == int(label)
            ]
            sampled_results.append(
                self._log_eval_plots_for_stiffness(
                    model=model,
                    device=device,
                    global_step=global_step,
                    stiffness_label=label,
                    label_candidates=label_candidates,
                    label_measured_candidates=label_measured_candidates,
                    available_labels=available_labels,
                    log_images=False,
                )
            )
            deterministic_results.append(
                self._log_eval_plots_for_stiffness(
                    model=model,
                    device=device,
                    global_step=global_step,
                    stiffness_label=label,
                    label_candidates=label_candidates,
                    label_measured_candidates=label_measured_candidates,
                    available_labels=available_labels,
                    log_images=False,
                    deterministic_decode=True,
                )
            )

        if len(sampled_results) == 1:
            wandb.log(
                {
                    "eval/prior_fan3d_low_high": wandb.Image(
                        sampled_results[0]["fan3d"] or sampled_results[0]["fan"]
                    ),
                    "eval/deterministic_fan_low_high": wandb.Image(deterministic_results[0]["fan"]),
                    "eval/deterministic_fan3d_low_high": wandb.Image(
                        deterministic_results[0]["fan3d"] or deterministic_results[0]["fan"]
                    ),
                },
                step=global_step,
            )
            for result in sampled_results + deterministic_results:
                plt.close(result["fan"])
                if result["fan3d"] is not None:
                    plt.close(result["fan3d"])
            return

        left_title = _eval_condition_label(
            low_label,
            override_stiffness_with_mode=self.override_stiffness_with_mode,
            use_stiffness_conditioning=self.use_stiffness_conditioning,
        )
        right_title = _eval_condition_label(
            high_label,
            override_stiffness_with_mode=self.override_stiffness_with_mode,
            use_stiffness_conditioning=self.use_stiffness_conditioning,
        )
        left_title = "low stiffness" if left_title is None else f"low stiffness / {left_title}"
        right_title = "high stiffness" if right_title is None else f"high stiffness / {right_title}"

        combined_sampled_fan3d = _build_side_by_side_figure(
            sampled_results[0]["fan3d"] or sampled_results[0]["fan"],
            sampled_results[1]["fan3d"] or sampled_results[1]["fan"],
            left_title=left_title,
            right_title=right_title,
            title=f"Eval prior 3D fan low vs high | step={global_step}",
        )
        combined_deterministic_fan = _build_side_by_side_figure(
            deterministic_results[0]["fan"],
            deterministic_results[1]["fan"],
            left_title=left_title,
            right_title=right_title,
            title=f"Eval deterministic decode fan low vs high | step={global_step}",
        )
        combined_deterministic_fan3d = _build_side_by_side_figure(
            deterministic_results[0]["fan3d"] or deterministic_results[0]["fan"],
            deterministic_results[1]["fan3d"] or deterministic_results[1]["fan"],
            left_title=left_title,
            right_title=right_title,
            title=f"Eval deterministic decode 3D fan low vs high | step={global_step}",
        )
        wandb.log(
            {
                "eval/prior_fan3d_low_high": wandb.Image(combined_sampled_fan3d),
                "eval/deterministic_fan_low_high": wandb.Image(combined_deterministic_fan),
                "eval/deterministic_fan3d_low_high": wandb.Image(combined_deterministic_fan3d),
            },
            step=global_step,
        )

        for result in sampled_results + deterministic_results:
            plt.close(result["fan"])
            if result["fan3d"] is not None:
                plt.close(result["fan3d"])
        plt.close(combined_sampled_fan3d)
        plt.close(combined_deterministic_fan)
        plt.close(combined_deterministic_fan3d)

    def eval(self, trainer, global_step, generate_plots=True):
        losses = []
        total_losses = []
        prior_l1_losses = []
        posterior_kl_losses = []
        prior_std_mean_vals = []
        posterior_std_mean_vals = []
        prior_entropy_vals = []
        posterior_entropy_vals = []
        action_l2, action_lsig = [], []
        sample_diversity_vals = []
        sample_endpoint_diversity_vals = []
        sample_end_direction_diversity_vals = []
        traj_variance_vals = []
        goal_distance_sum_vals = []
        dist_to_opt_traj_vals = []
        accuracy_list = []
        plot_candidates = [] if generate_plots else None
        plot_measured_candidates = [] if generate_plots else None
        raw_eval_index = 0
        test_dataset = getattr(self.test_loader, "dataset", None) if generate_plots else None

        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        was_training = model.training
        model.eval()

        with torch.no_grad():
            for batch in self.test_loader:
                # 1. データ受け取り
                imgs, obs, actions, mask, labels, arrangement_vectors, goal_vectors = _unpack_bc_batch(batch)

                # 2. GPU転送
                imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
                obs, actions, mask, labels = [ar.to(trainer.device_id) for ar in (obs, actions, mask, labels)]
                if arrangement_vectors is not None:
                    arrangement_vectors = arrangement_vectors.to(trainer.device_id)
                if goal_vectors is not None:
                    goal_vectors = goal_vectors.to(trainer.device_id)

                ac_flat = actions.reshape((actions.shape[0], -1))
                mask_flat = mask.reshape((mask.shape[0], -1))

                output_dict = model(
                    imgs,
                    obs,
                    ac_flat,
                    mask_flat,
                    class_labels=labels,
                    arrangement_vectors=arrangement_vectors,
                    goal_vectors=goal_vectors,
                )

                losses.append(output_dict["l1_loss"].item())
                if output_dict.get("total_loss") is not None:
                    # Eval total loss uses the model's exact training objective:
                    # reconstruction plus the configured KL term.
                    total_losses.append(output_dict["total_loss"].item())
                if output_dict.get("kl") is not None:
                    posterior_kl_losses.append(output_dict["kl"].item())
                if output_dict.get("prior_std_mean") is not None and model.latent_distribution != "categorical":
                    prior_std_mean_vals.append(output_dict["prior_std_mean"].item())
                if output_dict.get("posterior_std_mean") is not None and model.latent_distribution != "categorical":
                    posterior_std_mean_vals.append(output_dict["posterior_std_mean"].item())
                if output_dict.get("prior_entropy") is not None:
                    prior_entropy_vals.append(output_dict["prior_entropy"].item())
                if output_dict.get("posterior_entropy") is not None:
                    posterior_entropy_vals.append(output_dict["posterior_entropy"].item())

                pred_actions = self._predict_actions(
                    model,
                    imgs,
                    obs,
                    labels,
                    arrangement_vectors=arrangement_vectors,
                    goal_vectors=goal_vectors,
                )

                mask_den = mask.sum((1, 2)).clamp(min=1.0)
                prior_l1 = torch.abs(mask * (pred_actions - actions))
                prior_l1 = prior_l1.sum((1, 2)) / mask_den
                prior_l1_losses.append(prior_l1.mean().item())

                l2_delta = torch.square(mask * (pred_actions - actions))
                l2_delta = l2_delta.sum((1, 2)) / mask_den
                action_l2.append(l2_delta.mean().item())

                lsig = torch.logical_or(
                    torch.logical_and(actions > 0, pred_actions <= 0),
                    torch.logical_and(actions <= 0, pred_actions > 0),
                )
                lsig = (lsig.float() * mask).sum((1, 2)) / mask_den
                action_lsig.append(lsig.mean().item())

                sampled_eval_actions = self._sample_actions_for_plot(
                    model=model,
                    imgs=imgs,
                    obs=obs,
                    labels=labels,
                    num_samples=self.eval_plot_num_samples,
                    arrangement_vectors=arrangement_vectors,
                    goal_vectors=goal_vectors,
                )
                sample_diversity = _compute_sample_diversity(sampled_eval_actions, mask)
                if np.isfinite(sample_diversity):
                    sample_diversity_vals.append(sample_diversity)
                sample_endpoint_diversity = _compute_endpoint_diversity(sampled_eval_actions, mask)
                if np.isfinite(sample_endpoint_diversity):
                    sample_endpoint_diversity_vals.append(sample_endpoint_diversity)
                sampled_eval_actions_denorm = _decode_action_values(
                    sampled_eval_actions.detach().cpu().numpy(),
                    self._eval_plot_action_stats,
                    self.eval_plot_pose_mode,
                )
                sampled_eval_actions_denorm = torch.as_tensor(
                    sampled_eval_actions_denorm,
                    dtype=sampled_eval_actions.dtype,
                    device=sampled_eval_actions.device,
                )
                obs_denorm = _apply_grouped_transform(
                    obs.detach().cpu().numpy(),
                    self._eval_plot_state_stats,
                    inverse=True,
                )
                obs_denorm = torch.as_tensor(obs_denorm, dtype=obs.dtype, device=obs.device)

                cmd_start = _lowdim_command_start(
                    include_velocity=self.include_velocity,
                    include_tracking_error=self.include_tracking_error,
                )
                cmd_stop = cmd_start + sampled_eval_actions_denorm.shape[-1]
                sampled_abs_trajs = None
                if obs_denorm.shape[-1] >= cmd_stop:
                    current_cmd_pose = obs_denorm[:, -1, cmd_start:cmd_stop]
                    if self.eval_plot_pose_mode == "relative":
                        sampled_abs_trajs = relative_chunk_to_absolute_torch(
                            sampled_eval_actions_denorm, current_cmd_pose
                        )
                    else:
                        sampled_abs_np = pose_chunks_for_plot(
                            sampled_eval_actions_denorm.detach().cpu().numpy(),
                            current_cmd_pose.detach().cpu().numpy()[:, None, :],
                            self.eval_plot_pose_mode,
                        )
                        sampled_abs_trajs = torch.as_tensor(
                            sampled_abs_np,
                            dtype=sampled_eval_actions_denorm.dtype,
                            device=sampled_eval_actions_denorm.device,
                        )
                    traj_variance = _compute_traj_variance(
                        sampled_abs_trajs,
                        mask,
                        w_start=self.eval_traj_variance_w_start,
                        w_end=self.eval_traj_variance_w_end,
                    )
                    if np.isfinite(traj_variance):
                        traj_variance_vals.append(traj_variance)

                goal_distance_sum = _compute_goal_distance_sum(
                    sampled_actions=sampled_eval_actions_denorm,
                    obs=obs_denorm,
                    goal_frames=self.eval_plot_goal_frames,
                    pose_mode=self.eval_plot_pose_mode,
                    include_velocity=self.include_velocity,
                    include_tracking_error=self.include_tracking_error,
                    sampled_abs_trajs=sampled_abs_trajs,
                )
                if np.isfinite(goal_distance_sum):
                    goal_distance_sum_vals.append(goal_distance_sum)
                dist_to_opt_traj = _compute_dist_to_opt_traj(
                    sampled_actions=sampled_eval_actions_denorm,
                    obs=obs_denorm,
                    goal_frames=self.eval_plot_goal_frames,
                    pose_mode=self.eval_plot_pose_mode,
                    include_velocity=self.include_velocity,
                    include_tracking_error=self.include_tracking_error,
                    sampled_abs_trajs=sampled_abs_trajs,
                )
                if np.isfinite(dist_to_opt_traj):
                    dist_to_opt_traj_vals.append(dist_to_opt_traj)
                sample_end_direction_diversity = _compute_end_direction_diversity(sampled_eval_actions_denorm, mask)
                if np.isfinite(sample_end_direction_diversity):
                    sample_end_direction_diversity_vals.append(sample_end_direction_diversity)

                if generate_plots:
                    for batch_idx in range(actions.shape[0]):
                        # The batch label is the mode actually supplied to the policy. Use it
                        # for plot grouping and verify that dataset metadata stayed aligned.
                        sample_label = int(labels[batch_idx].detach().cpu().item())
                        meta = _extract_plot_metadata(
                            dataset=test_dataset,
                            sample_index=raw_eval_index,
                            sample_label=sample_label,
                        )

                        keep_step = meta["episode_step"] % self.eval_plot_prediction_stride == 0
                        within_step_limit = meta["episode_step"] < self.eval_plot_max_steps
                        if within_step_limit:
                            # Keep dense measured poses for plotting without changing prediction anchor stride.
                            measured_candidate = {
                                "obs": obs[batch_idx : batch_idx + 1].detach().cpu(),
                                "actions": actions[batch_idx : batch_idx + 1].detach().cpu(),
                                "mask": mask[batch_idx : batch_idx + 1].detach().cpu(),
                                "episode_id": int(meta["episode_id"]),
                                "episode_step": int(meta["episode_step"]),
                                "episode_length": int(meta["episode_length"]),
                                "stiffness_label": int(meta["stiffness_label"]),
                            }
                            plot_measured_candidates.append(measured_candidate)
                        if keep_step and within_step_limit:
                            candidate = {
                                "obs": obs[batch_idx : batch_idx + 1].detach().cpu(),
                                "actions": actions[batch_idx : batch_idx + 1].detach().cpu(),
                                "mask": mask[batch_idx : batch_idx + 1].detach().cpu(),
                                "labels": labels[batch_idx : batch_idx + 1].detach().cpu(),
                                "imgs": {
                                    cam_key: cam_tensor[batch_idx : batch_idx + 1].detach().cpu()
                                    for cam_key, cam_tensor in imgs.items()
                                },
                                "episode_id": int(meta["episode_id"]),
                                "episode_step": int(meta["episode_step"]),
                                "episode_length": int(meta["episode_length"]),
                                "stiffness_label": int(meta["stiffness_label"]),
                                **(
                                    {
                                        "arrangement_vectors": arrangement_vectors[batch_idx : batch_idx + 1]
                                        .detach()
                                        .cpu()
                                    }
                                    if arrangement_vectors is not None
                                    else {}
                                ),
                                **(
                                    {"goal_vectors": goal_vectors[batch_idx : batch_idx + 1].detach().cpu()}
                                    if goal_vectors is not None
                                    else {}
                                ),
                            }
                            plot_candidates.append(candidate)
                        raw_eval_index += 1

                if output_dict.get("logits") is not None:
                    logits = output_dict["logits"]
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == labels).float().mean().item()
                    accuracy_list.append(acc)

        if was_training:
            model.train()

        mean_val_loss = np.mean(losses)
        mean_total_loss = np.mean(total_losses) if total_losses else float("nan")
        mean_prior_l1 = np.mean(prior_l1_losses) if prior_l1_losses else float("nan")
        mean_posterior_kl = np.mean(posterior_kl_losses) if posterior_kl_losses else float("nan")
        mean_prior_std = np.mean(prior_std_mean_vals) if prior_std_mean_vals else float("nan")
        mean_posterior_std = np.mean(posterior_std_mean_vals) if posterior_std_mean_vals else float("nan")
        mean_prior_entropy = np.mean(prior_entropy_vals) if prior_entropy_vals else float("nan")
        mean_posterior_entropy = np.mean(posterior_entropy_vals) if posterior_entropy_vals else float("nan")
        mean_sample_diversity = np.mean(sample_diversity_vals) if sample_diversity_vals else float("nan")
        mean_sample_endpoint_diversity = (
            np.mean(sample_endpoint_diversity_vals) if sample_endpoint_diversity_vals else float("nan")
        )
        mean_sample_end_direction_diversity = (
            np.mean(sample_end_direction_diversity_vals) if sample_end_direction_diversity_vals else float("nan")
        )
        mean_traj_variance = np.mean(traj_variance_vals) if traj_variance_vals else float("nan")
        mean_goal_distance_sum = np.mean(goal_distance_sum_vals) if goal_distance_sum_vals else float("nan")
        mean_dist_to_opt_traj = np.mean(dist_to_opt_traj_vals) if dist_to_opt_traj_vals else float("nan")
        diversity_weights = {
            "sample": 1.0,
            "endpoint": 1.0,
            "end_direction": 1.0,
        }
        mean_sample_diversity_combined = (
            diversity_weights["sample"] * mean_sample_diversity
            + diversity_weights["endpoint"] * mean_sample_endpoint_diversity
            + diversity_weights["end_direction"] * mean_sample_end_direction_diversity
        )
        ac_l2 = np.mean(action_l2)
        ac_lsig = np.mean(action_lsig)

        # For Sweeping
        diversity_gap = (
            max(0.0, self.sweep_target_min_diversity - float(mean_sample_diversity))
            if np.isfinite(mean_sample_diversity)
            else float(self.sweep_target_min_diversity)
        )
        kl_gap = (
            max(0.0, self.sweep_target_min_kl - float(mean_posterior_kl))
            if np.isfinite(mean_posterior_kl)
            else float(self.sweep_target_min_kl)
        )
        base_prior_l1 = float(mean_prior_l1) if np.isfinite(mean_prior_l1) else 1e3
        sweep_score = base_prior_l1 + self.sweep_diversity_penalty * diversity_gap + self.sweep_kl_penalty * kl_gap

        if generate_plots:
            selected_all_candidates = _select_episode_plot_candidates(
                plot_candidates,
                max_steps=self.eval_plot_max_steps,
            )
            if len(selected_all_candidates) > 0:
                if self.use_stiffness_conditioning:
                    plot_label_arr = np.asarray(
                        [int(item["stiffness_label"]) for item in selected_all_candidates], dtype=np.int64
                    )
                    plot_label_counts = {
                        k: int(np.sum(plot_label_arr == k)) for k in range(1, self.stiffness_classes + 1)
                    }
                    plot_label_counts_str = ",".join([f"{k}:{v}" for k, v in plot_label_counts.items()])
                else:
                    plot_label_counts_str = "unconditioned"
            else:
                plot_label_counts_str = "none"
        else:
            selected_all_candidates = []
            plot_label_counts_str = "skipped"

        print(
            f"Step: {global_step}\tEval Total: {mean_total_loss:.4f}\t"
            f"Posterior L1: {mean_val_loss:.4f}\tPrior L1: {mean_prior_l1:.4f}\t"
            f"KL: {mean_posterior_kl:.4f}\tAction L2: {ac_l2:.3f}\tLSign: {ac_lsig:.4f}\t"
            f"prior_std: {mean_prior_std:.4f}\tpost_std: {mean_posterior_std:.4f}\t"
            f"prior_H: {mean_prior_entropy:.4f}\tpost_H: {mean_posterior_entropy:.4f}\t"
            f"sample_div: {mean_sample_diversity:.4f}\tsweep_score: {sweep_score:.4f}\t"
            f"goal_min_sum: {mean_goal_distance_sum:.4f}\t"
            # f"dist_to_opt_traj: {mean_dist_to_opt_traj:.4f}\t"
            f"sample_endpoint_div: {mean_sample_endpoint_diversity:.4f}\t"
            f"sample_end_direction_div: {mean_sample_end_direction_diversity:.4f}\t"
            f"traj_var_abs: {mean_traj_variance:.4f}\t"
            f"plot_steps: {len(selected_all_candidates)}\tplot_stride: {self.eval_plot_prediction_stride}\t"
            f"plot_samples: {self.eval_plot_num_samples if not self.goal_label else (self.eval_plot_num_samples // 3) * 3}"
            f"\tplot_label_counts: {plot_label_counts_str}"
        )

        if wandb.run is not None:
            log_dict = {
                # "eval/sample_diversity_combined": mean_sample_diversity_combined,
                "eval_diversity/sample_diversity": mean_sample_diversity,
                "eval_diversity/goal_min_dist_sum": mean_goal_distance_sum,
                # "eval/dist_to_opt_traj": mean_dist_to_opt_traj,
                "eval/prior_l1": mean_prior_l1,
                "eval/prior_entropy": mean_prior_entropy,
                "eval/total_loss": mean_total_loss,
                "eval/posterior_l1": mean_val_loss,
                "eval/posterior_entropy": mean_posterior_entropy,
                "eval_diversity/sample_endpoint_diversity": mean_sample_endpoint_diversity,
                "eval/posterior_kl": mean_posterior_kl,
                "eval/prior_lsig": ac_lsig,
                **(
                    {
                        "eval/prior_std_mean": mean_prior_std,
                        "eval/posterior_std_mean": mean_posterior_std,
                    }
                    if getattr(model, "latent_distribution", None) != "categorical"
                    and not getattr(model, "fixed_prior", False)
                    else {}
                ),
                "eval_diversity/sample_end_direction_diversity": mean_sample_end_direction_diversity,
                "eval_diversity/traj_variance": mean_traj_variance,
                # "eval/prior_l2": ac_l2,
                # "eval/sweep_score": sweep_score,
            }

            if accuracy_list:
                log_dict["eval/classification_accuracy"] = np.mean(accuracy_list)

            wandb.log(
                log_dict,
                step=global_step,
            )

            if generate_plots and len(selected_all_candidates) > 0:
                available_labels = sorted({int(item["stiffness_label"]) for item in plot_candidates})
                if self.use_stiffness_conditioning:
                    self._log_eval_low_high_plots(
                        model=model,
                        device=trainer.device_id,
                        global_step=global_step,
                        plot_candidates=plot_candidates,
                        plot_measured_candidates=plot_measured_candidates,
                        available_labels=available_labels,
                    )
                else:
                    self._log_eval_plots_for_stiffness(
                        model=model,
                        device=trainer.device_id,
                        global_step=global_step,
                        stiffness_label=None,
                        label_candidates=plot_candidates,
                        label_measured_candidates=plot_measured_candidates,
                        available_labels=available_labels,
                    )
