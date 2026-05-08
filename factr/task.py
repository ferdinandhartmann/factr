# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

import wandb
from factr.replay_buffer import IterableWrapper
from factr.utils import (
    apply_grouped_transform as _apply_grouped_transform,
)
from factr.utils import (
    load_norm_stats_from_buffer_path as _load_norm_stats_from_buffer_path,
)
from factr.utils_plot import (
    RPYPlotConfig,
    build_pose_3d_figure,
    build_pose_fan_figure,
    pose_chunks_for_plot,
)
from factr.utils_plot import (
    make_pose_dim_names as _shared_make_pose_dim_names,
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


def _make_pose_dim_names(dim):
    return _shared_make_pose_dim_names(int(dim))


# def _build_eval_pose_figure(true_actions, pred_actions, mask, title):
#     rpy_cfg = RPYPlotConfig(
#         subtract_pi=bool(RPY_SUBTRACT_PI),
#         subtract_pi_axis=int(RPY_SUBTRACT_PI_AXIS),
#         unit=str(RPY_PLOT_UNIT),
#     )
#     return build_pose_comparison_figure(
#         true_values=true_actions,
#         pred_values=pred_actions,
#         mask=mask,
#         title=title,
#         measured_values=None,
#         rpy_config=rpy_cfg,
#     )


def _build_eval_error_figure(chunk_mse, dim_mse):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(np.arange(len(chunk_mse)), chunk_mse, color="#377EB8", linewidth=1.8)
    axes[0].set_title("MSE by Action Chunk Step")
    axes[0].set_xlabel("chunk step")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.25)

    dim_names = _make_pose_dim_names(len(dim_mse))
    axes[1].bar(np.arange(len(dim_mse)), dim_mse, color="#FB9A99")
    axes[1].set_title("MSE by Pose Dimension")
    axes[1].set_xlabel("pose dimension")
    axes[1].set_ylabel("MSE")
    axes[1].set_xticks(np.arange(len(dim_mse)))
    axes[1].set_xticklabels(dim_names, rotation=45, ha="right")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    return fig


def _build_eval_trajectory_fan_figure(
    true_action_chunks,
    pred_action_chunks,
    mask_chunks,
    measured_pose=None,
    max_steps=300,
    stiffness_label=None,
    source_time_index=None,
    global_step=None,
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
        max_plot_steps=int(max_steps),
        stiffness_label=stiffness_label,
        global_step=global_step,
        plot_ground_truth_reconstructed=True,
        plot_geodesic_subplot=plot_geodesic_subplot,
        rpy_config=rpy_cfg,
    )


def _build_missing_stiffness_figure(stiffness_label, available_labels):
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 2.6))
    ax.axis("off")
    available_str = ", ".join(str(v) for v in available_labels) if len(available_labels) > 0 else "none"
    ax.text(0.01, 0.68, f"No sampled eval episodes for stiffness={stiffness_label}", fontsize=11)
    ax.text(0.01, 0.40, f"Available sampled stiffness labels: {available_str}", fontsize=10)
    ax.text(0.01, 0.14, "Adjust eval_plot_max_steps / eval_plot_prediction_stride to change coverage.", fontsize=9)
    fig.tight_layout()
    return fig


def _extract_plot_metadata(dataset, sample_index):
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
    stiffness_label = int(meta.get("stiffness_label", 0))
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
    return {
        "obs": obs,
        "actions": actions,
        "mask": mask,
        "labels": labels,
        "imgs": imgs,
        "time_index": plot_time_index,
    }


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
        sweep_target_min_diversity: float = 0.02,
        sweep_target_min_kl: float = 0.5,
        sweep_diversity_penalty: float = 2.0,
        sweep_kl_penalty: float = 0.05,
        stiffness_classes: int = 3,
    ):
        eval_plot_axis_limits = {
            "x": (0.2, 0.6),
            "y": (-0.4, 0.4),
            "z": (0.0, 0.7),
        }
        eval_plot_goal_frames = [
            # fourgoals_2
            {"name": "goal 1", "pose": [0.341, 0.240, 0.606, 0.999, -0.007, 0.013, -0.007, -1.000, -0.010]},
            {"name": "goal 2", "pose": [0.524, 0.226, 0.381, 1.000, 0.013, 0.025, 0.013, -1.000, 0.004]},
            {"name": "goal 3", "pose": [0.591, -0.336, -0.038, 0.907, -0.421, 0.037, -0.421, -0.907, 0.006]},
            {"name": "goal 4", "pose": [0.439, -0.239, -0.043, 0.905, -0.425, 0.023, -0.426, -0.904, 0.028]},
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
        self.eval_plot_pose_mode = str(eval_plot_pose_mode)
        self.eval_plot_rpy_config = RPYPlotConfig(
            subtract_pi=bool(eval_plot_rpy_subtract_pi),
            subtract_pi_axis=int(eval_plot_rpy_subtract_pi_axis),
            unit=str(eval_plot_rpy_plot_unit),
        )
        buffer_path = getattr(test_buffer, "buffer_path", None)
        self._eval_plot_state_stats, self._eval_plot_action_stats = _load_norm_stats_from_buffer_path(buffer_path)
        self.sweep_target_min_diversity = max(0.0, float(sweep_target_min_diversity))
        self.sweep_target_min_kl = max(0.0, float(sweep_target_min_kl))
        self.sweep_diversity_penalty = max(0.0, float(sweep_diversity_penalty))
        self.sweep_kl_penalty = max(0.0, float(sweep_kl_penalty))
        self.stiffness_classes = int(stiffness_classes)

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
    def _predict_actions(model, imgs, obs, labels):
        if getattr(model, "factr_baseline", False):
            try:
                pred_actions = model.get_actions_base(imgs, obs, class_labels=labels)
            except TypeError:
                pred_actions = model.get_actions_base(imgs, obs)
        else:
            try:
                pred_actions = model.get_actions_prior(imgs, obs, class_labels=labels, sample=False, num_samples=1)
            except TypeError:
                pred_actions = model.get_actions_prior(imgs, obs, sample=False, num_samples=1)

        if pred_actions.ndim == 4:
            pred_actions = pred_actions[:, 0]
        return pred_actions

    @staticmethod
    def _sample_actions_for_plot(model, imgs, obs, labels, num_samples):
        try:
            pred_actions = model.get_actions_prior(
                imgs,
                obs,
                class_labels=labels,
                sample=True,
                num_samples=num_samples,
            )
        except TypeError:
            pred_actions = model.get_actions_prior(
                imgs,
                obs,
                sample=True,
                num_samples=num_samples,
            )

        if pred_actions.ndim == 3:
            pred_actions = pred_actions.unsqueeze(1)
        return pred_actions

    def eval(self, trainer, global_step, generate_plots=True):
        losses = []
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
        l2_per_joint_all = []
        chunk_mse_all = []
        accuracy_list = []
        first_plot_sample = None
        plot_candidates = [] if generate_plots else None
        raw_eval_index = 0
        test_dataset = getattr(self.test_loader, "dataset", None) if generate_plots else None

        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        was_training = model.training
        model.eval()

        with torch.no_grad():
            for batch in self.test_loader:
                # 1. データ受け取り
                (imgs, obs), actions, mask, labels = batch

                # 2. GPU転送
                imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
                obs, actions, mask, labels = [ar.to(trainer.device_id) for ar in (obs, actions, mask, labels)]

                ac_flat = actions.reshape((actions.shape[0], -1))
                mask_flat = mask.reshape((mask.shape[0], -1))

                output_dict = model(imgs, obs, ac_flat, mask_flat, class_labels=labels)

                losses.append(output_dict["l1_loss"].item())
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

                pred_actions = self._predict_actions(model, imgs, obs, labels)

                mask_den = mask.sum((1, 2)).clamp(min=1.0)
                prior_l1 = torch.abs(mask * (pred_actions - actions))
                prior_l1 = prior_l1.sum((1, 2)) / mask_den
                prior_l1_losses.append(prior_l1.mean().item())

                l2_delta = torch.square(mask * (pred_actions - actions))
                l2_delta = l2_delta.sum((1, 2)) / mask_den
                action_l2.append(l2_delta.mean().item())

                l2_per_joint = (mask * (pred_actions - actions) ** 2).sum(1) / mask.sum(1).clamp(min=1.0)
                l2_per_joint_all.append(l2_per_joint.mean(0).cpu().numpy())
                chunk_sq = (mask * (pred_actions - actions) ** 2).sum(dim=(0, 2))
                chunk_den = mask.sum(dim=(0, 2)).clamp(min=1.0)
                chunk_mse_all.append((chunk_sq / chunk_den).cpu().numpy())

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
                )
                sample_diversity = _compute_sample_diversity(sampled_eval_actions, mask)
                if np.isfinite(sample_diversity):
                    sample_diversity_vals.append(sample_diversity)
                sample_endpoint_diversity = _compute_endpoint_diversity(sampled_eval_actions, mask)
                if np.isfinite(sample_endpoint_diversity):
                    sample_endpoint_diversity_vals.append(sample_endpoint_diversity)
                sampled_eval_actions_denorm = _apply_grouped_transform(
                    sampled_eval_actions.detach().cpu().numpy(),
                    self._eval_plot_action_stats,
                    inverse=True,
                )
                sampled_eval_actions_denorm = torch.as_tensor(
                    sampled_eval_actions_denorm,
                    dtype=sampled_eval_actions.dtype,
                    device=sampled_eval_actions.device,
                )
                sample_end_direction_diversity = _compute_end_direction_diversity(sampled_eval_actions_denorm, mask)
                if np.isfinite(sample_end_direction_diversity):
                    sample_end_direction_diversity_vals.append(sample_end_direction_diversity)
                sampled_eval_actions_denorm = _apply_grouped_transform(
                    sampled_eval_actions.detach().cpu().numpy(),
                    self._eval_plot_action_stats,
                    inverse=True,
                )
                sampled_eval_actions_denorm = torch.as_tensor(
                    sampled_eval_actions_denorm,
                    dtype=sampled_eval_actions.dtype,
                    device=sampled_eval_actions.device,
                )
                sample_end_direction_diversity = _compute_end_direction_diversity(sampled_eval_actions_denorm, mask)
                if np.isfinite(sample_end_direction_diversity):
                    sample_end_direction_diversity_vals.append(sample_end_direction_diversity)

                if generate_plots and first_plot_sample is None:
                    first_plot_sample = {
                        "true": actions[0].detach().cpu().numpy(),
                        "pred": pred_actions[0].detach().cpu().numpy(),
                        "mask": mask[0].detach().cpu().numpy(),
                    }

                if generate_plots:
                    for batch_idx in range(actions.shape[0]):
                        meta = _extract_plot_metadata(
                            dataset=test_dataset,
                            sample_index=raw_eval_index,
                        )

                        keep_step = meta["episode_step"] % self.eval_plot_prediction_stride == 0
                        within_step_limit = meta["episode_step"] < self.eval_plot_max_steps
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
                plot_label_arr = np.asarray(
                    [int(item["stiffness_label"]) for item in selected_all_candidates], dtype=np.int64
                )
                plot_label_counts = {k: int(np.sum(plot_label_arr == k)) for k in range(1, self.stiffness_classes + 1)}
                plot_label_counts_str = ",".join([f"{k}:{v}" for k, v in plot_label_counts.items()])
            else:
                plot_label_counts = {k: 0 for k in range(1, self.stiffness_classes + 1)}
                plot_label_counts_str = "none"
        else:
            selected_all_candidates = []
            plot_label_counts_str = "skipped"

        print(
            f"Step: {global_step}\tPosterior L1: {mean_val_loss:.4f}\tPrior L1: {mean_prior_l1:.4f}\t"
            f"KL: {mean_posterior_kl:.4f}\tAction L2: {ac_l2:.3f}\tLSign: {ac_lsig:.4f}\t"
            f"prior_std: {mean_prior_std:.4f}\tpost_std: {mean_posterior_std:.4f}\t"
            f"prior_H: {mean_prior_entropy:.4f}\tpost_H: {mean_posterior_entropy:.4f}\t"
            f"sample_div: {mean_sample_diversity:.4f}\tsweep_score: {sweep_score:.4f}\t"
            f"sample_endpoint_div: {mean_sample_endpoint_diversity:.4f}\t"
            f"sample_end_direction_div: {mean_sample_end_direction_diversity:.4f}\t"
            f"sample_end_direction_div: {mean_sample_end_direction_diversity:.4f}\t"
            f"plot_steps: {len(selected_all_candidates)}\tplot_stride: {self.eval_plot_prediction_stride}\t"
            f"plot_samples: {self.eval_plot_num_samples}\tplot_label_counts: {plot_label_counts_str}"
        )

        if wandb.run is not None:
            # for i, v in enumerate(l2_per_joint_mean):
            #     wandb.log({f"eval/joint{i + 1}_l2": v}, step=global_step)

            # if chunk_mse_mean is not None:
            #     for i, v in enumerate(chunk_mse_mean):
            #         wandb.log({f"eval/chunk_step_{i + 1}_mse": float(v)}, step=global_step)

            log_dict = {
                "eval/posterior_l1": mean_val_loss,
                "eval/prior_l1": mean_prior_l1,
                "eval/posterior_kl": mean_posterior_kl,
                "eval/prior_l2": ac_l2,
                "eval/prior_lsig": ac_lsig,
                **(
                    {
                        "eval/prior_std_mean": mean_prior_std,
                        "eval/posterior_std_mean": mean_posterior_std,
                    }
                    if getattr(model, "latent_distribution", None) != "categorical"
                    else {}
                ),
                "eval/prior_entropy": mean_prior_entropy,
                "eval/posterior_entropy": mean_posterior_entropy,
                "eval/sample_diversity": mean_sample_diversity,
                "eval/sample_endpoint_diversity": mean_sample_endpoint_diversity,
                "eval/sample_end_direction_diversity": mean_sample_end_direction_diversity,
                "eval/sample_diversity_combined": mean_sample_diversity_combined,
                # "eval/sweep_score": sweep_score,
            }

            if accuracy_list:
                log_dict["eval/classification_accuracy"] = np.mean(accuracy_list)
            # for stiffness_label, count in plot_label_counts.items():
            #     log_dict[f"eval/plot_anchor_count_stiffness_{int(stiffness_label)}"] = int(count)

            wandb.log(
                log_dict,
                step=global_step,
            )

            # if first_plot_sample is not None:
            #     fig = _build_eval_pose_figure(
            #         first_plot_sample["true"],
            #         first_plot_sample["pred"],
            #         first_plot_sample["mask"],
            #         title="Eval Example: Ground Truth vs Predicted Pose",
            #     )
            #     if fig is not None:
            #         wandb.log({"eval/prediction_example": wandb.Image(fig)}, step=global_step)
            #         plt.close(fig)

            # if chunk_mse_mean is not None:
            #     fig_err = _build_eval_error_figure(chunk_mse_mean, l2_per_joint_mean)
            #     if fig_err is not None:
            #         wandb.log({"eval/error_summary": wandb.Image(fig_err)}, step=global_step)
            #         plt.close(fig_err)

            if generate_plots and len(selected_all_candidates) > 0:
                if generate_plots:
                    plot_candidates = plot_candidates or []
                    available_labels = sorted({int(item["stiffness_label"]) for item in plot_candidates})

                    for stiffness_label in range(1, self.stiffness_classes + 1):
                        label_candidates = [
                            item for item in plot_candidates if int(item["stiffness_label"]) == int(stiffness_label)
                        ]
                        selected_stiff_candidates = _select_episode_plot_candidates(
                            label_candidates,
                            max_steps=self.eval_plot_max_steps,
                        )
                        if len(selected_stiff_candidates) == 0:
                            fig_missing = _build_missing_stiffness_figure(
                                stiffness_label=stiffness_label,
                                available_labels=available_labels,
                            )
                            if fig_missing is not None:
                                wandb.log(
                                    {f"eval/prior_fan_stiffness_{int(stiffness_label)}": wandb.Image(fig_missing)},
                                    step=global_step,
                                )
                                plt.close(fig_missing)
                            continue

                        stiffess_bundle = _stack_plot_candidates(selected_stiff_candidates, device=trainer.device_id)
                        sampled_actions_stiff = self._sample_actions_for_plot(
                            model=model,
                            imgs=stiffess_bundle["imgs"],
                            obs=stiffess_bundle["obs"],
                            labels=stiffess_bundle["labels"],
                            num_samples=self.eval_plot_num_samples,
                        )
                        sampled_actions_stiff_np = sampled_actions_stiff.detach().cpu().numpy()
                        measured_pose_stiff = (
                            stiffess_bundle["obs"][:, -1, : stiffess_bundle["actions"].shape[-1]].detach().cpu().numpy()
                        )

                        if int(stiffness_label) in (1, 2):
                            pose_dim = min(
                                9,
                                int(sampled_actions_stiff_np.shape[-1]),
                                int(measured_pose_stiff.shape[-1]),
                                int(stiffess_bundle["actions"].shape[-1]),
                            )
                            if pose_dim >= 9:
                                action_stats = self._eval_plot_action_stats
                                state_stats = self._eval_plot_state_stats
                                with torch.no_grad():
                                    pred_actions_stiff = self._predict_actions(
                                        model=model,
                                        imgs=stiffess_bundle["imgs"],
                                        obs=stiffess_bundle["obs"],
                                        labels=stiffess_bundle["labels"],
                                    )
                                pred_actions_stiff_np = pred_actions_stiff.detach().cpu().numpy()
                                actions_stiff_np = stiffess_bundle["actions"].detach().cpu().numpy()
                                obs_stiff_np = stiffess_bundle["obs"].detach().cpu().numpy()

                                if action_stats is not None:
                                    actions_stiff_np = _apply_grouped_transform(
                                        actions_stiff_np, action_stats, inverse=True
                                    )
                                    sampled_actions_stiff_np = _apply_grouped_transform(
                                        sampled_actions_stiff_np, action_stats, inverse=True
                                    )
                                    pred_actions_stiff_np = _apply_grouped_transform(
                                        pred_actions_stiff_np, action_stats, inverse=True
                                    )
                                if state_stats is not None:
                                    obs_stiff_np = _apply_grouped_transform(obs_stiff_np, state_stats, inverse=True)

                                measured_pose_stiff = obs_stiff_np[:, -1, :pose_dim]
                                actions_plot_np = pose_chunks_for_plot(
                                    actions_stiff_np[:, :, :pose_dim],
                                    measured_pose_stiff,
                                    self.eval_plot_pose_mode,
                                )
                                sampled_plot_np = pose_chunks_for_plot(
                                    sampled_actions_stiff_np[:, :, :, :pose_dim],
                                    measured_pose_stiff[:, None, :],
                                    self.eval_plot_pose_mode,
                                )
                                pred_plot_np = pose_chunks_for_plot(
                                    pred_actions_stiff_np[:, :, :pose_dim],
                                    measured_pose_stiff,
                                    self.eval_plot_pose_mode,
                                )

                                fig_stiff = _build_eval_trajectory_fan_figure(
                                    true_action_chunks=actions_plot_np,
                                    pred_action_chunks=sampled_plot_np,
                                    mask_chunks=stiffess_bundle["mask"].detach().cpu().numpy()[:, :, :pose_dim],
                                    measured_pose=measured_pose_stiff,
                                    max_steps=self.eval_plot_max_steps,
                                    stiffness_label=int(stiffness_label),
                                    source_time_index=stiffess_bundle["time_index"],
                                    global_step=global_step,
                                    plot_geodesic_subplot=self.eval_plot_geodesic_subplot,
                                    rpy_config=self.eval_plot_rpy_config,
                                )
                                wandb.log(
                                    {f"eval/prior_fan_stiffness_{int(stiffness_label)}": wandb.Image(fig_stiff)},
                                    step=global_step,
                                )
                                plt.close(fig_stiff)

                                true_pose_first = actions_plot_np[:, 0, :pose_dim]
                                pred_pose_first = pred_plot_np[:, 0, :pose_dim]
                                measured_pose_first = measured_pose_stiff
                                fig_stiff_3d = build_pose_3d_figure(
                                    measured_pose=measured_pose_first,
                                    true_pose=true_pose_first,
                                    pred_pose=pred_pose_first,
                                    sampled_pose_chunks=sampled_plot_np,
                                    prediction_stride=1,
                                    action_source=self.eval_plot_action_source,
                                    true_action_chunks=actions_plot_np,
                                    mask_chunks=stiffess_bundle["mask"].detach().cpu().numpy(),
                                    source_time_index=stiffess_bundle["time_index"],
                                    plot_ground_truth_reconstructed=True,
                                    axis_limits=self.eval_plot_axis_limits,
                                    view_elev=float(self.eval_plot_view_elev),
                                    view_azim=float(self.eval_plot_view_azim),
                                    goal_frames=self.eval_plot_goal_frames,
                                    show_plot=False,
                                )
                                if fig_stiff_3d is not None:
                                    action_key = str(self.eval_plot_action_source).strip().lower() or "prior"
                                    wandb.log(
                                        {
                                            f"eval/{action_key}_fan3d_stiffness_{int(stiffness_label)}": wandb.Image(
                                                fig_stiff_3d
                                            )
                                        },
                                        step=global_step,
                                    )
                                    plt.close(fig_stiff_3d)
                        else:
                            fig_stiff = _build_eval_trajectory_fan_figure(
                                true_action_chunks=stiffess_bundle["actions"].detach().cpu().numpy(),
                                pred_action_chunks=sampled_actions_stiff_np,
                                mask_chunks=stiffess_bundle["mask"].detach().cpu().numpy(),
                                measured_pose=measured_pose_stiff,
                                max_steps=self.eval_plot_max_steps,
                                stiffness_label=int(stiffness_label),
                                source_time_index=stiffess_bundle["time_index"],
                                global_step=global_step,
                                plot_geodesic_subplot=self.eval_plot_geodesic_subplot,
                                rpy_config=self.eval_plot_rpy_config,
                            )
                            wandb.log(
                                {f"eval/prior_fan_stiffness_{int(stiffness_label)}": wandb.Image(fig_stiff)},
                                step=global_step,
                            )
                            plt.close(fig_stiff)
