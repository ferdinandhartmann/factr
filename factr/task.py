# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

import wandb
from factr.plot_utils import (
    RPYPlotConfig,
    build_pose_comparison_figure,
    build_pose_fan_figure,
)
from factr.plot_utils import (
    make_pose_dim_names as _shared_make_pose_dim_names,
)
from factr.replay_buffer import IterableWrapper

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 150
RPY_SUBTRACT_PI = True
RPY_SUBTRACT_PI_AXIS = 0  # 0=roll, 1=pitch, 2=yaw
RPY_PLOT_UNIT = "deg"  # "rad" or "deg"


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


def _build_eval_pose_figure(true_actions, pred_actions, mask, title):
    rpy_cfg = RPYPlotConfig(
        subtract_pi=bool(RPY_SUBTRACT_PI),
        subtract_pi_axis=int(RPY_SUBTRACT_PI_AXIS),
        unit=str(RPY_PLOT_UNIT),
    )
    return build_pose_comparison_figure(
        true_values=true_actions,
        pred_values=pred_actions,
        mask=mask,
        title=title,
        measured_values=None,
        rpy_config=rpy_cfg,
    )


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
    max_steps=300,
    stiffness_label=None,
    source_time_index=None,
    global_step=None,
):
    if source_time_index is None:
        source_time_index = np.arange(true_action_chunks.shape[0])
    rpy_cfg = RPYPlotConfig(
        subtract_pi=bool(RPY_SUBTRACT_PI),
        subtract_pi_axis=int(RPY_SUBTRACT_PI_AXIS),
        unit=str(RPY_PLOT_UNIT),
    )
    return build_pose_fan_figure(
        true_action_chunks=true_action_chunks,
        pred_action_chunks=pred_action_chunks,
        mask_chunks=mask_chunks,
        source_time_index=np.asarray(source_time_index, dtype=np.int64),
        prediction_stride=1,
        measured_pose=None,
        max_plot_steps=int(max_steps),
        stiffness_label=stiffness_label,
        global_step=global_step,
        plot_ground_truth_reconstructed=True,
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


def _extract_plot_metadata(dataset, sample_index, fallback_label):
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
    stiffness_label = int(meta.get("stiffness_label", fallback_label))
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
        eval_plot_prediction_stride: int = 15,
        eval_plot_num_samples: int = 10,
        eval_diversity_num_samples: int = 10,
        eval_diversity_max_batches: int = 2,
        sweep_target_min_diversity: float = 0.02,
        sweep_target_min_kl: float = 0.5,
        sweep_diversity_penalty: float = 2.0,
        sweep_kl_penalty: float = 0.05,
        stiffness_classes: int = 3,
    ):
        self.n_cams, self.obs_dim, self.ac_dim = n_cams, obs_dim, ac_dim
        self.train_loader = _build_data_loader(train_buffer, batch_size, num_workers, is_train=True)
        self.eval_plot_max_steps = int(eval_plot_max_steps)
        self.eval_plot_prediction_stride = max(1, int(eval_plot_prediction_stride))
        self.eval_plot_num_samples = max(1, int(eval_plot_num_samples))
        self.eval_diversity_num_samples = max(2, int(eval_diversity_num_samples))
        self.eval_diversity_max_batches = max(1, int(eval_diversity_max_batches))
        self.sweep_target_min_diversity = max(0.0, float(sweep_target_min_diversity))
        self.sweep_target_min_kl = max(0.0, float(sweep_target_min_kl))
        self.sweep_diversity_penalty = max(0.0, float(sweep_diversity_penalty))
        self.sweep_kl_penalty = max(0.0, float(sweep_kl_penalty))
        self.stiffness_classes = max(1, int(stiffness_classes))

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
        diversity_batches_seen = 0
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
                if output_dict.get("prior_std_mean") is not None:
                    prior_std_mean_vals.append(output_dict["prior_std_mean"].item())
                if output_dict.get("posterior_std_mean") is not None:
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

                if diversity_batches_seen < self.eval_diversity_max_batches:
                    sampled_eval_actions = self._sample_actions_for_plot(
                        model=model,
                        imgs=imgs,
                        obs=obs,
                        labels=labels,
                        num_samples=self.eval_diversity_num_samples,
                    )
                    sample_diversity = _compute_sample_diversity(sampled_eval_actions, mask)
                    if np.isfinite(sample_diversity):
                        sample_diversity_vals.append(sample_diversity)
                    diversity_batches_seen += 1

                if generate_plots and first_plot_sample is None:
                    first_plot_sample = {
                        "true": actions[0].detach().cpu().numpy(),
                        "pred": pred_actions[0].detach().cpu().numpy(),
                        "mask": mask[0].detach().cpu().numpy(),
                    }

                if generate_plots:
                    for batch_idx in range(actions.shape[0]):
                        fallback_label = int(labels[batch_idx].detach().cpu().item())
                        meta = _extract_plot_metadata(
                            dataset=test_dataset,
                            sample_index=raw_eval_index,
                            fallback_label=fallback_label,
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
                "eval/prior_std_mean": mean_prior_std,
                "eval/posterior_std_mean": mean_posterior_std,
                "eval/prior_entropy": mean_prior_entropy,
                "eval/posterior_entropy": mean_posterior_entropy,
                "eval/sample_diversity": mean_sample_diversity,
                "eval/sweep_score": sweep_score,
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
                all_bundle = _stack_plot_candidates(selected_all_candidates, device=trainer.device_id)
                if all_bundle is not None:
                    sampled_actions_all = self._sample_actions_for_plot(
                        model=model,
                        imgs=all_bundle["imgs"],
                        obs=all_bundle["obs"],
                        labels=all_bundle["labels"],
                        num_samples=self.eval_plot_num_samples,
                    )
                    sampled_actions_all = sampled_actions_all.detach().cpu().numpy()
                    # fig_fan_all = _build_eval_trajectory_fan_figure(
                    #     true_action_chunks=all_bundle["actions"].detach().cpu().numpy(),
                    #     pred_action_chunks=sampled_actions_all,
                    #     mask_chunks=all_bundle["mask"].detach().cpu().numpy(),
                    #     max_steps=self.eval_plot_max_steps,
                    #     stiffness_label="all",
                    #     source_time_index=all_bundle["time_index"],
                    #     global_step=global_step,
                    # )
                    # if fig_fan_all is not None:
                    #     wandb.log({"eval/prior_fan_all": wandb.Image(fig_fan_all)}, step=global_step)
                    #     plt.close(fig_fan_all)

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

                        stiff_bundle = _stack_plot_candidates(selected_stiff_candidates, device=trainer.device_id)
                        sampled_actions_stiff = self._sample_actions_for_plot(
                            model=model,
                            imgs=stiff_bundle["imgs"],
                            obs=stiff_bundle["obs"],
                            labels=stiff_bundle["labels"],
                            num_samples=self.eval_plot_num_samples,
                        )
                        sampled_actions_stiff = sampled_actions_stiff.detach().cpu().numpy()
                        fig_stiff = _build_eval_trajectory_fan_figure(
                            true_action_chunks=stiff_bundle["actions"].detach().cpu().numpy(),
                            pred_action_chunks=sampled_actions_stiff,
                            mask_chunks=stiff_bundle["mask"].detach().cpu().numpy(),
                            max_steps=self.eval_plot_max_steps,
                            stiffness_label=int(stiffness_label),
                            source_time_index=stiff_bundle["time_index"],
                            global_step=global_step,
                        )
                        if fig_stiff is not None:
                            wandb.log(
                                {f"eval/prior_fan_stiffness_{int(stiffness_label)}": wandb.Image(fig_stiff)},
                                step=global_step,
                            )
                            plt.close(fig_stiff)
