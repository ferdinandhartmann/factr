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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200


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
    default_names = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6"]
    if dim <= len(default_names):
        return default_names[:dim]
    return [f"dim{i + 1}" for i in range(dim)]


def _build_eval_pose_figure(true_actions, pred_actions, mask, title):

    valid_rows = mask[:, 0] > 0
    if np.sum(valid_rows) < 2:
        return None

    true_valid = true_actions[valid_rows]
    pred_valid = pred_actions[valid_rows]
    time_index = np.arange(true_valid.shape[0])

    pose_dim = true_valid.shape[1]
    dim_names = _make_pose_dim_names(pose_dim)
    n_cols = min(3, pose_dim)
    n_rows = int(np.ceil(pose_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.8 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for dim in range(pose_dim):
        ax = axes[dim]
        ax.plot(
            time_index, true_valid[:, dim], color="black", linewidth=1.7, label="ground truth" if dim == 0 else None
        )
        ax.plot(
            time_index,
            pred_valid[:, dim],
            color="#E41A1C",
            linewidth=1.4,
            alpha=0.9,
            label="prediction" if dim == 0 else None,
        )
        ax.set_title(f"{dim_names[dim]}")
        ax.grid(alpha=0.25)

    for ax in axes[pose_dim:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    return fig


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
):

    total_plot_steps = int(max_steps)
    if total_plot_steps < 1:
        return None
    anchor_steps = min(true_action_chunks.shape[0], pred_action_chunks.shape[0], mask_chunks.shape[0])
    if anchor_steps < 1:
        return None

    true_action_chunks = true_action_chunks[:anchor_steps]
    pred_action_chunks = pred_action_chunks[:anchor_steps]
    mask_chunks = mask_chunks[:anchor_steps]
    if source_time_index is None:
        source_time_index = np.arange(anchor_steps)
    else:
        source_time_index = np.asarray(source_time_index, dtype=np.int64)[:anchor_steps]
        if source_time_index.shape[0] != anchor_steps:
            source_time_index = np.arange(anchor_steps)

    pose_dim = true_action_chunks.shape[-1]
    n_samples = pred_action_chunks.shape[1]
    time_colors = plt.cm.turbo(np.linspace(0.0, 1.0, anchor_steps))
    dim_names = _make_pose_dim_names(pose_dim)
    n_cols = min(3, pose_dim)
    n_rows = int(np.ceil(pose_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.8 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for dim in range(pose_dim):
        ax = axes[dim]
        has_sample_label = False
        for t in range(anchor_steps):
            valid_h = mask_chunks[t, :, 0] > 0
            if not np.any(valid_h):
                continue
            c_t = time_colors[t]
            t_base = int(source_time_index[t])
            horizon_idx = np.where(valid_h)[0]
            x_vals = t_base + horizon_idx
            within_window = x_vals < total_plot_steps
            if not np.any(within_window):
                continue
            x_vals = x_vals[within_window]
            for sample_idx in range(n_samples):
                ax.plot(
                    x_vals,
                    pred_action_chunks[t, sample_idx, horizon_idx[within_window], dim],
                    color=c_t,
                    linewidth=0.6,
                    alpha=0.7,
                    label=f"{n_samples}x{anchor_steps} sampled trajectories" if not has_sample_label else None,
                )
                has_sample_label = True

        ax.plot(
            source_time_index,
            true_action_chunks[:, 0, dim],
            color="black",
            linewidth=1.0,
            label="ground truth" if dim == 0 else None,
        )
        ax.set_xlim(0, total_plot_steps - 1)
        ax.set_title(f"{dim_names[dim]}")
        ax.grid(alpha=0.25)

    for ax in axes[pose_dim:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    stiff_str = "all" if stiffness_label is None else str(stiffness_label)
    fig.suptitle(
        f"Sampled Prior Trajectories vs Ground Truth | stiffness={stiff_str}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    return fig


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
    ):
        self.n_cams, self.obs_dim, self.ac_dim = n_cams, obs_dim, ac_dim
        self.train_loader = _build_data_loader(train_buffer, batch_size, num_workers, is_train=True)
        self.eval_plot_max_steps = int(eval_plot_max_steps)
        self.eval_plot_prediction_stride = max(1, int(eval_plot_prediction_stride))
        self.eval_plot_num_samples = max(1, int(eval_plot_num_samples))
        self.stiffness_classes = int(
            getattr(train_buffer, "stiffness_classes", getattr(test_buffer, "stiffness_classes", 3))
        )

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

    def eval(self, trainer, global_step):
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

    def eval(self, trainer, global_step):
        losses = []
        prior_l1_losses = []
        posterior_kl_losses = []
        prior_std_mean_vals = []
        posterior_std_mean_vals = []
        prior_entropy_vals = []
        posterior_entropy_vals = []
        action_l2, action_lsig = [], []
        l2_per_joint_all = []
        chunk_mse_all = []
        accuracy_list = []
        first_plot_sample = None
        plot_candidates = []
        raw_eval_index = 0
        test_dataset = getattr(self.test_loader, "dataset", None)

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

                if first_plot_sample is None:
                    first_plot_sample = {
                        "true": actions[0].detach().cpu().numpy(),
                        "pred": pred_actions[0].detach().cpu().numpy(),
                        "mask": mask[0].detach().cpu().numpy(),
                    }

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
        ac_l2 = np.mean(action_l2)
        ac_lsig = np.mean(action_lsig)
        l2_per_joint_mean = np.mean(np.stack(l2_per_joint_all, axis=0), axis=0)
        chunk_mse_mean = np.mean(np.stack(chunk_mse_all, axis=0), axis=0) if chunk_mse_all else None
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

        print(
            f"Step: {global_step}\tPosterior L1: {mean_val_loss:.4f}\tPrior L1: {mean_prior_l1:.4f}\t"
            f"KL: {mean_posterior_kl:.4f}\tAction L2: {ac_l2:.3f}\tLSign: {ac_lsig:.4f}\t"
            f"prior_std: {mean_prior_std:.4f}\tpost_std: {mean_posterior_std:.4f}\t"
            f"prior_H: {mean_prior_entropy:.4f}\tpost_H: {mean_posterior_entropy:.4f}\t"
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
                "eval/task_loss": mean_val_loss,
                "eval/prior_l1": mean_prior_l1,
                "eval/posterior_kl": mean_posterior_kl,
                "eval/action_l2": ac_l2,
                "eval/action_lsig": ac_lsig,
                "eval/prior_std_mean": mean_prior_std,
                "eval/posterior_std_mean": mean_posterior_std,
                "eval/prior_entropy": mean_prior_entropy,
                "eval/posterior_entropy": mean_posterior_entropy,
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

            if len(selected_all_candidates) > 0:
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
                    # )
                    # if fig_fan_all is not None:
                    #     wandb.log({"eval/prior_fan_all": wandb.Image(fig_fan_all)}, step=global_step)
                    #     plt.close(fig_fan_all)

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
                    )
                    if fig_stiff is not None:
                        wandb.log(
                            {f"eval/prior_fan_stiffness_{int(stiffness_label)}": wandb.Image(fig_stiff)},
                            step=global_step,
                        )
                        plt.close(fig_stiff)
