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

plt.rcParams["figure.dpi"] = 220
plt.rcParams["savefig.dpi"] = 220


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

    pin_memory = False
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda:0")
            pin_memory = True
        except Exception:
            pin_memory = False

    return DataLoader(
        buffer,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not isinstance(buffer, IterableDataset) and shuffle,
        pin_memory=pin_memory,
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


def _build_eval_pose_figure(true_actions, pred_actions, mask):
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
    fig.suptitle("Eval Example: Ground Truth vs Predicted Pose", fontsize=12)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    return fig


def _build_eval_trajectory_fan_figure(
    true_action_chunks,
    pred_action_chunks,
    mask_chunks,
    max_steps=400,
    stiffness_label=None,
    source_time_index=None,
):
    steps = min(int(max_steps), true_action_chunks.shape[0], pred_action_chunks.shape[0], mask_chunks.shape[0])
    if steps < 2:
        return None

    true_action_chunks = true_action_chunks[:steps]
    pred_action_chunks = pred_action_chunks[:steps]
    mask_chunks = mask_chunks[:steps]
    if source_time_index is None:
        source_time_index = np.arange(steps)
    else:
        source_time_index = np.asarray(source_time_index, dtype=np.int64)[:steps]
        if source_time_index.shape[0] != steps:
            source_time_index = np.arange(steps)

    pose_dim = true_action_chunks.shape[-1]
    n_samples = pred_action_chunks.shape[1]
    time_colors = plt.cm.turbo(np.linspace(0.0, 1.0, steps))
    dim_names = _make_pose_dim_names(pose_dim)
    n_cols = min(3, pose_dim)
    n_rows = int(np.ceil(pose_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.8 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for dim in range(pose_dim):
        ax = axes[dim]
        has_sample_label = False
        # Plot all sampled predicted trajectories:
        # for each timestep t, sample 10 trajectories over the action chunk.
        for t in range(steps):
            valid_h = mask_chunks[t, :, 0] > 0
            if not np.any(valid_h):
                continue
            c_t = time_colors[t]
            t_base = int(source_time_index[t])
            horizon_idx = np.where(valid_h)[0]
            x_vals = t_base + horizon_idx
            for sample_idx in range(n_samples):
                ax.plot(
                    x_vals,
                    pred_action_chunks[t, sample_idx, horizon_idx, dim],
                    color=c_t,
                    linewidth=0.5,
                    alpha=0.16,
                    label=f"{n_samples}x{steps} sampled trajectories" if not has_sample_label else None,
                )
                has_sample_label = True

        # Ground truth is plotted without de-normalization (buffer is already normalized).
        ax.plot(
            source_time_index,
            true_action_chunks[:, 0, dim],
            color="black",
            linewidth=1.0,
            label="ground truth" if dim == 0 else None,
        )
        ax.set_title(f"{dim_names[dim]}")
        ax.grid(alpha=0.25)

    for ax in axes[pose_dim:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    stiff_str = "unknown" if stiffness_label is None else str(int(stiffness_label))
    fig.suptitle(
        f"Eval Example: 10x{steps} Sampled Prior Trajectories vs Ground Truth | stiffness={stiff_str} (normalized units)",
        fontsize=12,
    )
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


def _build_ground_truth_gains_figure(gt_by_stiffness):
    valid_labels = [label for label, arr in gt_by_stiffness.items() if arr is not None and arr.shape[0] > 1]
    if len(valid_labels) == 0:
        return None

    first_label = valid_labels[0]
    pose_dim = gt_by_stiffness[first_label].shape[-1]
    dim_names = _make_pose_dim_names(pose_dim)
    n_cols = min(3, pose_dim)
    n_rows = int(np.ceil(pose_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.8 * n_rows), sharex=False)
    axes = np.array(axes).reshape(-1)
    label_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(valid_labels), 3)))

    for dim in range(pose_dim):
        ax = axes[dim]
        for idx, label in enumerate(valid_labels):
            gt_arr = gt_by_stiffness[label]
            ax.plot(
                np.arange(gt_arr.shape[0]),
                gt_arr[:, dim],
                color=label_colors[idx],
                linewidth=1.0,
                alpha=0.9,
                label=f"stiffness={int(label)} (n={gt_arr.shape[0]})" if dim == 0 else None,
            )
        ax.set_title(f"{dim_names[dim]}")
        ax.grid(alpha=0.25)
        ax.set_xlabel("test sample index")
        ax.set_ylabel("normalized cmd pose")

    for ax in axes[pose_dim:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(valid_labels)))
    fig.suptitle(
        "Ground Truth Pose Commands by Stiffness (all test-buffer samples, normalized units)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    return fig


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
        eval_plot_prediction_stride: int = 1,
        factr_baseline: bool = False,
    ):
        self.n_cams, self.obs_dim, self.ac_dim = n_cams, obs_dim, ac_dim
        self.eval_plot_prediction_stride = max(1, int(eval_plot_prediction_stride))
        self.train_loader = _build_data_loader(train_buffer, batch_size, num_workers, is_train=True)

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
            pred_actions = model.get_actions_base(imgs, obs, class_labels=labels)
        else:
            pred_actions = model.get_actions_prior(imgs, obs, class_labels=labels, sample=False, num_samples=1)

        if pred_actions.ndim == 4:
            pred_actions = pred_actions[:, 0]
        return pred_actions

    def eval(self, trainer, global_step):
        plot_max_steps = 400
        plot_by_stiffness = {}

        posterior_l1_losses = []
        prior_l1_losses = []
        posterior_kl_losses = []
        action_l2, action_lsig = [], []
        l2_per_pose_dim_all = []
        chunk_mse_all = []
        accuracy_list = []

        model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        was_training = model.training
        model.eval()
        try:
            eval_device = next(model.parameters()).device
        except StopIteration:
            eval_device = torch.device(trainer.device_id)
        print(f"Eval device: {eval_device}")

        with torch.no_grad():
            for batch in self.test_loader:
                # 1. データ受け取り
                (imgs, obs), actions, mask, labels = batch

                # 2. GPU転送
                imgs = {k: v.to(eval_device) for k, v in imgs.items()}
                obs, actions, mask, labels = [ar.to(eval_device) for ar in (obs, actions, mask, labels)]

                ac_flat = actions.reshape((actions.shape[0], -1))
                mask_flat = mask.reshape((mask.shape[0], -1))

                output_dict = model(imgs, obs, ac_flat, mask_flat, class_labels=labels)
                posterior_l1_losses.append(output_dict["l1_loss"].item())
                if output_dict.get("kl") is not None:
                    posterior_kl_losses.append(output_dict["kl"].item())

                pred_actions = self._predict_actions(model, imgs, obs, labels)
                mask_den = mask.sum((1, 2)).clamp(min=1.0)

                prior_l1 = torch.abs(mask * (pred_actions - actions))
                prior_l1 = prior_l1.sum((1, 2)) / mask_den
                prior_l1_losses.append(prior_l1.mean().item())

                l2_delta = torch.square(mask * (pred_actions - actions))
                l2_delta = l2_delta.sum((1, 2)) / mask_den
                action_l2.append(l2_delta.mean().item())

                l2_per_pose_dim = (mask * (pred_actions - actions) ** 2).sum(1) / mask.sum(1).clamp(min=1.0)
                l2_per_pose_dim_all.append(l2_per_pose_dim.mean(0).cpu().numpy())

                chunk_sq = (mask * (pred_actions - actions) ** 2).sum(dim=(0, 2))
                chunk_den = mask.sum(dim=(0, 2)).clamp(min=1.0)
                chunk_mse_all.append((chunk_sq / chunk_den).cpu().numpy())

                lsig = torch.logical_or(
                    torch.logical_and(actions > 0, pred_actions <= 0),
                    torch.logical_and(actions <= 0, pred_actions > 0),
                )
                lsig = (lsig.float() * mask).sum((1, 2)) / mask_den
                action_lsig.append(lsig.mean().item())

                for b_idx in range(obs.shape[0]):
                    label_val = int(labels[b_idx].item())
                    if label_val not in plot_by_stiffness:
                        plot_by_stiffness[label_val] = {
                            "obs": [],
                            "actions": [],
                            "mask": [],
                            "labels": [],
                            "imgs": {},
                            "count": 0,
                            "raw_count": 0,
                            "plot_indices": [],
                            "all_gt": [],
                        }
                    target = plot_by_stiffness[label_val]
                    target["all_gt"].append(actions[b_idx, 0].detach().cpu().numpy())
                    target["raw_count"] += 1
                    keep_for_plot = (target["raw_count"] - 1) % self.eval_plot_prediction_stride == 0
                    if keep_for_plot and target["count"] < plot_max_steps:
                        target["obs"].append(obs[b_idx : b_idx + 1].detach().cpu())
                        target["actions"].append(actions[b_idx : b_idx + 1].detach().cpu())
                        target["mask"].append(mask[b_idx : b_idx + 1].detach().cpu())
                        target["labels"].append(labels[b_idx : b_idx + 1].detach().cpu())
                        target["plot_indices"].append(target["raw_count"] - 1)
                        for cam_key, cam_tensor in imgs.items():
                            if cam_key not in target["imgs"]:
                                target["imgs"][cam_key] = []
                            target["imgs"][cam_key].append(cam_tensor[b_idx : b_idx + 1].detach().cpu())
                        target["count"] += 1

                if output_dict.get("logits") is not None:
                    logits = output_dict["logits"]
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == labels).float().mean().item()
                    accuracy_list.append(acc)

        if was_training:
            model.train()

        mean_prior_l1 = np.mean(prior_l1_losses)
        mean_posterior_l1 = np.mean(posterior_l1_losses)
        mean_posterior_kl = np.mean(posterior_kl_losses) if posterior_kl_losses else None
        ac_l2 = np.mean(action_l2)
        ac_lsig = np.mean(action_lsig)
        l2_per_pose_dim_mean = np.mean(np.stack(l2_per_pose_dim_all, axis=0), axis=0)
        chunk_mse_mean = np.mean(np.stack(chunk_mse_all, axis=0), axis=0)

        print(
            f"Step: {global_step}\tPrior L1: {mean_prior_l1:.4f}\tPosterior L1: {mean_posterior_l1:.4f}\tAction L2: {ac_l2:.3f}"
        )

        if wandb.run is not None:
            for i, v in enumerate(l2_per_pose_dim_mean):
                wandb.log({f"eval/pose_dim{i + 1}_l2": v}, step=global_step)

            log_dict = {
                "eval/task_loss": mean_prior_l1,
                "eval/prior_l1": mean_prior_l1,
                "eval/posterior_l1": mean_posterior_l1,
                "eval/action_l2": ac_l2,
                "eval/action_lsig": ac_lsig,
            }
            if mean_posterior_kl is not None:
                log_dict["eval/posterior_kl"] = mean_posterior_kl

            err_fig = _build_eval_error_figure(chunk_mse_mean, l2_per_pose_dim_mean)
            log_dict["eval/plot_error_summary"] = wandb.Image(err_fig)
            plt.close(err_fig)

            # Make a plot for gorund trouth of each stiffness level
            # gt_plot_dict = {
            #     label: np.asarray(target["all_gt"], dtype=np.float32)
            #     for label, target in plot_by_stiffness.items()
            #     if len(target["all_gt"]) > 1
            # }
            # gt_fig = _build_ground_truth_gains_figure(gt_plot_dict)
            # if gt_fig is not None:
            #     log_dict["eval/plot_ground_truth_all_gains"] = wandb.Image(gt_fig)
            #     plt.close(gt_fig)

            if hasattr(model, "get_actions_prior") and not getattr(model, "factr_baseline", False):
                for stiff_val in sorted(plot_by_stiffness.keys()):
                    target = plot_by_stiffness[stiff_val]
                    if target["count"] < 2:
                        continue

                    plot_obs = torch.cat(target["obs"], dim=0).to(eval_device)
                    plot_actions = torch.cat(target["actions"], dim=0).cpu().numpy()
                    plot_mask = torch.cat(target["mask"], dim=0).cpu().numpy()
                    plot_labels = torch.cat(target["labels"], dim=0).to(eval_device)
                    plot_imgs = {
                        cam_key: torch.cat(cam_batches, dim=0).to(eval_device)
                        for cam_key, cam_batches in target["imgs"].items()
                    }

                    sampled_actions = model.get_actions_prior(
                        imgs=plot_imgs,
                        obs=plot_obs,
                        class_labels=plot_labels,
                        sample=True,
                        num_samples=10,
                    )
                    if sampled_actions.ndim == 3:
                        sampled_actions = sampled_actions.unsqueeze(1)
                    sampled_actions = sampled_actions.detach().cpu().numpy()

                    traj_fig = _build_eval_trajectory_fan_figure(
                        true_action_chunks=plot_actions,
                        pred_action_chunks=sampled_actions,
                        mask_chunks=plot_mask,
                        max_steps=plot_max_steps,
                        stiffness_label=stiff_val,
                        source_time_index=target["plot_indices"],
                    )
                    if traj_fig is not None:
                        log_dict[f"eval/plot_example_gt_vs_pred_stiffness_{stiff_val}"] = wandb.Image(traj_fig)
                        plt.close(traj_fig)

            if accuracy_list:
                log_dict["eval/classification_accuracy"] = np.mean(accuracy_list)

            wandb.log(log_dict, step=global_step)
