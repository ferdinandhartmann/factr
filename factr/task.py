# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

import wandb
from factr.replay_buffer import IterableWrapper

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False


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
    if not _HAS_MPL:
        return None

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
        ax.plot(time_index, true_valid[:, dim], color="black", linewidth=1.7, label="ground truth" if dim == 0 else None)
        ax.plot(time_index, pred_valid[:, dim], color="#E41A1C", linewidth=1.4, alpha=0.9, label="prediction" if dim == 0 else None)
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
    if not _HAS_MPL:
        return None

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
    if not _HAS_MPL:
        return None

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
    stiff_str = "all" if stiffness_label is None else str(stiffness_label)
    fig.suptitle(
        f"Sampled Prior Trajectories vs Ground Truth | stiffness={stiff_str}",
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
        factr_baseline: bool = False,
        eval_plot_max_steps: int = 300,
        eval_plot_prediction_stride: int = 5,
        eval_plot_num_samples: int = 10,
    ):
        self.n_cams, self.obs_dim, self.ac_dim = n_cams, obs_dim, ac_dim
        self.train_loader = _build_data_loader(train_buffer, batch_size, num_workers, is_train=True)
        self.eval_plot_max_steps = int(eval_plot_max_steps)
        self.eval_plot_prediction_stride = max(1, int(eval_plot_prediction_stride))
        self.eval_plot_num_samples = max(1, int(eval_plot_num_samples))

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
        plot_samples = {"obs": [], "actions": [], "mask": [], "labels": [], "imgs": {}, "time_index": []}
        raw_eval_index = 0

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
                    if (
                        raw_eval_index % self.eval_plot_prediction_stride == 0
                        and len(plot_samples["obs"]) < self.eval_plot_max_steps
                    ):
                        plot_samples["obs"].append(obs[batch_idx : batch_idx + 1].detach().cpu())
                        plot_samples["actions"].append(actions[batch_idx : batch_idx + 1].detach().cpu())
                        plot_samples["mask"].append(mask[batch_idx : batch_idx + 1].detach().cpu())
                        plot_samples["labels"].append(labels[batch_idx : batch_idx + 1].detach().cpu())
                        plot_samples["time_index"].append(raw_eval_index)
                        for cam_key, cam_tensor in imgs.items():
                            if cam_key not in plot_samples["imgs"]:
                                plot_samples["imgs"][cam_key] = []
                            plot_samples["imgs"][cam_key].append(cam_tensor[batch_idx : batch_idx + 1].detach().cpu())
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

        print(
            f"Step: {global_step}\tPosterior L1: {mean_val_loss:.4f}\tPrior L1: {mean_prior_l1:.4f}\t"
            f"KL: {mean_posterior_kl:.4f}\tAction L2: {ac_l2:.3f}\tLSign: {ac_lsig:.4f}\t"
            f"prior_std: {mean_prior_std:.4f}\tpost_std: {mean_posterior_std:.4f}\t"
            f"prior_H: {mean_prior_entropy:.4f}\tpost_H: {mean_posterior_entropy:.4f}\t"
            f"plot_steps: {len(plot_samples['obs'])}\tplot_stride: {self.eval_plot_prediction_stride}\t"
            f"plot_samples: {self.eval_plot_num_samples}"
        )

        if wandb.run is not None:
            for i, v in enumerate(l2_per_joint_mean):
                wandb.log({f"eval/joint{i + 1}_l2": v}, step=global_step)
            if chunk_mse_mean is not None:
                for i, v in enumerate(chunk_mse_mean):
                    wandb.log({f"eval/chunk_step_{i + 1}_mse": float(v)}, step=global_step)

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

            wandb.log(log_dict, step=global_step)

            if _HAS_MPL and first_plot_sample is not None:
                fig = _build_eval_pose_figure(
                    first_plot_sample["true"],
                    first_plot_sample["pred"],
                    first_plot_sample["mask"],
                    title="Eval Example: Ground Truth vs Predicted Pose",
                )
                if fig is not None:
                    wandb.log({"eval/prediction_example": wandb.Image(fig)}, step=global_step)
                    plt.close(fig)

            if _HAS_MPL and chunk_mse_mean is not None:
                fig_err = _build_eval_error_figure(chunk_mse_mean, l2_per_joint_mean)
                if fig_err is not None:
                    wandb.log({"eval/error_summary": wandb.Image(fig_err)}, step=global_step)
                    plt.close(fig_err)

            if _HAS_MPL and len(plot_samples["obs"]) > 1:
                plot_obs = torch.cat(plot_samples["obs"], dim=0).to(trainer.device_id)
                plot_actions = torch.cat(plot_samples["actions"], dim=0)
                plot_mask = torch.cat(plot_samples["mask"], dim=0)
                plot_labels = torch.cat(plot_samples["labels"], dim=0).to(trainer.device_id)
                plot_imgs = {k: torch.cat(v, dim=0).to(trainer.device_id) for k, v in plot_samples["imgs"].items()}
                plot_time_idx = np.asarray(plot_samples["time_index"], dtype=np.int64)

                sampled_actions = self._sample_actions_for_plot(
                    model=model,
                    imgs=plot_imgs,
                    obs=plot_obs,
                    labels=plot_labels,
                    num_samples=self.eval_plot_num_samples,
                )
                sampled_actions = sampled_actions.detach().cpu().numpy()
                true_action_chunks = plot_actions.detach().cpu().numpy()
                mask_chunks = plot_mask.detach().cpu().numpy()
                label_arr = plot_labels.detach().cpu().numpy().reshape(-1)

                fig_fan_all = _build_eval_trajectory_fan_figure(
                    true_action_chunks=true_action_chunks,
                    pred_action_chunks=sampled_actions,
                    mask_chunks=mask_chunks,
                    max_steps=self.eval_plot_max_steps,
                    stiffness_label="all",
                    source_time_index=plot_time_idx,
                )
                if fig_fan_all is not None:
                    wandb.log({"eval/prior_fan_all": wandb.Image(fig_fan_all)}, step=global_step)
                    plt.close(fig_fan_all)

                for stiffness_label in sorted(np.unique(label_arr)):
                    idx = np.where(label_arr == stiffness_label)[0]
                    if idx.shape[0] < 2:
                        continue
                    fig_stiff = _build_eval_trajectory_fan_figure(
                        true_action_chunks=true_action_chunks[idx],
                        pred_action_chunks=sampled_actions[idx],
                        mask_chunks=mask_chunks[idx],
                        max_steps=self.eval_plot_max_steps,
                        stiffness_label=int(stiffness_label),
                        source_time_index=plot_time_idx[idx],
                    )
                    if fig_stiff is not None:
                        wandb.log({f"eval/prior_fan_stiffness_{int(stiffness_label)}": wandb.Image(fig_stiff)}, step=global_step)
                        plt.close(fig_stiff)
