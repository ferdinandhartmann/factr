import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

import wandb
from factr.replay_buffer import IterableWrapper

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 150


def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def _build_data_loader(buffer, batch_size, num_workers, is_train=False):
    if is_train and not isinstance(buffer, IterableDataset):
        buffer = IterableWrapper(buffer)

    return DataLoader(
        buffer,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not isinstance(buffer, IterableDataset),
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=is_train,
        worker_init_fn=_seed_worker,
    )


def _make_obs_dim_names(dim):
    names = [
        "pose_x",
        "pose_y",
        "pose_z",
        "pose_r1",
        "pose_r2",
        "pose_r3",
        "pose_r4",
        "pose_r5",
        "pose_r6",
        "vel_x",
        "vel_y",
        "vel_z",
        "vel_rx",
        "vel_ry",
        "vel_rz",
        "wrench_fx",
        "wrench_fy",
        "wrench_fz",
        "wrench_tx",
        "wrench_ty",
        "wrench_tz",
    ]
    if dim <= len(names):
        return names[:dim]
    return [f"obs_{idx + 1}" for idx in range(dim)]


def _build_prediction_figure(true_obs, pred_mean, pred_std, max_dims):
    n_dims = min(int(max_dims), true_obs.shape[1])
    if n_dims <= 0:
        return None

    n_cols = min(3, n_dims)
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.8 * n_rows), sharex=True)
    axes = np.array(axes).reshape(-1)
    time_axis = np.arange(true_obs.shape[0])
    dim_names = _make_obs_dim_names(true_obs.shape[1])

    for dim in range(n_dims):
        ax = axes[dim]
        ax.plot(time_axis, true_obs[:, dim], color="black", linewidth=1.3, label="ground truth" if dim == 0 else None)
        ax.plot(time_axis, pred_mean[:, dim], color="#E41A1C", linewidth=1.2, label="pred mean" if dim == 0 else None)
        lower = pred_mean[:, dim] - pred_std[:, dim]
        upper = pred_mean[:, dim] + pred_std[:, dim]
        ax.fill_between(time_axis, lower, upper, color="#FB9A99", alpha=0.3, label="pred std" if dim == 0 else None)
        ax.set_title(dim_names[dim])
        ax.grid(alpha=0.25)

    for ax in axes[n_dims:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)
    fig.suptitle("Observation Prediction (mean ± std)", fontsize=12)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    return fig


class ObsPredictionTask:
    def __init__(
        self,
        train_buffer,
        test_buffer,
        obs_dim,
        pose_action_dim,
        batch_size,
        num_workers,
        pred_horizon=30,
        eval_plot_max_steps=400,
        eval_plot_dims=9,
    ):
        self.obs_dim = int(obs_dim)
        self.pose_action_dim = int(pose_action_dim)
        self.pred_horizon = int(pred_horizon)
        self.eval_plot_max_steps = int(eval_plot_max_steps)
        self.eval_plot_dims = int(eval_plot_dims)

        self.train_loader = _build_data_loader(train_buffer, batch_size, num_workers, is_train=True)
        self.test_loader = DataLoader(
            test_buffer,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
        )

    def eval(self, trainer, global_step, generate_plots=True):
        sample_mse_vals = []
        mean_mse_vals = []
        pred_var_vals = []
        tracking_err_vals = []

        first_plot_chunk = None

        model_ref = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        was_training = model_ref.training
        model_ref.eval()

        with torch.no_grad():
            for batch in self.test_loader:
                obs_window, action_chunk, target_obs, target_mask, stiffness_labels, goal_labels = batch
                obs_window = obs_window.to(trainer.device_id)
                action_chunk = action_chunk.to(trainer.device_id)
                target_obs = target_obs.to(trainer.device_id)
                target_mask = target_mask.to(trainer.device_id)
                stiffness_labels = stiffness_labels.to(trainer.device_id)
                goal_labels = goal_labels.to(trainer.device_id)

                output = model_ref(
                    obs_window=obs_window,
                    action_chunk=action_chunk,
                    stiffness_labels=stiffness_labels,
                    goal_labels=goal_labels,
                    target_obs=target_obs,
                    target_mask=target_mask,
                )
                tracking_error = model_ref.compute_tracking_error(output["mean"], action_chunk)
                tracking_l2 = torch.linalg.norm(tracking_error, dim=-1)
                tracking_l2 = (tracking_l2 * target_mask).sum(dim=1) / torch.clamp(target_mask.sum(dim=1), min=1.0)

                sample_mse_vals.append(output["sample_mse"].item())
                mean_mse_vals.append(output["mean_mse"].item())
                pred_var_vals.append(output["var"].mean().item())
                tracking_err_vals.append(tracking_l2.mean().item())

                if generate_plots and first_plot_chunk is None:
                    first_plot_chunk = {
                        "true": target_obs[0].detach().cpu().numpy(),
                        "pred_mean": output["mean"][0].detach().cpu().numpy(),
                        "pred_std": output["std"][0].detach().cpu().numpy(),
                        "mask": target_mask[0].detach().cpu().numpy(),
                    }

        if was_training:
            model_ref.train()

        mean_sample_mse = float(np.mean(sample_mse_vals)) if len(sample_mse_vals) > 0 else float("nan")
        mean_mean_mse = float(np.mean(mean_mse_vals)) if len(mean_mse_vals) > 0 else float("nan")
        mean_pred_var = float(np.mean(pred_var_vals)) if len(pred_var_vals) > 0 else float("nan")
        mean_tracking_l2 = float(np.mean(tracking_err_vals)) if len(tracking_err_vals) > 0 else float("nan")

        print(
            f"Step: {global_step}\tEval sample_mse: {mean_sample_mse:.5f}\t"
            f"Eval mean_mse: {mean_mean_mse:.5f}\tPred var: {mean_pred_var:.5f}\t"
            f"Tracking L2: {mean_tracking_l2:.5f}"
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "eval/sample_mse": mean_sample_mse,
                    "eval/mean_mse": mean_mean_mse,
                    "eval/pred_var_mean": mean_pred_var,
                    "eval/tracking_error_l2": mean_tracking_l2,
                },
                step=global_step,
            )

            if generate_plots and first_plot_chunk is not None:
                valid = first_plot_chunk["mask"] > 0
                if np.any(valid):
                    true_obs = first_plot_chunk["true"][valid]
                    pred_mean = first_plot_chunk["pred_mean"][valid]
                    pred_std = first_plot_chunk["pred_std"][valid]

                    fig_obs = _build_prediction_figure(
                        true_obs=true_obs,
                        pred_mean=pred_mean,
                        pred_std=pred_std,
                        max_dims=self.eval_plot_dims,
                    )
                    if fig_obs is not None:
                        wandb.log({"eval/obs_prediction_plot": wandb.Image(fig_obs)}, step=global_step)
                        plt.close(fig_obs)
