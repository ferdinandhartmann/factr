import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

import wandb
from factr.goal_inference import GOAL_COLORS
from factr.obs_pred_plot_utils import build_obs_prediction_plot, collapse_obs_prediction_chunks
from factr.plot_utils import RPYPlotConfig, build_pose_fan_figure
from factr.replay_buffer import IterableWrapper

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 150
RPY_SUBTRACT_PI = True
RPY_SUBTRACT_PI_AXIS = 0  # 0=roll, 1=pitch, 2=yaw
RPY_PLOT_UNIT = "deg"  # "rad" or "deg"


def _build_eval_obs_fan_figure(
    true_obs_chunks,
    pred_obs_chunks,
    mask_chunks,
    source_time_index,
    max_steps,
    stiffness_label=None,
    global_step=None,
):
    if true_obs_chunks.shape[0] == 0:
        return None

    pose_dim = int(min(true_obs_chunks.shape[-1], pred_obs_chunks.shape[-1]))
    if pose_dim <= 0:
        return None

    mask_arr = np.asarray(mask_chunks, dtype=np.float32)
    if mask_arr.ndim == 2:
        mask_arr = np.repeat(mask_arr[:, :, None], pose_dim, axis=-1)
    elif mask_arr.ndim == 3:
        if mask_arr.shape[-1] < pose_dim:
            pad = np.repeat(mask_arr[:, :, -1:], pose_dim - mask_arr.shape[-1], axis=-1)
            mask_arr = np.concatenate([mask_arr, pad], axis=-1)
        else:
            mask_arr = mask_arr[:, :, :pose_dim]
    else:
        return None

    rpy_cfg = RPYPlotConfig(
        subtract_pi=bool(RPY_SUBTRACT_PI),
        subtract_pi_axis=int(RPY_SUBTRACT_PI_AXIS),
        unit=str(RPY_PLOT_UNIT),
    )
    return build_pose_fan_figure(
        true_action_chunks=np.asarray(true_obs_chunks, dtype=np.float32)[..., :pose_dim],
        pred_action_chunks=np.asarray(pred_obs_chunks, dtype=np.float32)[:, None, :, :pose_dim],
        mask_chunks=mask_arr,
        source_time_index=np.asarray(source_time_index, dtype=np.int64),
        prediction_stride=1,
        measured_pose=None,
        max_plot_steps=int(max_steps),
        stiffness_label=stiffness_label,
        global_step=global_step,
        plot_ground_truth_reconstructed=True,
        rpy_config=rpy_cfg,
    )


def _normalize_goal_labels_local(goal_labels, goal_classes):
    labels = np.asarray(goal_labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        return labels
    if np.min(labels) >= 1 and np.max(labels) <= int(goal_classes):
        return labels
    if np.min(labels) >= 0 and np.max(labels) < int(goal_classes):
        return labels + 1
    return np.clip(labels, 1, int(goal_classes))


def _extract_goal_label_from_candidate(candidate, goal_classes):
    goal_arr = candidate.get("goal_labels", None)
    if goal_arr is None:
        return None
    goal_np = np.asarray(goal_arr, dtype=np.int64).reshape(-1)
    if goal_np.size == 0:
        return None
    label = int(goal_np[0])
    if 1 <= label <= int(goal_classes):
        return label
    if 0 <= label < int(goal_classes):
        return label + 1
    return int(np.clip(label, 1, int(goal_classes)))


def _select_goal_examples_candidates(candidates, goal_classes, max_steps_per_episode):
    if len(candidates) == 0:
        return []

    episode_to_items = {}
    episode_order = []
    for item in candidates:
        ep_id = int(item["episode_id"])
        if ep_id not in episode_to_items:
            episode_to_items[ep_id] = []
            episode_order.append(ep_id)
        episode_to_items[ep_id].append(item)

    goal_to_episode = {}
    for ep_id in episode_order:
        ep_items = sorted(episode_to_items[ep_id], key=lambda v: int(v["episode_step"]))
        labels = []
        for item in ep_items:
            label = _extract_goal_label_from_candidate(item, goal_classes=goal_classes)
            if label is not None:
                labels.append(int(label))
        if len(labels) == 0:
            continue
        vals, cnts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
        ep_goal = int(vals[int(np.argmax(cnts))])
        if ep_goal not in goal_to_episode:
            goal_to_episode[ep_goal] = ep_id

    selected = []
    timeline_cursor = 0
    for goal_label in range(1, int(goal_classes) + 1):
        ep_id = goal_to_episode.get(goal_label, None)
        if ep_id is None:
            continue

        ep_items = sorted(episode_to_items[ep_id], key=lambda v: int(v["episode_step"]))
        for item in ep_items:
            if int(item["episode_step"]) >= int(max_steps_per_episode):
                continue
            out = dict(item)
            out["plot_time_index"] = int(timeline_cursor + int(item["episode_step"]))
            selected.append(out)

        episode_length = max(1, max(int(v["episode_length"]) for v in ep_items))
        timeline_cursor += int(episode_length)

    return selected


def _build_goal_examples_by_label_figure(
    goal_probabilities,
    true_goal_labels,
    pred_goal_labels,
    time_index,
    episode_ids,
    episode_steps,
    obs_anchor_pairs,
):
    probs = np.asarray(goal_probabilities, dtype=np.float32)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return None

    time_steps, goal_classes = probs.shape
    x = np.asarray(time_index, dtype=np.int64).reshape(-1)
    ep_ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    ep_steps = np.asarray(episode_steps, dtype=np.int64).reshape(-1)
    if x.shape[0] != time_steps or ep_ids.shape[0] != time_steps or ep_steps.shape[0] != time_steps:
        return None

    true_labels = _normalize_goal_labels_local(true_goal_labels, goal_classes=goal_classes)
    pred_labels = _normalize_goal_labels_local(pred_goal_labels, goal_classes=goal_classes)
    if true_labels.shape[0] != time_steps or pred_labels.shape[0] != time_steps:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = np.asarray(axes).reshape(-1)

    added_anchor_label = False
    for goal_label in range(1, 5):
        ax = axes[goal_label - 1]
        if goal_label > goal_classes:
            ax.axis("off")
            continue

        goal_hits = np.where(true_labels == goal_label)[0]
        if goal_hits.size == 0:
            ax.set_title(f"True goal {goal_label} | no episode in window")
            ax.set_ylim(-0.1, 1.1)
            ax.grid(alpha=0.25)
            continue

        chosen_episode = int(ep_ids[goal_hits[0]])
        ep_idx = np.where(ep_ids == chosen_episode)[0]
        ep_idx = ep_idx[np.argsort(x[ep_idx])]
        if ep_idx.size == 0:
            ax.set_title(f"True goal {goal_label} | empty selection")
            ax.set_ylim(-0.1, 1.1)
            ax.grid(alpha=0.25)
            continue

        x_ep = x[ep_idx]
        probs_ep = probs[ep_idx]
        true_ep = true_labels[ep_idx]
        pred_ep = pred_labels[ep_idx]
        steps_ep = ep_steps[ep_idx]

        seg_start = 0
        seg_goal = int(true_ep[0])
        for t in range(1, ep_idx.size + 1):
            boundary = t == ep_idx.size or int(true_ep[t]) != seg_goal
            if boundary:
                x0 = float(x_ep[seg_start]) - 0.5
                x1 = float(x_ep[t - 1]) + 0.5
                bg_color = GOAL_COLORS[(seg_goal - 1) % len(GOAL_COLORS)]
                ax.axvspan(x0, x1, color=bg_color, alpha=0.06, lw=0.0)
                if t < ep_idx.size:
                    seg_start = t
                    seg_goal = int(true_ep[t])

        for goal_idx in range(goal_classes):
            color = GOAL_COLORS[goal_idx % len(GOAL_COLORS)]
            ax.plot(
                x_ep,
                probs_ep[:, goal_idx],
                color=color,
                linewidth=1.0,
                alpha=0.9,
                label=f"P(goal={goal_idx + 1})" if goal_label == 1 else None,
            )

        anchor_mask = np.asarray(
            [(chosen_episode, int(step_val)) in obs_anchor_pairs for step_val in steps_ep],
            dtype=bool,
        )
        is_connected = bool(np.any(anchor_mask))
        if is_connected:
            anchor_label = "obs-pred anchor" if not added_anchor_label else None
            ax.scatter(
                x_ep[anchor_mask],
                np.full(int(np.sum(anchor_mask)), 1.04, dtype=np.float32),
                marker="v",
                s=20,
                color="black",
                label=anchor_label,
                clip_on=False,
                zorder=20,
            )
            added_anchor_label = True

        acc = float(np.mean(pred_ep == true_ep)) if pred_ep.size > 0 else float("nan")
        conn_str = "connected to obs plot" if is_connected else "not connected"
        ax.set_title(f"True goal {goal_label} | ep {chosen_episode} | {conn_str}")
        ax.text(0.01, 0.02, f"acc={acc:.2f}", transform=ax.transAxes, fontsize=8)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(alpha=0.25)
        ax.set_xlabel("episode timestep")
        if goal_label in (1, 3):
            ax.set_ylabel("probability")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(goal_classes + 1, 5),
            frameon=False,
        )

    fig.suptitle("Goal Inference by True Goal (1,2,3,4)", fontsize=12)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    return fig


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


def _stack_obs_plot_candidates(candidates, device):
    if len(candidates) == 0:
        return None

    obs_window = torch.cat([item["obs_window"] for item in candidates], dim=0).to(device)
    action_chunk = torch.cat([item["action_chunk"] for item in candidates], dim=0).to(device)
    target_obs = torch.cat([item["target_obs"] for item in candidates], dim=0).to(device)
    target_mask = torch.cat([item["target_mask"] for item in candidates], dim=0).to(device)
    stiffness_labels = torch.cat([item["stiffness_labels"] for item in candidates], dim=0).to(device)
    goal_labels = torch.cat([item["goal_labels"] for item in candidates], dim=0).to(device)
    time_index = np.asarray([int(item["plot_time_index"]) for item in candidates], dtype=np.int64)
    episode_ids = np.asarray([int(item["episode_id"]) for item in candidates], dtype=np.int64)
    episode_steps = np.asarray([int(item["episode_step"]) for item in candidates], dtype=np.int64)

    return {
        "obs_window": obs_window,
        "action_chunk": action_chunk,
        "target_obs": target_obs,
        "target_mask": target_mask,
        "stiffness_labels": stiffness_labels,
        "goal_labels": goal_labels,
        "time_index": time_index,
        "episode_ids": episode_ids,
        "episode_steps": episode_steps,
    }


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
        eval_plot_prediction_stride=10,
        eval_plot_dims=9,
    ):
        self.obs_dim = int(obs_dim)
        self.pose_action_dim = int(pose_action_dim)
        self.pred_horizon = int(pred_horizon)
        self.eval_plot_max_steps = int(eval_plot_max_steps)
        self.eval_plot_prediction_stride = max(1, int(eval_plot_prediction_stride))
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
        nll_per_elem_vals = []
        pred_var_vals = []
        tracking_err_vals = []
        goal_acc_vals = []
        goal_loglik_vals = []
        goal_loglik_per_elem_vals = []
        goal_loglik_margin_vals = []
        goal_entropy_vals = []
        goal_prob_sum_err_vals = []

        obs_plot_candidates = [] if generate_plots else None
        goal_plot_candidates = [] if generate_plots else None
        raw_eval_index = 0
        test_dataset = getattr(self.test_loader, "dataset", None) if generate_plots else None

        model_ref = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        goal_classes = int(getattr(model_ref, "goal_classes", 4))
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
                nll_per_elem_vals.append(output["nll_per_elem"].item())
                pred_var_vals.append(output["var"].mean().item())
                tracking_err_vals.append(tracking_l2.mean().item())

                if hasattr(model_ref, "infer_goals"):
                    goal_eval = model_ref.infer_goals(
                        obs_window=obs_window,
                        action_chunk=action_chunk,
                        stiffness_labels=stiffness_labels,
                        target_obs=target_obs,
                        target_mask=target_mask,
                        num_goal_samples=1,
                    )
                    valid_elements_per_sample = torch.clamp(
                        target_mask.sum(dim=1) * float(target_obs.shape[-1]),
                        min=1.0,
                    )
                    goal_loglik_all = goal_eval["log_likelihood_per_goal"]
                    true_goal_idx = model_ref.normalize_goal_labels(
                        goal_labels,
                        batch_size=goal_labels.shape[0],
                        device=goal_labels.device,
                    )
                    pred_goal_idx = torch.argmax(goal_eval["goal_posterior"], dim=-1)
                    goal_acc = (pred_goal_idx == true_goal_idx).float().mean()
                    true_goal_loglik = goal_loglik_all.gather(1, true_goal_idx.unsqueeze(-1)).squeeze(-1)
                    true_goal_loglik_per_elem = true_goal_loglik / valid_elements_per_sample
                    if goal_loglik_all.shape[1] > 1:
                        competing_goal_loglik = goal_loglik_all.clone()
                        competing_goal_loglik.scatter_(1, true_goal_idx.unsqueeze(-1), float("-inf"))
                        goal_loglik_margin = true_goal_loglik - competing_goal_loglik.max(dim=1).values
                    else:
                        goal_loglik_margin = torch.zeros_like(true_goal_loglik)
                    goal_entropy = -torch.sum(
                        goal_eval["goal_posterior"] * torch.log(goal_eval["goal_posterior"].clamp_min(1e-8)),
                        dim=-1,
                    ).mean()
                    goal_prob_sum_err = torch.abs(goal_eval["goal_posterior"].sum(dim=-1) - 1.0).mean()

                    goal_acc_vals.append(goal_acc.item())
                    goal_loglik_vals.append(true_goal_loglik.mean().item())
                    goal_loglik_per_elem_vals.append(true_goal_loglik_per_elem.mean().item())
                    goal_loglik_margin_vals.append(goal_loglik_margin.mean().item())
                    goal_entropy_vals.append(goal_entropy.item())
                    goal_prob_sum_err_vals.append(goal_prob_sum_err.item())

                if generate_plots:
                    for batch_idx in range(target_obs.shape[0]):
                        fallback_label = int(stiffness_labels[batch_idx].detach().cpu().item())
                        meta = _extract_plot_metadata(
                            dataset=test_dataset,
                            sample_index=raw_eval_index,
                            fallback_label=fallback_label,
                        )

                        keep_step = meta["episode_step"] % self.eval_plot_prediction_stride == 0
                        within_step_limit = meta["episode_step"] < self.eval_plot_max_steps
                        if within_step_limit:
                            candidate = {
                                "obs_window": obs_window[batch_idx : batch_idx + 1].detach().cpu(),
                                "action_chunk": action_chunk[batch_idx : batch_idx + 1].detach().cpu(),
                                "target_obs": target_obs[batch_idx : batch_idx + 1].detach().cpu(),
                                "target_mask": target_mask[batch_idx : batch_idx + 1].detach().cpu(),
                                "stiffness_labels": stiffness_labels[batch_idx : batch_idx + 1].detach().cpu(),
                                "goal_labels": goal_labels[batch_idx : batch_idx + 1].detach().cpu(),
                                "episode_id": int(meta["episode_id"]),
                                "episode_step": int(meta["episode_step"]),
                                "episode_length": int(meta["episode_length"]),
                                "stiffness_label": int(meta["stiffness_label"]),
                            }
                            goal_plot_candidates.append(candidate)
                            if keep_step:
                                obs_plot_candidates.append(candidate)
                        raw_eval_index += 1

        if generate_plots:
            selected_plot_candidates = _select_episode_plot_candidates(
                obs_plot_candidates,
                max_steps=self.eval_plot_max_steps,
            )
            selected_goal_plot_candidates = _select_episode_plot_candidates(
                goal_plot_candidates,
                max_steps=self.eval_plot_max_steps,
            )
            selected_goal_by_label_candidates = _select_goal_examples_candidates(
                goal_plot_candidates,
                goal_classes=goal_classes,
                max_steps_per_episode=self.eval_plot_max_steps,
            )
        else:
            selected_plot_candidates = []
            selected_goal_plot_candidates = []
            selected_goal_by_label_candidates = []

        plot_stiffness_str = "unknown"
        if len(selected_plot_candidates) > 0:
            stiffness_arr = np.asarray(
                [int(item["stiffness_label"]) for item in selected_plot_candidates],
                dtype=np.int64,
            )
            unique_stiffness = sorted(np.unique(stiffness_arr).tolist())
            if len(unique_stiffness) == 1:
                plot_stiffness_str = str(int(unique_stiffness[0]))
            else:
                counts = {int(label): int(np.sum(stiffness_arr == int(label))) for label in unique_stiffness}
                counts_str = ",".join([f"{k}:{v}" for k, v in counts.items()])
                plot_stiffness_str = f"mixed({counts_str})"

        mean_sample_mse = float(np.mean(sample_mse_vals)) if len(sample_mse_vals) > 0 else float("nan")
        mean_mean_mse = float(np.mean(mean_mse_vals)) if len(mean_mse_vals) > 0 else float("nan")
        mean_nll_per_elem = float(np.mean(nll_per_elem_vals)) if len(nll_per_elem_vals) > 0 else float("nan")
        mean_pred_var = float(np.mean(pred_var_vals)) if len(pred_var_vals) > 0 else float("nan")
        mean_tracking_l2 = float(np.mean(tracking_err_vals)) if len(tracking_err_vals) > 0 else float("nan")
        mean_goal_acc = float(np.mean(goal_acc_vals)) if len(goal_acc_vals) > 0 else float("nan")
        mean_goal_loglik = float(np.mean(goal_loglik_vals)) if len(goal_loglik_vals) > 0 else float("nan")
        mean_goal_loglik_per_elem = (
            float(np.mean(goal_loglik_per_elem_vals)) if len(goal_loglik_per_elem_vals) > 0 else float("nan")
        )
        mean_goal_loglik_margin = (
            float(np.mean(goal_loglik_margin_vals)) if len(goal_loglik_margin_vals) > 0 else float("nan")
        )
        mean_goal_entropy = float(np.mean(goal_entropy_vals)) if len(goal_entropy_vals) > 0 else float("nan")
        mean_goal_prob_sum_err = (
            float(np.mean(goal_prob_sum_err_vals)) if len(goal_prob_sum_err_vals) > 0 else float("nan")
        )

        print(
            f"Step: {global_step}\tEval sample_mse: {mean_sample_mse:.5f}\t"
            f"Eval mean_mse: {mean_mean_mse:.5f}\tNLL/elem: {mean_nll_per_elem:.5f}\tPred var: {mean_pred_var:.5f}\t"
            f"Tracking L2: {mean_tracking_l2:.5f}\tGoal acc: {mean_goal_acc:.5f}\t"
            f"Goal loglik: {mean_goal_loglik:.5f}\tGoal loglik/elem: {mean_goal_loglik_per_elem:.5f}\t"
            f"Goal margin: {mean_goal_loglik_margin:.5f}\tGoal H: {mean_goal_entropy:.5f}\t"
            f"Goal prob sum err: {mean_goal_prob_sum_err:.3e}\t"
            f"plot_anchors: {len(selected_plot_candidates)}\t"
            f"plot_stride: {self.eval_plot_prediction_stride}"
        )

        if wandb.run is not None:
            wandb.log(
                {
                    "eval/sample_mse": mean_sample_mse,
                    "eval/mean_mse": mean_mean_mse,
                    "eval/nll_per_elem": mean_nll_per_elem,
                    "eval/pred_var_mean": mean_pred_var,
                    "eval/tracking_error_l2": mean_tracking_l2,
                    "eval/goal_inference_acc": mean_goal_acc,
                    "eval/goal_true_loglik": mean_goal_loglik,
                    "eval/goal_true_loglik_per_elem": mean_goal_loglik_per_elem,
                    "eval/goal_loglik_margin": mean_goal_loglik_margin,
                    "eval/goal_posterior_entropy": mean_goal_entropy,
                    "eval/goal_prob_sum_error": mean_goal_prob_sum_err,
                },
                step=global_step,
            )

            if generate_plots:
                obs_anchor_pairs = set()

                if len(selected_plot_candidates) > 0:
                    plot_bundle = _stack_obs_plot_candidates(selected_plot_candidates, device=trainer.device_id)
                    if plot_bundle is not None:
                        with torch.no_grad():
                            plot_output = model_ref(
                                obs_window=plot_bundle["obs_window"],
                                action_chunk=plot_bundle["action_chunk"],
                                stiffness_labels=plot_bundle["stiffness_labels"],
                                goal_labels=plot_bundle["goal_labels"],
                                target_obs=plot_bundle["target_obs"],
                                target_mask=plot_bundle["target_mask"],
                            )

                        pred_sample = plot_output.get("sample", plot_output["mean"])
                        target_chunks = plot_bundle["target_obs"].detach().cpu().numpy()
                        pred_mean_chunks = plot_output["mean"].detach().cpu().numpy()
                        pred_std_chunks = plot_output["std"].detach().cpu().numpy()
                        pred_sample_chunks = pred_sample.detach().cpu().numpy()
                        target_mask_chunks = plot_bundle["target_mask"].detach().cpu().numpy()
                        anchor_steps = np.asarray(plot_bundle["time_index"], dtype=np.int64)

                        full_horizon_valid = np.sum(target_mask_chunks > 0.0, axis=1) >= int(self.pred_horizon)
                        inference_indices = np.where(full_horizon_valid)[0]
                        if inference_indices.size == 0 and target_chunks.shape[0] > 0:
                            inference_indices = np.asarray([0], dtype=np.int64)

                        target_plot = target_chunks[inference_indices]
                        pred_plot = pred_mean_chunks[inference_indices]
                        mask_plot = target_mask_chunks[inference_indices]
                        source_plot = anchor_steps[inference_indices]
                        anchor_episode_ids = plot_bundle["episode_ids"][inference_indices]
                        anchor_episode_steps = plot_bundle["episode_steps"][inference_indices]
                        obs_anchor_pairs = {
                            (int(ep_id), int(ep_step))
                            for ep_id, ep_step in zip(anchor_episode_ids.tolist(), anchor_episode_steps.tolist())
                        }

                        fig_obs_fan = _build_eval_obs_fan_figure(
                            true_obs_chunks=target_plot,
                            pred_obs_chunks=pred_plot,
                            mask_chunks=mask_plot,
                            source_time_index=source_plot,
                            max_steps=self.eval_plot_max_steps,
                            stiffness_label=plot_stiffness_str,
                            global_step=global_step,
                        )
                        if fig_obs_fan is not None:
                            wandb.log({"eval/obs_prediction_plot": wandb.Image(fig_obs_fan)}, step=global_step)
                            plt.close(fig_obs_fan)

                        # Keep a compact denoised summary plot for mean+-std over merged timeline.
                        collapsed = None
                        if target_plot.shape[0] > 0:
                            collapsed = collapse_obs_prediction_chunks(
                                true_chunks=target_plot,
                                pred_mean_chunks=pred_plot,
                                pred_std_chunks=pred_std_chunks[inference_indices],
                                pred_sample_chunks=pred_sample_chunks[inference_indices],
                                valid_mask=mask_plot,
                                anchor_steps=source_plot,
                                stride=1,
                            )
                        if collapsed is not None:
                            fig_obs_summary = build_obs_prediction_plot(
                                true_obs=collapsed["true"],
                                pred_mean=collapsed["pred_mean"],
                                pred_std=collapsed["pred_std"],
                                max_dims=int(collapsed["true"].shape[1]),
                                time_index=collapsed["time_index"],
                                pred_sample=collapsed["pred_sample"],
                                title=f"Observation Prediction (collapsed) | stiffness={plot_stiffness_str}",
                                rpy_subtract_pi=bool(RPY_SUBTRACT_PI),
                                rpy_subtract_pi_axis=int(RPY_SUBTRACT_PI_AXIS),
                                rpy_plot_unit=str(RPY_PLOT_UNIT),
                            )
                            if fig_obs_summary is not None:
                                wandb.log(
                                    {"eval/obs_prediction_plot_summary": wandb.Image(fig_obs_summary)}, step=global_step
                                )
                                plt.close(fig_obs_summary)

                # if hasattr(model_ref, "infer_goals") and len(selected_goal_plot_candidates) > 0:
                #     goal_plot_bundle = _stack_obs_plot_candidates(
                #         selected_goal_plot_candidates, device=trainer.device_id
                #     )
                #     if goal_plot_bundle is not None:
                #         with torch.no_grad():
                #             goal_plot_output = model_ref.infer_goals(
                #                 obs_window=goal_plot_bundle["obs_window"],
                #                 action_chunk=goal_plot_bundle["action_chunk"],
                #                 stiffness_labels=goal_plot_bundle["stiffness_labels"],
                #                 target_obs=goal_plot_bundle["target_obs"],
                #                 target_mask=goal_plot_bundle["target_mask"],
                #                 num_goal_samples=1,
                #             )

                #         episode_goal_probs = goal_plot_output["goal_posterior"].detach().cpu().numpy()
                #         pred_goal_labels_episode = np.argmax(episode_goal_probs, axis=-1).astype(np.int64) + 1
                #         true_goal_labels_episode = goal_plot_bundle["goal_labels"].detach().cpu().numpy().reshape(-1)
                #         goal_time_index = goal_plot_bundle["time_index"]

                #         fig_goal, _ = build_episode_goal_probability_figure(
                #             goal_probabilities=episode_goal_probs,
                #             true_goal_labels=true_goal_labels_episode,
                #             pred_goal_labels=pred_goal_labels_episode,
                #             time_index=goal_time_index,
                #             title_prefix="Goal Posterior over Episode (exact eval posterior, stride=1)",
                #         )
                #         wandb.log({"eval/goal_likelihood_plot": wandb.Image(fig_goal)}, step=global_step)
                #         plt.close(fig_goal)

                if hasattr(model_ref, "infer_goals") and len(selected_goal_by_label_candidates) > 0:
                    goal_by_label_bundle = _stack_obs_plot_candidates(
                        selected_goal_by_label_candidates,
                        device=trainer.device_id,
                    )
                    if goal_by_label_bundle is not None:
                        with torch.no_grad():
                            goal_by_label_output = model_ref.infer_goals(
                                obs_window=goal_by_label_bundle["obs_window"],
                                action_chunk=goal_by_label_bundle["action_chunk"],
                                stiffness_labels=goal_by_label_bundle["stiffness_labels"],
                                target_obs=goal_by_label_bundle["target_obs"],
                                target_mask=goal_by_label_bundle["target_mask"],
                                num_goal_samples=1,
                            )

                        probs_by_label = goal_by_label_output["goal_posterior"].detach().cpu().numpy()
                        pred_labels_by_label = np.argmax(probs_by_label, axis=-1).astype(np.int64) + 1
                        true_labels_by_label = goal_by_label_bundle["goal_labels"].detach().cpu().numpy().reshape(-1)

                        fig_goal_examples = _build_goal_examples_by_label_figure(
                            goal_probabilities=probs_by_label,
                            true_goal_labels=true_labels_by_label,
                            pred_goal_labels=pred_labels_by_label,
                            time_index=goal_by_label_bundle["time_index"],
                            episode_ids=goal_by_label_bundle["episode_ids"],
                            episode_steps=goal_by_label_bundle["episode_steps"],
                            obs_anchor_pairs=obs_anchor_pairs,
                        )
                        if fig_goal_examples is not None:
                            wandb.log(
                                {"eval/goal_likelihood_plot_by_goal": wandb.Image(fig_goal_examples)}, step=global_step
                            )
                            plt.close(fig_goal_examples)

        if was_training:
            model_ref.train()
