from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

GOAL_COLORS = ("#E41A1C", "#4DAF4A", "#377EB8", "#FF7F00")


def build_display_goal_posterior_over_time(
    log_likelihood_over_time,
    obs_dim=None,
    temperature=5.0,
    include_uniform_prior=True,
):
    """Build a visualization-friendly goal posterior trace from per-step log-likelihoods.

    The exact posterior for this model often becomes numerically sharp after the first
    predicted timestep because each step sums log-likelihood over all predicted dims.
    For plotting, we therefore average by obs dimension and apply a temperature so the
    curve still shows gradual belief updates.
    """
    log_likelihood_over_time = np.asarray(log_likelihood_over_time, dtype=np.float32)
    if log_likelihood_over_time.ndim != 2:
        raise ValueError(
            f"Expected log_likelihood_over_time with shape (T, G), got {tuple(log_likelihood_over_time.shape)}."
        )

    horizon, goal_classes = log_likelihood_over_time.shape
    if horizon == 0:
        return np.zeros((0, goal_classes), dtype=np.float32)

    obs_scale = float(obs_dim) if obs_dim is not None else 1.0
    scale = max(1.0, obs_scale * float(temperature))

    cumulative_logits = np.cumsum(log_likelihood_over_time / scale, axis=0)
    cumulative_logits -= np.max(cumulative_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(cumulative_logits)
    posterior = exp_logits / np.clip(np.sum(exp_logits, axis=-1, keepdims=True), 1e-8, None)

    if not include_uniform_prior:
        return posterior.astype(np.float32)

    uniform = np.full((1, goal_classes), 1.0 / float(goal_classes), dtype=np.float32)
    return np.concatenate([uniform, posterior.astype(np.float32)], axis=0)


def build_goal_likelihood_figure(
    log_likelihood_over_time,
    posterior_over_time=None,
    true_goal_label=None,
    pred_goal_label=None,
    title_prefix="Goal Inference",
):
    log_likelihood_over_time = np.asarray(log_likelihood_over_time, dtype=np.float32)
    if log_likelihood_over_time.ndim != 2:
        raise ValueError(
            f"Expected log_likelihood_over_time with shape (T, G), got {tuple(log_likelihood_over_time.shape)}."
        )

    horizon, goal_classes = log_likelihood_over_time.shape
    loglik_time_axis = np.arange(1, horizon + 1, dtype=np.int64)
    has_posterior = posterior_over_time is not None

    if has_posterior:
        posterior_over_time = np.asarray(posterior_over_time, dtype=np.float32)
        if posterior_over_time.ndim != 2:
            raise ValueError(
                f"Expected posterior_over_time shape (T, {goal_classes}), got {tuple(posterior_over_time.shape)}."
            )
        if posterior_over_time.shape[1] != goal_classes:
            raise ValueError(
                f"Expected posterior_over_time goal axis size {goal_classes}, got {posterior_over_time.shape[1]}."
            )
        if posterior_over_time.shape[0] == horizon:
            posterior_time_axis = np.arange(1, horizon + 1, dtype=np.int64)
        elif posterior_over_time.shape[0] == horizon + 1:
            posterior_time_axis = np.arange(0, horizon + 1, dtype=np.int64)
        else:
            raise ValueError(
                "Expected posterior_over_time time axis to have length "
                f"{horizon} or {horizon + 1}, got {posterior_over_time.shape[0]}."
            )

    if has_posterior:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.6, 4))
        axes = [ax]

    ax_ll = axes[0]
    for goal_idx in range(goal_classes):
        color = GOAL_COLORS[goal_idx % len(GOAL_COLORS)]
        ax_ll.plot(
            loglik_time_axis,
            log_likelihood_over_time[:, goal_idx],
            color=color,
            linewidth=1.0,
            label=f"goal {goal_idx + 1}",
        )
    ax_ll.set_title("Log-Likelihood per Timestamp")
    ax_ll.set_xlabel("horizon step")
    ax_ll.set_ylabel("log p(obs_t | goal)")
    ax_ll.grid(alpha=0.25)

    if has_posterior:
        ax_post = axes[1]
        for goal_idx in range(goal_classes):
            color = GOAL_COLORS[goal_idx % len(GOAL_COLORS)]
            ax_post.plot(
                posterior_time_axis,
                posterior_over_time[:, goal_idx],
                color=color,
                linewidth=1.5,
                label=f"goal {goal_idx + 1}",
            )
        ax_post.set_title("Posterior over Time")
        ax_post.set_xlabel("observed horizon step")
        ax_post.set_ylabel("p(goal | obs_1:t)")
        ax_post.set_ylim(-0.1, 1.1)
        ax_post.grid(alpha=0.25)

    handles, labels = ax_ll.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=min(goal_classes, 4), frameon=False
        )

    title = title_prefix
    if true_goal_label is not None or pred_goal_label is not None:
        title += f" | true={true_goal_label} pred={pred_goal_label}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    return fig


def _normalize_goal_labels(goal_labels, goal_classes):
    labels = np.asarray(goal_labels, dtype=np.int64).reshape(-1)
    if labels.size == 0:
        raise ValueError("goal_labels must not be empty.")
    if np.min(labels) >= 1 and np.max(labels) <= goal_classes:
        out = labels
    elif np.min(labels) >= 0 and np.max(labels) < goal_classes:
        out = labels + 1
    else:
        out = np.clip(labels, 1, goal_classes)
    return out.astype(np.int64)


def normalize_goal_probabilities(goal_probabilities):
    probs = np.asarray(goal_probabilities, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"Expected goal_probabilities shape (T, G), got {tuple(probs.shape)}.")
    sums_raw = np.sum(probs, axis=-1)
    abs_err_raw = np.abs(sums_raw - 1.0)
    denom = np.clip(sums_raw, 1e-8, None)
    probs_norm = probs / denom[:, None]
    return probs_norm, sums_raw, abs_err_raw


def build_episode_goal_probability_figure(
    goal_probabilities,
    true_goal_labels,
    pred_goal_labels=None,
    time_index=None,
    title_prefix="Goal Posterior Over Episode",
):
    probs, prob_sums, prob_abs_err = normalize_goal_probabilities(goal_probabilities)
    time_steps, goal_classes = probs.shape

    true_labels = _normalize_goal_labels(true_goal_labels, goal_classes)
    if true_labels.shape[0] != time_steps:
        raise ValueError(f"Expected {time_steps} true goal labels, got {true_labels.shape[0]}.")

    if pred_goal_labels is None:
        pred_labels = np.argmax(probs, axis=-1).astype(np.int64) + 1
    else:
        pred_labels = _normalize_goal_labels(pred_goal_labels, goal_classes)
        if pred_labels.shape[0] != time_steps:
            raise ValueError(f"Expected {time_steps} predicted goal labels, got {pred_labels.shape[0]}.")

    if time_index is None:
        x = np.arange(time_steps, dtype=np.int64)
    else:
        x = np.asarray(time_index, dtype=np.int64).reshape(-1)
        if x.shape[0] != time_steps:
            raise ValueError(f"Expected time_index length {time_steps}, got {x.shape[0]}.")

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.4))

    seg_start = 0
    seg_goal = int(true_labels[0])
    for t in range(1, time_steps + 1):
        boundary = t == time_steps or int(true_labels[t]) != seg_goal
        if boundary:
            x0 = float(x[seg_start]) - 0.5
            x1 = float(x[t - 1]) + 0.5
            bg_color = GOAL_COLORS[(seg_goal - 1) % len(GOAL_COLORS)]
            ax.axvspan(x0, x1, color=bg_color, alpha=0.06, lw=0.0)
            if t < time_steps:
                seg_start = t
                seg_goal = int(true_labels[t])

    for goal_idx in range(goal_classes):
        color = GOAL_COLORS[goal_idx % len(GOAL_COLORS)]
        ax.plot(
            x,
            probs[:, goal_idx],
            color=color,
            linewidth=1.0,
            alpha=0.9,
            label=f"P(goal={goal_idx + 1})",
        )

    correct_mask = pred_labels == true_labels
    correct_count = int(np.sum(correct_mask))
    accuracy = float(correct_count / max(1, int(time_steps)))

    ax.set_title("Goal Probability at Each Episode Timestep")
    ax.set_xlabel("episode timestep")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=min(goal_classes, 4), frameon=False)

    ax.text(
        0.01,
        0.99,
        f"accuracy: {accuracy * 100.0:.1f}% ({correct_count}/{int(time_steps)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#9E9E9E", "alpha": 0.85},
    )

    max_err = float(np.max(prob_abs_err))
    mean_err = float(np.mean(prob_abs_err))
    title = f"{title_prefix} | max|sum(p)-1|={max_err:.2e} mean={mean_err:.2e}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    payload = {
        "goal_probability": probs,
        "goal_probability_sum": prob_sums,
        "goal_probability_abs_error": prob_abs_err,
        "true_goal_label": true_labels,
        "pred_goal_label": pred_labels,
        "correct_mask": correct_mask,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": int(time_steps),
        "max_sum_error": max_err,
        "mean_sum_error": mean_err,
    }
    return fig, payload


def _extract_goal_label_from_candidate(candidate, goal_classes):
    goal_value = candidate.get("goal_label", None)
    if goal_value is None:
        goal_value = candidate.get("goal_labels", None)
    if goal_value is None:
        return None

    goal_np = np.asarray(goal_value, dtype=np.int64).reshape(-1)
    if goal_np.size == 0:
        return None

    label = int(goal_np[0])
    if 1 <= label <= int(goal_classes):
        return label
    if 0 <= label < int(goal_classes):
        return label + 1
    return int(np.clip(label, 1, int(goal_classes)))


def select_goal_examples_candidates(candidates, goal_classes, max_steps_per_episode: Optional[int] = None):
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
        ep_items = sorted(episode_to_items[ep_id], key=lambda value: int(value["episode_step"]))
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

        ep_items = sorted(episode_to_items[ep_id], key=lambda value: int(value["episode_step"]))
        for item in ep_items:
            if max_steps_per_episode is not None and int(item["episode_step"]) >= int(max_steps_per_episode):
                continue
            out = dict(item)
            out["plot_time_index"] = int(timeline_cursor + int(item["episode_step"]))
            selected.append(out)

        episode_length = max(1, max(int(value["episode_length"]) for value in ep_items))
        timeline_cursor += int(episode_length)

    return selected


def build_goal_examples_by_label_figure(
    goal_probabilities,
    true_goal_labels,
    pred_goal_labels,
    time_index,
    episode_ids,
    episode_steps,
    obs_anchor_pairs,
    episode_name_by_id=None,
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

    true_labels = _normalize_goal_labels(true_goal_labels, goal_classes)
    pred_labels = _normalize_goal_labels(pred_goal_labels, goal_classes)
    if true_labels.shape[0] != time_steps or pred_labels.shape[0] != time_steps:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = np.asarray(axes).reshape(-1)

    anchor_pairs = set() if obs_anchor_pairs is None else set(obs_anchor_pairs)
    added_anchor_label = False
    for goal_label in range(1, 5):
        ax = axes[goal_label - 1]
        if goal_label > goal_classes:
            ax.axis("off")
            continue

        goal_hits = np.where(true_labels == goal_label)[0]
        if goal_hits.size == 0:
            ax.set_title(f"True goal {goal_label} | no episode in selection")
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
        for idx in range(1, ep_idx.size + 1):
            boundary = idx == ep_idx.size or int(true_ep[idx]) != seg_goal
            if boundary:
                x0 = float(x_ep[seg_start]) - 0.5
                x1 = float(x_ep[idx - 1]) + 0.5
                bg_color = GOAL_COLORS[(seg_goal - 1) % len(GOAL_COLORS)]
                ax.axvspan(x0, x1, color=bg_color, alpha=0.06, lw=0.0)
                if idx < ep_idx.size:
                    seg_start = idx
                    seg_goal = int(true_ep[idx])

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
            [(chosen_episode, int(step_val)) in anchor_pairs for step_val in steps_ep],
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
        episode_name = str(chosen_episode)
        if isinstance(episode_name_by_id, dict):
            episode_name = str(episode_name_by_id.get(chosen_episode, episode_name))
        ax.set_title(f"True goal {goal_label} | {episode_name} | {conn_str}")
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
