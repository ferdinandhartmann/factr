import matplotlib.pyplot as plt
import numpy as np

GOAL_COLORS = ("#E41A1C", "#4DAF4A", "#377EB8", "#FF7F00")


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
            "Expected log_likelihood_over_time with shape (T, G), "
            f"got {tuple(log_likelihood_over_time.shape)}."
        )

    horizon, goal_classes = log_likelihood_over_time.shape
    time_axis = np.arange(horizon)
    has_posterior = posterior_over_time is not None

    if has_posterior:
        posterior_over_time = np.asarray(posterior_over_time, dtype=np.float32)
        if posterior_over_time.shape != (horizon, goal_classes):
            raise ValueError(
                "Expected posterior_over_time shape "
                f"({horizon}, {goal_classes}), got {tuple(posterior_over_time.shape)}."
            )

    if has_posterior:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.6, 4))
        axes = [ax]

    ax_ll = axes[0]
    for goal_idx in range(goal_classes):
        color = GOAL_COLORS[goal_idx % len(GOAL_COLORS)]
        ax_ll.plot(
            time_axis,
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
                time_axis,
                posterior_over_time[:, goal_idx],
                color=color,
                linewidth=1.0,
                label=f"goal {goal_idx + 1}",
            )
        ax_post.set_title("Posterior over Time")
        ax_post.set_xlabel("horizon step")
        ax_post.set_ylabel("p(goal | obs_1:t)")
        ax_post.set_ylim(0.0, 1.0)
        ax_post.grid(alpha=0.25)

    handles, labels = ax_ll.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=min(goal_classes, 4), frameon=False)

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

    ax.set_title("Goal Probability at Each Episode Timestep")
    ax.set_xlabel("episode timestep")
    ax.set_ylabel("probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=min(goal_classes, 4), frameon=False)

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
        "max_sum_error": max_err,
        "mean_sum_error": mean_err,
    }
    return fig, payload
