#!/usr/bin/env python3

"""Evaluate one low-dimensional CVAE episode and create thesis-style figures.

The model architecture is read from the run's ``exp_config.yaml`` and the raw
topic layout plus normalization statistics are read from ``rollout_config.yaml``.
"""

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from factr.utils_plot import pose_chunks_for_plot, relative_chunk_from_absolute  # noqa: E402
from scripts import eval_single_episode_lowdim as eval_utils  # noqa: E402

DPI = 300
VIEW_ELEV = 25
VIEW_AZIM = -55

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "scripts/eval_params.yaml"

COLORS = {
    "leading": "#289ADC",
    "following": "#009E73",
    "commanded": "#A4A4A4",
    "samples": "tab:orange",
    "selected": "#004381",
    "goal": "#6F6F6F",
    "target": "#C43C39",
}

GOAL_GROUP_COLORS = {
    0: "#CE2900",  # muted rose
    1: "#E7C400",  # warm gold
    2: "#06BB00",  # soft cyan
}

ROLE_COLOR_MAP = mcolors.LinearSegmentedColormap.from_list(
    "following_to_leading",
    [COLORS["following"], COLORS["leading"]],
)

GOALS = {
    1: np.array([0.388, -0.041, 0.031]),
    2: np.array([0.382, -0.038, 0.179]),
    3: np.array([0.371, -0.048, 0.316]),
    4: np.array([0.560, -0.365, 0.035]),
    5: np.array([0.555, -0.371, 0.178]),
    6: np.array([0.544, -0.355, 0.323]),
    7: np.array([0.257, -0.376, 0.039]),
    8: np.array([0.260, -0.376, 0.181]),
    9: np.array([0.254, -0.380, 0.327]),
}


def apply_thesis_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def parse_target_goal(path):
    match = re.search(r"(?:^|_)goal_(\d+)(?:_|\.|$)", path.stem)
    return int(match.group(1)) if match else None


def choose_anchor_indices(count, requested_count):
    if count == 0:
        return np.empty(0, dtype=np.int64)
    requested_count = min(max(1, requested_count), count)
    return np.unique(np.rint(np.linspace(0, count - 1, requested_count)).astype(np.int64))


def add_measured_path(ax, data):
    points = data["measured"]
    if points.shape[0] < 2:
        return

    # Use one continuous line so the dash pattern does not restart at every
    # measured segment and accidentally appear solid.
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=COLORS["commanded"],
        linewidth=1.0,
        linestyle=(0, (5, 3)),
        dash_capstyle="butt",
        alpha=1.0,
        zorder=5,
    )


def add_predictions(ax, data, anchor_count, samples_per_anchor):
    sample_indices = choose_anchor_indices(len(data["predictions"]), anchor_count)
    displayed_xyz = []

    for batch_index in sample_indices:
        samples = data["predictions"][batch_index]
        policy_indices = choose_anchor_indices(samples.shape[0], samples_per_anchor or samples.shape[0])
        sample_goal_groups = data.get("sample_goal_groups")
        for policy_index in policy_indices:
            xyz = samples[policy_index, :, :3]
            finite = np.all(np.isfinite(xyz), axis=1)
            if np.count_nonzero(finite) < 2:
                continue
            xyz = xyz[finite]
            displayed_xyz.append(xyz)
            sample_color = GOAL_GROUP_COLORS[int(sample_goal_groups[policy_index])] if sample_goal_groups is not None else COLORS["samples"]
            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                color=sample_color,
                linewidth=0.9,
                alpha=0.62,
                zorder=9,
            )

        selected = np.asarray(data["selected_predictions"][batch_index], dtype=np.float64)
        if selected.ndim != 2 or selected.shape[1] < 3:
            continue
        xyz = selected[:, :3]
        finite = np.all(np.isfinite(xyz), axis=1)
        if np.count_nonzero(finite) < 2:
            continue
        xyz = xyz[finite]
        displayed_xyz.append(xyz)
        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color=COLORS["selected"],
            linewidth=1.15,
            alpha=1.0,
            zorder=12,
        )

    return displayed_xyz


def _select_device(gpu_id):
    if str(gpu_id).strip().lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and gpu_id is not None:
        return torch.device(f"cuda:{int(gpu_id)}")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.inference_mode()
def _extract_latent_statistics(model, obs, actions, labels, arrangements, goals, device):
    """Return distribution means and variances at every evaluation timestep."""
    obs_t = torch.from_numpy(obs).float().to(device)
    actions_t = torch.from_numpy(actions).float().to(device)
    labels_t = torch.from_numpy(labels).long().to(device)
    arrangements_t = torch.from_numpy(arrangements).float().to(device) if arrangements is not None else None
    goals_t = torch.from_numpy(goals).float().to(device) if goals is not None else None

    context_all = model._build_context_tokens(
        obs_t,
        class_labels=labels_t,
        arrangement_vectors=arrangements_t,
        goal_vectors=goals_t,
    )
    prior = model._prior(model._build_z_context(context_all[:, :-1]))
    posterior = model.posterior(model._posterior_context(context_all), actions_t)

    if str(model.latent_distribution).lower() == "gaussian":
        return {
            "labels": [f"$z_{{{i}}}$" for i in range(prior["mu"].shape[-1])],
            "prior_mean": prior["mu"].cpu().numpy(),
            "prior_var": prior["logvar"].exp().cpu().numpy(),
            "posterior_mean": posterior["mu"].cpu().numpy(),
            "posterior_var": posterior["logvar"].exp().cpu().numpy(),
        }

    # A categorical latent has no scalar Gaussian mean. Treat category indices
    # as values and plot their exact expectation and variance per z variable.
    category_values = torch.arange(prior["logits"].shape[-1], device=device, dtype=prior["logits"].dtype)

    def categorical_moments(logits):
        probabilities = torch.softmax(logits, dim=-1)
        mean = (probabilities * category_values).sum(dim=-1)
        variance = (probabilities * (category_values - mean.unsqueeze(-1)).square()).sum(dim=-1)
        return mean.cpu().numpy(), variance.cpu().numpy()

    prior_mean, prior_var = categorical_moments(prior["logits"])
    posterior_mean, posterior_var = categorical_moments(posterior["logits"])
    return {
        "labels": [f"$z_{{{i}}}$" for i in range(prior_mean.shape[-1])],
        "prior_mean": prior_mean,
        "prior_var": prior_var,
        "posterior_mean": posterior_mean,
        "posterior_var": posterior_var,
    }


def make_z_figure(statistics, time_seconds):
    apply_thesis_style()
    latent_dim = statistics["prior_mean"].shape[-1]
    fig = plt.figure(figsize=(14, max(4.5, 1.85 * latent_dim)))
    grid = fig.add_gridspec(
        latent_dim,
        2,
        hspace=0.32,
        wspace=0.20,
    )
    axes = np.empty((latent_dim, 2), dtype=object)
    for dim in range(latent_dim):
        axes[dim, 0] = fig.add_subplot(grid[dim, 0], sharex=axes[0, 0] if dim else None)
        axes[dim, 1] = fig.add_subplot(grid[dim, 1], sharex=axes[0, 1] if dim else None)
    prior_color = "#DD8452"  # muted orange
    posterior_color = "#4C72B0"  # muted blue

    mean_limits = (-2.9, 2.9)
    variance_limits = (-0.1, 1.1)

    for dim, latent_label in enumerate(statistics["labels"]):
        mean_ax, variance_ax = axes[dim]
        mean_ax.plot(time_seconds, statistics["prior_mean"][:, dim], color=prior_color, linewidth=2.3)
        mean_ax.plot(time_seconds, statistics["posterior_mean"][:, dim], color=posterior_color, linewidth=2.3)
        variance_ax.plot(time_seconds, statistics["prior_var"][:, dim], color=prior_color, linewidth=2.3)
        variance_ax.plot(time_seconds, statistics["posterior_var"][:, dim], color=posterior_color, linewidth=2.3)
        mean_ax.set_ylabel(latent_label, fontsize=15.75)
        mean_ax.set_ylim(mean_limits)
        mean_ax.tick_params(axis="both", labelsize=14.625)
        mean_ax.grid(True)
        variance_ax.set_ylim(variance_limits)
        variance_ax.tick_params(axis="both", labelsize=14.625)
        variance_ax.grid(True)

    axes[0, 0].set_title("Mean", fontsize=18, fontweight="normal")
    axes[0, 1].set_title("Variance", fontsize=18, fontweight="normal")
    axes[-1, 0].set_xlabel("Time [s]", fontsize=16.3125)
    axes[-1, 1].set_xlabel("Time [s]", fontsize=16.3125)

    handles = [
        Line2D([0], [0], color=prior_color, linewidth=1.4, label="Prior $p(z|c)$"),
        Line2D([0], [0], color=posterior_color, linewidth=1.4, label="Posterior $q(z|c,a)$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.962), fontsize=20)
    fig.suptitle("Prior and posterior latent statistics", y=0.995, fontsize=20.25, fontweight="semibold")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.05)
    return fig


def make_average_attention_figure(attention, memory_labels):
    """Match the attention-video grid using matrices averaged over the episode."""
    apply_thesis_style()
    attention_label_aliases = {"stiffness": "mode", "command": "pose cmd", "velocity": "ee vel"}
    display_labels = [attention_label_aliases.get(label, label) for label in memory_labels]
    if attention.ndim != 5:
        raise ValueError(f"Expected attention shape (steps, layers, heads, action_tokens, memory_tokens), got {attention.shape}.")
    _, num_layers, num_heads, num_action_tokens, num_memory_tokens = attention.shape
    if num_memory_tokens != len(memory_labels):
        raise ValueError(f"Attention/token-label mismatch: {num_memory_tokens} != {len(memory_labels)}")

    # Preserve every layer/head from the video, changing only the time-varying
    # matrices into their complete-episode averages.
    episode_average = attention.mean(axis=0)
    fig = plt.figure(
        figsize=(
            max(4.8 * (num_heads + 1), 14.5),
            max(4.5 * num_layers + 1.0, 13.5),
        )
    )
    grid = fig.add_gridspec(
        num_layers,
        num_heads + 2,
        width_ratios=[1.0] * num_heads + [0.04, 1.0],
        hspace=0.30,
        wspace=0.18,
    )
    action_ticks = np.unique(np.rint(np.linspace(0, num_action_tokens - 1, min(5, num_action_tokens))).astype(int))

    def style_attention_axis(ax, show_x_labels, show_y_labels, emphasize=False):
        ax.set_xticks(
            np.arange(num_memory_tokens),
            labels=display_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=24,
        )
        ax.set_yticks(action_ticks)
        ax.tick_params(axis="x", labelbottom=show_x_labels)
        ax.tick_params(axis="y", labelsize=22.5, labelleft=show_y_labels)
        ax.tick_params(axis="both", width=0.8, length=3.5)
        border_color = COLORS["selected"] if emphasize else "#6F6F6F"
        border_width = 1.2 if emphasize else 0.8
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(border_width)

    first_image = None
    for layer_index in range(num_layers):
        for head_index in range(num_heads):
            ax = fig.add_subplot(grid[layer_index, head_index])
            image = ax.imshow(
                episode_average[layer_index, head_index],
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
                cmap="turbo",
            )
            first_image = image if first_image is None else first_image
            ax.set_title(f"L{layer_index + 1} · H{head_index + 1}", fontsize=25.5, pad=14)
            style_attention_axis(
                ax,
                show_x_labels=layer_index == num_layers - 1,
                show_y_labels=head_index == 0,
            )
            if head_index == 0:
                ax.set_ylabel("Action-chunk step", fontsize=24.75, labelpad=12)

        spacer_ax = fig.add_subplot(grid[layer_index, num_heads])
        spacer_ax.axis("off")
        mean_ax = fig.add_subplot(grid[layer_index, -1])
        mean_ax.imshow(
            episode_average[layer_index].mean(axis=0),
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="turbo",
        )
        mean_ax.set_title(f"L{layer_index + 1} · Mean", fontsize=25.5, pad=14)
        style_attention_axis(
            mean_ax,
            show_x_labels=layer_index == num_layers - 1,
            show_y_labels=True,
            emphasize=True,
        )
        mean_ax.set_ylabel("Action-chunk step", fontsize=24.75, labelpad=12)

    # Overall mean subplot intentionally disabled for the thesis figure.
    # The code is kept here so it can be restored later if needed.
    # total_ax = fig.add_subplot(grid[-1, -1])
    # total_ax.imshow(
    #     episode_average.mean(axis=(0, 1)),
    #     aspect="auto",
    #     vmin=0.0,
    #     vmax=1.0,
    #     cmap="turbo",
    # )
    # total_ax.set_title("Overall mean\n(all layers and heads)", fontsize=12.5, pad=8, linespacing=1.0)
    # total_ax.set_xlabel("Decoder memory token", fontsize=12.5, labelpad=7)
    # total_ax.set_ylabel("Action-chunk step", fontsize=12, labelpad=6)
    # style_attention_axis(total_ax, show_x_labels=True, show_y_labels=True, emphasize=True)

    colorbar_ax = fig.add_axes([0.972, 0.19, 0.013, 0.66])
    colorbar = fig.colorbar(first_image, cax=colorbar_ax)
    colorbar.set_label("Cross-attention weight", fontsize=24.75, labelpad=16)
    colorbar.ax.tick_params(labelsize=22.5, width=1.0, length=5)
    fig.suptitle(
        "Episode-Averaged Decoder Cross-Attention",
        y=0.97,
        fontsize=30,
        fontweight="semibold",
    )
    # fig.text(
    #     0.5,
    #     0.952,
    #     "Prior policy · averaged over all episode steps",
    #     ha="center",
    #     va="top",
    #     fontsize=22.5,
    #     color="#4A4A4A",
    # )
    fig.subplots_adjust(left=0.055, right=0.955, bottom=0.13, top=0.90)
    return fig


def evaluate_episode(
    run_dir,
    checkpoint_name,
    episode_path,
    rollout_config_path,
    num_samples,
    gpu_id,
    plot_average_attention,
    sample_goal_groups_equally,
):
    exp_config_path = run_dir / "exp_config.yaml"
    checkpoint_path = run_dir / checkpoint_name
    if not exp_config_path.is_file():
        raise FileNotFoundError(f"Experiment YAML not found: {exp_config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cfg = eval_utils._load_run_cfg(exp_config_path)
    if rollout_config_path is None:
        run_rollout_config = run_dir / "rollout_config.yaml"
        buffer_path = Path(str(cfg.test_buffer_path)).expanduser()
        rollout_config_path = run_rollout_config if run_rollout_config.is_file() else buffer_path.parent / "rollout_config.yaml"
    rollout_cfg = eval_utils._load_rollout_config(rollout_config_path)
    raw = eval_utils._load_raw_episode_to_arrays(episode_path, rollout_cfg)

    include_velocity = bool(OmegaConf.select(cfg, "agent.include_velocity", default=True))
    include_tracking = bool(OmegaConf.select(cfg, "agent.include_tracking_error", default=True))
    states = eval_utils._lowdim_filter_state_features(raw["states"], include_velocity, include_tracking)
    state_stats = eval_utils._lowdim_state_stats_without_features(rollout_cfg.get("norm_stats", {}).get("state"), include_velocity, include_tracking)
    action_stats = rollout_cfg.get("norm_stats", {}).get("action")
    action_chunk_mode = eval_utils._canonical_action_mode(OmegaConf.select(cfg, "action_chunk_mode", default="absolute"))
    actions = raw["actions"]
    if action_chunk_mode == "delta" and len(actions) > 1:
        delta_actions = np.zeros_like(actions)
        delta_actions[1:] = actions[1:] - actions[:-1]
        actions = delta_actions

    obs_window = int(cfg.obs_window)
    ac_chunk = int(cfg.ac_chunk)
    action_offset = int(OmegaConf.select(cfg, "task.test_buffer.action_index_offset", default=0))
    samples = eval_utils._build_eval_samples_from_raw_episode(states, actions, raw["episode_label"], obs_window, ac_chunk, action_offset)
    obs_norm, _ = eval_utils._ensure_normalized(samples["obs"], state_stats, "apply", "state")
    command_start = eval_utils._lowdim_command_start(include_velocity, include_tracking)
    command_anchor = samples["obs"][:, -1, command_start : command_start + 9]
    if action_chunk_mode == "relative":
        action_input = relative_chunk_from_absolute(samples["actions"], command_anchor)
    else:
        action_input = samples["actions"]
    actions_norm, _ = eval_utils._ensure_normalized(action_input, action_stats, "apply", "action")

    arrangements = None
    if bool(OmegaConf.select(cfg, "agent.use_arrangement_conditioning", default=False)):
        if raw["arrangement_vector"] is None:
            raise ValueError("Checkpoint requires arrangement conditioning, but the episode has no arrangement topic.")
        arrangements = np.repeat(raw["arrangement_vector"][None], len(samples["steps"]), axis=0).astype(np.float32)
    goals = None
    if bool(OmegaConf.select(cfg, "agent.goal_label", default=False)):
        if raw["goal_vectors"] is None:
            raise ValueError("Checkpoint requires goal conditioning, but the episode has no goal topic.")
        goals = raw["goal_vectors"][samples["steps"]]

    device = _select_device(gpu_id)
    print(f"Using device: {device}")
    model = eval_utils._load_model(cfg, checkpoint_path, device)
    obs_t = torch.from_numpy(obs_norm).float().to(device)
    labels_t = torch.from_numpy(samples["labels"]).long().to(device)
    arrangement_t = torch.from_numpy(arrangements).float().to(device) if arrangements is not None else None
    goals_t = torch.from_numpy(goals).float().to(device) if goals is not None else None
    with torch.inference_mode():
        sample_goal_groups = None
        if sample_goal_groups_equally:
            if not bool(getattr(model, "goal_label", False)):
                raise ValueError("Equal goal-group sampling requires a goal-conditioned checkpoint.")
            if num_samples % 3 != 0:
                raise ValueError("Equal goal-group sampling requires num_samples to be divisible by 3.")
            samples_per_goal_group = num_samples // 3
            print(f"Equal goal-group sampling: {samples_per_goal_group} samples per group ({num_samples} total).")
            goal_basis = torch.eye(3, device=device, dtype=obs_t.dtype)
            grouped_predictions = []
            for goal_group_index in range(3):
                group_condition = goal_basis[goal_group_index].unsqueeze(0).expand(obs_t.shape[0], -1)
                grouped_predictions.append(
                    model.get_actions_prior(
                        {},
                        obs_t,
                        class_labels=labels_t,
                        arrangement_vectors=arrangement_t,
                        goal_vectors=group_condition,
                        sample=True,
                        num_samples=samples_per_goal_group,
                    )
                )
            predicted_norm = torch.cat(grouped_predictions, dim=1).cpu().numpy()
            sample_goal_groups = np.repeat(np.arange(3, dtype=np.int64), samples_per_goal_group)
        else:
            predicted_norm = (
                model.get_actions_prior(
                    {},
                    obs_t,
                    class_labels=labels_t,
                    arrangement_vectors=arrangement_t,
                    goal_vectors=goals_t,
                    sample=True,
                    num_samples=num_samples,
                )
                .cpu()
                .numpy()
            )
        selected_norm = (
            model.get_actions_prior(
                {},
                obs_t,
                class_labels=labels_t,
                arrangement_vectors=arrangement_t,
                goal_vectors=goals_t,
                sample=False,
                num_samples=1,
            )[:, 0]
            .cpu()
            .numpy()
        )

    predicted = eval_utils._apply_grouped_transform(predicted_norm, action_stats, inverse=True)
    selected = eval_utils._apply_grouped_transform(selected_norm, action_stats, inverse=True)
    plot_anchor = command_anchor if action_chunk_mode == "relative" else samples["obs"][:, -1, :9]
    predicted_plot = pose_chunks_for_plot(predicted[..., :9], plot_anchor[:, None], action_chunk_mode)
    selected_plot = pose_chunks_for_plot(selected[..., :9], plot_anchor, action_chunk_mode)

    processing_cfg = rollout_cfg.get("processing_config") or {}
    frequency_key = "target_downsampling_freq" if processing_cfg.get("downsample", False) else "data_frequency"
    frequency = float(processing_cfg.get(frequency_key) or 1.0)
    time_seconds = samples["steps"].astype(np.float64) / frequency
    class_count = int(OmegaConf.select(cfg, "agent.stiffness_classes", default=1))
    role_value = (samples["labels"] - 1) / max(1, class_count - 1)
    plot_data = {
        "measured": samples["obs"][:, -1, :3],
        "measured_time": time_seconds,
        "commanded": samples["actions"][:, 0, :3],
        "stiffness": role_value.astype(np.float64),
        "stiffness_time": time_seconds,
        "predictions": predicted_plot[..., :3],
        "selected_predictions": selected_plot[..., :3],
        "sample_goal_groups": sample_goal_groups,
    }
    latent_stats = _extract_latent_statistics(model, obs_norm, actions_norm, samples["labels"], arrangements, goals, device)
    attention_data = None
    if plot_average_attention:
        print("Collecting episode-average decoder cross-attention...")
        cross_attention = eval_utils._collect_decoder_cross_attention(
            model=model,
            obs=obs_t,
            actions=torch.from_numpy(actions_norm).float().to(device),
            labels=labels_t,
            arrangement_vectors=arrangement_t,
            action_source="prior",
            goal_vectors=goals_t,
        )
        attention_data = {
            "values": cross_attention,
            "memory_labels": eval_utils._decoder_memory_token_labels(model),
        }
    return plot_data, latent_stats, time_seconds, rollout_config_path, attention_data


def automatic_limits(arrays, padding_fraction=0.035):
    finite_parts = []
    for values in arrays:
        values = np.asarray(values, dtype=np.float64).reshape(-1, 3)
        finite = values[np.all(np.isfinite(values), axis=1)]
        if finite.size:
            finite_parts.append(finite)
    points = np.concatenate(finite_parts, axis=0)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    spans = np.maximum(maxs - mins, 0.05)
    padding = padding_fraction * spans
    mins -= padding
    maxs += padding
    mins[2] = max(0.0, mins[2])
    return mins, maxs


def style_3d_axis(ax, mins, maxs):
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)
    ax.set_xlabel("$x$ [m]", labelpad=8)
    ax.set_ylabel("$y$ [m]", labelpad=8)
    ax.set_zlabel("$z$ [m]", labelpad=7)
    ax.view_init(elev=float(VIEW_ELEV), azim=float(VIEW_AZIM))
    ax.grid(True)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MultipleLocator(0.1))
        axis.pane.fill = True
        axis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        axis.pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))


def add_goals(ax, target_goal, floor_z):
    for goal_id, xyz in GOALS.items():
        is_target = goal_id == target_goal
        color = COLORS["target"] if is_target else COLORS["goal"]
        ax.plot(
            [xyz[0], xyz[0]],
            [xyz[1], xyz[1]],
            [floor_z, xyz[2]],
            color=color,
            linestyle=":" if is_target else (0, (2, 3)),
            linewidth=1.15 if is_target else 0.65,
            alpha=0.9 if is_target else 0.45,
            zorder=2,
        )
        ax.scatter(
            *xyz,
            marker="*" if is_target else "D",
            s=92 if is_target else 22,
            color=color if is_target else "white",
            edgecolor=color,
            linewidth=1.0,
            depthshade=False,
            zorder=12,
        )
        ax.text(
            xyz[0],
            xyz[1],
            xyz[2] + 0.012,
            f"G{goal_id}",
            color=color,
            fontsize=8 if is_target else 7,
            fontweight="bold" if is_target else "normal",
            ha="center",
            va="bottom",
            zorder=13,
        )


def make_figure(data, source_path, anchor_count, samples_per_anchor):
    apply_thesis_style()
    target_goal = parse_target_goal(source_path)

    fig = plt.figure(figsize=(7.5, 5.7))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    ax.plot(
        data["commanded"][:, 0],
        data["commanded"][:, 1],
        data["commanded"][:, 2],
        color=COLORS["leading"],
        linewidth=1.4,
        linestyle="solid",
        alpha=0.9,
        zorder=5,
    )
    add_measured_path(ax, data)
    prediction_xyz = add_predictions(ax, data, anchor_count, samples_per_anchor)

    limit_arrays = [data["measured"], data["commanded"], np.stack(list(GOALS.values()))]
    limit_arrays.extend(prediction_xyz)
    mins, maxs = automatic_limits(limit_arrays)
    add_goals(ax, target_goal, mins[2])

    ax.scatter(*data["measured"][0], marker="o", s=32, color="white", edgecolor="#202020", linewidth=1.0, depthshade=False, zorder=14)
    ax.scatter(*data["measured"][-1], marker="s", s=34, color="#202020", edgecolor="white", linewidth=0.6, depthshade=False, zorder=14)

    style_3d_axis(ax, mins, maxs)
    ax.set_title(
        "Evaluation of Policy Model (leading mode) with different goal samples",
        y=1.055,
        pad=8,
        fontsize=10,
        fontweight="normal",
    )
    # ax.text2D(
    #     0.5,
    #     1.012,
    #     "green = following",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="bottom",
    #     fontsize=7.5,
    #     fontweight="normal",
    #     color="#4A4A4A",
    # )

    handles = [
        Line2D([0], [0], color=COLORS["commanded"], linewidth=1.0, linestyle=(0, (5, 3)), label="Measured trajectory"),
        Line2D(
            [0],
            [0],
            color=COLORS["leading"],
            linewidth=1.4,
            linestyle="-",
            label="Commanded trajectory",
        ),
    ]
    if data.get("sample_goal_groups") is None:
        handles.append(Line2D([0], [0], color=COLORS["samples"], linewidth=1.4, alpha=0.8, label="Prior samples"))
    else:
        sample_goal_groups = np.asarray(data["sample_goal_groups"])
        goal_group_labels = tuple(
            f"Base goal {group_index + 1} ({np.count_nonzero(sample_goal_groups == group_index)} samples)" for group_index in range(3)
        )
        handles.extend(
            Line2D(
                [0],
                [0],
                color=GOAL_GROUP_COLORS[group_index],
                linewidth=1.4,
                alpha=0.8,
                label=group_label,
            )
            for group_index, group_label in enumerate(goal_group_labels)
        )
    handles.extend(
        [
            Line2D([0], [0], color=COLORS["selected"], linewidth=1.3, label="Deterministic (1 sample)"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=COLORS["goal"], markersize=5, label="Goals"),
        ]
    )
    if target_goal is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=COLORS["target"],
                markeredgecolor=COLORS["target"],
                markersize=9,
                label=f"Target goal G{target_goal}",
            )
        )
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#202020", markersize=5, label="Start"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#202020", markeredgecolor="#202020", markersize=5, label="End"),
        ]
    )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.6,
    )
    # Keep the z label inside the canvas and separate the x label from the legend.
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.19)
    return fig


def _project_path(value):
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_thesis_config(config_path):
    with open(config_path, "r") as config_file:
        all_config = yaml.safe_load(config_file)
    shared = dict(all_config.get("shared") or {})
    thesis = dict(all_config.get("eval_single_episode_thesis") or {})
    if not thesis:
        raise KeyError(f"Missing eval_single_episode_thesis section in {config_path}")
    use_episode_list = thesis.get("use_episode_list", shared.get("use_episode_list", False))
    if bool(use_episode_list):
        raise ValueError("eval_single_episode_thesis supports one episode; set use_episode_list: false")

    run_name = thesis.get("run_name") or shared.get("run_name")
    run_dir = _project_path(thesis.get("run_dir"))
    if run_dir is None:
        dataset_name = str(thesis.get("dataset_name") or shared["dataset_name"])
        buffer_set = str(thesis.get("buffer_set_name") or shared.get("buffer_set_name", "auto"))
        project_name = dataset_name if buffer_set.lower() == "auto" else buffer_set
        project_prefix = thesis.get("dataset_project_prefix", shared.get("dataset_project_prefix", ""))
        project_key = f"{project_prefix}{project_name}"
        if run_name is None:
            raise KeyError("Missing run_name in the shared or eval_single_episode_thesis YAML section")
        run_dir = PROJECT_ROOT / "checkpoints" / project_key / str(run_name) / "rollout"
    elif run_name is None:
        run_name = run_dir.parent.name if run_dir.name == "rollout" else run_dir.name

    checkpoint_name = thesis.get("checkpoint_name") or shared.get("checkpoint_name", "latest_ckpt.ckpt")
    gpu_id = thesis.get("gpu_id")
    if gpu_id is None:
        gpu_id = shared.get("gpu_id")
    return {
        **thesis,
        "run_dir": run_dir,
        "raw_episode_dir": _project_path(thesis.get("raw_episode_dir")),
        "episode_file_name": thesis.get("episode_file_name") or shared.get("episode_file_name"),
        "rollout_config_path": _project_path(thesis.get("rollout_config_path")),
        "output_stem": _project_path(thesis.get("output_stem")),
        "run_name": str(run_name),
        "checkpoint_name": str(checkpoint_name),
        "gpu_id": gpu_id,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Optional YAML episode filename/path override")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Evaluation YAML")
    parser.add_argument("--run-dir", type=Path, help="Optional YAML run_dir override")
    parser.add_argument("--checkpoint", help="Optional YAML checkpoint_name override")
    parser.add_argument("--rollout-config", type=Path, help="rollout_config.yaml; inferred from the run YAML when omitted")
    parser.add_argument("--output", type=Path, help="Optional YAML output_stem override")
    parser.add_argument("--gpu-id", help='CUDA index or "cpu"; overrides YAML')
    parser.add_argument("--num-samples", type=int, help="Optional YAML num_samples override")
    parser.add_argument(
        "--equal-goal-sampling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override YAML equal sampling across the three goal groups",
    )
    parser.add_argument("--prediction-anchors", default=20, type=int, help="Optional YAML prediction_anchors override")
    parser.add_argument("--samples-per-anchor", default=40, type=int, help="Optional YAML samples_per_anchor override")
    parser.add_argument("--no-pdf", default=True, action="store_true", help="Override YAML and only save PNG versions")
    return parser.parse_args()


def _resolve_episode_path(input_override, raw_episode_dir, episode_file_name):
    if input_override is not None:
        candidate = input_override.expanduser()
        if not candidate.is_absolute() and candidate.parent == Path("."):
            candidate = raw_episode_dir / candidate
        return candidate.resolve()

    relative_episode = Path(str(episode_file_name))
    candidate = raw_episode_dir / relative_episode
    if candidate.is_file() or len(relative_episode.parts) < 2:
        return candidate.resolve()

    # Dataset-qualified names such as boxlift_1_follow/ep_29.pkl map to the
    # standard raw-data layout boxlift_1_follow/data/ep_29.pkl.
    dataset_dir = relative_episode.parts[0]
    episode_tail = Path(*relative_episode.parts[1:])
    return (raw_episode_dir / dataset_dir / "data" / episode_tail).resolve()


def main():
    global DPI, VIEW_AZIM, VIEW_ELEV
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    script_cfg = _load_thesis_config(config_path)

    source_path = _resolve_episode_path(
        args.input,
        script_cfg["raw_episode_dir"],
        script_cfg["episode_file_name"],
    )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    run_dir = (args.run_dir or script_cfg["run_dir"]).expanduser().resolve()
    run_name = (run_dir.parent.name if run_dir.name == "rollout" else run_dir.name) if args.run_dir else script_cfg["run_name"]
    checkpoint_name = args.checkpoint or script_cfg["checkpoint_name"]
    rollout_config_path = args.rollout_config or script_cfg["rollout_config_path"]
    rollout_config_path = rollout_config_path.expanduser().resolve() if rollout_config_path else None
    gpu_id = args.gpu_id if args.gpu_id is not None else script_cfg.get("gpu_id")
    num_samples = args.num_samples if args.num_samples is not None else int(script_cfg["num_samples"])
    prediction_anchors = args.prediction_anchors if args.prediction_anchors is not None else int(script_cfg["prediction_anchors"])
    samples_per_anchor = args.samples_per_anchor if args.samples_per_anchor is not None else int(script_cfg["samples_per_anchor"])
    save_pdf = bool(script_cfg.get("save_pdf", True)) and not args.no_pdf
    plot_average_attention = bool(script_cfg.get("plot_average_attention", True))
    sample_goal_groups_equally = args.equal_goal_sampling if args.equal_goal_sampling is not None else bool(script_cfg.get("sample_goal_groups_equally", False))
    DPI = int(script_cfg.get("dpi", DPI))
    VIEW_ELEV = float(script_cfg.get("view_elev", VIEW_ELEV))
    VIEW_AZIM = float(script_cfg.get("view_azim", VIEW_AZIM))
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    dataset_episode_name = f"{source_path.parent.parent.name}_{source_path.stem}"

    configured_output = script_cfg.get("output_stem")
    output_stem = (
        args.output.expanduser()
        if args.output
        else configured_output or run_dir.parent / "episode_eval_thesis" / source_path.stem / f"3d_prediction_thesis_{dataset_episode_name}_{run_name}"
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    data, latent_statistics, time_seconds, resolved_rollout_config, attention_data = evaluate_episode(
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        episode_path=source_path,
        rollout_config_path=rollout_config_path,
        num_samples=num_samples,
        gpu_id=gpu_id,
        plot_average_attention=plot_average_attention,
        sample_goal_groups_equally=sample_goal_groups_equally,
    )
    print(f"Using script config: {config_path}")
    print(f"Using experiment config: {run_dir / 'exp_config.yaml'}")
    print(f"Using rollout config: {resolved_rollout_config}")
    fig = make_figure(data, source_path, prediction_anchors, samples_per_anchor)
    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved {png_path}")
    if save_pdf:
        pdf_path = output_stem.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved {pdf_path}")
    plt.close(fig)

    z_stem = output_stem.parent / f"z_stats_thesis_{dataset_episode_name}_{run_name}"
    z_fig = make_z_figure(latent_statistics, time_seconds)
    z_png_path = z_stem.with_suffix(".png")
    z_fig.savefig(z_png_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved {z_png_path}")
    if save_pdf:
        z_pdf_path = z_stem.with_suffix(".pdf")
        z_fig.savefig(z_pdf_path, bbox_inches="tight")
        print(f"Saved {z_pdf_path}")
    plt.close(z_fig)

    if attention_data is not None:
        attention_stem = output_stem.parent / f"{dataset_episode_name}_ca_ep_average_{run_name}"
        attention_fig = make_average_attention_figure(
            attention_data["values"],
            attention_data["memory_labels"],
        )
        attention_png_path = attention_stem.with_suffix(".png")
        attention_fig.savefig(attention_png_path, dpi=DPI, bbox_inches="tight")
        print(f"Saved {attention_png_path}")
        if save_pdf:
            attention_pdf_path = attention_stem.with_suffix(".pdf")
            attention_fig.savefig(attention_pdf_path, bbox_inches="tight")
            print(f"Saved {attention_pdf_path}")
        plt.close(attention_fig)


if __name__ == "__main__":
    main()
