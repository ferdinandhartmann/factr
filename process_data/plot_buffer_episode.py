#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

FIXED_Y_LIMITS = {
    "EE Position": (-3.0, 3.0),
    "EE Orientation": (-3.0, 3.0),
    "EE Velocity": (-3.0, 3.0),
    "Tracking Error": (-2.0, 2.0),
    "External Wrench": (-5.0, 5.0),
    "Action Position": (-3.0, 3.0),
    "Action Orientation": (-3.0, 3.0),
    "Action Stiffness": (-1.0, 1.0),
    "Goals": (-0.05, 1.05),
}


def load_rollout_config(buf_path, rollout_config_path=None):
    if rollout_config_path is None:
        rollout_config_path = Path(buf_path).parent / "rollout_config.yaml"
    rollout_config_path = Path(rollout_config_path)
    if not rollout_config_path.exists():
        return None
    with open(rollout_config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_episode_names(buf_path):
    rollout_config = load_rollout_config(buf_path)
    if rollout_config is None:
        return None
    split_config = rollout_config.get("split_config", {})
    buffer_name = Path(buf_path).name
    if buffer_name == split_config.get("test_buffer"):
        return split_config.get("test_episodes")
    if buffer_name == split_config.get("train_buffer"):
        return split_config.get("train_episodes")
    return split_config.get("episodes")


def _inverse_group_transform(values, group):
    gtype = group.get("type", "identity")
    if gtype == "min_max":
        mins = np.asarray(group.get("min", []), dtype=float)
        maxs = np.asarray(group.get("max", []), dtype=float)
        return (values + 1.0) * 0.5 * (maxs - mins) + mins
    if gtype in ("gaussian", "gaussian_clip", "zscore_clip"):
        mean = np.asarray(group.get("mean", []), dtype=float)
        std = np.asarray(group.get("std", []), dtype=float)
        return values * std + mean
    if gtype in ("fixed_scale", "fixed_scale_clip"):
        scales = np.asarray(group.get("scales", []), dtype=float)
        return values * scales
    if gtype == "log1p_zscore_clip":
        mean = np.asarray(group.get("mean", []), dtype=float)
        std = np.asarray(group.get("std", []), dtype=float)
        log_values = values * std + mean
        return np.sign(log_values) * np.expm1(np.abs(log_values))
    if gtype == "log1p":
        return np.sign(values) * np.expm1(np.abs(values))
    return values


def denormalize_states(states, norm_stats):
    if not norm_stats or norm_stats.get("mode") != "grouped":
        return states
    states = states.copy()
    for group in norm_stats.get("groups", []):
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = int(indices[0]), int(indices[1])
        states[:, start:stop] = _inverse_group_transform(states[:, start:stop], group)
    return states


def denormalize_actions(actions, action_stats):
    if not action_stats:
        return actions
    if action_stats.get("mode") != "grouped":
        mean = np.asarray(action_stats.get("mean", []), dtype=float)
        std = np.asarray(action_stats.get("std", []), dtype=float)
        if mean.size == actions.shape[1] and std.size == actions.shape[1]:
            return actions * std + mean
        return actions
    actions = actions.copy()
    for group in action_stats.get("groups", []):
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = int(indices[0]), int(indices[1])
        actions[:, start:stop] = _inverse_group_transform(actions[:, start:stop], group)
    return actions


def build_state_labels(state_dim, norm_stats):
    labels = [f"{i + 1}" for i in range(state_dim)]
    if not norm_stats or norm_stats.get("mode") != "grouped":
        return labels
    name_map = {
        "ee_position": ["EE_Pos_x", "EE_Pos_y", "EE_Pos_z"],
        "ee_orientation": ["EE_Rot_c1_x", "EE_Rot_c1_y", "EE_Rot_c1_z", "EE_Rot_c2_x", "EE_Rot_c2_y", "EE_Rot_c2_z"],
        "ee_velocity": ["EE_Vel_x", "EE_Vel_y", "EE_Vel_z", "EE_Vel_rx", "EE_Vel_ry", "EE_Vel_rz"],
        "tracking_error": [
            "Tracking_Err_x",
            "Tracking_Err_y",
            "Tracking_Err_z",
            "Tracking_Err_rx",
            "Tracking_Err_ry",
            "Tracking_Err_rz",
        ],
        "external_wrench": [
            "Ext_Wrench_fx",
            "Ext_Wrench_fy",
            "Ext_Wrench_fz",
            "Ext_Wrench_tx",
            "Ext_Wrench_ty",
            "Ext_Wrench_tz",
        ],
    }
    for group in norm_stats.get("groups", []):
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = int(indices[0]), int(indices[1])
        group_name = group.get("name", "group")
        names = name_map.get(group_name, [])
        if len(names) != stop - start:
            names = [f"{group_name}_{i}" for i in range(stop - start)]
        for i in range(stop - start):
            labels[start + i] = names[i]
    return labels


def build_action_labels(action_dim, action_stats):
    labels = [f"Act {i + 1}" for i in range(action_dim)]
    if not action_stats or action_stats.get("mode") != "grouped":
        return labels
    name_map = {
        "ee_position": ["EE_Pos_x", "EE_Pos_y", "EE_Pos_z"],
        "ee_orientation": ["EE_Rot_c1_x", "EE_Rot_c1_y", "EE_Rot_c1_z", "EE_Rot_c2_x", "EE_Rot_c2_y", "EE_Rot_c2_z"],
        "impedance_stiffness": [
            "Stiff_transl_x",
            "Stiff_transl_y",
            "Stiff_transl_z",
            "Stiff_rot_x",
            "Stiff_rot_y",
            "Stiff_rot_z",
        ],
    }
    for group in action_stats.get("groups", []):
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = int(indices[0]), int(indices[1])
        group_name = group.get("name", "group")
        names = name_map.get(group_name, [])
        if len(names) != stop - start:
            names = [f"{group_name}_{i}" for i in range(stop - start)]
        for i in range(stop - start):
            idx = start + i
            if 0 <= idx < action_dim:
                labels[idx] = names[i]
    return labels


def build_state_groups(state_dim, norm_stats):
    if norm_stats and norm_stats.get("mode") == "grouped":
        title_map = {
            "ee_position": "EE Position",
            "ee_orientation": "EE Orientation",
            "ee_velocity": "EE Velocity",
            "tracking_error": "Tracking Error",
            "external_wrench": "External Wrench",
        }
        groups = []
        for group in norm_stats.get("groups", []):
            indices = group.get("indices", None)
            if not indices or len(indices) != 2:
                continue
            start, stop = int(indices[0]), int(indices[1])
            dims = list(range(start, min(stop, state_dim)))
            if dims:
                name = str(group.get("name", "group"))
                groups.append((title_map.get(name, name), dims))
        if groups:
            return groups
    return [("State", list(range(state_dim)))]


def build_action_groups(action_dim, action_stats):
    if action_stats and action_stats.get("mode") == "grouped":
        title_map = {
            "ee_position": "Action Position",
            "ee_orientation": "Action Orientation",
            "impedance_stiffness": "Action Stiffness",
        }
        groups = []
        for group in action_stats.get("groups", []):
            indices = group.get("indices", None)
            if not indices or len(indices) != 2:
                continue
            start, stop = int(indices[0]), int(indices[1])
            dims = list(range(start, min(stop, action_dim)))
            if dims:
                name = str(group.get("name", "group"))
                groups.append((title_map.get(name, name), dims))
        if groups:
            return groups
    return [("Actions", list(range(action_dim)))]


def print_state_minmax(states, labels=None, title=None):
    mins = states.min(axis=0)
    maxs = states.max(axis=0)
    if title:
        print(title)
    if labels is None:
        labels = [f"state[{i:02d}]" for i in range(states.shape[1])]
    for i, (mn, mx) in enumerate(zip(mins, maxs)):
        print(f"  {labels[i]}: min={mn: .6g}  max={mx: .6g}")


def plot_grouped_series(
    title,
    x,
    data,
    labels,
    groups,
    output_path,
    color,
    y_limit_mode="auto",
    y_label_fontsize=12,
    subplot_top=0.985,
    suptitle_y=0.995,
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    grouped_dims = []
    for group_title, group_dims in groups:
        dims = [dim for dim in group_dims if 0 <= dim < data.shape[1]]
        if dims:
            grouped_dims.append((group_title, dims))
    if not grouped_dims:
        grouped_dims = [("Series", list(range(data.shape[1])))]

    total_rows = sum(len(dims) for _, dims in grouped_dims) + max(0, len(grouped_dims) - 1)
    height_ratios = []
    for group_index, (_, dims) in enumerate(grouped_dims):
        height_ratios.extend([1.0] * len(dims))
        if group_index < len(grouped_dims) - 1:
            height_ratios.append(0.35)

    fig = plt.figure(figsize=(12, 1.8 * (total_rows - (len(grouped_dims) - 1)) + 0.25 * (len(grouped_dims) - 1)))
    gs = GridSpec(total_rows, 1, height_ratios=height_ratios, hspace=0.18)

    axes = []
    row = 0
    for group_index, (group_title, dims) in enumerate(grouped_dims):
        first_ax = None
        if str(y_limit_mode) == "fixed":
            group_ylim = FIXED_Y_LIMITS.get(group_title)
        elif str(y_limit_mode) == "auto":
            group_values = data[:, dims]
            ymin = float(np.min(group_values))
            ymax = float(np.max(group_values))
            pad = 0.05 * max(1e-6, ymax - ymin)
            group_ylim = (ymin - pad, ymax + pad)
        else:
            group_ylim = None
        for dim in dims:
            ax = fig.add_subplot(gs[row, 0], sharex=axes[0] if axes else None)
            if first_ax is None:
                first_ax = ax
            ax.plot(x, data[:, dim], color=color, linewidth=1.0, alpha=0.7)
            ax.set_ylabel(str(labels[dim]).replace("_", " "), fontsize=y_label_fontsize)
            if group_ylim is not None:
                ax.set_ylim(group_ylim[0], group_ylim[1])
            ax.grid(True, alpha=0.3)
            axes.append(ax)
            row += 1
        if first_ax is not None:
            first_ax.set_title(group_title, fontsize=13, pad=8)
        if group_index < len(grouped_dims) - 1:
            spacer = fig.add_subplot(gs[row, 0])
            spacer.axis("off")
            row += 1

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xlabel("Frame index", fontsize=10)

    fig.tight_layout(pad=0.05)
    fig.subplots_adjust(top=float(subplot_top), bottom=0.04)
    fig.suptitle(title, fontsize=14, y=float(suptitle_y))
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def load_buffer(buf_path):
    with open(buf_path, "rb") as f:
        return pickle.load(f)


def extract_episode_data(trajectory):
    states = []
    actions = []
    goals = []

    for obs, action, _ in trajectory:
        states.append(np.asarray(obs["state"], dtype=float))
        actions.append(np.asarray(action, dtype=float))
        if "goals" in obs:
            goals.append(np.asarray(obs["goals"], dtype=float))

    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)
    goals_array = np.stack(goals, axis=0) if goals else None
    return states, actions, goals_array


def list_episodes(buf_path):
    buffer = load_buffer(buf_path)
    episode_names = resolve_episode_names(buf_path)
    print(f"Loaded buffer: {buf_path}")
    print(f"Number of episodes in buffer: {len(buffer)}")
    for idx, trajectory in enumerate(buffer):
        episode_name = None if episode_names is None else episode_names[idx]
        if episode_name is None:
            print(f"episode_index={idx} length={len(trajectory)}")
        else:
            print(f"episode_index={idx} episode_name={episode_name} length={len(trajectory)}")


def plot_episode(buf_path, episode_index, output_dir=None, step=1, denormalize=False, y_limit_mode="auto"):
    buf_path = Path(buf_path)
    buffer = load_buffer(buf_path)
    num_episodes = len(buffer)
    episode_names = resolve_episode_names(buf_path)
    episode_name = (
        f"episode_{int(episode_index):03d}" if episode_names is None else str(episode_names[int(episode_index)])
    )

    print(f"Loaded buffer: {buf_path}")
    print(f"Number of episodes in buffer: {num_episodes}")
    print(f"Selected episode index: {episode_index}")
    print(f"Selected episode name: {episode_name}")

    trajectory = buffer[int(episode_index)]
    states, actions, goals = extract_episode_data(trajectory)

    rollout_config = load_rollout_config(buf_path)
    norm_stats = None
    action_stats = None
    if rollout_config:
        norm_stats = rollout_config.get("norm_stats", {}).get("state", None)
        action_stats = rollout_config.get("norm_stats", {}).get("action", None)

    if denormalize and norm_stats:
        states = denormalize_states(states, norm_stats)
    if denormalize and action_stats:
        actions = denormalize_actions(actions, action_stats)

    if step > 1:
        states = states[::step]
        actions = actions[::step]
        if goals is not None:
            goals = goals[::step]

    t = np.arange(states.shape[0])
    dataset_name = buf_path.parent.name

    if output_dir is None:
        output_dir = buf_path.parent / "visualizations" / episode_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_labels = build_state_labels(states.shape[1], norm_stats)
    action_labels = build_action_labels(actions.shape[1], action_stats)

    print_state_minmax(
        states,
        labels=state_labels,
        title=f"\nState min/max for {episode_name}:",
    )

    state_plot = output_dir / f"{dataset_name}_{episode_name}_states.png"
    plot_grouped_series(
        f"{dataset_name} {episode_name} States",
        t,
        states,
        state_labels,
        build_state_groups(states.shape[1], norm_stats),
        state_plot,
        color="blue",
        y_limit_mode=y_limit_mode,
    )
    print(f"Saved {state_plot}")

    action_plot = output_dir / f"{dataset_name}_{episode_name}_actions.png"
    plot_grouped_series(
        f"{dataset_name} {episode_name} Actions",
        t,
        actions,
        action_labels,
        build_action_groups(actions.shape[1], action_stats),
        action_plot,
        color="red",
        y_limit_mode=y_limit_mode,
        subplot_top=0.96,
        suptitle_y=0.992,
    )
    print(f"Saved {action_plot}")

    if goals is not None:
        goal_plot = output_dir / f"{dataset_name}_{episode_name}_goals.png"
        goal_labels = [f"Goal {i + 1}" for i in range(goals.shape[1])]
        plot_grouped_series(
            f"{dataset_name} {episode_name} Goals",
            t[: goals.shape[0]],
            goals,
            goal_labels,
            [("Goals", list(range(goals.shape[1])))],
            goal_plot,
            color="black",
            y_limit_mode=y_limit_mode,
            subplot_top=0.90,
            suptitle_y=0.992,
        )
        print(f"Saved {goal_plot}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buffer",
        default="/home/ferdinand/activeinference/factr/process_data/processed_data/fourgoals_2_allgauss_noclip_cmdinput/buf_test.pkl",
    )
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--denormalize",
        default=0,
        type=int,
        help="Whether to denormalize the states and actions using the rollout config stats.",
    )
    parser.add_argument("--y-limit-mode", default="fixed", choices=["auto", "fixed"])
    args = parser.parse_args()

    if args.list_episodes:
        list_episodes(args.buffer)
        return
    if args.episode_index is None:
        raise ValueError("Set --episode-index or use --list-episodes.")

    plot_episode(
        buf_path=args.buffer,
        episode_index=args.episode_index,
        output_dir=args.output_dir,
        step=args.step,
        denormalize=bool(args.denormalize),
        y_limit_mode=str(args.y_limit_mode),
    )


if __name__ == "__main__":
    main()
