#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# FACTR buffer visualizer (for tuple-based buffer.pkl)
# ---------------------------------------------------------------------------

import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_buffer(buf_path):
    buf_path = Path(buf_path)
    if not buf_path.exists():
        raise FileNotFoundError(f"❌ Buffer file not found: {buf_path}")
    with open(buf_path, "rb") as f:
        buf = pickle.load(f)
    print(f"Loaded buffer with {len(buf)} trajectories")
    return buf


def load_rollout_config(buf_path, rollout_config_path=None):
    if rollout_config_path is None:
        rollout_config_path = Path(buf_path).parent / "rollout_config.yaml"
    rollout_config_path = Path(rollout_config_path)
    if not rollout_config_path.exists():
        return None
    with open(rollout_config_path, "r") as f:
        return yaml.safe_load(f)


def _inverse_group_transform(values, group):
    gtype = group.get("type", "identity")
    if gtype == "min_max":
        mins = np.asarray(group.get("min", []), dtype=float)
        maxs = np.asarray(group.get("max", []), dtype=float)
        return (values + 1.0) * 0.5 * (maxs - mins) + mins
    if gtype in ("gaussian", "zscore_clip"):
        mean = np.asarray(group.get("mean", []), dtype=float)
        std = np.asarray(group.get("std", []), dtype=float)
        return values * std + mean
    if gtype == "fixed_scale_clip":
        scales = np.asarray(group.get("scales", []), dtype=float)
        return values * scales
    if gtype == "log1p_zscore_clip":
        mean = np.asarray(group.get("mean", []), dtype=float)
        std = np.asarray(group.get("std", []), dtype=float)
        log_values = values * std + mean
        return np.sign(log_values) * (np.expm1(np.abs(log_values)))
    if gtype == "log1p":
        return np.sign(values) * (np.expm1(np.abs(values)))
    return values


def denormalize_states(states, norm_stats):
    if not norm_stats or norm_stats.get("mode") != "grouped":
        return states
    states = states.copy()
    for group in norm_stats.get("groups", []):
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = indices
        sl = slice(int(start), int(stop))
        states[:, sl] = _inverse_group_transform(states[:, sl], group)
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
        start, stop = indices
        sl = slice(int(start), int(stop))
        actions[:, sl] = _inverse_group_transform(actions[:, sl], group)
    return actions


def build_state_labels(state_dim, norm_stats):
    labels = [f"{i + 1}" for i in range(state_dim)]
    if not norm_stats or norm_stats.get("mode") != "grouped":
        return labels

    name_map = {
        "ee_position": ["ee_pos_x", "ee_pos_y", "ee_pos_z"],
        "ee_orientation": [
            "ee_rot_c1_x",
            "ee_rot_c1_y",
            "ee_rot_c1_z",
            "ee_rot_c2_x",
            "ee_rot_c2_y",
            "ee_rot_c2_z",
        ],
        "ee_velocity": ["ee_vel_x", "ee_vel_y", "ee_vel_z", "ee_vel_rx", "ee_vel_ry", "ee_vel_rz"],
        "tracking_error": ["trk_x", "trk_y", "trk_z", "trk_rx", "trk_ry", "trk_rz"],
        "external_wrench": ["w_fx", "w_fy", "w_fz", "w_tx", "w_ty", "w_tz"],
    }

    for group in norm_stats.get("groups", []):
        indices = group.get("indices", None)
        if not indices or len(indices) != 2:
            continue
        start, stop = indices
        start, stop = int(start), int(stop)
        group_name = group.get("name", "group")
        names = name_map.get(group_name, [])
        length = stop - start
        if len(names) != length:
            names = [f"{group_name}_{i}" for i in range(length)]
        for i in range(length):
            labels[start + i] = names[i]
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
            dims = [d for d in range(start, min(stop, state_dim))]
            if not dims:
                continue
            name = str(group.get("name", "group"))
            title = title_map.get(name, name.replace("_", " ").title())
            groups.append((title, dims))
        if groups:
            return groups

    if state_dim >= 27:
        dims = list(range(27))
        return [
            ("EE Pose", dims[0:9]),
            ("EE Velocity", dims[9:15]),
            ("Tracking Error", dims[15:21]),
            ("External Wrench", dims[21:27]),
        ]
    return [("State", list(range(state_dim)))]


def plot_grouped_series(title, x, data, labels, groups, output_path, color, y_label_fontsize=12):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator

    labels = [str(l).replace("_", " ") for l in labels]
    grouped_dims = []
    data_dim = data.shape[1]
    for gtitle, gdims in groups:
        selected = [d for d in gdims if 0 <= d < data_dim]
        if selected:
            grouped_dims.append((gtitle, selected))
    if not grouped_dims:
        grouped_dims = [("Series", list(range(data_dim)))]

    total_rows = sum(len(gdims) for _, gdims in grouped_dims) + max(0, len(grouped_dims) - 1)
    height_ratios = []
    for idx, (_, gdims) in enumerate(grouped_dims):
        height_ratios.extend([1.0] * len(gdims))
        if idx < len(grouped_dims) - 1:
            height_ratios.append(0.35)

    fig = plt.figure(figsize=(12, 1.8 * (total_rows - (len(grouped_dims) - 1)) + 0.25 * (len(grouped_dims) - 1)))
    gs = GridSpec(total_rows, 1, height_ratios=height_ratios, hspace=0.18)

    axes = []
    row = 0
    for group_index, (gtitle, gdims) in enumerate(grouped_dims):
        first_ax = None
        for dim in gdims:
            ax = fig.add_subplot(gs[row, 0], sharex=axes[0] if axes else None)
            if first_ax is None:
                first_ax = ax
            ax.plot(x, data[:, dim], color=color, linewidth=1.0, alpha=0.7)
            ax.set_ylabel(labels[dim], fontsize=y_label_fontsize)
            ax.grid(True, alpha=0.3)
            axes.append(ax)
            row += 1
        if first_ax is not None:
            first_ax.set_title(gtitle, loc="center", fontsize=13, pad=8)
        if gdims:
            last_ax = axes[-1]
            last_ax.set_xlabel("Frame index", fontsize=10)
            for ax in axes[-len(gdims) : -1]:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False, bottom=False)
            last_ax.tick_params(axis="x", labelbottom=True, bottom=True)
        if group_index < len(grouped_dims) - 1:
            spacer = fig.add_subplot(gs[row, 0])
            spacer.axis("off")
            row += 1

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout(pad=0.05)
    fig.subplots_adjust(top=0.985, bottom=0.04)
    fig.suptitle(title, fontsize=14, y=0.995)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_camera_images(buffer, output_dir, plot_index=20):
    """
    Save decoded RGB images from enc_cam_0 at regular intervals.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    saved = 0

    for traj_idx, traj in enumerate(buffer):
        if not traj or not isinstance(traj[0], tuple):
            continue

        for step_idx, (obs, action, reward) in enumerate(traj):
            if step_idx != plot_index:
                continue

            if "enc_cam_0" not in obs:
                continue

            img_enc = obs["enc_cam_0"]

            # decode JPEG bytes
            if isinstance(img_enc, np.ndarray):
                img = cv2.imdecode(img_enc, cv2.IMREAD_COLOR)
            elif isinstance(img_enc, bytes):
                img = cv2.imdecode(np.frombuffer(img_enc, np.uint8), cv2.IMREAD_COLOR)
            else:
                print(f"⚠️ Unexpected image format in traj {traj_idx}, step {step_idx}")
                continue

            if img is None:
                print(f"⚠️ Could not decode image at traj {traj_idx}, step {step_idx}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            filename = output_dir / f"traj{traj_idx:02d}_frame{step_idx:03d}.jpg"
            cv2.imwrite(str(filename), img)
            saved += 1

    print(f"✅ Saved {saved} images to: {output_dir}")


def plot_buffer(buf_path, output_dir=None, step=1, rollout_config_path=None, denormalize=True):
    dataset_name = buf_path.split("/")[-2]
    print(f"Checking buffer from {dataset_name}")

    buf_path = Path(buf_path)
    buffer = load_buffer(buf_path)

    rollout_config = load_rollout_config(buf_path, rollout_config_path=rollout_config_path)
    norm_stats = None
    action_stats = None
    if rollout_config:
        norm_stats = rollout_config.get("norm_stats", {}).get("state", None)
        action_stats = rollout_config.get("norm_stats", {}).get("action", None)

    if output_dir is None:
        output_dir = buf_path.parent / "visualizations"
    output_dir.mkdir(exist_ok=True, parents=True)

    all_states = []
    all_actions = []
    all_goals = []

    for traj_idx, traj in enumerate(buffer):
        # each traj = list of (obs_dict, action, reward)
        if not traj or not isinstance(traj[0], tuple):
            print(f"⚠️ Skipping traj {traj_idx}: not a tuple-based trajectory")
            continue

        states = []
        actions = []
        goals = []
        for entry in traj:
            try:
                obs_dict, action, reward = entry
                if isinstance(obs_dict, dict) and "state" in obs_dict:
                    states.append(np.array(obs_dict["state"], dtype=float))
                if isinstance(obs_dict, dict):
                    if "goals" in obs_dict:
                        goals.append(np.array(obs_dict["goals"], dtype=float))
                    elif "goal" in obs_dict:
                        goals.append(np.array(obs_dict["goal"], dtype=float))
                actions.append(np.array(action, dtype=float))
            except Exception as e:
                print(f"⚠️ Skipping bad entry in traj {traj_idx}: {e}")
                continue

        if len(states) > 0 and len(actions) > 0:
            min_len = min(len(states), len(actions))
            states = np.stack(states[:min_len])
            actions = np.stack(actions[:min_len])
            all_states.append(states)
            all_actions.append(actions)
            if len(goals) > 0:
                goals = np.stack(goals[:min_len])
                all_goals.append(goals)

    if not all_states or not all_actions:
        raise ValueError("❌ No valid 'states' or 'actions' found in buffer!")

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    goals = np.concatenate(all_goals, axis=0) if all_goals else None
    print(f"📊 States shape: {states.shape}, Actions shape: {actions.shape}")

    if denormalize and norm_stats:
        states = denormalize_states(states, norm_stats)
    if denormalize and action_stats:
        actions = denormalize_actions(actions, action_stats)

    if step > 1:
        states = states[::step]
        actions = actions[::step]
        if goals is not None:
            goals = goals[::step]

    t = np.arange(len(states))

    # -------------------------

    state_labels = build_state_labels(states.shape[1], norm_stats)

    state_groups = build_state_groups(states.shape[1], norm_stats)

    # Plot: States
    out_state = output_dir / f"{dataset_name}_buffer_states.png"
    plot_grouped_series(
        f"All States from Buffer of {dataset_name}",
        t,
        states,
        state_labels,
        state_groups,
        out_state,
        color="blue",
        y_label_fontsize=12,
    )
    print(f"✅ Saved {out_state}")

    # Plot: Actions
    out_act = output_dir / f"{dataset_name}_buffer_actions.png"
    action_labels = [f"Act {j + 1}" for j in range(actions.shape[1])]
    plot_grouped_series(
        f"All Actions from Buffer of {dataset_name}",
        t,
        actions,
        action_labels,
        [("Actions", list(range(actions.shape[1])))],
        out_act,
        color="red",
        y_label_fontsize=12,
    )
    print(f"✅ Saved {out_act}")

    # Plot: States zoomed in
    from_first_datapoints = 1000
    only_first_datapoints = 1800
    out_state = output_dir / f"{dataset_name}_buffer_states_zoomed.png"
    plot_grouped_series(
        f"All States from Buffer of {dataset_name} (first {only_first_datapoints} datapoints)",
        t[from_first_datapoints:only_first_datapoints],
        states[from_first_datapoints:only_first_datapoints],
        state_labels,
        state_groups,
        out_state,
        color="blue",
        y_label_fontsize=12,
    )
    print(f"✅ Saved {out_state}")

    # Plot: Actions zoomed in
    out_state = output_dir / f"{dataset_name}_buffer_actions_zoomed.png"
    plot_grouped_series(
        f"All Actions from Buffer of {dataset_name} (first {only_first_datapoints} datapoints)",
        t[from_first_datapoints:only_first_datapoints],
        actions[from_first_datapoints:only_first_datapoints],
        action_labels,
        [("Actions", list(range(actions.shape[1])))],
        out_state,
        color="red",
        y_label_fontsize=12,
    )
    print(f"✅ Saved {out_state}")

    # Plot: Goals (if present)
    if goals is not None:
        goals = np.atleast_2d(goals)
        if goals.shape[0] != len(t):
            goals = goals[: len(t)]
        out_goal = output_dir / f"{dataset_name}_buffer_goals.png"
        goal_labels = [f"Goal {j + 1}" for j in range(goals.shape[1])]
        plot_grouped_series(
            f"Goals from Buffer of {dataset_name}",
            t[: goals.shape[0]],
            goals,
            goal_labels,
            [("Goals", list(range(goals.shape[1])))],
            out_goal,
            color="black",
            y_label_fontsize=12,
        )
        print(f"✅ Saved {out_goal}")

    # save_camera_images(buffer, output_dir / "camera_images")

    print("🎯 Done: All buffer plots created.")


if __name__ == "__main__":
    buf_path = "/home/ferdinand/activeinference/factr/process_data/training_data/fourgoals_1_newnorm_train/buf.pkl"

    plot_buffer(buf_path)
