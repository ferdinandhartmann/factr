import pickle
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np


def load_data(data_path):
    """Load data from pkl file."""
    data_path = Path(data_path)
    if data_path.suffix == ".pkl":
        with open(data_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}. Must be .pkl")


def extract_topic_data(pkl_data, topic_name):
    """Extract data for a specific topic from pkl format."""
    if topic_name not in pkl_data["data"]:
        return None, None

    data = pkl_data["data"][topic_name]
    timestamps = pkl_data["timestamps"][topic_name]

    return data, timestamps


def safe_extract_vector(data_list: List[Any], key: str, dim: int) -> np.ndarray:
    """
    Safely extracts fixed-size vector data from the list of dictionary entries.
    Guarantees a 2D array (N, dim), padding with NaN for corrupt/missing entries.
    """
    processed_data = []

    if data_list is None:
        print("Data list is None. Returning empty array.")
        return np.full((0, dim), np.nan, dtype=np.float32)

    for d in data_list:
        if key in d:
            value = d[key]
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == dim:
                processed_data.append(value)
            else:
                print(f"Corrupt or unexpected data length: {value}")
                processed_data.append([np.nan] * dim)
        else:
            print(f"Missing key: {key} in data entry: {d}")
            processed_data.append([np.nan] * dim)

    result = np.array(processed_data, dtype=np.float32)
    print(f"Processed data shape: {result.shape}")
    return result


def plot_all_traj_in_one_plot(data_path, output_dir, step=1):
    """
    Overlay plots using ORIGINAL INDICES as the x-axis (no timestamps).
    For each file, topics are trimmed to the same min length so they share the same x.
    `step` optionally downsamples (e.g., step=5 plots every 5th point).
    """
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset_name = Path(data_path).parent.stem
    print(f"Dataset name: {dataset_name}")

    print("Creating allplot")

    topics = {
        "external_torques_broadcaster": "/franka_robot_state_broadcaster/external_joint_torques",
        "measured_joints": "/franka_robot_state_broadcaster/measured_joint_states",
        "pose_command": "/cartesian_impedance_controller/pose_command",
        "ee_velocity": "/cartesian_impedance_controller/ee_velocity",
        "tracking_error": "/cartesian_impedance_controller/tracking_error",
        "external_wrench_stiffness": "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
        "goal": "/goal",
        "robot_state": "/franka_robot_state_broadcaster/robot_state",
    }

    pkl_files = sorted(data_path.glob("*.pkl"))
    if not pkl_files:
        print("⚠️ No PKL files found!")
        return

    entries = []
    for pkl_file in pkl_files[1:]:
        # try:
        pkl_data = load_data(pkl_file)

        # Extract per-topic raw data (ignore timestamps entirely)
        meas_data, _ = extract_topic_data(pkl_data, topics["measured_joints"])
        cmd_data, _ = extract_topic_data(pkl_data, topics["pose_command"])
        ee_vel_data, _ = extract_topic_data(pkl_data, topics["ee_velocity"])
        tracking_err_data, _ = extract_topic_data(pkl_data, topics["tracking_error"])
        wrench_data, _ = extract_topic_data(pkl_data, topics["external_wrench_stiffness"])
        goal_data, _ = extract_topic_data(pkl_data, topics["goal"])
        torq_brd_data, _ = extract_topic_data(pkl_data, topics["external_torques_broadcaster"])

        # Convert to fixed-size arrays; allow None if missing
        meas_pos = safe_extract_vector(meas_data, "position", 7)  # (N,7)
        cmd_pose = safe_extract_vector(cmd_data, "ee_pose_commanded", 9)  # (N,9)
        brd_torq = safe_extract_vector(torq_brd_data, "effort", 7)  # (N,7)
        ee_vel = safe_extract_vector(ee_vel_data, "data", 6)  # (N,6)
        tracking_err = safe_extract_vector(tracking_err_data, "data", 6)  # (N,6)
        wrench_stiff = safe_extract_vector(wrench_data, "external_wrench", 6)  # (N,6)

        # Trim by min length so each group shares the SAME x indices within this file
        def trim_minlen(arrs):
            valid = [a for a in arrs if a is not None and len(a) > 0]
            if not valid:
                return None, []
            minlen = min(a.shape[0] for a in valid)
            trimmed = [(a[:minlen] if a is not None else None) for a in arrs]
            x = np.arange(minlen)
            return x, trimmed

        # Group 1: positions (measured joints / commanded EE pose)
        x_pos, (meas_pos_t, cmd_pose_t) = trim_minlen([meas_pos, cmd_pose])

        # Group 2: torques (broadcaster / observed)
        # x_tq, (brd_torq_t) = trim_minlen([brd_torq])
        x_tq, trimmed_tqs = trim_minlen([brd_torq])
        brd_torq_t = trimmed_tqs[0] if trimmed_tqs else None

        # Group 3: 6D signals (ee velocity / tracking error / wrench)
        x_6d, (ee_vel_t, tracking_err_t, wrench_stiff_t) = trim_minlen([ee_vel, tracking_err, wrench_stiff])

        # # Downsample if requested
        # if step > 1:
        #     if x_pos is not None:
        #         x_pos = x_pos[::step]
        #         if meas_pos_t is not None: meas_pos_t = meas_pos_t[::step]
        #         if cmd_pos_t is not None:  cmd_pos_t  = cmd_pos_t[::step]
        #         # if obs_pos_t is not None:  obs_pos_t  = obs_pos_t[::step]
        #     if x_tq is not None:
        #         x_tq = x_tq[::step]
        #         if brd_torq_t is not None: brd_torq_t = brd_torq_t[::step]
        #         # if obs_torq_t is not None: obs_torq_t = obs_torq_t[::step]

        def extract_goal_value(goal_list):
            if not goal_list:
                return None
            first = goal_list[0]
            if isinstance(first, dict) and "goal" in first:
                try:
                    return int(first["goal"])
                except (TypeError, ValueError):
                    return None
            return None

        goal_value = extract_goal_value(goal_data)

        if any(v is not None for v in [cmd_pose_t, brd_torq_t, meas_pos_t, ee_vel_t, tracking_err_t, wrench_stiff_t]):
            entries.append(
                {
                    "name": pkl_file.stem,
                    "x_pos": x_pos,
                    "meas_pos": meas_pos_t,
                    "cmd_pose": cmd_pose_t,
                    # 'obs_pos': obs_pos_t,
                    "x_tq": x_tq,
                    "brd_torq": brd_torq_t,
                    "x_6d": x_6d,
                    "ee_vel": ee_vel_t,
                    "tracking_err": tracking_err_t,
                    "wrench_stiff": wrench_stiff_t,
                    "goal": goal_value,
                    # 'obs_torq': obs_torq_t
                }
            )
        else:
            print(f"⚠️ {pkl_file.name}: no usable arrays after trimming.")

        # except Exception as e:
        #     print(f"❌ Error in {pkl_file.name}: {e}")

    if not entries:
        print("⚠️ No usable data to plot.")
        return

    # Determine y-axis limits based on the min and max of commanded EE pose
    y_min_pos = []
    y_max_pos = []

    for j in range(9):
        # all_obs_pos_j = np.concatenate([e['obs_pos'][:, j] for e in entries if e['obs_pos'] is not None], axis=0)
        cmd_pose_series = [e["cmd_pose"][:, j] for e in entries if e["cmd_pose"] is not None]
        if cmd_pose_series:
            all_cmd_pose_j = np.concatenate(cmd_pose_series, axis=0)
            y_min_pos.append(all_cmd_pose_j.min() - 0.1)
            y_max_pos.append(all_cmd_pose_j.max() + 0.1)
        else:
            meas_series = [e["meas_pos"][:, j] for e in entries if e["meas_pos"] is not None]
            if meas_series:
                all_meas_pos_j = np.concatenate(meas_series, axis=0)
                y_min_pos.append(all_meas_pos_j.min() - 0.1)
                y_max_pos.append(all_meas_pos_j.max() + 0.1)
            else:
                y_min_pos.append(-1.0)
                y_max_pos.append(1.0)
    # Overlay plot: Positions
    # ------------------------------
    # fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    # fig.suptitle(f"All Joint Positions of /obs topic of {dataset_name} dataset", fontsize=16, y=0.96)

    # for j in range(7):
    #     ax = axes[j]
    #     for e in entries:
    #         if e['x_pos'] is None:
    #             continue
    #         if e['obs_pos'] is not None:
    #             ax.plot(e['x_pos'], e['obs_pos'][:, j],   alpha=0.3, linewidth=1.0, color='blue', label=f"{e['name']} obs")
    #             ax.set_ylim(y_min_pos[j], y_max_pos[j])
    #     ax.set_ylabel(f"J{j+1} [rad]")
    #     ax.grid(True, alpha=0.3)
    #     # if j == 0:
    #     #     # one combined legend (deduplicate labels)
    #     #     handles, labels = ax.get_legend_handles_labels()
    #     #     uniq = dict(zip(labels, handles))
    #     #     ax.legend(uniq.values(), uniq.keys(), fontsize=7, ncol=3)
    # axes[-1].set_xlabel("Dataset Index")
    # plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    # out_pos = output_dir / f"allplot_{dataset_name}_positions_observed.png"
    # plt.savefig(out_pos, dpi=150)
    # plt.close(fig)
    # print(f"✅ Saved {out_pos}")

    colours_direction = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels_direction = ["goal 1", "goal 2", "goal 3", "goal 4"]
    goal_to_color = {1: colours_direction[0], 2: colours_direction[1], 3: colours_direction[2], 4: colours_direction[3]}
    goal_to_label = {1: labels_direction[0], 2: labels_direction[1], 3: labels_direction[2], 4: labels_direction[3]}

    fig, axes = plt.subplots(9, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f"All Commanded EE Poses of /pose_command topic of {dataset_name} dataset", fontsize=16, y=0.96)

    labels_added = set()
    for j in range(9):
        ax = axes[j]
        for idx, e in enumerate(entries):
            if e["x_pos"] is None:
                continue
            if e["cmd_pose"] is not None:
                goal_value = e.get("goal")
                color = goal_to_color.get(goal_value, "#7f7f7f")
                label = goal_to_label.get(goal_value, "unknown goal")
                show_label = label if label not in labels_added else None
                ax.plot(e["x_pos"], e["cmd_pose"][:, j], alpha=0.5, linewidth=1.0, color=color, label=show_label)
                if show_label:
                    labels_added.add(label)
                ax.set_ylim(y_min_pos[j], y_max_pos[j])
        ax.set_ylabel(f"EE {j + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
    axes[-1].set_xlabel("Dataset Index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    out_pos = output_dir / f"allplot_{dataset_name}_ee_pose_command.png"
    plt.savefig(out_pos, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {out_pos}")

    # fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    # fig.suptitle(f"All Joint Positions of /traj and /obs topic of {dataset_name} dataset", fontsize=16, y=0.96)

    # for j in range(7):
    #     ax = axes[j]
    #     for e in entries[:20]:
    #         if e['x_pos'] is None:
    #             continue
    #         if e['cmd_pos'] is not None:
    #             ax.plot(e['x_pos'], e['cmd_pos'][:, j],   alpha=0.5, linewidth=1.8, color='red', label=f"{e['name']} cmd")
    #         if e['obs_pos'] is not None:
    #             ax.plot(e['x_pos'], e['obs_pos'][:, j],   alpha=0.5, linewidth=1.8, color='blue', label=f"{e['name']} obs")
    #     ax.set_ylabel(f"J{j+1} [rad]")
    #     ax.grid(True, alpha=0.3)
    # axes[-1].set_xlabel("Dataset Index")
    # plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    # out_pos = output_dir / f"allplot_{dataset_name}_positions_traj_obs.png"
    # plt.savefig(out_pos, dpi=150)
    # plt.close(fig)
    # print(f"✅ Saved {out_pos}")

    # ------------------------------
    # Overlay plot: Torques
    # ------------------------------
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"All Joint Torques of /broadcast topic of {dataset_name} dataset", fontsize=16, y=0.96)

    labels_added = set()
    for j in range(7):
        ax = axes[j]
        for idx, e in enumerate(entries):
            if e["x_tq"] is None:
                continue
            if e["brd_torq"] is not None:
                goal_value = e.get("goal")
                color = goal_to_color.get(goal_value, "#7f7f7f")
                label = goal_to_label.get(goal_value, "unknown goal")
                show_label = label if label not in labels_added else None
                ax.plot(e["x_tq"], e["brd_torq"][:, j], alpha=0.4, linewidth=1.0, color=color, label=show_label)
                if show_label:
                    labels_added.add(label)
        ax.set_ylabel(f"J{j + 1} [Nm]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
    axes[-1].set_xlabel("Dataset Index")
    plt.tight_layout(rect=(0.03, 0.03, 0.97, 0.96))
    out_tq = output_dir / f"allplot_{dataset_name}_torques_broadcasted.png"
    plt.savefig(out_tq, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {out_tq}")

    # ------------------------------
    # Overlay plot: EE Velocity (6D)
    # ------------------------------
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"EE Velocity of /ee_velocity topic of {dataset_name} dataset", fontsize=16, y=0.96)

    labels_added = set()
    for j in range(6):
        ax = axes[j]
        for idx, e in enumerate(entries):
            if e["x_6d"] is None:
                continue
            if e["ee_vel"] is not None:
                goal_value = e.get("goal")
                color = goal_to_color.get(goal_value, "#7f7f7f")
                label = goal_to_label.get(goal_value, "unknown goal")
                show_label = label if label not in labels_added else None
                ax.plot(e["x_6d"], e["ee_vel"][:, j], alpha=0.4, linewidth=1.0, color=color, label=show_label)
                if show_label:
                    labels_added.add(label)
        ax.set_ylabel(f"V{j + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
    axes[-1].set_xlabel("Dataset Index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    out_vel = output_dir / f"allplot_{dataset_name}_ee_velocity.png"
    plt.savefig(out_vel, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {out_vel}")

    # ------------------------------
    # Overlay plot: Tracking Error (6D)
    # ------------------------------
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Tracking Error of /tracking_error topic of {dataset_name} dataset", fontsize=16, y=0.96)

    labels_added = set()
    for j in range(6):
        ax = axes[j]
        for idx, e in enumerate(entries):
            if e["x_6d"] is None:
                continue
            if e["tracking_err"] is not None:
                goal_value = e.get("goal")
                color = goal_to_color.get(goal_value, "#7f7f7f")
                label = goal_to_label.get(goal_value, "unknown goal")
                show_label = label if label not in labels_added else None
                ax.plot(e["x_6d"], e["tracking_err"][:, j], alpha=0.4, linewidth=1.0, color=color, label=show_label)
                if show_label:
                    labels_added.add(label)
        ax.set_ylabel(f"E{j + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
    axes[-1].set_xlabel("Dataset Index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    out_err = output_dir / f"allplot_{dataset_name}_tracking_error.png"
    plt.savefig(out_err, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {out_err}")

    # ------------------------------
    # Overlay plot: Wrench in stiffness frame (6D)
    # ------------------------------
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"External Wrench in Stiffness Frame of {dataset_name} dataset", fontsize=16, y=0.96)

    labels_added = set()
    for j in range(6):
        ax = axes[j]
        for idx, e in enumerate(entries):
            if e["x_6d"] is None:
                continue
            if e["wrench_stiff"] is not None:
                goal_value = e.get("goal")
                color = goal_to_color.get(goal_value, "#7f7f7f")
                label = goal_to_label.get(goal_value, "unknown goal")
                show_label = label if label not in labels_added else None
                ax.plot(e["x_6d"], e["wrench_stiff"][:, j], alpha=0.4, linewidth=1.0, color=color, label=show_label)
                if show_label:
                    labels_added.add(label)
        ax.set_ylabel(f"W{j + 1}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=10)
    axes[-1].set_xlabel("Dataset Index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    out_wrench = output_dir / f"allplot_{dataset_name}_wrench_stiffness.png"
    plt.savefig(out_wrench, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {out_wrench}")

    # fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    # fig.suptitle(f"All Joint Torques of /obs topic of {dataset_name} dataset", fontsize=16, y=0.96)

    # for j in range(7):
    #     ax = axes[j]
    #     for e in entries:
    #         if e['x_tq'] is None:
    #             continue
    #         if e['obs_torq'] is not None:
    #             ax.plot(e['x_tq'], e['obs_torq'][:, j], alpha=0.3, linewidth=1.0, color="red", label=f"{e['name']} obs")
    #     ax.set_ylabel(f"J{j+1} [Nm]")
    #     ax.grid(True, alpha=0.3)
    # axes[-1].set_xlabel("Dataset Index")
    # plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    # out_tq = output_dir / f"allplot_{dataset_name}_torques_observed.png"
    # plt.savefig(out_tq, dpi=150)
    # plt.close(fig)
    # print(f"✅ Saved {out_tq}")

    print("🎯 Done: All all-plots created.")


if __name__ == "__main__":
    ################### All Data in one plot ###################

    dataset_folder = Path("/home/ferdinand/activeinference/factr/process_data/data_to_process/fourgoals_1/data/")
    output_folder_allplots = dataset_folder.parent / "visualizations" / "all_in_one_plots"
    plot_all_traj_in_one_plot(dataset_folder, output_folder_allplots)
