
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import sys
from typing import List, Any

def load_data(data_path):
    """Load data from pkl file."""
    data_path = Path(data_path)
    if data_path.suffix == '.pkl':
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}. Must be .pkl")

def extract_topic_data(pkl_data, topic_name):
    """Extract data for a specific topic from pkl format."""
    if topic_name not in pkl_data['data']:
        return None, None
    
    data = pkl_data['data'][topic_name]
    timestamps = pkl_data['timestamps'][topic_name]
    
    return data, timestamps

def safe_extract_7d_data(data_list: List[Any], key: str) -> np.ndarray:
    """
    Safely extracts 7-dimensional joint data ('position' or 'effort') 
    from the list of dictionary entries. 
    Guarantees a 2D array (N, 7), padding with NaN for corrupt/missing entries.
    """
    processed_data = []
        
    if data_list is None:
        print("Data list is None. Returning empty array.")
        return np.full((0, 7), np.nan, dtype=np.float32)

    for d in data_list:
        if key in d:
            value = d[key]
            # Check if the value is a sequence of length 7
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 7:
                processed_data.append(value)
            else:
                # Corrupt or unexpected length (e.g., gripper data accidentally logged)
                print(f"Corrupt or unexpected data length: {value}")
                processed_data.append([np.nan] * 7)
        else:
            # Missing key
            print(f"Missing key: {key} in data entry: {d}")
            processed_data.append([np.nan] * 7)
            
    # Convert to NumPy array with float dtype to allow NaNs
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
        # 'franka_state': '/franka/right/obs_franka_state',
        'external_torques_broadcaster': '/franka_robot_state_broadcaster/external_joint_torques',
        # 'franka_torque_leader': '/franka/right/obs_franka_torque',
        # 'impedance_cmd': '/joint_impedance_command_controller/joint_trajectory',
        'impedance_cmd': '/joint_impedance_dynamic_gain_controller/joint_impedance_command',
        'measured_joints': '/franka_robot_state_broadcaster/measured_joint_states',
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
        meas_data, _ = extract_topic_data(pkl_data, topics['measured_joints'])
        cmd_data, _ = extract_topic_data(pkl_data, topics['impedance_cmd'])
        # obs_state_data, _ = extract_topic_data(pkl_data, topics['franka_state'])
        torq_brd_data, _ = extract_topic_data(pkl_data, topics['external_torques_broadcaster'])
        # torq_obs_data, _ = extract_topic_data(pkl_data, topics['franka_torque_leader'])

        # Convert to (N, 7) arrays; allow None if missing
        meas_pos = safe_extract_7d_data(meas_data, 'position')            # (N,7) or None
        cmd_pos = safe_extract_7d_data(cmd_data, 'position')              # (N,7) or None
        # obs_pos = safe_extract_7d_data(obs_state_data, 'position')        # (N,7) or None
        brd_torq = safe_extract_7d_data(torq_brd_data, 'effort')          # (N,7) or None
        # obs_torq = safe_extract_7d_data(torq_obs_data, 'effort')          # (N,7) or None

        # Trim by min length so each group shares the SAME x indices within this file
        def trim_minlen(arrs):
            valid = [a for a in arrs if a is not None and len(a) > 0]
            if not valid:
                return None, []
            minlen = min(a.shape[0] for a in valid)
            trimmed = [(a[:minlen] if a is not None else None) for a in arrs]
            x = np.arange(minlen)
            return x, trimmed

        # Group 1: positions (measured / commanded / observed)
        x_pos, (meas_pos_t, cmd_pos_t) = trim_minlen([meas_pos, cmd_pos])

        # Group 2: torques (broadcaster / observed)
        # x_tq, (brd_torq_t) = trim_minlen([brd_torq])
        x_tq, trimmed_tqs = trim_minlen([brd_torq])
        brd_torq_t = trimmed_tqs[0] if trimmed_tqs else None

        # Downsample if requested
        if step > 1:
            if x_pos is not None:
                x_pos = x_pos[::step]
                if meas_pos_t is not None: meas_pos_t = meas_pos_t[::step]
                if cmd_pos_t is not None:  cmd_pos_t  = cmd_pos_t[::step]
                # if obs_pos_t is not None:  obs_pos_t  = obs_pos_t[::step]
            if x_tq is not None:
                x_tq = x_tq[::step]
                if brd_torq_t is not None: brd_torq_t = brd_torq_t[::step]
                # if obs_torq_t is not None: obs_torq_t = obs_torq_t[::step]

        if any(v is not None for v in [cmd_pos_t, brd_torq_t]):
            entries.append({
                'name': pkl_file.stem,
                'x_pos': x_pos,
                'meas_pos': meas_pos_t,
                'cmd_pos': cmd_pos_t,
                # 'obs_pos': obs_pos_t,
                'x_tq': x_tq,
                'brd_torq': brd_torq_t,
                # 'obs_torq': obs_torq_t
            })
        else:
            print(f"⚠️ {pkl_file.name}: no usable arrays after trimming.")

        # except Exception as e:
        #     print(f"❌ Error in {pkl_file.name}: {e}")

    if not entries:
        print("⚠️ No usable data to plot.")
        return

    # Determine y-axis limits based on the min and max of obs_pos and cmd_pos for each joint
    y_min_pos = []
    y_max_pos = []

    for j in range(7):
        # all_obs_pos_j = np.concatenate([e['obs_pos'][:, j] for e in entries if e['obs_pos'] is not None], axis=0)
        all_cmd_pos_j = np.concatenate([e['cmd_pos'][:, j] for e in entries if e['cmd_pos'] is not None], axis=0)
        y_min_pos.append(all_cmd_pos_j.min() - 0.1)
        y_max_pos.append(all_cmd_pos_j.max() + 0.1)
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

    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"All Joint Positions of /traj topic of {dataset_name} dataset", fontsize=16, y=0.96)

    for j in range(7):
        ax = axes[j]
        for e in entries:
            if e['x_pos'] is None: 
                continue
            if e['cmd_pos'] is not None:
                ax.plot(e['x_pos'], e['cmd_pos'][:, j],   alpha=0.3, linewidth=1.0, color='red', label=f"{e['name']} cmd")
                ax.set_ylim(y_min_pos[j], y_max_pos[j])
        ax.set_ylabel(f"J{j+1} [rad]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Dataset Index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    out_pos = output_dir / f"allplot_{dataset_name}_positions_traj.png"
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

    for j in range(7):
        ax = axes[j]
        for e in entries:
            if e['x_tq'] is None:
                continue
            if e['brd_torq'] is not None:
                ax.plot(e['x_tq'], e['brd_torq'][:, j], alpha=0.3, linewidth=1.0, color="blue", label=f"{e['name']} brd")
        ax.set_ylabel(f"J{j+1} [Nm]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Dataset Index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    out_tq = output_dir / f"allplot_{dataset_name}_torques_broadcasted.png"
    plt.savefig(out_tq, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {out_tq}")

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




if __name__ == '__main__':

    ################### All Data in one plot ###################

    dataset_folder = Path("/home/ferdinand/factr/process_data/data_to_process/20251112/data/")
    output_folder_allplots = dataset_folder.parent / "visualizations" / "all_in_one_plots"
    plot_all_traj_in_one_plot(dataset_folder, output_folder_allplots)
