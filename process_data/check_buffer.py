#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# FACTR buffer visualizer (for tuple-based buffer.pkl)
# ---------------------------------------------------------------------------

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2



def load_buffer(buf_path):
    buf_path = Path(buf_path)
    if not buf_path.exists():
        raise FileNotFoundError(f"❌ Buffer file not found: {buf_path}")
    with open(buf_path, "rb") as f:
        buf = pickle.load(f)
    print(f"Loaded buffer with {len(buf)} trajectories")
    return buf

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

def plot_buffer(buf_path, output_dir=None, step=1):
    dataset_name = buf_path.split("/")[-2]
    print(f"Checking buffer from {dataset_name}")
    
    buf_path = Path(buf_path)
    buffer = load_buffer(buf_path)

    if output_dir is None:
        output_dir = buf_path.parent / "visualizations"
    output_dir.mkdir(exist_ok=True, parents=True)

    all_states = []
    all_actions = []

    for traj_idx, traj in enumerate(buffer):
        # each traj = list of (obs_dict, action, reward)
        if not traj or not isinstance(traj[0], tuple):
            print(f"⚠️ Skipping traj {traj_idx}: not a tuple-based trajectory")
            continue

        states = []
        actions = []
        for entry in traj:
            try:
                obs_dict, action, reward = entry
                if isinstance(obs_dict, dict) and "state" in obs_dict:
                    states.append(np.array(obs_dict["state"], dtype=float))
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

    if not all_states or not all_actions:
        raise ValueError("❌ No valid 'states' or 'actions' found in buffer!")

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    print(f"📊 States shape: {states.shape}, Actions shape: {actions.shape}")

    if step > 1:
        states = states[::step]
        actions = actions[::step]

    t = np.arange(len(states))

    # -------------------------

    # Plot: States
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"All States (Ext. Torques) from Buffer of {dataset_name}", fontsize=16, y=0.96)
    for j in range(min(7, states.shape[1])):
        ax = axes[j]
        ax.plot(t, states[:, j], color="blue", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(f"State {j+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    out_state = output_dir / f"{dataset_name}_buffer_states.png"
    plt.savefig(out_state, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {out_state}")

    # Plot: Actions
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"All Actions (Joint Angles) from Buffer of {dataset_name}", fontsize=16, y=0.96)
    for j in range(min(7, actions.shape[1])):
        ax = axes[j]
        ax.plot(t, actions[:, j], color="red", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(f"Act {j+1} [rad]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    out_act = output_dir / f"{dataset_name}_buffer_actions.png"
    plt.savefig(out_act, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {out_act}")

    # Plot: States zoomed in
    only_first_datapoints = 900
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"All States (Ext. Torques) from Buffer of {dataset_name} (first {only_first_datapoints} datapoints)", fontsize=16, y=0.96)
    for j in range(min(7, states.shape[1])):
        ax = axes[j]
        ax.plot(t[:only_first_datapoints], states[:only_first_datapoints, j], color="blue", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(f"State {j+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    out_state = output_dir / f"{dataset_name}_buffer_states_zoomed.png"
    plt.savefig(out_state, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {out_state}")

    # Plot: Actions zoomed in
    only_first_datapoints = 900
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"All Actions (Joint Angles) from Buffer of {dataset_name} (first {only_first_datapoints} datapoints)", fontsize=16, y=0.96)
    for j in range(min(7, actions.shape[1])):
        ax = axes[j]
        ax.plot(t[:only_first_datapoints], actions[:only_first_datapoints, j], color="red", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(f"Act {j+1} [rad]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    out_state = output_dir / f"{dataset_name}_buffer_actions_zoomed.png"
    plt.savefig(out_state, dpi=150)
    plt.close(fig)
    print(f"✅ Saved {out_state}")


    save_camera_images(buffer, output_dir / "camera_images")

    print("🎯 Done: All buffer plots created.")

if __name__ == "__main__":

    buf_path = "/home/ferdinand/factr/process_data/training_data/20251024_60_25hz_filt/buf.pkl"

    plot_buffer(buf_path)
