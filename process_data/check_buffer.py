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

    if step > 1:
        states = states[::step]
        actions = actions[::step]
        if goals is not None:
            goals = goals[::step]

    t = np.arange(len(states))

    # -------------------------
    
    plot_length_scale_state = 1.2
    plot_length_scale_action = 1.3

    # Plot: States
    fig, axes = plt.subplots(states.shape[1], 1, figsize=(12, plot_length_scale_state*states.shape[1]), sharex=True)
    fig.suptitle(f"All States from Buffer of {dataset_name}", fontsize=16, y=0.96)
    for j in range(states.shape[1]):
        ax = axes[j]
        ax.plot(t, states[:, j], color="blue", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(f"{j+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame index")
    plt.tight_layout(rect=(0.03, 0.03, 0.97, 0.96))
    out_state = output_dir / f"{dataset_name}_buffer_states.png"
    plt.savefig(out_state, dpi=300)
    plt.close(fig)
    print(f"✅ Saved {out_state}")

    # Plot: Actions
    fig, axes = plt.subplots(actions.shape[1], 1, figsize=(12, plot_length_scale_action*actions.shape[1]), sharex=True)
    fig.suptitle(f"All Actions from Buffer of {dataset_name}", fontsize=16, y=0.96)
    for j in range(actions.shape[1]):
        ax = axes[j]
        ax.plot(t, actions[:, j], color="red", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(f"Act {j+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame index")
    plt.tight_layout(rect=(0.03, 0.03, 0.97, 0.96))
    out_act = output_dir / f"{dataset_name}_buffer_actions.png"
    plt.savefig(out_act, dpi=300)
    plt.close(fig)
    print(f"✅ Saved {out_act}")

    # Plot: States zoomed in
    from_first_datapoints = 1000
    only_first_datapoints = 1800
    fig, axes = plt.subplots(states.shape[1], 1, figsize=(12, plot_length_scale_state*states.shape[1]), sharex=True)
    fig.suptitle(f"All States from Buffer of {dataset_name} (first {only_first_datapoints} datapoints)", fontsize=16, y=0.96)
    for j in range(states.shape[1]):
        ax = axes[j]
        ax.plot(t[from_first_datapoints:only_first_datapoints], states[from_first_datapoints:only_first_datapoints, j], color="blue", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(f"{j+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame index")
    plt.tight_layout(rect=(0.03, 0.03, 0.97, 0.96))
    out_state = output_dir / f"{dataset_name}_buffer_states_zoomed.png"
    plt.savefig(out_state, dpi=300)
    plt.close(fig)
    print(f"✅ Saved {out_state}")

    # Plot: Actions zoomed in
    fig, axes = plt.subplots(actions.shape[1], 1, figsize=(12, plot_length_scale_action*actions.shape[1]), sharex=True)
    fig.suptitle(f"All Actions from Buffer of {dataset_name} (first {only_first_datapoints} datapoints)", fontsize=16, y=0.96)
    for j in range(actions.shape[1]):
        ax = axes[j]
        ax.plot(t[from_first_datapoints:only_first_datapoints], actions[from_first_datapoints:only_first_datapoints, j], color="red", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(f"Act {j+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame index")
    plt.tight_layout(rect=(0.03, 0.03, 0.97, 0.96))
    out_state = output_dir / f"{dataset_name}_buffer_actions_zoomed.png"
    plt.savefig(out_state, dpi=300)
    plt.close(fig)
    print(f"✅ Saved {out_state}")

    # Plot: Goals (if present)
    if goals is not None:
        goals = np.atleast_2d(goals)
        if goals.shape[0] != len(t):
            goals = goals[:len(t)]
        fig, axes = plt.subplots(goals.shape[1], 1, figsize=(12, 4), sharex=True)
        fig.suptitle(f"Goals from Buffer of {dataset_name}", fontsize=16, y=0.96)
        if goals.shape[1] == 1:
            axes = [axes]
        for j in range(goals.shape[1]):
            ax = axes[j]
            ax.plot(t, goals[:, j], color="black", linewidth=1.0, alpha=0.7)
            ax.set_ylabel(f"Goal {j+1}")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Frame index")
        plt.tight_layout(rect=(0.03, 0.03, 0.97, 0.96))
        out_goal = output_dir / f"{dataset_name}_buffer_goals.png"
        plt.savefig(out_goal, dpi=300)
        plt.close(fig)
        print(f"✅ Saved {out_goal}")

    # save_camera_images(buffer, output_dir / "camera_images")

    print("🎯 Done: All buffer plots created.")

if __name__ == "__main__":

    buf_path = "/home/ferdinand/activeinference/factr/process_data/training_data/fourgoals_1/buf.pkl"

    plot_buffer(buf_path)
