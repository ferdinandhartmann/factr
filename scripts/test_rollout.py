import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from hydra.utils import instantiate
import yaml
import os
import copy
from typing import List, Any

from test_rollout_utils import (
    preprocess_image, load_episode_from_buffer, get_all_joint_cmds_np, 
    calculate_franka_fk, load_and_extract_raw_data, load_pkl, load_and_prepare_policy
)

import plotly.graph_objects as go
import plotly.io as pio


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

plt.rcParams.update(plt.rcParamsDefault)

# ---------- CONFIG ---------- # Select model, checkpoint, and episode here
model_name = "20251112_60_25hz_filt2_7dof_s42_ac25_b64_lr0.0002_iter6000_"
checkpoint = "latest"
episode_names = ["ep_08", "ep_25", "ep_40", "ep_45", "ep_61", "ep_62", "ep_63", "ep_64", "ep_65", "ep_66"] # List of episode names to test

downsample = True # from 50Hz to 25Hz
vs_all_plot = False # whether to load all joint commands from dataset for visualization
use_buffer = False  # !!!!!!!!! somehow different results, i dont know why. load from buffer.pkl instead of raw PKL files 
remove_joints = [] # zero-indexed joints to remove
interactive_3d_plot = True
endeffector_plot = True

# ---------- PATHS & DEVICE ----------
FACTR_REPO = Path(__file__).resolve().parents[1]
print(f"Detected FACTR repo folder: {FACTR_REPO}")

# Checkpoint and configs
CKPT_PATH = None
if checkpoint == "latest":
    CKPT_PATH = Path(f"{FACTR_REPO}/scripts/checkpoints/{model_name}/rollout/latest_ckpt.ckpt")
else:
    CKPT_PATH = Path(f"{FACTR_REPO}/scripts/checkpoints/{model_name}/{checkpoint}.ckpt")
EXP_CFG_PATH = Path(f"{FACTR_REPO}/scripts/checkpoints/{model_name}/rollout/exp_config.yaml")
ROLLOUT_CFG_PATH = Path(f"{FACTR_REPO}/scripts/checkpoints/{model_name}/rollout/rollout_config.yaml")

# Raw data
RAW_DATA_PATH_TRAIN = Path(f"{FACTR_REPO}/process_data/raw_data_train/20251112_60/")
RAW_DATA_PATH_EVAL = Path(f"{FACTR_REPO}/process_data/raw_data_eval/20251112_7/")

# Buffer
BUF_PATH = Path(f"{FACTR_REPO}/process_data/training_data/20251112_60_25hz_filt2/buf.pkl")

# Output folder
output_folder = Path(f"{FACTR_REPO}/scripts/test_rollout_output/{model_name}_{checkpoint}")
output_folder.mkdir(parents=True, exist_ok=True)

# Topics in PKL files
image_topic = "/realsense/front/im"
obs_topic = "/franka_robot_state_broadcaster/external_joint_torques"
# action_topic = "/joint_impedance_command_controller/joint_trajectory"
action_topic = "/joint_impedance_dynamic_gain_controller/joint_impedance_command"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------- MAIN CODE ---------------------------- #

# Load Policy
policy = load_and_prepare_policy(EXP_CFG_PATH, CKPT_PATH, DEVICE)

# --- 3. Load Normalization Stats (NEW SECTION) ---
with open(ROLLOUT_CFG_PATH, 'r') as f:
    rollout_config = yaml.safe_load(f)

# Torque
obs_mean = torch.tensor(rollout_config['norm_stats']['state']['mean']).float().to(DEVICE)
obs_std = torch.tensor(rollout_config['norm_stats']['state']['std']).float().to(DEVICE)

# Policy output (action)
action_mean = torch.tensor(rollout_config['norm_stats']['action']['mean']).float().to(DEVICE)
action_std = torch.tensor(rollout_config['norm_stats']['action']['std']).float().to(DEVICE)
print(f"✅ Loaded normalization stats from {ROLLOUT_CFG_PATH}")

if vs_all_plot:
    print("Loading all joint commands from dataset for visualization...")
    joint_cmds_all, joint_cmds_all_norm = get_all_joint_cmds_np(RAW_DATA_PATH_TRAIN, action_mean, action_std)
    print(f"✅ Loaded and normalized all joint commands from dataset folder.")

image_obs = []
torque_obs = []
true_actions = []
use_indicies = [i for i in range(7) if i not in remove_joints]
print(f"Using joint indices: {use_indicies} (removed joints: {remove_joints})")

# --- Load arrays from PKL file ---
for episode_name in episode_names:
    use_eval = False
    if use_buffer:
        episode_idx = int(episode_name.split("_")[1])  # ep_40 → 40
        try:
            image_obs, torque_obs, true_actions = load_episode_from_buffer(BUF_PATH, episode_idx)
            print(f"Size from buffer - Images: {image_obs.shape}, Torque: {torque_obs.shape}, Actions: {true_actions.shape}")
        except Exception as e:
            print(f"ℹ️ Info: Could not load episode {episode_name} from buffer. Error: {e}, loading from raw PKL instead.")
            RAW_DATA_PATH = RAW_DATA_PATH_EVAL / f"{episode_name}.pkl"
            image_obs, torque_obs, true_actions = load_and_extract_raw_data(RAW_DATA_PATH, downsample=downsample, image_topic=image_topic, obs_topic=obs_topic, action_topic=action_topic)
            use_eval = True

    else:
        RAW_DATA_PATH = RAW_DATA_PATH_TRAIN / f"{episode_name}.pkl"
        if not RAW_DATA_PATH.exists():
            RAW_DATA_PATH = RAW_DATA_PATH_EVAL / f"{episode_name}.pkl"
        if not RAW_DATA_PATH.exists():
            print(f"Required PKL file not found: {RAW_DATA_PATH}, skipping this episode.")
            break 
        image_obs, torque_obs, true_actions = load_and_extract_raw_data(RAW_DATA_PATH, downsample=downsample, image_topic=image_topic, obs_topic=obs_topic, action_topic=action_topic)

    # -----------------------------
    # INFERENCE
    # -----------------------------
    pred_actions = []
    pred_actions_norm = []
    true_action_list = []
    normalized_true_action_list = []
    attn_image, attn_force = [], []
    attn_layer_vectors_list = []

    action_mean_red = action_mean[use_indicies]
    action_std_red = action_std[use_indicies]

    N = min(len(true_actions), len(torque_obs), len(image_obs))
    print(f"🚀 Running inference on {N} samples...")

    for i in tqdm(range(N)):
        img = image_obs[i]
        torque = torque_obs[i]

        img_tensor = preprocess_image(img).to(DEVICE)

        # Normalize Torque Observation
        torque_tensor = torch.from_numpy(torque).float().to(DEVICE)
        torque_tensor = (torque_tensor - obs_mean) / obs_std  # normalization
        torque_tensor = torque_tensor.unsqueeze(0)
        torque_tensor = torque_tensor[:, use_indicies] 

        # Normalize Ground Truth Action or denormalize from buffer
        true_action = true_actions[i]
        true_action_tensor = torch.from_numpy(np.array(true_action)).float().to(DEVICE)
        normalized_true_action = []
        if not use_buffer or use_eval:
            normalized_true_action = (true_action_tensor - action_mean) / action_std # normalization
        else:
            true_action = true_action_tensor * action_std + action_mean  # denormalize from buffer
            true_action = true_action.unsqueeze(0).cpu().numpy()[0]
            normalized_true_action = true_action_tensor
        true_action_list.append(true_action)
        normalized_true_action_list.append(normalized_true_action.unsqueeze(0).cpu().numpy()[0])

        with torch.no_grad():
            pred_action, cross_w = policy.get_actions({"cam0": img_tensor}, torque_tensor, return_weights=True)
            # print(f"size of cross_w: {cross_w.shape}")
            # corss_w shape = (L, B, H, NQ​, NS​)
            #               =([6, 1, 8, 25, 2])
            # 6 Decoder layers
            # 1 Batch size
            # 8 Attention heads
            # 25 Number of action chunks (NQ)
            # 2 Number of source tokens (NS) - 1 image + 1 torque

            if cross_w is not None:
                # Canonicalize attention (5D tensor: L, B, H, Nq, Ns)
                # Remove Batch dimension (B=1) -> (L, H, Nq, Ns)
                cross_w_no_batch = cross_w.squeeze(dim=1) 
                # Mean over Heads (H) -> (L, Nq, Ns)
                attn_heads_mean = cross_w_no_batch.mean(dim=1) 
                # Mean over Queries (Nq) -> (L, Ns)
                attn_layer_vector = attn_heads_mean.mean(dim=1) 
                # This vector is (L=6, Ns=2): rows are layers, columns are tokens [Image, Torque]
                attn_layer_vector_np = attn_layer_vector.cpu().numpy()
                attn_layer_vectors_list.append(attn_layer_vector_np)

                # size of cross_w: torch.Size([6, 1, 8, 25, 2])
                # --- Layer Attention Vector (Timestep 238) ---
                # Shape: torch.Size([6, 2])
                # [Image, Torque] Attention per Layer 1-6:
                # [[0.4363075  0.5636925 ]
                # [0.3447853  0.65521467]
                # [0.38640937 0.6135906 ]
                # [0.55851334 0.44148666]
                # [0.46212408 0.5378759 ]
                # [0.51679295 0.48320693]]

                # Mean over all Layers (L) -> (Ns). This is used to maintain the time-series plot.
                attn_mean_combined = attn_layer_vector.mean(dim=0)                
                N_tokens = attn_mean_combined.shape[-1]
                if N_tokens == 2:
                    # Token 0 is Image, Token 1 is Torque
                    img_attn = attn_mean_combined[0].item()
                    torque_attn = attn_mean_combined[1].item()
                    attn_image.append(img_attn)
                    attn_force.append(torque_attn)
                else:
                    print(f"Warning: Expected 2 source tokens but found {N_tokens}. Skipping attention logging.")

            # test_rollout.py → agent.get_actions() → policy.get_actions() → model.get_actions()
            pred_action_norm = pred_action.cpu().numpy()[0]
            pred_action = pred_action * action_std_red + action_mean_red # Inverse normalization 
            pred_action = pred_action.cpu().numpy()[0]

        pred_actions.append(pred_action)
        pred_actions_norm.append(pred_action_norm)

    pred_actions = np.array(pred_actions)
    pred_actions_norm = np.array(pred_actions_norm)
    true_actions_normalized = np.array(normalized_true_action_list)
    true_actions = np.array(true_action_list)
    attn_layer_vectors_stacked = np.array(attn_layer_vectors_list)
    attn_image_np = np.array(attn_image)
    attn_force_np = np.array(attn_force)

    if len(remove_joints) > 0:
        for remove_joint in sorted(remove_joints):
            pred_actions = np.insert(pred_actions, remove_joint, 0, axis=2)
            pred_actions_norm = np.insert(pred_actions_norm, remove_joint, 0, axis=2)

    print("Finished inference")
    print(f"Pred shape: {pred_actions.shape}, True shape: {true_actions.shape}")

    t = np.arange(len(pred_actions))  # <-- ensure t matches N
    dof_dims = pred_actions.shape[2]
    pred_dims = pred_actions.shape[1]
    print(f"Number of Frames: {t.shape[0]}, dof_dims: {dof_dims}, pred_dims: {pred_dims}")

    # Get max and min of each joint dimension
    max_mins = []
    for d in range(dof_dims): 
        max_val = np.max(pred_actions[:, :, d])
        min_val = np.min(pred_actions[:, :, d])
        max_mins.append((max_val, min_val))
    max_y_diff = max(np.abs(m[0] - m[1]) for m in max_mins)

    # Calculate L2 Loss
    l2_loss = np.mean(np.linalg.norm(pred_actions_norm[:, 0, use_indicies] - true_actions[:, use_indicies], axis=1))
    print(f"Average L2 Loss for all joints: {l2_loss:.6f}")

    # Calculate End-Effector Positions
    q_gt = true_actions[:, :]
    q_pred = pred_actions[:, 0, :] 
    predicted_pos = calculate_franka_fk(q_pred)
    ground_truth_pos = calculate_franka_fk(q_gt)
    ef_error = np.linalg.norm(predicted_pos - ground_truth_pos, axis=1)


    ######################################
    #             PLOTTING 
    ######################################
    if interactive_3d_plot == True:
        # --- Plot 5: 3D Trajectory (The New Plot) ---
        fig = go.Figure()
        # Plot Ground Truth 3D path
        fig.add_trace(go.Scatter3d(
            x=ground_truth_pos[:, 0],
            y=ground_truth_pos[:, 1],
            z=ground_truth_pos[:, 2],
            mode='lines',
            name='GT Trajectory',
            line=dict(color='red', width=4)
        ))
        # Plot Predicted 3D path
        fig.add_trace(go.Scatter3d(
            x=predicted_pos[:, 0],
            y=predicted_pos[:, 1],
            z=predicted_pos[:, 2],
            mode='lines',
            name='Predicted Trajectory',
            line=dict(color='blue', width=4)
        ))
        # Mark start and end points
        fig.add_trace(go.Scatter3d(x=[ground_truth_pos[0, 0]], y=[ground_truth_pos[0, 1]], z=[ground_truth_pos[0, 2]], mode='markers', name='GT Start', marker=dict(color='black', size=6, symbol='circle')))
        fig.add_trace(go.Scatter3d(x=[ground_truth_pos[-1, 0]], y=[ground_truth_pos[-1, 1]], z=[ground_truth_pos[-1, 2]], mode='markers', name='GT End', marker=dict(color='red', size=8, symbol='x')))
        fig.add_trace(go.Scatter3d(x=[predicted_pos[0, 0]], y=[predicted_pos[0, 1]], z=[predicted_pos[0, 2]], mode='markers', name='Pred Start', marker=dict(color='darkgray', size=6, symbol='circle')))
        fig.add_trace(go.Scatter3d(x=[predicted_pos[-1, 0]], y=[predicted_pos[-1, 1]], z=[predicted_pos[-1, 2]], mode='markers', name='Pred End', marker=dict(color='blue', size=8, symbol='x')));

        # Set axis labels, limits, and camera view
        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                xaxis=dict(range=[0.3, 0.6]),
                yaxis=dict(range=[-0.15, 0.15]),
                zaxis=dict(range=[0, 0.55]),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.5)  
                )
            ),
            title=f"End-Effector 3D Trajectory Comparison — {episode_name}",
            legend=dict(x=0.8, y=0.9),
        )

        fig.show() # auto-open in browser
        # plotly_path = f"{output_folder}/tr_efp_{episode_name}.html"
        # pio.write_html(fig, file=plotly_path, auto_open=False)
        # print(f"✅ Saved interactive 3D plot: {plotly_path}")

    if endeffector_plot == True:
        fig = plt.figure(figsize=(15, 15))
        fig.suptitle(f"End-Effector Plots of Ground-Truth and Prediction of {episode_name}", fontsize=16)
        fig.text(0.5, 0.95, f"model: {model_name}", fontsize=11, ha='center', va='top')
                
        ax_x = fig.add_subplot(5, 2, 1) # Plot 1: X vs Time (Top-Left)
        ax_y = fig.add_subplot(5, 2, 2, sharex=ax_x) # Plot 2: Y vs Time (Top-Right)
        ax_z = fig.add_subplot(5, 2, 3, sharex=ax_x) # Plot 3: Z vs Time (Middle-Left)
        ax_err = fig.add_subplot(5, 2, 4, sharex=ax_x) # Plot 4: Positional Error (Middle-Right)
        ax_3d = fig.add_subplot(5, 2, (5, 10), projection='3d') # Plot 5: 3D Trajectory (Bottom, spanning two columns)
        axes = [ax_x, ax_y, ax_z] 
        labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)']
        
        # --- Plot 1, 2, 3: X, Y, Z Axes over Time ---
        for i in range(3):
            axes[i].plot(ground_truth_pos[:, i], label=f'GT {labels[i][0]}', color='red', alpha=0.7, linewidth=1.5)
            axes[i].plot(predicted_pos[:, i], label=f'Predicted {labels[i][0]}', color='blue', linewidth=1.5)
            axes[i].set_ylabel(labels[i])
            axes[i].legend(loc='lower right')
            axes[i].grid(True, alpha=0.4)
            if i == 2:
                axes[i].set_xlabel("Timestep") # Only label X-axis on the bottom-most time plot

        # --- Plot 4: Positional Error ---
        ax_err.plot(ef_error, label='Positional Error (m)', color='black', linewidth=1.5)
        ax_err.set_ylim(0, 0.2)
        ax_x.set_title("End-Effector x-Position Over Time")
        ax_y.set_title("End-Effector y-Position Over Time")
        ax_z.set_title("End-Effector z-Position Over Time")
        ax_err.set_title("End-Effector Positional Error")
        ax_err.set_xlabel("Timestep")
        ax_z.set_xlabel("Timestep")
        ax_err.set_ylabel("Error (m)")
        ax_err.legend()
        ax_err.grid(True, alpha=0.4)

        # --- Plot 5: 3D Trajectory (The New Plot) ---
        # Plot Ground Truth 3D path
        ax_3d.plot(ground_truth_pos[:, 0], ground_truth_pos[:, 1], ground_truth_pos[:, 2], label='GT Trajectory', color='red', linewidth=1.2)
        ax_3d.plot(predicted_pos[:, 0], predicted_pos[:, 1], predicted_pos[:, 2], label='Predicted Trajectory', color='blue', linewidth=1.2)
        # Mark start and end points
        ax_3d.scatter(ground_truth_pos[0, 0], ground_truth_pos[0, 1], ground_truth_pos[0, 2], c='k', marker='o', s=20, label='GT Start')
        ax_3d.scatter(ground_truth_pos[-1, 0], ground_truth_pos[-1, 1], ground_truth_pos[-1, 2], c='r', marker='x', s=50, label='GT End')
        ax_3d.scatter(predicted_pos[0, 0], predicted_pos[0, 1], predicted_pos[0, 2], c='darkgray', marker='o', s=20, label='Pred Start')
        ax_3d.scatter(predicted_pos[-1, 0], predicted_pos[-1, 1], predicted_pos[-1, 2], c='blue', marker='x', s=50, label='Pred End')
        
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_xlim([0.3, 0.6])
        ax_3d.set_ylim([-0.15, 0.15])
        ax_3d.set_zlim([0, 0.55])
        ax_3d.set_title("End-Effector 3D Trajectory Comparison")
        ax_3d.legend(loc='best')
        ax_3d.grid(True, alpha=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
        ax_3d.view_init(elev=7.5, azim=-60)  # Elevation (up-down) and Azimuth (left-right)

        combined_path = f"{output_folder}/tr_efp_{episode_name}.png"
        plt.savefig(combined_path, dpi=250)
        plt.close(fig)
        print(f"✅ Saved end-effector position plots: {combined_path}")


    ############### Combined: Attention plot + 6 evenly spaced images
    # num_imgs = 6
    N_img = len(image_obs)
    # frame_indices = np.linspace(0, N_img - 1, num_imgs, dtype=int)
    desired_indices  = np.array([0, 50, 100, 150, 200, 250])
    frame_indices = np.clip(desired_indices, 0, N_img - 1)

    fig = plt.figure(figsize=(12, 7))

    # ----- 1. Attention plot (top, larger space) -----
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)  # Allocate 2/3 of the height
    ax1.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.6)
    for layer_idx in range(attn_layer_vectors_stacked.shape[1]):
        # ax1.plot(
        #     attn_layer_vectors_stacked[:, layer_idx, 0],
        #     label=f"Layer {layer_idx + 1} - Image",
        #     linestyle="--",
        #     linewidth=1.2,
        #     alpha=0.7,
        #     color=f"C{layer_idx}",
        # )
        # ax1.plot(
        #     attn_layer_vectors_stacked[:, layer_idx, 1],
        #     label=f"Layer {layer_idx + 1} - Torque",
        #     linewidth=1.2,
        #     alpha=0.7,
        #     color=f"C{layer_idx}",
        # )
        linestyle = "-" if layer_idx == 0 else "-" if layer_idx == attn_layer_vectors_stacked.shape[1] - 1 else "--"
        colour = "blue" if layer_idx == 0 else "black" if layer_idx == attn_layer_vectors_stacked.shape[1] - 1 else f"C{layer_idx}"
        alpha = 1.0 if layer_idx == 0 else 1.0 if layer_idx == attn_layer_vectors_stacked.shape[1] - 1 else 0.7
        ax1.plot(
            (attn_layer_vectors_stacked[:, layer_idx, 1] - attn_layer_vectors_stacked[:, layer_idx, 0]),
            label=f"Layer {layer_idx + 1} - (Torque - Image)",
            linewidth=1.8,
            linestyle=linestyle,
            color=colour,
            alpha=alpha,
        )
    # ax1.plot(attn_force_np, label="Torque attention (mean)", linewidth=1.5, color="black")
    # ax1.plot(attn_image_np, label="Image attention (mean)", linewidth=1.5, color="darkgray")
    # ax1.plot((attn_force_np - attn_image_np), label="(Torque - Image) attention mean", linewidth=2.0, color="black")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Mean attention weight")
    ax1.set_title(f"Attention to Force and Image — {episode_name}")
    ax1.legend(loc="lower right")
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.4)
    # Add labels for -1 and 1 limits
    ax1.text(ax1.get_xlim()[0]+0.1, -0.9, "100% Image", va="center", ha="left", fontsize=12, color="gray")
    ax1.text(ax1.get_xlim()[0]+0.1, 0.9, "100% Torque", va="center", ha="left", fontsize=12, color="gray")

    # ----- 2. Image strip (bottom, 1/3 space) -----
    gs = plt.GridSpec(3, len(frame_indices))  # 3 rows x num_imgs columns
    start_row = 2  # start at row 2 to stay below attention plot

    for k, idx in enumerate(frame_indices):
        ax = fig.add_subplot(gs[start_row:, k])  # row 2
        img = image_obs[idx]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"t = {idx}", fontsize=9, pad=4)

    # plt.subplots_adjust(hspace=0.5)  # Adjust spacing between plots
    plt.tight_layout()

    # Save combined figure
    combined_path = f"{output_folder}/tr_att_{episode_name}.png"
    plt.savefig(combined_path, dpi=250)
    plt.close(fig)
    print(f"✅ Saved attention + images combined plot: {combined_path}")




    # Visualization (unchanged content) 
    fig, axes = plt.subplots(dof_dims, 1, figsize=(12, 2 * dof_dims), sharex=True)
    fig.suptitle(f"FACTR Prediction vs Ground Truth for episode {episode_name}, y-plot-range: {max_y_diff:.1f}", fontsize=16, y=0.98)
    fig.text(0.5, 0.95, f"model: {model_name}", fontsize=11, ha='center', va='top')
    for d in range(dof_dims):
        ax = axes[d]
        ax.plot(t, true_actions[:, d], label="Ground Truth Joint Pos.", linewidth=2.0, color="red")
        ax.set_ylabel(f"J{d+1} Pos. [rad]")
        # every subplot should have same abs difference between y-limits
        mid = (max_mins[d][0] + max_mins[d][1]) / 2.0
        ax.set_ylim(mid - max_y_diff/2 - 0.06*max_y_diff, mid + max_y_diff/2 + 0.06*max_y_diff)
        for i in range(pred_dims):
            ax.plot(t + i, pred_actions[:, i, d], label="Predicted Joint Pos.", linewidth=0.8, alpha=0.3, color="blue")
            if i == 0:
                ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    save_path = f"{output_folder}/tr_rad_pred_{episode_name}.png"
    plt.savefig(save_path, dpi=250)
    plt.close(fig)
    print(f"✅ Saved plot to {save_path}")


    # Visualization (normalized) 
    fig, axes = plt.subplots(dof_dims, 1, figsize=(12, 2 * dof_dims), sharex=True)
    fig.suptitle(f"Normalized FACTR Prediction vs Ground Truth for episode {episode_name}", fontsize=16, y=0.98)
    fig.text(0.5, 0.95, f"model: {model_name}", fontsize=11, ha='center', va='top')
    for d in range(dof_dims):
        ax = axes[d]
        ax.plot(t, true_actions_normalized[:, d], label="Normalized Ground Truth Joint Pos.", linewidth=2.0, color="red")
        ax.set_ylabel(f"J{d+1} Pos. norm.")
        ax.set_ylim(-3.0, 3.0)
        for i in range(pred_dims):
            ax.plot(t + i, pred_actions_norm[:, i, d], label="Normalized Predicted Joint Pos.", linewidth=0.8, alpha=0.3, color="blue")
            if i == 0:
                ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    save_path = f"{output_folder}/tr_norm_pred_{episode_name}.png"
    plt.savefig(save_path, dpi=250)
    plt.close(fig)
    print(f"✅ Saved plot to {save_path}")

    if vs_all_plot:
        # Plot overlay of all dataset trajectories and FACTR predictions in rad
        fig, axes = plt.subplots(dof_dims, 1, figsize=(12, 2 * dof_dims), sharex=True)
        fig.suptitle(f"Joint Positions vs FACTR Predictions episode {episode_name}, y-plot-range: {max_y_diff:.1f}", fontsize=16, y=0.98)
        fig.text(0.5, 0.95, f"model: {model_name}", fontsize=11, ha='center', va='top')
        for d in range(dof_dims):
            ax = axes[d]
            # Dataset trajectories
            for ep_idx, ep_data in enumerate(joint_cmds_all):
                t_ep = np.arange(ep_data.shape[0])
                ax.plot(t_ep, ep_data[:, d], color="red", alpha=0.3, linewidth=1.0, label="Joint Pos. from Dataset" if (d == 0 and ep_idx == 0) else None)
            # Ground truth
            t_pred = np.arange(pred_actions.shape[0])
            ax.plot(t_pred, true_actions[:, d], label="Ground Truth Joint Pos.", linewidth=2, color="black", alpha=0.8)
            mid = (max_mins[d][0] + max_mins[d][1]) / 2.0
            ax.set_ylim(mid - max_y_diff/2 - 0.06*max_y_diff, mid + max_y_diff/2 + 0.06*max_y_diff)
            # Predictions
            for i in range(pred_dims):
                ax.plot(t_pred + i, pred_actions[:, i, d], color="blue", alpha=0.4, linewidth=0.8, label="FACTR prediction" if (d == 0 and i == 0) else None)
            ax.set_ylabel(f"J{d+1} Pos. [rad]")
            if d == 0:
                ax.legend(loc="upper right", fontsize=10)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Timestep")
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
        save_path = f"{output_folder}/tr_rad_pred_vs_all_{episode_name}.png"
        plt.savefig(save_path, dpi=250)
        plt.close(fig)
        print(f"✅ Saved overlay plot: {save_path}")


        # Plot overlay of all dataset trajectories and FACTR predictions normalized
        fig, axes = plt.subplots(dof_dims, 1, figsize=(12, 2 * dof_dims), sharex=True)
        fig.suptitle(f"Normalized Joint Positions vs FACTR Predictions for episode {episode_name}", fontsize=16, y=0.98)
        fig.text(0.5, 0.95, f"model: {model_name}", fontsize=11, ha='center', va='top')
        for d in range(dof_dims):
            ax = axes[d]
            # Dataset trajectories
            for ep_idx, ep_data in enumerate(joint_cmds_all_norm):
                t_ep = np.arange(ep_data.shape[0])
                ax.plot(t_ep, ep_data[:, d], color="red", alpha=0.3, linewidth=1.0, label="Normalized Joint Pos. from Dataset" if (d == 0 and ep_idx == 0) else None)
            # Ground truth
            t_pred = np.arange(pred_actions_norm.shape[0])
            ax.plot(t_pred, true_actions_normalized[:, d], label="Normalized Ground Truth Joint Pos.", linewidth=2, color="black", alpha=0.8)
            # Predictions
            for i in range(pred_dims):
                ax.plot(t_pred + i, pred_actions_norm[:, i, d], color="blue", alpha=0.4, linewidth=0.8, label="Normalized FACTR prediction" if (d == 0 and i == 0) else None)
            ax.set_ylabel(f"J{d+1} pos. norm.")
            if d == 0:
                ax.legend(loc="upper right", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-3.0, 3.0)
        axes[-1].set_xlabel("Timestep")
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
        save_path = f"{output_folder}/tr_norm_pred_vs_all_{episode_name}.png"
        plt.savefig(save_path, dpi=250)
        plt.close(fig)
        print(f"✅ Saved overlay plot: {save_path}")

    print("")

