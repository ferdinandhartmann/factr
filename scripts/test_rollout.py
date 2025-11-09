import torch
import pickle
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from omegaconf import OmegaConf
import yaml
import os
import copy
from typing import List, Any

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

# ---------- CONFIG ---------- # Select model, checkpoint, and episode here
model_name = "20251107_60_25hz_s40_ac25_b64_lr0.00025_20000"
checkpoint = "latest"
episode_names = ["ep_62", "ep_63", "ep_64"] # List of episode names to test

# ---------- PATHS & DEVICE ----------
CKPT_PATH = None
if checkpoint == "latest":
    CKPT_PATH = Path(f"scripts/checkpoints/{model_name}/rollout/latest_ckpt.ckpt")
else:
    CKPT_PATH = Path(f"scripts/checkpoints/{model_name}/{checkpoint}.ckpt")
EXP_CFG_PATH = Path(f"scripts/checkpoints/{model_name}/rollout/exp_config.yaml")
ROLLOUT_CFG_PATH = Path(f"scripts/checkpoints/{model_name}/rollout/rollout_config.yaml")

# RAW_DATA_PATH = Path(f"/home/ferdinand/factr/process_data/raw_data_train/20251024_60/{episode_name}.pkl")
# if not RAW_DATA_PATH.exists():
#     RAW_DATA_PATH = Path(f"/home/ferdinand/factr/process_data/raw_data_eval/20251024_4/{episode_name}.pkl")
# if not RAW_DATA_PATH.exists():
#     raise FileNotFoundError(f"Required PKL file not found: {RAW_DATA_PATH}.")

dataset_folder = Path("/home/ferdinand/factr/process_data/raw_data_train/20251107_60/")

output_folder = Path(f"/home/ferdinand/factr/scripts/test_rollout_output/{model_name}_{checkpoint}")
output_folder.mkdir(parents=True, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_extract_raw_data(pkl_path: Path):
    """
    Load the raw pickle file and extract/process image, state (torque), 
    and action data, ensuring images are proper NumPy arrays.
    """
    with open(pkl_path, "rb") as f:
        raw_data = pickle.load(f)
    print(f"✅ Loaded raw data from {pkl_path}")
    
    image_obs, torque_obs, actions = [], [], []

    # Define topic names used in the raw data
    image_topic = "/realsense/front/im"
    obs_topic = "/franka_robot_state_broadcaster/external_joint_torques"
    action_topic = "/joint_impedance_command_controller/joint_trajectory"

    if "data" not in raw_data:
        raise ValueError("Unknown data structure: 'data' key not found in raw PKL file.")

    entries = raw_data["data"]
    
    # --- 1. Extract Data ---
    for topic, values in entries.items():
        if image_topic in topic:
            print(f"Extracting image data from {topic} ({len(values)} frames)")
            imgs = []
            for v in values:
                if isinstance(v, dict) and "data" in v and "height" in v and "width" in v:
                    try:
                        # 1. Load raw buffer as a flat uint8 array
                        img_flat = np.frombuffer(v["data"], dtype=np.uint8)
                        
                        # 2. Reshape the 1D buffer into a 2D/3D image array (H, W, C)
                        # -1 calculates the number of channels (C) automatically.
                        img = img_flat.reshape((v["height"], v["width"], -1))
                        
                        # 3. Validation: Ensure it is a valid multi-dimensional array
                        if img.ndim >= 2 and img.dtype == np.uint8:
                            imgs.append(img)
                        else:
                            print(f"⚠️ Image skip: Reshape failed or incorrect type.")
                    except Exception as e:
                        # Catches errors like dimension mismatch during reshape
                        print(f"⚠️ Image skip due to reshape error: {e}")
            image_obs.extend(imgs)
            
        elif action_topic in topic:
            for v in values:
                if isinstance(v, dict) and "position" in v:
                    actions.append(v["position"])
            print(f"Extracting joint actions from {topic} ({len(actions)} commands)")

        elif obs_topic in topic:
            for v in values:
                if isinstance(v, dict) and "effort" in v:
                    torque_obs.append(v["effort"])
            print(f"Extracting observations from {topic} ({len(torque_obs)} commands)")

    # --- 2. Handle Missing/Downsampling ---
    # Convert lists to NumPy arrays
    torque_obs = np.array(torque_obs)
    actions = np.array(actions)
    final_image_obs = image_obs # Use dtype=object to hold different HxWxC arrays

    # Dummy state check (simplified, assuming this is handled robustly in the full script)
    if len(torque_obs) == 0 and len(actions) > 0:
        torque_obs = np.zeros((len(actions), 7))
    elif len(torque_obs) == 0 and len(final_image_obs) > 0:
        torque_obs = np.zeros((len(final_image_obs), 7))

    # # Downsampling logic (as per your original script's output, simplified for this function block)
    # target_freq = 25.0
    # avg_freq = 50.0 # Assumed default from your log output
    
    # if avg_freq > target_freq:
    #     step = int(np.floor(avg_freq / target_freq)) # step = 2
    #     final_image_obs = final_image_obs[::step]
    #     torque_obs = torque_obs[::step]
    #     actions = actions[::step]
    #     print(f"🔻 Downsampled from ~{avg_freq:.1f} Hz to ~{target_freq:.1f} Hz (step={step})")

    # Match array lengths
    N = min(len(final_image_obs), len(torque_obs), len(actions))
    final_image_obs = final_image_obs[:N]
    torque_obs = torque_obs[:N]
    actions = actions[:N]
        
    print(f"✅ Extracted image_obs: {len(final_image_obs)} | torque_obs: {torque_obs.shape} | actions: {actions.shape}")
    
    # The returned image array elements are now guaranteed to be np.uint8 arrays
    return final_image_obs, torque_obs, actions

def preprocess_image(img):
    # Handle grayscale or depth
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    img = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) 
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std # with standard normalization
    return img_tensor.to(DEVICE)

def load_pkl(path: Path):
    """Load pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def extract_topic(pkl_data, topic: str):
    """Return (data, timestamps) for topic if exists."""
    return (
        pkl_data['data'].get(topic, []),
        pkl_data['timestamps'].get(topic, []),
    )

def extract_7d(data_list: List[Any], key: str) -> np.ndarray:
    """Extract (N,7) data safely, padding invalid with NaN."""
    arr = []
    for d in data_list:
        val = d.get(key, [np.nan] * 7)
        if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 7:
            arr.append(val)
        else:
            arr.append([np.nan] * 7)
    return np.array(arr, dtype=np.float32)
def get_all_joint_cmds_np(data_path, action_mean, action_std):
    """
    Load each episode's joint commands separately (not concatenated)
    and normalize them using rollout_config.yaml stats.
    Returns:
        cmds_per_episode: list of np.ndarray, each (T_i, 7)
        cmds_per_episode_norm: list of np.ndarray, each (T_i, 7)
    """
    data_path = Path(data_path)
    topic = "/joint_impedance_command_controller/joint_trajectory"
    pkl_files = sorted(data_path.glob("*.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No PKL files found in {data_path}")

    cmds_per_episode = []
    cmds_per_episode_norm = []

    # Convert mean/std to numpy if they’re tensors
    if isinstance(action_mean, torch.Tensor):
        action_mean = action_mean.cpu().numpy()
    if isinstance(action_std, torch.Tensor):
        action_std = action_std.cpu().numpy()

    for pkl_file in pkl_files:
        try:
            data = load_pkl(pkl_file)
            cmd_data, _ = extract_topic(data, topic)
            cmd_pos = extract_7d(cmd_data, "position")

            if len(cmd_pos) == 0:
                continue

            cmd_norm = (cmd_pos - action_mean) / action_std
            cmds_per_episode.append(cmd_pos)
            cmds_per_episode_norm.append(cmd_norm)

        except Exception as e:
            print(f"⚠️ Skipping {pkl_file.name}: {e}")

    print(f"✅ Loaded {len(cmds_per_episode)} episodes from {data_path}")
    return cmds_per_episode, cmds_per_episode_norm



# ----------------------------

cfg = OmegaConf.load(EXP_CFG_PATH)

# 🩹 Patch Hydra-only expressions manually
if "task" in cfg:
    if "cam_indexes" in cfg.task:
        cfg.task.n_cams = len(cfg.task.cam_indexes)
    if "train_buffer" in cfg.task:
        cfg.task.train_buffer.transform = "medium"
        cfg.task.train_buffer.past_frames = cfg.img_chunk - 1
    if "test_buffer" in cfg.task:
        cfg.task.test_buffer.transform = "preproc"
        cfg.task.test_buffer.past_frames = cfg.img_chunk - 1
if "curriculum" in cfg:
    cfg.curriculum.max_step = cfg.max_iterations

# 🚫 Safely remove Hydra-only keys without triggering interpolation
# Use dict-like interface instead of OmegaConf accessors
raw_dict = OmegaConf.to_container(cfg, resolve=False)

if "rt" in raw_dict and isinstance(raw_dict["rt"], str) and raw_dict["rt"].startswith("${hydra:"):
    cfg["rt"] = "agent/features"
if "hydra" in raw_dict:
    cfg.pop("hydra", None)

OmegaConf.resolve(cfg)

agent_cfg = cfg.agent  # model definition lives here
policy = instantiate(agent_cfg)

# --- 2. Load checkpoint ---
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
state_dict = ckpt["model"]

# 🩹 Remove 'module.' prefix if present (for DataParallel checkpoints)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

missing, unexpected = policy.load_state_dict(new_state_dict, strict=False)
print(f"✅ Loaded policy from {CKPT_PATH}, step {ckpt['global_step']}")
if len(missing) > 0 or len(unexpected) > 0:
    print(f"⚠️ Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

policy.eval()
policy.to(DEVICE)
policy = torch.compile(policy)

print(f"✅ Loaded policy from {CKPT_PATH}, step {ckpt['global_step']}")


# --- 3. Load Normalization Stats (NEW SECTION) ---
try:
    with open(ROLLOUT_CFG_PATH, 'r') as f:
        rollout_config = yaml.safe_load(f)

    # Torque
    obs_mean = torch.tensor(rollout_config['norm_stats']['state']['mean']).float().to(DEVICE)
    obs_std = torch.tensor(rollout_config['norm_stats']['state']['std']).float().to(DEVICE)

    # Policy output (action)
    action_mean = torch.tensor(rollout_config['norm_stats']['action']['mean']).float().to(DEVICE)
    action_std = torch.tensor(rollout_config['norm_stats']['action']['std']).float().to(DEVICE)
    print(f"✅ Loaded normalization stats from {ROLLOUT_CFG_PATH}")

except Exception as e:
    print(f"⚠️ Warning: Could not load normalization stats from {ROLLOUT_CFG_PATH}. Running inference with unnormalized torque/actions. Error: {e}")
    # Define identity stats if loading fails to prevent script crash
    obs_mean, obs_std = 0., 1.
    action_mean, action_std = 0., 1.

print("Loading all joint commands from dataset for visualization...")
joint_cmds_all, joint_cmds_all_norm = get_all_joint_cmds_np(dataset_folder, action_mean, action_std)
print(f"✅ Loaded and normalized all joint commands from dataset folder.")

# --- Load arrays from PKL file ---
for episode_name in episode_names:
    RAW_DATA_PATH = Path(f"/home/ferdinand/factr/process_data/raw_data_train/20251107_60/{episode_name}.pkl")
    if not RAW_DATA_PATH.exists():
        RAW_DATA_PATH = Path(f"/home/ferdinand/factr/process_data/raw_data_eval/20251107_8/{episode_name}.pkl")
    if not RAW_DATA_PATH.exists():
        print(f"Required PKL file not found: {RAW_DATA_PATH}, skipping this episode.")
        break 

    image_obs, torque_obs, true_actions = load_and_extract_raw_data(RAW_DATA_PATH)

    # -----------------------------
    # INFERENCE
    # -----------------------------
    pred_actions = []
    pred_actions_norm = []
    normalized_true_action_list = []
    attn_image, attn_force = [], []

    N = min(len(true_actions), len(torque_obs), len(image_obs))
    print(f"🚀 Running inference on {N} samples...")

    for i in tqdm(range(N)):
        img = image_obs[i]
        torque = torque_obs[i]

        img_tensor = preprocess_image(img)

        # Normalize Torque Observation
        torque_tensor = torch.from_numpy(torque).float().to(DEVICE)
        torque_tensor = (torque_tensor - obs_mean) / obs_std  # normalization
        torque_tensor = torque_tensor.unsqueeze(0)

        # Normalize Ground Truth Action
        true_action_tensor = torch.from_numpy(true_actions[i]).float().to(DEVICE)
        normalized_true_action = (true_action_tensor - action_mean) / action_std  # normalization
        normalized_true_action = normalized_true_action.unsqueeze(0)
        normalized_true_action_list.append(normalized_true_action.cpu().numpy()[0])

        with torch.no_grad():
            pred_action, cross_w = policy.get_actions({"cam0": img_tensor}, torque_tensor, return_weights=True)
            # test_rollout.py → agent.get_actions() → policy.get_actions() → model.get_actions()
            pred_action_norm = pred_action.cpu().numpy()[0]
            pred_action = pred_action * action_std + action_mean # Inverse normalization 
            pred_action = pred_action.cpu().numpy()[0]

            if cross_w is not None:
                # Canonicalize attention
                if cross_w.dim() == 3:
                    cross_w = cross_w.unsqueeze(0)  # (1, H, Tq, Tk)
                attn_heads_mean = cross_w.mean(1)   # mean over heads → (B, Tq, Tk)
                attn_mean = attn_heads_mean.mean(1) # mean over decoder queries → (B, Tk)

                N_images = 1  # since you only have cam0 and no past frames
                img_attn = attn_mean[..., :N_images].mean().item()
                torque_attn = attn_mean[..., N_images:].mean().item()

                attn_image.append(img_attn)
                attn_force.append(torque_attn)

        pred_actions.append(pred_action)
        pred_actions_norm.append(pred_action_norm)

    pred_actions = np.array(pred_actions)
    pred_actions_norm = np.array(pred_actions_norm)
    true_actions_normalized = np.array(normalized_true_action_list)
    true_actions = true_actions[:N]

    print("✅ Finished inference")
    print(f"Pred shape: {pred_actions.shape}, True shape: {true_actions.shape}")

    with torch.no_grad():
        if isinstance(pred_actions, np.ndarray):
            if pred_actions.ndim == 3:
                pred_action = pred_actions[:, :, :]
        if isinstance(pred_action_norm, np.ndarray):
            if pred_action_norm.ndim == 3:
                pred_action_norm = pred_action_norm[:, :, :]
            

    # EVALUATION (MSE) and VISUALIZATION
    t = np.arange(len(pred_action))  # <-- ensure t matches N
    dof_dims = pred_action.shape[2]
    pred_dims = pred_action.shape[1]
    print(f"Number of Frames: {t.shape[0]}, dof_dims: {dof_dims}, pred_dims: {pred_dims}")

    max_mins = []
    for d in range(dof_dims):  # Use dof_dims for joint dimensions
        # Get max and min of each joint dimension
        max_val = np.max(pred_action[:, :, d])
        min_val = np.min(pred_action[:, :, d])
        max_mins.append((max_val, min_val))

    max_y_diff = max(np.abs(m[0] - m[1]) for m in max_mins)

    # Calculate L2 Loss
    # Ensure pred_action and true_actions have compatible shapes
    l2_loss = np.mean(np.linalg.norm(pred_actions_norm[:, 0, :] - true_actions, axis=1))
    print(f"\nAverage L2 Loss for all joints: {l2_loss:.6f}\n")


    # Plot attention weights over time
    plt.figure(figsize=(10,5))
    plt.plot(attn_force, label="Torque attention", linewidth=1.8)
    plt.plot(attn_image, label="Image attention", linewidth=1.8)
    plt.xlabel("Timestep")
    plt.ylabel("Mean attention weight")
    plt.title(f"Attention to Force and Image of {episode_name}")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    attn_path = f"{output_folder}/tr_att_{episode_name}.png"
    plt.savefig(attn_path, dpi=200)
    print(f"✅ Saved attention plot to {attn_path}")


    # Visualization (unchanged content) 
    fig, axes = plt.subplots(dof_dims, 1, figsize=(12, 2 * dof_dims), sharex=True)
    fig.suptitle(f"FACTR Prediction vs Ground Truth\nModel {model_name}, episode {episode_name}, y-plot-range: {max_y_diff:.1f}", fontsize=16, y=0.96)
    for d in range(dof_dims):
        ax = axes[d]
        ax.plot(t, true_actions[:, d], label="Ground Truth Joint Pos.", linewidth=2.5, color="red")
        ax.set_ylabel(f"J{d+1} Pos. [rad]")
        # every subplot should have same abs difference between y-limits
        mid = (max_mins[d][0] + max_mins[d][1]) / 2.0
        ax.set_ylim(mid - max_y_diff/2 - 0.04*max_y_diff, mid + max_y_diff/2 + 0.04*max_y_diff)
        for i in range(pred_dims):
            ax.plot(t + i, pred_action[:, i, d], label="Predicted Joint Pos.", linewidth=0.8, alpha=0.3, color="blue")
            if i == 0:
                ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    save_path = f"{output_folder}/tr_rad_pred_{episode_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved plot to {save_path}")


    # Visualization (normalized) 
    fig, axes = plt.subplots(dof_dims, 1, figsize=(12, 2 * dof_dims), sharex=True)
    fig.suptitle(f"Normalized FACTR Prediction vs Ground Truth\nModel {model_name}, episode {episode_name}, y-plot-range: {max_y_diff:.1f}", fontsize=16, y=0.96)
    for d in range(dof_dims):
        ax = axes[d]
        ax.plot(t, true_actions_normalized[:, d], label="Ground Truth Joint Pos.", linewidth=2.5, color="red")
        ax.set_ylabel(f"J{d+1} Pos. norm.")
        # ax.set_ylim(-2.8, 2.8)
        for i in range(pred_dims):
            ax.plot(t + i, pred_actions_norm[:, i, d], label="Normalized Predicted Joint Pos.", linewidth=0.8, alpha=0.3, color="blue")
            if i == 0:
                ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    save_path = f"{output_folder}/tr_norm_pred_{episode_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved plot to {save_path}")


    # Plot overlay of all dataset trajectories and FACTR predictions (un-normalized)
    fig, axes = plt.subplots(dof_dims, 1, figsize=(12, 2 * dof_dims), sharex=True)
    fig.suptitle(f"Joint Positions vs FACTR Predictions\nModel {model_name}, episode {episode_name}", fontsize=16, y=0.96)
    for d in range(dof_dims):
        ax = axes[d]
        # Dataset trajectories
        for ep_idx, ep_data in enumerate(joint_cmds_all_norm):
            t_ep = np.arange(ep_data.shape[0])
            ax.plot(t_ep, ep_data[:, d], color="red", alpha=0.3, linewidth=1.0, label="Joint Pos. from Dataset" if (d == 0 and ep_idx == 0) else None)
        # Ground truth
        t_pred = np.arange(pred_action.shape[0])
        ax.plot(t_pred, true_actions[:, d], label="Ground Truth Joint Pos. normalized", linewidth=1.2, color="black", alpha=0.8)
        # Predictions
        for i in range(pred_dims):
            ax.plot(t_pred + i, pred_action[:, i, d], color="blue", alpha=0.4, linewidth=0.8, label="Normalized FACTR prediction" if (d == 0 and i == 0) else None)
        ax.set_ylabel(f"J{d+1} Pos. [rad]")
        if d == 0:
            ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    save_path = f"{output_folder}/tr_norm_pred_vs_all_{episode_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved overlay plot: {save_path}")


    # Plot overlay of all dataset trajectories and FACTR predictions (normalized)
    fig, axes = plt.subplots(dof_dims, 1, figsize=(12, 2 * dof_dims), sharex=True)
    fig.suptitle(f"Normalized Joint Positions vs FACTR Predictions\nModel {model_name}, episode {episode_name}", fontsize=16, y=0.96)
    for d in range(dof_dims):
        ax = axes[d]
        # Dataset trajectories
        for ep_idx, ep_data in enumerate(joint_cmds_all):
            t_ep = np.arange(ep_data.shape[0])
            ax.plot(t_ep, ep_data[:, d], color="red", alpha=0.3, linewidth=1.0, label="Normalized Joint Pos. from Dataset" if (d == 0 and ep_idx == 0) else None)
        # Ground truth
        t_pred = np.arange(pred_actions_norm.shape[0])
        ax.plot(t_pred, true_actions[:, d], label="Ground Truth Joint Pos. normalized", linewidth=1.2, color="black", alpha=0.8)
        # Predictions
        for i in range(pred_dims):
            ax.plot(t_pred + i, pred_actions_norm[:, i, d], color="blue", alpha=0.4, linewidth=0.8, label="Normalized FACTR prediction" if (d == 0 and i == 0) else None)
        ax.set_ylabel(f"J{d+1} pos. norm.")
        if d == 0:
            ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
    save_path = f"{output_folder}/tr_rad_pred_vs_all_{episode_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved overlay plot: {save_path}")

