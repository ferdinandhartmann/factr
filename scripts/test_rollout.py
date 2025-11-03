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
model_name = "20251024_60_25hz_b64_lr27_d12_3"
checkpoint = "latest"
episode_names = ["ep_4", "ep_7", "ep_62", "ep_64"]  # List of episode names to test

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

dataset_folder = Path("/home/ferdinand/factr/process_data/raw_data_train/20251024_60/")


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
    image_topic = "/realsense/arm/im"
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

def get_all_joint_cmds(data_path):
    """Extract all joint commands from PKL files in a folder."""
    # --- Paths and setup ---
    data_path = Path(dataset_folder)
    dataset_name = data_path.parent.stem
    topic = "/joint_impedance_command_controller/joint_trajectory"

    print(f"Dataset: {dataset_name}")

    pkl_files = sorted(data_path.glob("*.pkl"))
    if not pkl_files:
        print("⚠️ No PKL files found.")
        exit()

    entries = []
    for pkl_file in pkl_files[1:]:
        try:
            data = load_pkl(pkl_file)
            cmd_data, _ = extract_topic(data, topic)
            cmd_pos = extract_7d(cmd_data, "position")
            if len(cmd_pos) == 0:
                continue
            x = np.arange(len(cmd_pos))
            entries.append({"x": x, "cmd_pos": cmd_pos})
        except Exception as e:
            print(f"❌ Error in {pkl_file.name}: {e}")

    if not entries:
        print("⚠️ No usable data found.")
        exit()

    # --- Compute normalization stats ---
    cmd_concat = np.concatenate([e["cmd_pos"] for e in entries], axis=0)
    means = np.nanmean(cmd_concat, axis=0)
    stds = np.nanstd(cmd_concat, axis=0)
    stds[stds < 1e-8] = 1.0

    print("✅ Computed normalization stats")

    # --- Normalize ---
    for e in entries:
        e["cmd_norm"] = (e["cmd_pos"] - means) / stds

    return entries

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
joint_cmds_normalized = get_all_joint_cmds(dataset_folder)
print(f"✅ Loaded and normalized all joint commands from dataset folder.")

# --- Load arrays from PKL file ---
for episode_name in episode_names:
    RAW_DATA_PATH = Path(f"/home/ferdinand/factr/process_data/raw_data_train/20251024_60/{episode_name}.pkl")
    if not RAW_DATA_PATH.exists():
        RAW_DATA_PATH = Path(f"/home/ferdinand/factr/process_data/raw_data_eval/20251024_4/{episode_name}.pkl")
    if not RAW_DATA_PATH.exists():
        print(f"Required PKL file not found: {RAW_DATA_PATH}, skipping this episode.")
        break 

    image_obs, torque_obs, true_action = load_and_extract_raw_data(RAW_DATA_PATH)

    # -----------------------------
    # INFERENCE
    # -----------------------------
    pred_actions = []
    normalized_true_action_list = []

    N = min(len(true_action), len(torque_obs), len(image_obs))
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
        true_action_tensor = torch.from_numpy(true_action[i]).float().to(DEVICE)
        normalized_true_action = (true_action_tensor - action_mean) / action_std  # normalization
        normalized_true_action = normalized_true_action.unsqueeze(0)
        normalized_true_action_list.append(normalized_true_action.cpu().numpy()[0])

        with torch.no_grad():
            pred_action = policy.get_actions({"cam0": img_tensor}, torque_tensor)
            # pred_action = pred_action * action_std + action_mean # Inverse normalization 
            pred_action = pred_action.cpu().numpy()[0]

        pred_actions.append(pred_action)

    pred_actions = np.array(pred_actions)
    true_action = np.array(normalized_true_action_list)
    # true_action = true_action[:N]
    # true_action = normalized_true_action.cpu().numpy()


    print("✅ Finished inference")
    print(f"Pred shape: {pred_actions.shape}, True shape: {true_action.shape}")

    with torch.no_grad():
        if isinstance(pred_actions, np.ndarray):
            if pred_actions.ndim == 3:
                pred_action = pred_actions[:, :, :]


    # EVALUATION (MSE) and VISUALIZATION
    t = np.arange(len(pred_action))  # <-- ensure t matches N
    dof_dims = pred_action.shape[2]
    pred_dims = pred_action.shape[1]
    print(f"Number of Frames: {t.shape[0]}, dof_dims: {dof_dims}, pred_dims: {pred_dims}")

    max_mins = []
    # MSE per joint over all predictions and the whole trajectory
    for d in range(dof_dims):  # Use dof_dims for joint dimensions
        # mse = np.mean((pred_action[:, :, d] - true_action[:, d, None]) ** 2)  # Compute MSE over all predictions and trajectory
        # print(f"MSE Joint {d+1} over all predictions and whole trajectory: {mse:.6f}")
        # Get max and min of each joint dimension
        max_val = np.max(pred_action[:, :, d])
        min_val = np.min(pred_action[:, :, d])
        max_mins.append((max_val, min_val))

    max_y_diff = max(np.abs(m[0] - m[1]) for m in max_mins)

    # # Calculate L2 Loss
    # l2_loss = np.mean(np.linalg.norm(pred_action.reshape(-1, dof_dims) - true_action, axis=1))
    # print(f"\nAverage L2 Loss for all joints: {l2_loss:.6f}"


    # Visualization
    plt.figure(figsize=(10, 2 * dof_dims))
    for d in range(dof_dims):
        plt.subplot(dof_dims, 1, d + 1)
        plt.plot(t, true_action[:, d], label="Ground Truth Joint Pos.", linewidth=2.5, color="red")
        # plt.ylabel(f"Pos. Joint {d+1} [rad]")
        plt.ylabel(f"Norm. Pos. Joint {d+1}")
        # every subplot should have same abs difference between y-limits
        # mid = (max_mins[d][0] + max_mins[d][1]) / 2.0
        # plt.ylim(mid - max_y_diff/2 - 0.04*max_y_diff, mid + max_y_diff/2 + 0.04*max_y_diff)
        plt.ylim(-2.5, 2.5)
        for i in range (pred_dims):    
            plt.plot(t + i, pred_action[:, i, d], label="Unnormalized Predicted Joint Pos.", linewidth=0.8, alpha=0.3, color="blue")
            if i == 0:
                plt.legend(loc="upper right")  
        if d == 0:
            plt.title(f"FACTR Prediction vs Ground Truth of {episode_name}")
        plt.grid(True, alpha=0.4)
    plt.xlabel("Frame")
    plt.tight_layout()
    save_path = f"/home/ferdinand/factr/scripts/test_rollout_output/test_rollout_{model_name}_{checkpoint}_{episode_name}.png"
    plt.savefig(save_path)
    print(f"✅ Saved plot to {save_path}")


    # Plot overlay of all dataset trajectories and FACTR predictions (normalized)
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"Normalized Joint Positions vs FACTR Predictions\nModel {model_name}, episode {episode_name}", fontsize=16, y=0.96)

    for j in range(7):
        ax = axes[j]
        # Dataset trajectories (red)
        for i, cmd in enumerate(joint_cmds_normalized):
            ax.plot(cmd["x"], cmd["cmd_norm"][:, j], color="red", alpha=0.3, linewidth=1.0, label="Normalized Joint Pos. from Dataset" if i == 0 else None)
        # True (ground truth) (black)
        t_pred = np.arange(pred_action.shape[0])
        ax.plot(t_pred, true_action[:, j], label="Ground Truth Joint Pos. normalized", linewidth=1.2, color="black", alpha=0.8)
        # Predictions (blue)
        for i in range(pred_action.shape[1]):
            ax.plot(t_pred + i, pred_action[:, i, j], color="blue", alpha=0.4, linewidth=0.8,
                    label="Unnormalized FACTR prediction" if (j == 0 and i == 0) else None)
        ax.set_ylabel(f"J{j+1} normalized pos.")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frame index")
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])

    out_path = f"/home/ferdinand/factr/scripts/test_rollout_output/all_and_pred_norm_{model_name}_{episode_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved overlay plot: {out_path}")

