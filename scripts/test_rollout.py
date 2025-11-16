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
model_name = "20251112_60_25hz_filt2_7dof_s42_ac25_b64_lr0.00018_iter3000_"
checkpoint = "latest"
episode_names = ["ep_8", "ep_25", "ep_40", "ep_45", "ep_61", "ep_62", "ep_63", "ep_64", "ep_65", "ep_66"] # List of episode names to test

downsample = True # from 50Hz to 25Hz
vs_all_plot = True # whether to load all joint commands from dataset for visualization
use_buffer = True  # load from buffer.pkl instead of raw PKL files
remove_joints = [] # zero-indexed joints to remove

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


def load_and_extract_raw_data(pkl_path: Path, image_topic="/realsense/front/im", obs_topic="/franka_robot_state_broadcaster/external_joint_torques", action_topic="/joint_impedance_dynamic_gain_controller/joint_impedance_command"):
    """
    Load the raw pickle file and extract/process image, state (torque), 
    and action data, ensuring images are proper NumPy arrays.
    """
    with open(pkl_path, "rb") as f:
        raw_data = pickle.load(f)
    print(f"✅ Loaded raw data from {pkl_path}")
    
    image_obs, torque_obs, actions = [], [], []

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

    if downsample:
        # Downsampling logic (as per your original script's output, simplified for this function block)
        target_freq = 25.0
        avg_freq = 50.0 # Assumed default from your log output  
        step = int(np.floor(avg_freq / target_freq)) # step = 2
        final_image_obs = final_image_obs[::step]
        torque_obs = torque_obs[::step]
        actions = actions[::step]
        print(f"🔻 Downsampled from ~{avg_freq:.1f} Hz to ~{target_freq:.1f} Hz (step={step})")

    # Match array lengths
    N = min(len(final_image_obs), len(torque_obs), len(actions))
    final_image_obs = final_image_obs[:N]
    torque_obs = torque_obs[:N]
    actions = actions[:N]
        
    print(f"✅ Extracted image_obs: {len(final_image_obs)} | torque_obs: {torque_obs.shape} | actions: {actions.shape}")
    
    # The returned image array elements are now guaranteed to be np.uint8 arrays
    return final_image_obs, torque_obs, actions

def preprocess_image(img):

    # --- FIX: ensure uint8 RGB ---
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    elif img.dtype != np.uint8:
        # assume float image in [0,1]
        img = (img * 255).clip(0, 255).astype(np.uint8)

    # Handle grayscale
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)

    if img.shape[:2] != (224, 224):
        img = cv2.resize(img, (224, 224))

    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.

    # normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std

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
    topic = "/joint_impedance_dynamic_gain_controller/joint_impedance_command"
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

            if downsample:
                target_freq = 25.0
                avg_freq = 50.0 # Assumed default from your log output  
                step = int(np.floor(avg_freq / target_freq)) # step = 2
                # Downsample from 50Hz to 25Hz
                cmd_pos = cmd_pos[::step]
                # print(f"🔻 Downsampled joint commands in {pkl_file.name} from ~{avg_freq:.1f} Hz to ~{target_freq:.1f} Hz (step={step})")

            cmd_norm = (cmd_pos - action_mean) / action_std
            cmds_per_episode.append(cmd_pos)
            cmds_per_episode_norm.append(cmd_norm)

        except Exception as e:
            print(f"⚠️ Skipping {pkl_file.name}: {e}")

    print(f"✅ Loaded {len(cmds_per_episode)} episodes from {data_path}")
    return cmds_per_episode, cmds_per_episode_norm


def load_episode_from_buffer(buf_path, episode_idx, cam_key="enc_cam_0"):

    with open(buf_path, "rb") as f:
        traj_list = pickle.load(f)

    traj = traj_list[episode_idx-1]  # zero-indexed

    images = []
    states = []
    actions = []

    for (obs, act, done) in traj:

        # ----- 1. Load image -----
        if cam_key not in obs:
            raise KeyError(f"{cam_key} not found in obs dict keys={obs.keys()}")

        enc = obs[cam_key]   # encoded JPEG bytes (np.uint8 array)
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)  # decode JPEG
        if img is None:
            raise ValueError("Failed to decode JPEG from buffer!")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB

        images.append(img)

        # ----- 2. Low-dim state -----
        if "state" not in obs:
            raise KeyError("obs['state'] missing from buffer step")
        states.append(np.array(obs["state"], dtype=float))

        # ----- 3. Action -----
        actions.append(np.array(act, dtype=float))

    images = np.stack(images)
    states = np.stack(states)
    actions = np.stack(actions)

    return images, states, actions

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
    print(f"⚠️ Missing keys: {len(missing)}, Unexpected keys: {len( )}")

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
            print(f"⚠️ Warning: Could not load episode {episode_name} from buffer. Error: {e}, loading from raw PKL instead.")
            RAW_DATA_PATH = RAW_DATA_PATH_EVAL / f"{episode_name}.pkl"
            image_obs, torque_obs, true_actions = load_and_extract_raw_data(RAW_DATA_PATH, image_topic=image_topic, obs_topic=obs_topic, action_topic=action_topic)
            use_eval = True

    else:
        RAW_DATA_PATH = RAW_DATA_PATH_TRAIN / f"{episode_name}.pkl"
        if not RAW_DATA_PATH.exists():
            RAW_DATA_PATH = RAW_DATA_PATH_EVAL / f"{episode_name}.pkl"
        if not RAW_DATA_PATH.exists():
            print(f"Required PKL file not found: {RAW_DATA_PATH}, skipping this episode.")
            break 
        image_obs, torque_obs, true_actions = load_and_extract_raw_data(RAW_DATA_PATH, image_topic=image_topic, obs_topic=obs_topic, action_topic=action_topic)

    # -----------------------------
    # INFERENCE
    # -----------------------------
    pred_actions = []
    pred_actions_norm = []
    true_action_list = []
    normalized_true_action_list = []
    attn_image, attn_force = [], []

    action_mean_red = action_mean[use_indicies]
    action_std_red = action_std[use_indicies]

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
            # test_rollout.py → agent.get_actions() → policy.get_actions() → model.get_actions()
            pred_action_norm = pred_action.cpu().numpy()[0]
            pred_action = pred_action * action_std_red + action_mean_red # Inverse normalization 
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
    true_actions = np.array(true_action_list)

    if len(remove_joints) > 0:
        for remove_joint in sorted(remove_joints):
            pred_actions = np.insert(pred_actions, remove_joint, 0, axis=2)
            pred_actions_norm = np.insert(pred_actions_norm, remove_joint, 0, axis=2)

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
    l2_loss = np.mean(np.linalg.norm(pred_actions_norm[:, 0, use_indicies] - true_actions[:, use_indicies], axis=1))
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
        ax.plot(t, true_actions[:, d], label="Ground Truth Joint Pos.", linewidth=2.0, color="red")
        ax.set_ylabel(f"J{d+1} Pos. [rad]")
        # every subplot should have same abs difference between y-limits
        mid = (max_mins[d][0] + max_mins[d][1]) / 2.0
        ax.set_ylim(mid - max_y_diff/2 - 0.06*max_y_diff, mid + max_y_diff/2 + 0.06*max_y_diff)
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
    fig.suptitle(f"Normalized FACTR Prediction vs Ground Truth\nModel {model_name}, episode {episode_name}", fontsize=16, y=0.96)
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
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved plot to {save_path}")

    if vs_all_plot:
        # Plot overlay of all dataset trajectories and FACTR predictions in rad
        fig, axes = plt.subplots(dof_dims, 1, figsize=(12, 2 * dof_dims), sharex=True)
        fig.suptitle(f"Joint Positions vs FACTR Predictions\nModel {model_name}, episode {episode_name}, y-plot-range: {max_y_diff:.1f}", fontsize=16, y=0.96)
        for d in range(dof_dims):
            ax = axes[d]
            # Dataset trajectories
            for ep_idx, ep_data in enumerate(joint_cmds_all):
                t_ep = np.arange(ep_data.shape[0])
                ax.plot(t_ep, ep_data[:, d], color="red", alpha=0.3, linewidth=1.0, label="Joint Pos. from Dataset" if (d == 0 and ep_idx == 0) else None)
            # Ground truth
            t_pred = np.arange(pred_action.shape[0])
            ax.plot(t_pred, true_actions[:, d], label="Ground Truth Joint Pos.", linewidth=2, color="black", alpha=0.8)
            mid = (max_mins[d][0] + max_mins[d][1]) / 2.0
            ax.set_ylim(mid - max_y_diff/2 - 0.06*max_y_diff, mid + max_y_diff/2 + 0.06*max_y_diff)
            # Predictions
            for i in range(pred_dims):
                ax.plot(t_pred + i, pred_action[:, i, d], color="blue", alpha=0.4, linewidth=0.8, label="FACTR prediction" if (d == 0 and i == 0) else None)
            ax.set_ylabel(f"J{d+1} Pos. [rad]")
            if d == 0:
                ax.legend(loc="upper right", fontsize=10)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Timestep")
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.96])
        save_path = f"{output_folder}/tr_rad_pred_vs_all_{episode_name}.png"
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✅ Saved overlay plot: {save_path}")


        # Plot overlay of all dataset trajectories and FACTR predictions normalized
        fig, axes = plt.subplots(dof_dims, 1, figsize=(12, 2 * dof_dims), sharex=True)
        fig.suptitle(f"Normalized Joint Positions vs FACTR Predictions\nModel {model_name}, episode {episode_name}", fontsize=16, y=0.96)
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
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"✅ Saved overlay plot: {save_path}")

