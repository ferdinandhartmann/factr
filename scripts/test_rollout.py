# ---------------------------------------------------------------------------
# FACTR policy test script on recorded training data (.pkl)
# ---------------------------------------------------------------------------
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
from tqdm import tqdm
import os
import yaml

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")


# ---------- CONFIG ----------
model_name = "20251024_50hz_60_batch64_lr2e4"
# CKPT_PATH = Path(f"scripts/checkpoints/{model_name}/rollout/latest_ckpt.ckpt")
EXP_CFG_PATH = Path(f"scripts/checkpoints/{model_name}/rollout/exp_config.yaml")
CKPT_PATH = Path(f"scripts/checkpoints/{model_name}/ckpt_005000.ckpt")
# EXP_CFG_PATH = Path(f"scripts/checkpoints/{model_name}/exp_config.yaml")
ROLLOUT_CFG_PATH = Path(f"scripts/checkpoints/{model_name}/rollout/rollout_config.yaml") 

base_name = "ep_64"
DATA_PATH = Path(f"/home/ferdinand/factr/process_data/converted_pkls_for_test/converted_{base_name}/")
image_file = DATA_PATH / f"{base_name}_image_obs.npy"
torque_file = DATA_PATH / f"{base_name}_torque_obs.npy"
actions_file = DATA_PATH / f"{base_name}_actions.npy"

if not image_file.exists() or not torque_file.exists() or not actions_file.exists():
    raise FileNotFoundError("One or more required .npy files are missing. Please check the DATA_PATH and base_name.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Torque is the 'obs' observation
    obs_mean = torch.tensor(rollout_config['norm_stats']['state']['mean']).float().to(DEVICE)
    obs_std = torch.tensor(rollout_config['norm_stats']['state']['std']).float().to(DEVICE)

    # Policy output (action) denormalization
    action_mean = torch.tensor(rollout_config['norm_stats']['action']['mean']).float().to(DEVICE)
    action_std = torch.tensor(rollout_config['norm_stats']['action']['std']).float().to(DEVICE)
    print(f"✅ Loaded normalization stats from {ROLLOUT_CFG_PATH}")

except Exception as e:
    print(f"⚠️ Warning: Could not load normalization stats from {ROLLOUT_CFG_PATH}. Running inference with unnormalized torque/actions. Error: {e}")
    # Define identity stats if loading fails to prevent script crash
    obs_mean, obs_std = 0., 1.
    action_mean, action_std = 0., 1.

# Load arrays
image_obs = np.load(image_file, allow_pickle=True)
torque_obs = np.load(torque_file)
true_action = np.load(actions_file)
print(f"✅ Loaded image_obs: {len(image_obs)} | torque_obs: {torque_obs.shape} | actions: {true_action.shape}")

# --- 4. Preprocess image helper ---
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

# -----------------------------
# INFERENCE
# -----------------------------
pred_actions = []

N = min(len(true_action), len(torque_obs), len(image_obs))
print(f"🚀 Running inference on {N} samples...")

for i in tqdm(range(N)):
    img = image_obs[i]
    torque = torque_obs[i]

    img_tensor = preprocess_image(img)

    # 🚨 FIXED: Normalize Torque Observation
    torque_tensor = torch.from_numpy(torque).float().to(DEVICE)
    torque_tensor = (torque_tensor - obs_mean) / obs_std # normalization
    torque_tensor = torque_tensor.unsqueeze(0) # Add batch dim

    # torque_tensor = torch.from_numpy(torque).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_action = policy.get_actions({"cam0": img_tensor}, torque_tensor)
        pred_action = pred_action * action_std + action_mean # <-- Inverse normalization 
        pred_action = pred_action.cpu().numpy()[0]

    pred_actions.append(pred_action)

pred_actions = np.array(pred_actions)
true_action = true_action[:N]

print("✅ Finished inference")
print(f"Pred shape: {pred_actions.shape}, True shape: {true_action.shape}")

with torch.no_grad():
    if isinstance(pred_actions, np.ndarray):
        if pred_actions.ndim == 3:
            pred_action = pred_actions[:, :, :]
        
    # print(f"Predicted action shape: {pred_action.shape}")
    # print(f"\nLast prediction check - Pred: {pred_action}, True: {true_action[-1]}")

# -----------------------------
# EVALUATION (MSE) and VISUALIZATION
# -----------------------------

t = np.arange(len(pred_action))  # <-- ensure t matches N
dof_dims = pred_action.shape[2]
pred_dims = pred_action.shape[1]
print(f"Number of Frames: {t.shape[0]}, dof_dims: {dof_dims}, pred_dims: {pred_dims}")

# for i in range (pred_dims):
#     mse = np.mean((pred_action[:, i, :] - true_action) ** 2, axis=0)
#     print(f"\n📊 Mean Squared Error per action dimension for pred. {100-i}:, Average {np.mean(mse):.6f}")
    # for j, val in enumerate(mse):
    #     print(f"  Joint {j+1}: {val:.6f}")


plt.figure(figsize=(10, 2 * dof_dims))
for d in range(dof_dims):
    plt.subplot(dof_dims, 1, d + 1)
    plt.plot(t, true_action[:, d], label="Ground Truth Joint Pos.", linewidth=2.5, color="red")
    plt.ylabel(f"Pos. Joint {d+1} [rad]")
    for i in range (pred_dims):    
        plt.plot(t + i, pred_action[:, i, d], label="Predicted Joint Pos.", linewidth=0.8, alpha=0.3, color="blue")
        if i == 0:
            plt.legend(loc="upper right")  
    if d == 0:
        plt.title(f"FACTR Policy Prediction vs Ground Truth ({pred_dims} pred. timesteps) of episode: {base_name}")
plt.xlabel("Frame")
plt.tight_layout()
save_path = f"/home/ferdinand/factr/scripts/test_rollout_output/test_rollout_{model_name}_{base_name}.png"
plt.savefig(save_path)
print(f"\n✅ Saved plot to {save_path}")