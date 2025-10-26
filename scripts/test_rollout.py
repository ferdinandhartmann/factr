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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")


# ---------- CONFIG ----------
CKPT_PATH = Path("scripts/checkpoints/test/rollout/latest_ckpt.ckpt")
EXP_CFG_PATH = Path("scripts/checkpoints/test/rollout/exp_config_offline.yaml")

base_name = "ep_38"
DATA_PATH = Path(f"/home/ferdinand/factr/process_data/converted_pkls_for_test/converted_{base_name}/")
image_file = DATA_PATH / f"{base_name}_image_obs.npy"
state_file = DATA_PATH / f"{base_name}_state_obs.npy"
actions_file = DATA_PATH / f"{base_name}_actions.npy"

if not image_file.exists() or not state_file.exists() or not actions_file.exists():
    raise FileNotFoundError("One or more required .npy files are missing. Please check the DATA_PATH and base_name.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

number_of_last_pred_toshow = 100

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
policy.load_state_dict(ckpt["model"])
policy.eval()
policy.to(DEVICE)

print(f"✅ Loaded policy from {CKPT_PATH}, step {ckpt['global_step']}")
policy = torch.compile(policy)

# Load arrays
image_obs = np.load(image_file, allow_pickle=True)
state_obs = np.load(state_file)
true_action = np.load(actions_file)
print(f"✅ Loaded image_obs: {len(image_obs)} | state_obs: {state_obs.shape} | actions: {true_action.shape}")

# --- 4. Preprocess image helper ---
def preprocess_image(img):
    # Handle grayscale or depth
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    img = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.to(DEVICE)

# -----------------------------
# INFERENCE
# -----------------------------
pred_actions = []

N = min(len(true_action), len(state_obs), len(image_obs))
print(f"🚀 Running inference on {N} samples...")

for i in tqdm(range(N)):
    img = image_obs[i]
    state = state_obs[i]

    img_tensor = preprocess_image(img)
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_action = policy.get_actions({"cam0": img_tensor}, state_tensor)
        pred_action = pred_action.cpu().numpy()[0]

    pred_actions.append(pred_action)

pred_actions = np.array(pred_actions)
true_action = true_action[:N]

print("✅ Finished inference")
print(f"Pred shape: {pred_actions.shape}, True shape: {true_action.shape}")

with torch.no_grad():
    if isinstance(pred_actions, np.ndarray):
        if pred_actions.ndim == 3:
            pred_action = pred_actions[:, -number_of_last_pred_toshow:, :]
        
    print(f"Predicted action shape: {pred_action.shape}")
    print(f"\nLast prediction check - Pred: {pred_action}, True: {true_action[-1]}")

# -----------------------------
# EVALUATION (MSE) and VISUALIZATION
# -----------------------------

t = np.arange(len(pred_action))  # <-- ensure t matches N
dof_dims = pred_action.shape[2]
pred_dims = pred_action.shape[1]
print(f"Number of Frames: {t.shape[0]}, dof_dims: {dof_dims}, pred_dims: {pred_dims}")

for i in range (pred_dims):
    mse = np.mean((pred_action[:, i, :] - true_action) ** 2, axis=0)
    print(f"\n📊 Mean Squared Error per action dimension for pred. {100-i}:, Average {np.mean(mse):.6f}")
    for j, val in enumerate(mse):
        print(f"  Joint {j+1}: {val:.6f}")


plt.figure(figsize=(10, 2 * dof_dims))
for d in range(dof_dims):
    plt.subplot(dof_dims, 1, d + 1)
    plt.plot(t, true_action[:, d], label="Ground Truth", linewidth=2)
    plt.ylabel(f"Pos. Joint {d+1} [rad]")
    plt.legend(loc="upper right")  
    for i in range (pred_dims):    
        plt.plot(t, pred_action[:, i, d], "--", label="Predicted", linewidth=2)
    if d == 0:
        plt.title(f"FACTR Policy Prediction vs Ground Truth of {number_of_last_pred_toshow} last prediction (of 100) of episode: {base_name}")
plt.xlabel("Frame")
plt.tight_layout()
plt.savefig(f"/home/ferdinand/factr/scripts/test_rollout_{base_name}.png")
print(f"\n✅ Saved plot to /home/ferdinand/factr/scripts/test_rollout_output/test_rollout_{base_name}.png")