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
import roboticstoolbox as rtb
from mpl_toolkits.mplot3d import Axes3D

def load_and_prepare_policy(exp_cfg_path, ckpt_path, device):
    """
    Load and prepare the policy model from the experiment configuration and checkpoint.

    Args:
        exp_cfg_path (Path): Path to the experiment configuration file.
        ckpt_path (Path): Path to the model checkpoint file.
        device (str): Device to load the model onto ("cuda" or "cpu").

    Returns:
        policy (torch.nn.Module): The loaded and prepared policy model.
    """
    cfg = OmegaConf.load(exp_cfg_path)

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

    # Safely remove Hydra-only keys without triggering interpolation
    # Use dict-like interface instead of OmegaConf accessors
    raw_dict = OmegaConf.to_container(cfg, resolve=False)

    if "rt" in raw_dict and isinstance(raw_dict["rt"], str) and raw_dict["rt"].startswith("${hydra:"):
        cfg["rt"] = "agent/features"
    if "hydra" in raw_dict:
        cfg.pop("hydra", None)

    OmegaConf.resolve(cfg)

    agent_cfg = cfg.agent  # model definition lives here
    policy = instantiate(agent_cfg)

    # --- Load checkpoint ---
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"]

    # 🩹 Remove 'module.' prefix if present (for DataParallel checkpoints)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = policy.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Loaded policy from {ckpt_path}, step {ckpt['global_step']}")
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"⚠️ Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    policy.eval()
    policy.to(device)
    policy = torch.compile(policy)
    print(f"✅ Policy ready for inference on {device}")

    return policy

def load_and_extract_raw_data(pkl_path: Path, downsample=False, image_topic="/realsense/front/im", obs_topic="/franka_robot_state_broadcaster/external_joint_torques", action_topic="/joint_impedance_dynamic_gain_controller/joint_impedance_command"):
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

    return img_tensor

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

def get_all_joint_cmds_np(data_path, action_mean, action_std, downsample=False):
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

    print(f"✅ Loaded buffer from {buf_path}, total episodes: {len(traj_list)}, episode_idx: {episode_idx}")
    
    traj = traj_list[episode_idx-1]  # zero-indexed

    print(f"--- Extracting episode {episode_idx}, length: {len(traj)} steps ---")

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



def calculate_franka_fk(joint_angles_timeseries):
    """
    Calculates the end-effector position (x, y, z) for the Franka Panda robot.
    """
    robot = rtb.models.Panda() # 1. Load the Franka Panda model (which FR3 uses)
    
    num_timesteps = joint_angles_timeseries.shape[0]
    ee_positions = np.zeros((num_timesteps, 3))

    for t in range(num_timesteps):
        q = joint_angles_timeseries[t, :]
        T = robot.fkine(q)
        ee_positions[t, :] = T.t 
        
    return ee_positions