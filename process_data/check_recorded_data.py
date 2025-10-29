import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Any, List
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import math
import sys


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
    
    for d in data_list:
        if key in d:
            value = d[key]
            # Check if the value is a sequence of length 7
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 7:
                processed_data.append(value)
            else:
                # Corrupt or unexpected length (e.g., gripper data accidentally logged)
                processed_data.append([np.nan] * 7)
        else:
            # Missing key
            processed_data.append([np.nan] * 7)
            
    # Convert to NumPy array with float dtype to allow NaNs
    return np.array(processed_data, dtype=np.float32)

def plot_joint_data(pkl_data, output_dir):
    """
    Creates the two requested combined plots from a single PKL file:
    1. Commanded Position, Measured Position, and Broadcaster Torques.
    2. Observed State Position and Observed Torque.
    """
    print("Creating combined plots...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Topic Mapping ---
    topics = {
        'franka_state': '/franka/right/obs_franka_state',
        'external_torques_broadcaster': '/franka_robot_state_broadcaster/external_joint_torques',
        'franka_torque_leader': '/franka/right/obs_franka_torque',
        'impedance_cmd': '/joint_impedance_command_controller/joint_trajectory',
        'measured_joints': '/franka_robot_state_broadcaster/measured_joint_states',
    }
    
    data_dict = {}
    for key, topic in topics.items():
        data, timestamps = extract_topic_data(pkl_data, topic)
        if data is not None and len(data) > 0:
            data_dict[key] = {
                'data': data,
                'timestamps': np.array(timestamps) / 1e9  # Convert to seconds
            }
            # Convert to relative time
            if len(data_dict[key]['timestamps']) > 0:
                data_dict[key]['timestamps'] -= data_dict[key]['timestamps'][0]

    # Get data using safe extraction
    broadcaster_states = safe_extract_7d_data(data_dict['measured_joints']['data'], 'position')
    commanded_pos = safe_extract_7d_data(data_dict['impedance_cmd']['data'], 'position')
    broadcaster_torques = safe_extract_7d_data(data_dict['external_torques_broadcaster']['data'], 'effort')

    # ----------------------------------------------------------------------
    # 1. PLOT: Commanded Position, Measured Position, and Observed Position
    # ----------------------------------------------------------------------
    required_keys_1 = ['measured_joints', 'impedance_cmd', 'franka_state']
    if all(k in data_dict for k in required_keys_1):
        print("  Plotting Combined Joint Position Plot...")
        fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
        fig.suptitle('Combined Joint Data: Commanded Pos, Measured Pos, and Broadcaster Torque', fontsize=16)

        # Get data
        broadcaster_states =  safe_extract_7d_data(data_dict['measured_joints']['data'], 'position')
        broadcaster_ts = data_dict['measured_joints']['timestamps']
        
        commanded_pos = safe_extract_7d_data(data_dict['impedance_cmd']['data'], 'position')
        commanded_ts = data_dict['impedance_cmd']['timestamps']
        
        observed_states = safe_extract_7d_data(data_dict['franka_state']['data'], 'position')
        observed_ts = data_dict['franka_state']['timestamps']

        for i in range(7):
            ax1 = axes[i]
            # Plot Positions/Commands on left Y-axis (ax1)
            line1, = ax1.plot(broadcaster_ts, broadcaster_states[:, i], linewidth=3, label='/franka_robot_state_broadcaster/measured_joint_states [rad]', alpha=0.7, color='blue')
            line2, = ax1.plot(commanded_ts, commanded_pos[:, i], linewidth=2, label='Commanded Pos to controller (joint_trajectory) [rad]', alpha=0.7, color='cyan')
            line3, = ax1.plot(observed_ts, observed_states[:, i], linewidth=2, label='/franka/right/obs_franka_state [rad]', alpha=0.7, color='red')
            ax1.set_ylabel(f'J{i+1} Pos [rad]', fontsize=10, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.grid(True, alpha=0.3)

            if i == 0:
                lines = [line1, line2, line3]
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right', fontsize=8)

            if i == 6:
                ax1.set_xlabel('Time [s]', fontsize=10)
        
        plt.tight_layout()
        output_path = output_dir / 'combined_pos.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved to {output_path}")
        plt.close()
    else:
        print("  ⚠️ Skipping Plot 1: Missing one or more required topics for Combined Pos/Cmd/Torque.")


    # ----------------------------------------------------------------------
    # 2. PLOT: Broadcasted Torque and Observed Torque
    # ----------------------------------------------------------------------
    required_keys_2 = ['external_torques_broadcaster', 'franka_torque_leader']
    if all(k in data_dict for k in required_keys_2):
        print("  Plotting Combined Joint Torque Plot...")
        fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
        fig.suptitle('FACTR Observation Comparison: State Position vs. Observed Torque', fontsize=16)

        # Get data
        broadcaster_torques = safe_extract_7d_data(data_dict['external_torques_broadcaster']['data'], 'effort')
        broadcaster_ts = data_dict['franka_state']['timestamps']
        
        franka_leader_torques = safe_extract_7d_data(data_dict['franka_torque_leader']['data'], 'effort')
        franka_leader_ts = data_dict['franka_torque_leader']['timestamps']
        
        for i in range(7):
            ax1 = axes[i]
            line1, = ax1.plot(broadcaster_ts, broadcaster_torques[:, i], linewidth=2, label='Broadcasted Torque [Nm]', alpha=0.7, color='darkgreen')
            line2, = ax1.plot(franka_leader_ts, franka_leader_torques[:, i], linewidth=2, label='Observed Torque [Nm]', alpha=0.7, color='orange')
            ax1.set_ylabel(f'J{i+1} Torque [Nm]', fontsize=10, color='orange')
            ax1.tick_params(axis='y', labelcolor='orange')

            if i == 0:
                lines = [line1, line2]
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right', fontsize=8)

            if i == 6:
                ax1.set_xlabel('Time [s]', fontsize=10)
        
        plt.tight_layout()
        output_path = output_dir / 'combined_torque.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved to {output_path}")
        plt.close()
    else:
        print("  ⚠️ Skipping Plot 2: Missing one or more required topics for FACTR Observation Comparison.")
    



def gaussian_2d_kernel(
    kernel_size: int,
    sigma: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generates a 2D Gaussian kernel tensor."""
    
    kernel_radius = kernel_size // 2
    x = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=dtype)
    
    gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    
    kernel_2d = torch.outer(gaussian_1d, gaussian_1d)
    return kernel_2d


def gaussian_2d_smoothing(
    img: torch.Tensor,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Apply 2D Gaussian smoothing (blur) to a batch of images using F.conv2d.
    Args:
        img: Tensor of shape (..., C, H, W).
        scale: Controls the standard deviation (sigma) of the Gaussian kernel. 
               Larger scale corresponds to more smoothing.
    Returns:
        blurred: Tensor of the same shape as img.
    """
    if scale <= 0:
        return img

    sigma = scale
    
    # Calculate kernel size (must be odd). 
    kernel_size = max(3, 2 * math.ceil(3 * sigma) + 1)

    kernel_2d = gaussian_2d_kernel(kernel_size, sigma, device=img.device, dtype=img.dtype)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    C = img.shape[-3]
    kernel_2d = kernel_2d.repeat(C, 1, 1, 1)  # shape: (C, 1, kH, kW)

    padding = kernel_size // 2
    
    original_shape = img.shape
    # Flatten the batch/time dimensions: (B*T, C, H, W)
    batch_shape = original_shape[:-3]
    spatial_shape = original_shape[-2:]
    batch_size = int(torch.prod(torch.tensor(batch_shape)))
    img_reshaped = img.view(batch_size, C, *spatial_shape)
    
    blurred = F.conv2d(img_reshaped, kernel_2d, groups=C, padding=padding)
    
    return blurred.view(*original_shape)

def visualize_curriculum_steps(pkl_data, output_dir, topic_name='/realsense/arm/im'):
    """
    Applies the FACTR visual curriculum (downsample and blur) to the first image 
    and saves the 6 resulting degradation levels (Scale 0 to 5) using PyTorch for blurring.
    """
    print(f"🖼️ Visualizing curriculum steps for {topic_name}...")
    
    # --- Data Extraction (Unchanged) ---
    image_data, _ = extract_topic_data(pkl_data, topic_name)
    
    if image_data is None or len(image_data) == 0:
        print(f"❌ No image data found for topic: {topic_name}. Skipping curriculum visualization.")
        return

    first_image = image_data[0]
    
    if 'data' not in first_image or first_image['encoding'] != 'rgb8':
        print(f"❌ First image is not 'rgb8' or missing data key. Skipping curriculum visualization.")
        return

    width = first_image['width']
    height = first_image['height']
    raw_data = first_image['data']
    
    # Convert raw bytes to numpy array
    try:
        base_img_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3)
    except ValueError as e:
        print(f"❌ Could not reshape image data: {e}. Skipping.")
        return

    for scale in range(6):
                
        img = base_img_array.copy()

        # Target size for ViT input 224x224
        target_h, target_w = 224, 224 
        # 1. Downsample (to the ViT input size, as is done in FACTR preprocessing)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. Convert to PyTorch Tensor for blurring
        # From (H, W, C) to (1, C, H, W) and scale from [0, 255] to [0, 1] (float)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        if scale > 0:
            # We use the blur_sigma from the degradation schedule as the scale argument
            blurred_tensor = gaussian_2d_smoothing(img_tensor, scale=scale)
        else:
            blurred_tensor = img_tensor
        
        # Convert tensor (1, C, H, W) to (H, W, C) numpy array
        processed_img_array = (blurred_tensor.squeeze(0).permute(1, 2, 0) * 255.0).clamp(0, 255).byte().cpu().numpy()

        img_pil = Image.fromarray(processed_img_array)
        output_path = output_dir / f"curriculum_scale_{scale}.png"
        img_pil.save(output_path)
        print(f" 	Scale {scale} saved: {output_path.name}")
        
    print("✅ Curriculum visualization complete!")





if __name__ == '__main__':

    ################# Single File Visualization #################

    episode_name = "ep_4"  ## SELECT EPISODE HERE ####
    pkl_path = Path(f"/home/ferdinand/factr/process_data/data_to_process/20251024/data/{episode_name}.pkl")
        
    base_dir = pkl_path.parent.parent / "visualizations"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    episode_dir = base_dir / pkl_path.stem
    if not episode_dir.exists():
        episode_dir.mkdir(parents=True, exist_ok=True)
    output_dir = episode_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Visualizing data from: {pkl_path}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    pkl_path = pkl_path.with_suffix('.pkl')
    if not pkl_path.exists():
        print(f"❌ PKL file not found: {pkl_path}")
        sys.exit(1)
    try:
        pkl_data = load_data(pkl_path)
        plot_joint_data(pkl_data, output_dir)
        visualize_curriculum_steps(pkl_data, output_dir, topic_name='/realsense/arm/im')
    except Exception as e:
        print(f"❌ Error processing {pkl_path.name}: {e}")

    print(f"🎯 Single file visualization complete!")

