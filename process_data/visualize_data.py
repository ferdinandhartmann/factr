#!/usr/bin/env python3

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import cv2
from PIL import Image
import os
import imageio.v2 as imageio

def load_data(data_path):
    """Load data from pkl or json file"""
    data_path = Path(data_path)
    
    if data_path.suffix == '.pkl':
        with open(data_path, 'rb') as f:
            return pickle.load(f)
    elif data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def extract_topic_data(pkl_data, topic_name):
    """Extract data for a specific topic from pkl format"""
    if topic_name not in pkl_data['data']:
        return None, None
    
    data = pkl_data['data'][topic_name]
    timestamps = pkl_data['timestamps'][topic_name]
    
    return data, timestamps


def create_image_gif(pkl_data, output_path, topic_name='/realsense/front/im', fps=50):
    """Create Video from image data"""
    print(f"Creating Video for {topic_name}...")
    
    image_data, timestamps = extract_topic_data(pkl_data, topic_name)
    
    if image_data is None:
        print(f"No data found for topic: {topic_name}")
        return
    
    images = []
    for i, data in enumerate(image_data):
        if 'data' in data:
            # Convert ROS Image message data to numpy array
            width = data['width']
            height = data['height']
            encoding = data['encoding']
            raw_data = data['data']
            
            if encoding == 'rgb8':
                # RGB image
                img_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 3)
            elif encoding == '32FC1':
                # Depth image
                img_array = np.frombuffer(raw_data, dtype=np.float32).reshape(height, width)
                # Normalize depth for visualization
                img_array = (img_array - np.nanmin(img_array)) / (np.nanmax(img_array) - np.nanmin(img_array))
                img_array = (img_array * 255).astype(np.uint8)
                img_array = cv2.applyColorMap(img_array, cv2.COLORMAP_JET)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                print(f"Unsupported encoding: {encoding}")
                continue
            
            # Add step number text
            img_pil = Image.fromarray(img_array)
            images.append(img_pil)
            
            if i % 10 == 0:
                print(f"  Processed {i+1}/{len(image_data)} frames")
    
    if images:
        print(f"Saving Video with {len(images)} frames to {output_path}")
        # Resize images to reduce size before saving
        resized_images = [img.resize((img.width // 2, img.height // 2)) for img in images]
        imageio.mimsave(output_path, resized_images, fps=fps)
        # images[0].save(
        #     output_path,
        #     save_all=True,
        #     append_images=images[1:],
        #     duration=int(1000/fps),
        #     loop=0
        # )
        print(f"✅ Video saved to {output_path}")
    else:
        print("❌ No images to save")


def plot_joint_data(pkl_data, output_dir):
    """Plot Franka joint positions, external torques, and impedance commands"""
    print("Creating plots...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data for each topic
    topics = {
        'franka_state': '/franka/right/obs_franka_state',
        'external_torques': '/franka_robot_state_broadcaster/external_joint_torques',
        'external_torques_leader': '/franka/right/obs_franka_torque',
        # 'impedance_cmd': '/joint_impedance_command_controller/joint_trajectory',
        'impedance_cmd': '/joint_impedance_dynamic_gain_controller/joint_impedance_command',
        'measured_joints': '/franka_robot_state_broadcaster/measured_joint_states',
        'gripper_cmd': '/factr_teleop/right/cmd_gripper_pos',
        'gripper_state': '/gripper/right/obs_gripper_state'
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
    
    # 1. Plot Franka Joint Positions
    if 'franka_state' in data_dict:
        print("  Plotting Franka joint positions...")
        fig, axes = plt.subplots(7, 1, figsize=(12, 14))
        fig.suptitle('Franka Joint Positions', fontsize=16)
        
        positions = np.array([d['position'] for d in data_dict['franka_state']['data']])
        timestamps = data_dict['franka_state']['timestamps']
        
        for i in range(7):
            axes[i].plot(timestamps, positions[:, i], linewidth=2)
            axes[i].set_ylabel(f'Joint {i+1} [rad]', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            if i == 6:
                axes[i].set_xlabel('Time [s]', fontsize=10)
        
        plt.tight_layout()
        output_path = output_dir / 'franka_joint_positions.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved to {output_path}")
        plt.close()
    
    # 2. Plot External Joint Torques
    if 'external_torques' in data_dict:
        print("  Plotting external joint torques...")
        fig, axes = plt.subplots(7, 1, figsize=(12, 14))
        fig.suptitle('External Joint Torques', fontsize=16)
        
        torques = np.array([d['effort'] for d in data_dict['external_torques']['data']])
        timestamps = data_dict['external_torques']['timestamps']
        
        for i in range(7):
            axes[i].plot(timestamps, torques[:, i], linewidth=2, color='red')
            axes[i].set_ylabel(f'Joint {i+1} [Nm]', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            if i == 6:
                axes[i].set_xlabel('Time [s]', fontsize=10)

        # torques_leader = np.array([d['effort'] for d in data_dict['external_torques_leader']['data']])
        # timestamps_leader = data_dict['external_torques_leader']['timestamps']

        # for i in range(7):
        #     axes[i].plot(timestamps_leader, torques_leader[:, i], linewidth=2, color='blue')
        #     axes[i].set_ylabel(f'Leader? Joint {i+1} [Nm]', fontsize=10)
        #     axes[i].grid(True, alpha=0.3)
        #     if i == 6:
        #         axes[i].set_xlabel('Time [s]', fontsize=10)
        
        plt.tight_layout()
        output_path = output_dir / 'external_joint_torques.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved to {output_path}")
        plt.close()
    
    # # 3. Plot Impedance Controller Commands
    # if 'impedance_cmd' in data_dict:
    #     print("  Plotting impedance controller commands...")
    #     fig, axes = plt.subplots(7, 1, figsize=(12, 14))
    #     fig.suptitle('Impedance Controller Joint Commands', fontsize=16)
        
    #     commands = np.array([d['position'] for d in data_dict['impedance_cmd']['data']])
    #     timestamps = data_dict['impedance_cmd']['timestamps']
        
    #     for i in range(7):
    #         axes[i].plot(timestamps, commands[:, i], linewidth=2, color='green')
    #         axes[i].set_ylabel(f'Joint {i+1} [rad]', fontsize=10)
    #         axes[i].grid(True, alpha=0.3)
    #         if i == 6:
    #             axes[i].set_xlabel('Time [s]', fontsize=10)
        
    #     plt.tight_layout()
    #     output_path = output_dir / 'impedance_commands.png'
    #     plt.savefig(output_path, dpi=150, bbox_inches='tight')
    #     print(f"  ✅ Saved to {output_path}")
    #     plt.close()
    
    # 4. Combined plot: Measured vs Commanded positions
    if 'measured_joints' in data_dict and 'impedance_cmd' in data_dict:
        print("  Plotting measured vs commanded positions...")
        fig, axes = plt.subplots(7, 1, figsize=(12, 14))
        fig.suptitle('Joint Positions: Measured vs Commanded', fontsize=16)
        
        measured_pos = np.array([d['position'] for d in data_dict['measured_joints']['data']])
        measured_ts = data_dict['measured_joints']['timestamps']
        
        commanded_pos = np.array([d['position'] for d in data_dict['impedance_cmd']['data']])
        commanded_ts = data_dict['impedance_cmd']['timestamps']
        
        for i in range(7):
            axes[i].plot(measured_ts, measured_pos[:, i], linewidth=2, label='Measured', alpha=0.7)
            axes[i].plot(commanded_ts, commanded_pos[:, i], linewidth=2, label='Commanded', alpha=0.7, linestyle='--')
            axes[i].set_ylabel(f'Joint {i+1} [rad]', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend(loc='upper right', fontsize=8)
            if i == 6:
                axes[i].set_xlabel('Time [s]', fontsize=10)
        
        plt.tight_layout()
        output_path = output_dir / 'measured_vs_commanded.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved to {output_path}")
        plt.close()
    
    # 5. Plot Gripper State
    if 'gripper_state' in data_dict:
        print("  Plotting gripper state...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        fig.suptitle('Gripper Position', fontsize=16)
        
        gripper_pos = np.array([d['position'][0] if len(d['position']) > 0 else 0.0 
                                for d in data_dict['gripper_state']['data']])
        timestamps = data_dict['gripper_state']['timestamps']
        
        ax.plot(timestamps, gripper_pos, linewidth=2, color='purple')
        ax.set_ylabel('Gripper Position [0-1]', fontsize=10)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        output_path = output_dir / 'gripper_state.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved to {output_path}")
        plt.close()
    
    # 6. Plot Gripper Command (if available)
    if 'gripper_cmd' in data_dict:
        print("  Plotting gripper command...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        fig.suptitle('Gripper Command', fontsize=16)
        
        gripper_cmd = np.array([d['position'][0] if len(d['position']) > 0 else 0.0 
                                for d in data_dict['gripper_cmd']['data']])
        timestamps = data_dict['gripper_cmd']['timestamps']
        
        ax.plot(timestamps, gripper_cmd, linewidth=2, color='orange')
        ax.set_ylabel('Gripper Command [0-1]', fontsize=10)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        output_path = output_dir / 'gripper_command.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved to {output_path}")
        plt.close()
    
    # 7. Combined Gripper Plot
    if 'gripper_state' in data_dict and 'gripper_cmd' in data_dict:
        print("  Plotting gripper state vs command...")
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        fig.suptitle('Gripper: State vs Command', fontsize=16)
        
        gripper_state = np.array([d['position'][0] if len(d['position']) > 0 else 0.0 
                                   for d in data_dict['gripper_state']['data']])
        state_ts = data_dict['gripper_state']['timestamps']
        
        gripper_cmd = np.array([d['position'][0] if len(d['position']) > 0 else 0.0 
                                for d in data_dict['gripper_cmd']['data']])
        cmd_ts = data_dict['gripper_cmd']['timestamps']
        
        ax.plot(state_ts, gripper_state, linewidth=2, label='State', alpha=0.7, color='purple')
        ax.plot(cmd_ts, gripper_cmd, linewidth=2, label='Command', alpha=0.7, linestyle='--', color='orange')
        ax.set_ylabel('Gripper Position [0-1]', fontsize=10)
        ax.set_xlabel('Time [s]', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        output_path = output_dir / 'gripper_state_vs_command.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved to {output_path}")
        plt.close()
    
    print("✅ All plots created!")


def visualize_data(data_path, output_dir=None):
    """Main function to visualize collected data"""
    data_path = Path(data_path)
    
    # Set output directory
    if output_dir is None:
        output_dir = data_path.parent / f"{data_path.stem}_visualizations"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Visualizing data from: {data_path}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    # Load pkl data (contains actual image data)
    pkl_path = data_path.with_suffix('.pkl')
    if not pkl_path.exists():
        print(f"❌ PKL file not found: {pkl_path}")
        return
    
    pkl_data = load_data(pkl_path)
    
    # Create GIFs for images
    create_image_gif(pkl_data, output_dir / 'camera_rgb.mp4', '/realsense/front/im', fps=25)
    # create_image_gif(pkl_data, output_dir / 'camera_depth.mp4', '/realsense/arm/depth', fps=50)
    # Create plots
    plot_joint_data(pkl_data, output_dir)
    
    print("=" * 60)
    print(f"✅ Visualization complete! Check {output_dir}")


if __name__ == '__main__':
    import sys

    base_data_dir = Path("/home/ferdinand/factr/process_data/data_to_process/20251112/data")
    base_output_dir = Path("/home/ferdinand/factr/process_data/data_to_process/20251112/visualizations")

    # Allow overriding from CLI
    if len(sys.argv) >= 2:
        base_data_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        base_output_dir = Path(sys.argv[2])

    # Create output base directory
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Get all .pkl files (sorted for consistent numbering)
    pkl_files = sorted(base_data_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"❌ No .pkl files found in {base_data_dir}")
        sys.exit(1)

    print(f"Found {len(pkl_files)} .pkl files in {base_data_dir}")
    
    # Loop over all PKL files and visualize
    for idx, pkl_path in enumerate(pkl_files, start=0):
        output_dir = base_output_dir / pkl_path.stem
        
        # Skip if visualization already exists
        if output_dir.exists() and any(output_dir.iterdir()):
            print("=" * 80)
            print(f"⚡ Skipping file {idx}/{len(pkl_files)}: {pkl_path.name} (already visualized)")
            continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print(f"📁 Processing file {idx}/{len(pkl_files)}: {pkl_path.name}")
        print(f"📊 Output directory: {output_dir}")
        print("=" * 80)

        try:
            visualize_data(pkl_path, output_dir)
        except Exception as e:
            print(f"❌ Error processing {pkl_path.name}: {e}")
            continue


    print("=" * 80)
    print(f"✅ All visualizations completed! Check {base_output_dir}")

