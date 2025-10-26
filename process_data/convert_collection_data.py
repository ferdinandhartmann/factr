#!/usr/bin/env python3

import pickle
import json
import numpy as np
from pathlib import Path
import argparse
import os


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


def extract_joint_gripper_commands_from_json(json_data):
    """Extract joint and gripper command data from JSON format"""
    joint_commands = []
    gripper_commands = []
    # timestamps = []
    
    # Topics for joint and gripper commands
    joint_topic = '/joint_impedance_command_controller/joint_trajectory'
    gripper_topic = '/factr_teleop/right/cmd_gripper_pos'
    
    for entry in json_data:
        if entry.get('topic') == joint_topic:
            # Extract joint position commands
            if 'position' in entry and entry['position']:
                joint_commands.append(entry['position'])
                # timestamps.append(entry.get('timestamp', 0.0))
        elif entry.get('topic') == gripper_topic:
            # Extract gripper commands
            if 'position' in entry and entry['position']:
                gripper_commands.append(entry['position'])
    
    return {
        'joint_commands': np.array(joint_commands) if joint_commands else np.array([]),
        'gripper_commands': np.array(gripper_commands) if gripper_commands else np.array([]),
        # 'timestamps': np.array(timestamps) if timestamps else np.array([])
    }


def extract_all_data_from_pkl(pkl_data):
    """Extract all data from pickle format including joint, gripper, and image data"""
    extracted_data = {}
    
    # Topics to extract
    topics = {
        'joint_commands': '/joint_impedance_command_controller/joint_trajectory',
        'gripper_commands': '/factr_teleop/right/cmd_gripper_pos',
        # 'franka_state': '/franka/right/obs_franka_state',
        # 'external_torques': '/franka_robot_state_broadcaster/external_joint_torques',
        # 'measured_joints': '/franka_robot_state_broadcaster/measured_joint_states',
        # 'gripper_state': '/gripper/right/obs_gripper_state'
    }
    
    # Image topics (common patterns)
    image_topics = []
    
    if 'data' in pkl_data and 'timestamps' in pkl_data:
        # Extract structured data
        for data_type, topic in topics.items():
            if topic in pkl_data['data']:
                data_list = []
                for data_point in pkl_data['data'][topic]:
                    if isinstance(data_point, dict):
                        if 'position' in data_point:
                            data_list.append(data_point['position'])
                        elif 'velocity' in data_point:
                            data_list.append(data_point['velocity'])
                        elif 'effort' in data_point:
                            data_list.append(data_point['effort'])
                        else:
                            # For other structured data
                            data_list.append(data_point)
                    else:
                        data_list.append(data_point)
                
                if data_list:
                    extracted_data[data_type] = np.array(data_list)
                    # Also save timestamps for this topic
                    # if topic in pkl_data['timestamps']:
                    #     extracted_data[f'{data_type}_timestamps'] = np.array(pkl_data['timestamps'][topic])
        
        # Extract image data
        for topic in pkl_data['data'].keys():
            if any(img_keyword in topic.lower() for img_keyword in ['image', 'camera', 'rgb', 'depth']):
                image_data = []
                for data_point in pkl_data['data'][topic]:
                    if isinstance(data_point, dict) and 'data' in data_point:
                        # Convert image data to numpy array
                        img_data = np.frombuffer(data_point['data'], dtype=np.uint8)
                        if 'width' in data_point and 'height' in data_point:
                            # Reshape if dimensions are available
                            try:
                                img_data = img_data.reshape((data_point['height'], data_point['width'], -1))
                            except:
                                pass  # Keep as 1D if reshape fails
                        image_data.append(img_data)
                
                if image_data:
                    # Use topic name as key, replacing '/' with '_'
                    topic_key = topic.replace('/', '_').replace('\\', '_')
                    extracted_data[f'images{topic_key}'] = np.array(image_data)
                    # if topic in pkl_data['timestamps']:
                    #     extracted_data[f'images_{topic_key}_timestamps'] = np.array(pkl_data['timestamps'][topic])
    
    return extracted_data


def save_all_data_to_npy(data_dict, output_dir, base_name):
    """Save all extracted data to npy files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Save all data types
    for data_type, data_array in data_dict.items():
        if len(data_array) > 0:
            # Create filename
            filename = f"{base_name}_{data_type}.npy"
            filepath = output_dir / filename
            
            # Save data
            np.save(filepath, data_array)
            saved_files.append(str(filepath))
            
            # Print info
            if 'image' in data_type.lower():
                print(f"✅ Saved {data_type}: {filepath} (shape: {data_array.shape}, dtype: {data_array.dtype})")
            else:
                print(f"✅ Saved {data_type}: {filepath} (shape: {data_array.shape})")
    
    return saved_files


def convert_data_file(input_file, output_dir=None):
    """Convert a single data file to npy format"""
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Set output directory - create npy folder structure
    if output_dir is None:
        # Create npy folder in raw_data directory
        raw_data_dir = input_path.parent.parent  # Go up to raw_data level
        npy_dir = raw_data_dir / "npy"
        # Create subdirectory with same name as input directory
        input_dir_name = input_path.parent.name
        output_dir = npy_dir / input_dir_name
    else:
        output_dir = Path(output_dir)
    
    # Load data
    print(f"📁 Loading data from: {input_file}")
    data = load_data(input_path)
    
    # Extract data based on file format
    if input_path.suffix == '.json':
        extracted_data = extract_joint_gripper_commands_from_json(data)
    elif input_path.suffix == '.pkl':
        extracted_data = extract_all_data_from_pkl(data)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Generate base name for output files
    base_name = input_path.stem
    
    # Save to npy files
    print(f"💾 Saving to directory: {output_dir}")
    saved_files = save_all_data_to_npy(extracted_data, output_dir, base_name)
    
    # Print summary
    print(f"\n📊 Summary:")
    for data_type, data_array in extracted_data.items():
        if len(data_array) > 0:
            print(f"  {data_type}: {len(data_array)} samples (shape: {data_array.shape})")
    print(f"  Total saved files: {len(saved_files)}")
    
    return saved_files


def convert_directory(input_dir, output_dir=None, file_pattern="*.pkl"):
    """Convert all matching files in a directory"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Set output directory - create npy folder structure
    if output_dir is None:
        # Create npy folder in raw_data directory
        raw_data_dir = input_path.parent.parent  # Go up to raw_data level
        npy_dir = raw_data_dir / "npy"
        # Create subdirectory with same name as input directory
        input_dir_name = input_path.name
        output_dir = npy_dir / input_dir_name
    else:
        output_dir = Path(output_dir)
    
    # Find all matching files
    if file_pattern == "*.json":
        files = list(input_path.glob("*.json"))
    elif file_pattern == "*.pkl":
        files = list(input_path.glob("*.pkl"))
    else:
        files = list(input_path.glob(file_pattern))
    
    if not files:
        print(f"⚠️  No files found matching pattern: {file_pattern}")
        return []
    
    print(f"🔍 Found {len(files)} files to convert")
    print(f"📁 Input directory: {input_path}")
    print(f"📁 Output directory: {output_dir}")
    
    all_saved_files = []
    for file_path in files:
        print(f"\n{'='*50}")
        print(f"Processing: {file_path.name}")
        try:
            saved_files = convert_data_file(file_path, output_dir)
            all_saved_files.extend(saved_files)
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
    
    print(f"\n🎉 Conversion completed!")
    print(f"Total files processed: {len(files)}")
    print(f"Total npy files created: {len(all_saved_files)}")
    print(f"Output directory: {output_dir}")
    
    return all_saved_files


def main():
    parser = argparse.ArgumentParser(description="Convert data collection files to npy format")
    parser.add_argument("input", help="Input file or directory path")
    parser.add_argument("-o", "--output", help="Output directory (default: input_dir/npy_output)")
    parser.add_argument("-p", "--pattern", default="*.pkl", 
                       help="File pattern for directory conversion (default: *.pkl)")
    parser.add_argument("--json", action="store_true", 
                       help="Convert JSON files instead of pickle files")
    
    args = parser.parse_args()
    
    # Set file pattern based on arguments
    if args.json:
        args.pattern = "*.json"
    
    input_path = Path(args.input)
    
    try:
        if input_path.is_file():
            # Convert single file
            convert_data_file(input_path, args.output)
        elif input_path.is_dir():
            # Convert directory
            convert_directory(input_path, args.output, args.pattern)
        else:
            print(f"❌ Input path does not exist: {input_path}")
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
