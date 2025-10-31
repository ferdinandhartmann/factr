#!/usr/bin/env python3
"""
Convert a single FACTR .pkl recording file into numpy arrays usable for policy testing.
Usage:
    python convert_single_pkl.py /path/to/ep_2.pkl -o ./converted_ep_2
"""

import pickle
import numpy as np
from pathlib import Path
import argparse
import cv2



def load_pkl(pkl_path):
    """Load the pickle file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"✅ Loaded {pkl_path}")
    print(f"Keys: {list(data.keys())}")
    return data


def extract_factr_data(raw_data):
    """
    Extract image, state, and action data from a FACTR pickle file.
    Handles both 'data' style (rosbag-format) and preprocessed formats.
    """
    image_obs, torque_obs, actions = [], [], []


    ########################### Topic names (customize as needed) ###########################
    image_topic = "/realsense/arm/im"
    obs_topic = "/franka_robot_state_broadcaster/external_joint_torques"
    action_topic = "/joint_impedance_command_controller/joint_trajectory"



    # Case 2: raw teleop data (nested under "data")
    if "data" in raw_data:
        print("Detected raw teleop format.")
        entries = raw_data["data"]

        # each topic (e.g. /camera/color/image_raw, /joint_impedance_command_controller/joint_trajectory)
        for topic, values in entries.items():
            if image_topic in topic:
                print(f"Extracting image data from {topic} ({len(values)} frames)")
                imgs = []
                for v in values:
                    if isinstance(v, dict) and "data" in v:
                        img = np.frombuffer(v["data"], dtype=np.uint8)
                        if "height" in v and "width" in v:
                            try:
                                img = img.reshape((v["height"], v["width"], -1))
                            except:
                                pass
                        imgs.append(img)
                image_obs.extend(imgs)
            elif action_topic in topic:
                print(f"Extracting joint actions from {topic} ({len(values)} commands)")
                for v in values:
                    if isinstance(v, dict) and "position" in v:
                        actions.append(v["position"])
            elif obs_topic in topic:
                print(f"Extracting observations from {topic} ({len(values)} commands)")
                for v in values:
                    if isinstance(v, dict) and "effort" in v:
                        torque_obs.append([v["effort"]])

        # Create dummy state if not found
        torque_obs = np.zeros((len(actions), 7)) if len(actions) > 0 else np.zeros((len(image_obs), 7))

    else:
        raise ValueError("Unknown data structure in .pkl file.")

    # Downsample to target rate (50 Hz) if data frequency is higher
    target_freq = 50.0
    timestamps = [v["timestamp"] for topic, values in raw_data["data"].items() for v in values if "timestamp" in v]
    if len(timestamps) > 1:
        time_diffs = np.diff(sorted(timestamps))
        avg_freq = 1.0 / np.mean(time_diffs)
        print(f"Detected average frequency: {avg_freq:.1f} Hz")
    else:
        avg_freq = 50  # Default to 50 Hz if timestamps are insufficient
        print(f"defaulting average frequency to: {avg_freq:.1f} Hz")
    if avg_freq > target_freq:
        step = int(np.floor(avg_freq / target_freq))
        image_obs = image_obs[::step]
        torque_obs = torque_obs[::step]
        actions = actions[::step]
        avg_freq = avg_freq / step
        print(f"🔻 Downsampled from ~{avg_freq * step:.1f} Hz to ~{avg_freq:.1f} Hz (step={step})")

    return np.array(image_obs), np.array(torque_obs), np.array(actions)


def save_arrays(output_dir, base_name, image_obs, torque_obs, actions):
    """Save numpy arrays to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    if len(image_obs) > 0:
        paths["image_obs"] = output_dir / f"{base_name}_image_obs.npy"
        np.save(paths["image_obs"], image_obs)
    if len(torque_obs) > 0:
        paths["torque_obs"] = output_dir / f"{base_name}_torque_obs.npy"
        np.save(paths["torque_obs"], torque_obs)
    if len(actions) > 0:
        paths["actions"] = output_dir / f"{base_name}_actions.npy"
        np.save(paths["actions"], actions)

    for k, v in paths.items():
        print(f"💾 Saved {k}: {v} ({np.load(v).shape})")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Convert a single FACTR pickle recording to .npy arrays")
    parser.add_argument("input", nargs="?", help="Path to the .pkl file")
    parser.add_argument("-o", "--output", help="Output directory", default="./converted_output")
    args = parser.parse_args()

    if not args.input:
        episode = "ep_61" ######################## SELECT EPISODE HERE ########################
        # args.input = f"/home/ferdinand/factr/process_data/raw_data/20251024_train/{episode}.pkl"
        args.input = f"/home/ferdinand/factr/process_data/raw_data_evaluation/20251024/{episode}.pkl"
        args.output = f"/home/ferdinand/factr/process_data/converted_pkls_for_test/converted_{episode}/"
    
    # Ensure the output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = Path(args.input)
    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")

    data = load_pkl(pkl_path)
    image_obs, torque_obs, actions = extract_factr_data(data)
    save_arrays(args.output, pkl_path.stem, image_obs, torque_obs, actions)

    print("\n✅ Conversion complete.")
    print(f"Output saved in: {args.output}")


if __name__ == "__main__":
    main()
