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
    image_obs, state_obs, actions = [], [], []

    # Case 1: FACTR processed buffer
    if all(k in raw_data for k in ["image_obs", "state_obs", "actions"]):
        print("Detected processed FACTR format.")
        image_obs = np.array(raw_data["image_obs"])
        state_obs = np.array(raw_data["state_obs"])
        actions = np.array(raw_data["actions"])
        return image_obs, state_obs, actions

    # Case 2: raw teleop data (nested under "data")
    elif "data" in raw_data:
        print("Detected raw teleop format.")
        entries = raw_data["data"]

        # each topic (e.g. /camera/color/image_raw, /joint_impedance_command_controller/joint_trajectory)
        for topic, values in entries.items():
            if "/realsense/arm/im" in topic:
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
            elif "joint_trajectory" in topic:
                print(f"Extracting joint actions from {topic} ({len(values)} commands)")
                for v in values:
                    if isinstance(v, dict) and "position" in v:
                        actions.append(v["position"])
            elif "gripper" in topic:
                print(f"Extracting gripper actions from {topic} ({len(values)} commands)")
                for v in values:
                    if isinstance(v, dict) and "position" in v:
                        actions.append([v["position"]])

        # Create dummy state if not found
        state_obs = np.zeros((len(actions), 7)) if len(actions) > 0 else np.zeros((len(image_obs), 7))

    else:
        raise ValueError("Unknown data structure in .pkl file.")

    return np.array(image_obs), np.array(state_obs), np.array(actions)


def save_arrays(output_dir, base_name, image_obs, state_obs, actions):
    """Save numpy arrays to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    if len(image_obs) > 0:
        paths["image_obs"] = output_dir / f"{base_name}_image_obs.npy"
        np.save(paths["image_obs"], image_obs)
    if len(state_obs) > 0:
        paths["state_obs"] = output_dir / f"{base_name}_state_obs.npy"
        np.save(paths["state_obs"], state_obs)
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
        episode = "ep_38" ######################## SELECT EPISODE HERE ########################
        args.input = f"/home/ferdinand/factr/process_data/raw_data/20251024/{episode}.pkl"
        args.output = f"/home/ferdinand/factr/process_data/converted_pkls_for_test/converted_{episode}/"
    
    # Ensure the output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = Path(args.input)
    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")

    data = load_pkl(pkl_path)
    image_obs, state_obs, actions = extract_factr_data(data)
    save_arrays(args.output, pkl_path.stem, image_obs, state_obs, actions)

    print("\n✅ Conversion complete.")
    print(f"Output saved in: {args.output}")


if __name__ == "__main__":
    main()
