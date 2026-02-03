#!/usr/bin/env python3
import pickle
import numpy as np
import cv2
from PIL import Image
import imageio.v2 as imageio
from pathlib import Path


def load_pkl(pkl_path):
    """Load pickle file"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def extract_topic_images(pkl_data, topic_name):
    """Extract image arrays from a given topic"""
    if "data" not in pkl_data or topic_name not in pkl_data["data"]:
        print(f"❌ Topic {topic_name} not found.")
        return []

    frames = []
    for msg in pkl_data["data"][topic_name]:
        if not isinstance(msg, dict) or "data" not in msg:
            continue

        width, height, encoding = msg.get("width"), msg.get("height"), msg.get("encoding")
        raw = msg["data"]

        if encoding == "rgb8":
            img = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        elif encoding == "32FC1":
            depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
            # Normalize and apply colormap for visualization
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-8)
            depth = (depth * 255).astype(np.uint8)
            img = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            continue

        frames.append(img)
    return frames


def save_gif(frames, output_path, fps=25, scale=0.5, drop_ratio=2):
    """
    Save list of numpy images as a low-res GIF with frame skipping.
    drop_ratio=2 means keep every 2nd frame, etc.
    """
    if not frames:
        print(f"⚠️ No frames to save for {output_path}")
        return

    # Subsample frames to reduce total number but maintain real-time pace
    frames = frames[::drop_ratio]

    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        resized.append(Image.fromarray(frame))

    print(f"💾 Saving {len(resized)} frames to {output_path} (fps={fps}, drop_ratio={drop_ratio})")
    imageio.mimsave(output_path, resized, fps=fps)
    print(f"✅ Saved GIF: {output_path}")



def main():
    episode = "ep_0_no_movement"
    data_fps = 50
    gif_fps = 10
    scale_factor = 0.4
    pkl_path = Path(f"/home/ferdinand/factr_project/factr/process_data/data_to_process/20251107/data/{episode}.pkl")
    output_dir = Path("/home/ferdinand/factr_project/factr/process_data/data_to_process/20251107/visualizations/a_gifs")
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"📁 Loading {pkl_path}")
    data = load_pkl(pkl_path)

    # RGB
    rgb_frames = extract_topic_images(data, "/realsense/front/im")
    save_gif(rgb_frames, output_dir / f"{episode}_rgb_preview.gif", fps=gif_fps, scale=scale_factor, drop_ratio=int(data_fps/gif_fps))

    # Depth
    depth_frames = extract_topic_images(data, "/realsense/front/depth")
    save_gif(depth_frames, output_dir / f"{episode}_depth_preview.gif", fps=gif_fps, scale=scale_factor, drop_ratio=int(data_fps/gif_fps))

    print("🎉 Done! GIFs saved in:", output_dir)


if __name__ == "__main__":
    main()
