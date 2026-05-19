import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tensorflow_datasets as tfds


def _load_episode(builder, split, episode_index):
    dataset = builder.as_dataset(split=split)
    for idx, episode in enumerate(tfds.as_numpy(dataset)):
        if idx == episode_index:
            return episode
    raise IndexError(f"Episode index {episode_index} out of range")


def _stack_frames(step):
    cam_1 = step["observation"]["exterior_image_1_left"]
    cam_2 = step["observation"]["exterior_image_2_left"]
    cam_3 = step["observation"]["wrist_image_left"]
    return np.concatenate((cam_1, cam_2, cam_3), axis=1)


def main():
    parser = argparse.ArgumentParser(description="Render DROID camera triptych video.")
    parser.add_argument(
        "--input_path",
        default="/home/ferdinand/activeinference/factr/process_data/data_to_process/droid_100/1.0.0",
        help="TFDS builder directory with dataset_info.json",
    )
    parser.add_argument("--split", default="train", help="Dataset split to read")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to render")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument(
        "--output",
        default="/home/ferdinand/activeinference/factr/plots/droid_triptych.mp4",
        help="Output video path (.mp4 or .gif)",
    )
    args = parser.parse_args()

    builder = tfds.builder_from_directory(args.input_path)
    episode = _load_episode(builder, args.split, args.episode)

    frames = [_stack_frames(step) for step in episode["steps"]]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".gif":
        imageio.mimsave(output_path, frames, duration=1.0 / args.fps)
    else:
        imageio.mimwrite(output_path, frames, fps=args.fps, codec="libx264")

    print(f"Saved {len(frames)} frames to {output_path}")


if __name__ == "__main__":
    main()
