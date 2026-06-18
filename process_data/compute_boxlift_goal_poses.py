#!/usr/bin/env python3
"""
Compute average measured EE goal poses for the boxlift lead/follow data.

For each episode, this reads the last 1 second of
/franka_robot_state_broadcaster/robot_state ee_pose samples, averages that
segment, then averages those episode means per /goal label.
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_DATA_DIRS = [
    Path("/home/ferdinand/activeinference/factr/process_data/data_to_process/boxlift_1_lead/data"),
    Path("/home/ferdinand/activeinference/factr/process_data/data_to_process/boxlift_1_follow/data"),
]
DEFAULT_OUTPUT = Path("/home/ferdinand/activeinference/factr/process_data/data_to_process/boxlift_1_goal_poses.json")
DEFAULT_LINES_OUTPUT = Path(
    "/home/ferdinand/activeinference/factr/process_data/data_to_process/boxlift_1_goal_poses.txt"
)

GOAL_TOPIC = "/goal"
POSE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
POSE_KEY = "ee_pose"
LAST_SECONDS = 1.0
EXPECTED_GOALS = range(1, 10)


def _natural_key(path: Path) -> Tuple[Any, ...]:
    stem = path.stem
    parts: List[Any] = []
    for chunk in stem.replace("-", "_").split("_"):
        parts.append(int(chunk) if chunk.isdigit() else chunk)
    return tuple(parts)


def _load_pkl(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _goal_from_episode(pkl_data: Dict[str, Any], path: Path) -> int:
    entries = pkl_data.get("data", {}).get(GOAL_TOPIC, [])
    for entry in reversed(entries):
        value = entry.get("goal") if isinstance(entry, dict) else entry
        if value is None:
            continue
        return int(value)
    raise ValueError(f"{path.name}: no usable {GOAL_TOPIC} entries found")


def _pose_array_from_episode(pkl_data: Dict[str, Any], path: Path) -> np.ndarray:
    entries = pkl_data.get("data", {}).get(POSE_TOPIC, [])
    poses = []
    for entry in entries:
        if not isinstance(entry, dict) or POSE_KEY not in entry:
            continue
        pose = np.asarray(entry[POSE_KEY], dtype=np.float64).reshape(-1)
        if pose.size != 9:
            raise ValueError(f"{path.name}: expected 9D {POSE_KEY}, got {pose.size}D")
        poses.append(pose)
    if not poses:
        raise ValueError(f"{path.name}: no usable {POSE_TOPIC} entries found")
    return np.stack(poses, axis=0)


def _timestamps_seconds(pkl_data: Dict[str, Any], path: Path, count: int) -> np.ndarray:
    ts_entries = pkl_data.get("timestamps", {}).get(POSE_TOPIC, [])
    if len(ts_entries) != count:
        raise ValueError(f"{path.name}: pose/timestamp length mismatch ({count} poses, {len(ts_entries)} timestamps)")
    timestamps = np.asarray(ts_entries, dtype=np.float64)
    return (timestamps - timestamps[0]) * 1e-9


def _last_second_pose_mean(pkl_data: Dict[str, Any], path: Path) -> np.ndarray:
    poses = _pose_array_from_episode(pkl_data, path)
    timestamps = _timestamps_seconds(pkl_data, path, len(poses))

    # Keep exactly the final time window so episodes with different lengths align.
    window_start = timestamps[-1] - LAST_SECONDS
    mask = timestamps >= window_start
    if not np.any(mask):
        mask[-1] = True
    return np.mean(poses[mask], axis=0)


def _iter_pkl_paths(data_dirs: Iterable[Path]) -> Iterable[Path]:
    for data_dir in data_dirs:
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        yield from sorted(data_dir.glob("*.pkl"), key=_natural_key)


def compute_goal_poses(data_dirs: Iterable[Path]) -> Dict[str, Any]:
    per_goal_episode_means: Dict[int, List[np.ndarray]] = defaultdict(list)

    for path in _iter_pkl_paths(data_dirs):
        pkl_data = _load_pkl(path)
        goal = _goal_from_episode(pkl_data, path)
        per_goal_episode_means[goal].append(_last_second_pose_mean(pkl_data, path))

    missing_goals = [goal for goal in EXPECTED_GOALS if goal not in per_goal_episode_means]
    if missing_goals:
        raise ValueError(f"Missing expected goals: {missing_goals}")

    goals = {}
    for goal in EXPECTED_GOALS:
        episode_means = np.stack(per_goal_episode_means[goal], axis=0)
        averaged_pose = np.mean(episode_means, axis=0)
        goals[str(goal)] = {
            "ee_pose": [round(float(v), 3) for v in averaged_pose],
            "num_files": int(episode_means.shape[0]),
        }

    return {
        "source_dirs": [str(path) for path in data_dirs],
        "goal_topic": GOAL_TOPIC,
        "pose_topic": POSE_TOPIC,
        "pose_key": POSE_KEY,
        "last_seconds": LAST_SECONDS,
        "goals": goals,
    }


def write_pose_lines(result: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for goal in EXPECTED_GOALS:
            pose = result["goals"][str(goal)]["ee_pose"]
            pose_text = ", ".join(f"{value:.3f}" for value in pose)
            f.write(f"[{pose_text}] # Boxlift goal {goal}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write the computed goal poses.")
    parser.add_argument(
        "--lines-output",
        type=Path,
        default=DEFAULT_LINES_OUTPUT,
        help="Where to write one pose per line for easy copying into configs.",
    )
    parser.add_argument(
        "data_dirs",
        nargs="*",
        type=Path,
        default=DEFAULT_DATA_DIRS,
        help="Folders containing episode .pkl files. Defaults to boxlift_1 lead and follow data folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_goal_poses(args.data_dirs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    write_pose_lines(result, args.lines_output)

    print(f"Wrote averaged goal poses to {args.output}")
    print(f"Wrote one-line goal poses to {args.lines_output}")


if __name__ == "__main__":
    main()
