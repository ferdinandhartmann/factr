#!/usr/bin/env python3

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Folder containing .pkl files to update.
PKL_FOLDER = Path("~/activeinference/factr/process_data/data_to_process/boxlift_2_lead/data").expanduser()

DESIRED_RAW_TOPIC = "/cartesian_admittance_controller/pose_desired_raw"
COMMANDED_TOPIC = "/cartesian_admittance_controller/pose_command"
OUTPUT_TOPIC = "/admittance_offset"
POSE_KEY = "ee_pose_commanded"
OUTPUT_KEY = "ee_offset"

OVERWRITE_EXISTING_TOPIC: bool = True
OUTPUT: str = "admittance_offset_assignments.json"
TARGET_FILE_SUFFIXES: Tuple[str, ...] = (".pkl",)


def _load_pkl(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pkl(path: Path, data: Dict[str, Any]) -> None:
    with path.open("wb") as f:
        pickle.dump(data, f)


def _to_seconds(timestamps: List[Any]) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64)
    if ts.size and np.nanmax(ts) > 1e12:
        ts = ts * 1e-9
    return ts


def _topic_pose_series(pkl_data: Dict[str, Any], topic: str) -> Tuple[np.ndarray, np.ndarray]:
    if topic not in pkl_data.get("data", {}) or topic not in pkl_data.get("timestamps", {}):
        raise KeyError(f"Missing topic or timestamps for {topic}")

    values = []
    timestamps = []
    raw_timestamps = _to_seconds(pkl_data["timestamps"][topic])
    for message, timestamp in zip(pkl_data["data"][topic], raw_timestamps):
        if not isinstance(message, dict) or POSE_KEY not in message:
            continue
        value = np.asarray(message[POSE_KEY], dtype=np.float64)
        if value.shape != (9,):
            raise ValueError(f"Expected 9D {POSE_KEY} on {topic}, got shape {value.shape}")
        if not np.all(np.isfinite(value)):
            continue
        values.append(value)
        timestamps.append(float(timestamp))

    if not values:
        raise ValueError(f"No valid {POSE_KEY} samples found on {topic}")
    return np.asarray(timestamps, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _interpolate_series(
    target_timestamps: np.ndarray,
    source_timestamps: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_timestamps, source_timestamps, source_values[:, axis])
            for axis in range(source_values.shape[1])
        ]
    )


def _inject_admittance_offset(pkl_data: Dict[str, Any]) -> int:
    if "data" not in pkl_data or not isinstance(pkl_data["data"], dict):
        raise ValueError("Missing 'data' dict in pkl structure")
    if "timestamps" not in pkl_data or not isinstance(pkl_data["timestamps"], dict):
        raise ValueError("Missing 'timestamps' dict in pkl structure")

    if OUTPUT_TOPIC in pkl_data["data"] and not OVERWRITE_EXISTING_TOPIC:
        return 0

    desired_timestamps, desired_poses = _topic_pose_series(pkl_data, DESIRED_RAW_TOPIC)
    commanded_timestamps, commanded_poses = _topic_pose_series(pkl_data, COMMANDED_TOPIC)

    # Use desired_raw timestamps as the output grid, then subtract the commanded pose.
    commanded_on_desired_grid = _interpolate_series(desired_timestamps, commanded_timestamps, commanded_poses)
    offsets = desired_poses - commanded_on_desired_grid

    pkl_data["data"][OUTPUT_TOPIC] = [{OUTPUT_KEY: offset.astype(float).tolist()} for offset in offsets]
    pkl_data["timestamps"][OUTPUT_TOPIC] = pkl_data["timestamps"][DESIRED_RAW_TOPIC]
    return len(offsets)


def main() -> None:
    folder = PKL_FOLDER
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    all_pkl_paths = sorted(folder.glob("*.pkl"))
    target_pkl_paths = [p for p in all_pkl_paths if p.name.endswith(TARGET_FILE_SUFFIXES)]

    print(f"Found {len(target_pkl_paths)} target .pkl files in {folder}")
    if not target_pkl_paths:
        print("WARNING: No .pkl files matched TARGET_FILE_SUFFIXES. No files were changed.")
        return

    results = []
    successful_updates = 0
    for pkl_path in target_pkl_paths:
        print(f"Processing {pkl_path.name} ...")
        try:
            pkl_data = _load_pkl(pkl_path)
            count = _inject_admittance_offset(pkl_data)
            if count > 0 or OVERWRITE_EXISTING_TOPIC:
                _save_pkl(pkl_path, pkl_data)
                successful_updates += 1
                print(f"  success: wrote {count} offset entries")
            else:
                print("  skipped: topic already exists")

            results.append(
                {
                    "file": pkl_path.name,
                    "output_topic": OUTPUT_TOPIC,
                    "desired_raw_topic": DESIRED_RAW_TOPIC,
                    "commanded_topic": COMMANDED_TOPIC,
                    "output_key": OUTPUT_KEY,
                    "number_of_offset_entries": count,
                }
            )
        except Exception as exc:
            print(f"  error: {exc}")
            results.append({"file": pkl_path.name, "error": str(exc)})

    if successful_updates == 0:
        print("WARNING: No files were successfully updated. Assignments file was not saved.")
        return

    output_path = folder / OUTPUT
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved offset assignments to {output_path}")
    print(f"Successfully updated {successful_updates}/{len(target_pkl_paths)} files.")


if __name__ == "__main__":
    main()
