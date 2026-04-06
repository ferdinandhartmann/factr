#!/usr/bin/env python3

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


ROBOT_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"

# Folder containing .pkl files
PKL_FOLDER = Path("~/activeinference/factr/process_data/data_to_process/fourgoals_2/data").expanduser()

# Configuration variables (replaces command-line arguments)
# Set these at the top of the file to change behavior instead of using CLI args
PRESET_PATH: Optional[Path] = None
OUTPUT: str = "goal_assignments.json"
TOP_K: int = 1
W_POS: float = 1.0
W_ROT: float = 1.0

# Whether to update data files in place
UPDATE_FILES: bool = True

# Optional manual goal assignment for files ending with "_stiff.pkl".
# If provided, goals are assigned in sorted filename order for stiff files.
# Example: ["goal_1", "goal_3", "goal_2"] or [1, 3, 2]
STIFF_GOALS_LIST: Optional[List[Any]] = [
    1,
    4,
    1,
    3,
    1,
    4,
    1,
    4,
    3,
    2,
    3,
    1,
    2,
    1,
    2,
    4,
    3,
    4,
    4,
    1,
    2,
    1,
    4,
    3,
    1,
    4,
    1,
    3,
    4,
    4,
    3,
    2,
    4,
    4,
    1,
    3,
    1,
    1,
    2,
    3,
    1,
    3,
    3,
    2,
    3,
    1,
    3,
    2,
    2,
    2,
    3,
    3,
    4,
    1,
    3,
    2,
    2,
    2,
    2,
    2,
]


def _rot6d_to_rotmat(rot6d: List[float]) -> np.ndarray:
    a1 = np.array(rot6d[:3], dtype=np.float64)
    a2 = np.array(rot6d[3:6], dtype=np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def _rotmat_to_rot6d(rot: np.ndarray) -> List[float]:
    return [
        float(rot[0, 0]),
        float(rot[1, 0]),
        float(rot[2, 0]),
        float(rot[0, 1]),
        float(rot[1, 1]),
        float(rot[2, 1]),
    ]


def _quaternion_from_rotation_matrix(rot: np.ndarray) -> List[float]:
    trace = np.trace(rot)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    else:
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s

    return [float(x), float(y), float(z), float(w)]


def _rotation_matrix_from_quaternion(q: List[float]) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _pose_to_9d(pose: List[float]) -> List[float]:
    if len(pose) == 9:
        return pose
    if len(pose) == 7:
        p = pose[:3]
        q = pose[3:7]
        rot = _rotation_matrix_from_quaternion(q)
        rot6d = _rotmat_to_rot6d(rot)
        return p + rot6d
    raise ValueError(f"Unsupported pose length: {len(pose)}")


def _load_pkl(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pkl(path: Path, data: Dict[str, Any]) -> None:
    with path.open("wb") as f:
        pickle.dump(data, f)


def _get_last_ee_pose(pkl_data: Dict[str, Any]) -> List[float]:
    if "data" not in pkl_data or ROBOT_STATE_TOPIC not in pkl_data["data"]:
        raise ValueError(f"Missing robot_state topic: {ROBOT_STATE_TOPIC}")
    entries = pkl_data["data"][ROBOT_STATE_TOPIC]
    for entry in reversed(entries):
        if isinstance(entry, dict) and "ee_pose" in entry:
            return entry["ee_pose"]
    raise ValueError("No ee_pose found in robot_state entries")


def _distance(pose_9d: np.ndarray, target_9d: np.ndarray, w_pos: float, w_rot: float) -> float:
    pos_diff = pose_9d[:3] - target_9d[:3]
    rot_diff = pose_9d[3:] - target_9d[3:]
    return float(w_pos * np.linalg.norm(pos_diff) + w_rot * np.linalg.norm(rot_diff))


def _format_goal_name(goal_name: Any) -> str:
    if isinstance(goal_name, int):
        return str(goal_name)
    if isinstance(goal_name, str) and goal_name.startswith("goal_"):
        suffix = goal_name.split("goal_", 1)[1]
        if suffix.isdigit():
            return suffix
    return str(goal_name)


def _is_stiff_file(path: Path) -> bool:
    return path.name.endswith("_stiff.pkl")


def _goal_from_stiff_list(
    path: Path, stiff_file_order: List[Path], stiff_goals_list: Optional[List[Any]]
) -> Optional[Any]:
    if not _is_stiff_file(path):
        return None
    if stiff_goals_list is None:
        return None
    try:
        idx = stiff_file_order.index(path)
    except ValueError as exc:
        raise ValueError(f"Stiff file not found in stiff ordering: {path.name}") from exc
    if idx >= len(stiff_goals_list):
        raise ValueError(
            f"STIFF_GOALS_LIST has {len(stiff_goals_list)} entries but needs at least {len(stiff_file_order)} "
            f"for stiff files. Missing assignment for {path.name}."
        )
    return stiff_goals_list[idx]


def _inject_goal_entries_after_steps(data_list: List[Any], goal_name: str) -> int:
    if not isinstance(data_list, list):
        raise ValueError("Expected a list of timestep entries")
    last_index_by_step: Dict[int, int] = {}
    last_timestamp_by_step: Dict[int, float] = {}
    for idx, entry in enumerate(data_list):
        if isinstance(entry, dict) and "step" in entry and "timestamp" in entry:
            step = entry["step"]
            ts = entry["timestamp"]
            if isinstance(step, int) and isinstance(ts, (int, float)):
                last_index_by_step[step] = idx
                last_timestamp_by_step[step] = float(ts)

    goal_value = _format_goal_name(goal_name)
    count = 0
    new_list: List[Any] = []
    for idx, entry in enumerate(data_list):
        new_list.append(entry)
        if isinstance(entry, dict) and "step" in entry:
            step = entry["step"]
            if step in last_index_by_step and last_index_by_step[step] == idx:
                ts = last_timestamp_by_step.get(step)
                if ts is not None:
                    new_list.append({"goal": goal_value})
                    count += 1

    data_list[:] = new_list
    return count


def _inject_goal_into_pkl(pkl_data: Dict[str, Any], goal_name: str) -> int:
    if "data" not in pkl_data:
        raise ValueError("Missing 'data' in pkl structure")
    data = pkl_data["data"]
    if isinstance(data, list):
        return _inject_goal_entries_after_steps(data, goal_name)
    if isinstance(data, dict):
        timestamps = pkl_data.get("timestamps") if isinstance(pkl_data.get("timestamps"), dict) else {}
        base_ts_list = None
        if isinstance(timestamps, dict):
            if ROBOT_STATE_TOPIC in timestamps and isinstance(timestamps[ROBOT_STATE_TOPIC], list):
                base_ts_list = timestamps[ROBOT_STATE_TOPIC]
            else:
                for v in timestamps.values():
                    if isinstance(v, list):
                        base_ts_list = v
                        break

        if base_ts_list is None:
            max_len = 0
            for entries in data.values():
                if isinstance(entries, list):
                    max_len = max(max_len, len(entries))
            base_ts_list = [None] * max_len

        goal_value = _format_goal_name(goal_name)
        goal_entries = []
        goal_timestamps = []
        for idx, ts in enumerate(base_ts_list):
            goal_entries.append({"goal": goal_value})
            goal_timestamps.append(ts)

        data["/goal"] = goal_entries
        if isinstance(timestamps, dict):
            timestamps["/goal"] = goal_timestamps
        return len(goal_entries)
    raise ValueError("Unsupported 'data' type in pkl structure")


def _load_structured_file(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        with path.open("r") as f:
            return json.load(f)
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read .yaml/.yml files")
        with path.open("r") as f:
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _save_structured_file(path: Path, data: Any) -> None:
    if path.suffix.lower() == ".json":
        with path.open("w") as f:
            json.dump(data, f, indent=2)
        return
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to write .yaml/.yml files")
        with path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        return
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _inject_goal_into_structured_list(data: Any, goal_name: str) -> int:
    if not isinstance(data, list):
        raise ValueError("Expected a list of timestep entries")
    return _inject_goal_entries_after_steps(data, goal_name)


def _load_presets(preset_path: Optional[Path]) -> List[Dict[str, Any]]:
    if preset_path is None:
        return [
            # fourgoals_1
            # {"index": "goal_1", "pose": [0.384, -0.26, 0.181, 0.998, 0.014, 0.065, 0.011, -0.999, 0.05]},
            # {"index": "goal_2", "pose": [0.656, -0.155, -0.023, 0.996, -0.054, 0.065, -0.057, -0.997, 0.045]},
            # {"index": "goal_3", "pose": [0.51, 0.269, -0.028, -0.015, 1.0, -0.027, 1.0, 0.015, 0.013]},
            # {"index": "goal_4", "pose": [0.466, 0.361, 0.48, 0.998, 0.056, 0.032, 0.034, -0.034, -0.999]},
            # fourgoals_2
            {"index": "goal_1", "pose": [0.341, 0.240, 0.606, 0.999, -0.007, 0.013, -0.007, -1.000, -0.010]},  # 33
            {"index": "goal_2", "pose": [0.524, 0.226, 0.381, 1.000, 0.013, 0.025, 0.013, -1.000, 0.004]},  # 27
            {"index": "goal_3", "pose": [0.591, -0.336, -0.038, 0.907, -0.421, 0.037, -0.421, -0.907, 0.006]},  # 31
            {"index": "goal_4", "pose": [0.439, -0.239, -0.043, 0.905, -0.425, 0.023, -0.426, -0.904, 0.028]},  # 29
        ]
    with preset_path.open("r") as f:
        presets = json.load(f)
    if not isinstance(presets, list):
        raise ValueError("Preset file must be a list of objects with 'index' and 'pose'")
    return presets


def main() -> None:
    # Use module-level configuration variables instead of CLI args
    preset_path = PRESET_PATH
    output_name = OUTPUT
    top_k = TOP_K
    w_pos = W_POS
    w_rot = W_ROT
    update_files = UPDATE_FILES
    stiff_goals_list = STIFF_GOALS_LIST

    folder = PKL_FOLDER
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    presets = _load_presets(preset_path)
    if len(presets) == 0:
        raise SystemExit("No presets provided")

    preset_9d = []
    for p in presets:
        if "index" not in p or "pose" not in p:
            raise SystemExit("Each preset must include 'index' and 'pose'")
        pose_9d = _pose_to_9d(p["pose"])
        preset_9d.append({"index": p["index"], "pose": np.array(pose_9d, dtype=np.float64)})

    all_pkl_paths = sorted(folder.glob("*.pkl"))
    stiff_file_order = [p for p in all_pkl_paths if _is_stiff_file(p)]

    results = []
    for pkl_path in all_pkl_paths:
        try:
            pkl_data = _load_pkl(pkl_path)
            ee_pose = _get_last_ee_pose(pkl_data)
            ee_pose_9d = np.array(_pose_to_9d(ee_pose), dtype=np.float64)

            distances = []
            for preset in preset_9d:
                dist = _distance(ee_pose_9d, preset["pose"], w_pos, w_rot)
                distances.append({"index": preset["index"], "distance": dist})

            distances.sort(key=lambda x: x["distance"])
            closest = distances[: max(1, top_k)]
            manual_goal = _goal_from_stiff_list(pkl_path, stiff_file_order, stiff_goals_list)
            if manual_goal is not None:
                goal_index = manual_goal
                closest = [{"index": manual_goal, "distance": None, "source": "manual_stiff_list"}]
            else:
                goal_index = closest[0]["index"]
            updated_counts = {}

            if update_files:
                pkl_count = _inject_goal_into_pkl(pkl_data, goal_index)
                _save_pkl(pkl_path, pkl_data)
                updated_counts["pkl_entries"] = pkl_count

                for ext in (".json", ".yaml", ".yml"):
                    structured_path = pkl_path.with_suffix(ext)
                    if structured_path.exists():
                        structured_data = _load_structured_file(structured_path)
                        structured_count = _inject_goal_into_structured_list(structured_data, goal_index)
                        _save_structured_file(structured_path, structured_data)
                        updated_counts[structured_path.suffix.lstrip(".") + "_entries"] = structured_count

            results.append(
                {
                    "file": pkl_path.name,
                    "last_ee_pose": ee_pose_9d.tolist(),
                    "closest_goal": closest,
                    "number_of_added_goal_entries": updated_counts,
                }
            )
        except Exception as exc:
            results.append({"file": pkl_path.name, "error": str(exc)})

    # Compute summary counts: primary assigned goal (closest[0]) per file
    summary_counts: Dict[str, int] = {}
    for r in results:
        if "closest_goal" in r and isinstance(r["closest_goal"], list) and len(r["closest_goal"]) > 0:
            index = r["closest_goal"][0].get("index")
            if index is not None:
                summary_counts[index] = summary_counts.get(index, 0) + 1

    # Append summary to results and save
    results.append({"summary": summary_counts})

    output_path = folder / output_name
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Print the summary to stdout
    print(f"Saved assignments to {output_path}")
    print("Summary counts per goal:")
    for name, cnt in summary_counts.items():
        print(f"  {name}: {cnt}")


if __name__ == "__main__":
    main()
