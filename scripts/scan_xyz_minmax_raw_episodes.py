#!/usr/bin/env python3
"""
Scan raw episode .pkl files and report global min/max of measured and commanded XYZ.

Raw episodes are expected to be pickled dicts with at least:
- raw["data"][topic] = list of messages (dicts or arrays)
- raw["timestamps"][topic] = list/array of timestamps (not used here)

This script is intentionally robust to slightly different message formats.
"""

import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

# ---------------------------------------------------------------------------
# User Config (edit these variables, then run this script directly)
# ---------------------------------------------------------------------------
RAW_EPISODE_DIR = Path(
    "/home/ferdinand/activeinference/factr/process_data/data_to_process/fourgoals_1/data"
)

# Topics/keys used by FACTR raw data
MEASURED_TOPIC = "/franka_robot_state_broadcaster/robot_state"
MEASURED_KEYS = ("ee_pose",)

COMMANDED_TOPIC = "/cartesian_impedance_controller/pose_command"
COMMANDED_KEYS = ("ee_pose_commanded", "ee_pose", "pose")

# Optional: save a JSON next to this script
SAVE_JSON = True
OUT_JSON_PATH = Path(__file__).with_suffix(".out.json")


def _iter_numbers(obj: Any):
    # Yield scalar numbers from nested list/tuple/array-ish structures.
    if obj is None:
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_numbers(v)
        return
    # numpy arrays / torch tensors often have tolist()
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        yield from _iter_numbers(tolist())
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_numbers(v)
        return
    # Fallback: try iterating
    try:
        for v in obj:
            yield from _iter_numbers(v)
    except TypeError:
        return


def _first3_floats(obj: Any) -> Optional[Tuple[float, float, float]]:
    vals = []
    for v in _iter_numbers(obj):
        if math.isfinite(v):
            vals.append(v)
            if len(vals) == 3:
                return (vals[0], vals[1], vals[2])
    return None


def _extract_xyz_from_msg(msg: Any, keys: Iterable[str]) -> Optional[Tuple[float, float, float]]:
    """
    Return xyz (3,) if possible, else None.

    Supports:
    - dict msg with known keys
    - dict msg with fallback "data"
    - list/tuple/np.ndarray msg with >= 3 entries
    """
    if isinstance(msg, dict):
        for key in keys:
            if key in msg and msg[key] is not None:
                xyz = _first3_floats(msg[key])
                if xyz is not None:
                    return xyz
        if "data" in msg and msg["data"] is not None:
            xyz = _first3_floats(msg["data"])
            if xyz is not None:
                return xyz
        return None

    return _first3_floats(msg)


def _update_minmax(
    cur_min: Tuple[float, float, float],
    cur_max: Tuple[float, float, float],
    xyz_list: Iterable[Tuple[float, float, float]],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], int]:
    count = 0
    min_x, min_y, min_z = cur_min
    max_x, max_y, max_z = cur_max
    for x, y, z in xyz_list:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue
        count += 1
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)
    return (min_x, min_y, min_z), (max_x, max_y, max_z), count


def _format_vec(v: Tuple[float, float, float]) -> str:
    return f"[{v[0]: .5f}, {v[1]: .5f}, {v[2]: .5f}]"


def main() -> None:
    if not RAW_EPISODE_DIR.exists():
        raise FileNotFoundError(f"RAW_EPISODE_DIR not found: {RAW_EPISODE_DIR}")

    episode_files = sorted(RAW_EPISODE_DIR.glob("*.pkl"))
    if len(episode_files) == 0:
        raise RuntimeError(f"No .pkl files found in: {RAW_EPISODE_DIR}")

    meas_min: Tuple[float, float, float] = (math.inf, math.inf, math.inf)
    meas_max: Tuple[float, float, float] = (-math.inf, -math.inf, -math.inf)
    cmd_min: Tuple[float, float, float] = (math.inf, math.inf, math.inf)
    cmd_max: Tuple[float, float, float] = (-math.inf, -math.inf, -math.inf)

    meas_count = 0
    cmd_count = 0
    skipped = []

    for ep_path in episode_files:
        try:
            with open(ep_path, "rb") as f:
                raw = pickle.load(f)
        except Exception as e:
            skipped.append({"file": ep_path.name, "reason": f"pickle_load_failed: {e}"})
            continue

        data = raw.get("data", {})
        if MEASURED_TOPIC not in data or COMMANDED_TOPIC not in data:
            skipped.append({"file": ep_path.name, "reason": "missing_topic"})
            continue

        meas_msgs = data.get(MEASURED_TOPIC, [])
        cmd_msgs = data.get(COMMANDED_TOPIC, [])

        meas_xyz_list = []
        for msg in meas_msgs:
            xyz = _extract_xyz_from_msg(msg, MEASURED_KEYS)
            if xyz is not None:
                meas_xyz_list.append(xyz)
        meas_min, meas_max, added = _update_minmax(meas_min, meas_max, meas_xyz_list)
        meas_count += int(added)

        cmd_xyz_list = []
        for msg in cmd_msgs:
            xyz = _extract_xyz_from_msg(msg, COMMANDED_KEYS)
            if xyz is not None:
                cmd_xyz_list.append(xyz)
        cmd_min, cmd_max, added = _update_minmax(cmd_min, cmd_max, cmd_xyz_list)
        cmd_count += int(added)

    print(f"Scanned episodes: {len(episode_files)} in {RAW_EPISODE_DIR}")
    print(f"Measured XYZ samples (finite): {meas_count}")
    print(f"Commanded XYZ samples (finite): {cmd_count}")

    if not all(math.isfinite(v) for v in meas_min + meas_max):
        print("Measured XYZ: no finite samples found.")
    else:
        print(f"Measured XYZ min: {_format_vec(meas_min)}")
        print(f"Measured XYZ max: {_format_vec(meas_max)}")

    if not all(math.isfinite(v) for v in cmd_min + cmd_max):
        print("Commanded XYZ: no finite samples found.")
    else:
        print(f"Commanded XYZ min: {_format_vec(cmd_min)}")
        print(f"Commanded XYZ max: {_format_vec(cmd_max)}")

    if len(skipped) > 0:
        print(f"Skipped: {len(skipped)} episodes (missing topics / load errors).")

    if SAVE_JSON:
        payload = {
            "raw_episode_dir": str(RAW_EPISODE_DIR),
            "num_files": int(len(episode_files)),
            "measured_topic": MEASURED_TOPIC,
            "commanded_topic": COMMANDED_TOPIC,
            "measured_xyz_finite_count": int(meas_count),
            "commanded_xyz_finite_count": int(cmd_count),
            "measured_xyz_min": list(meas_min) if all(math.isfinite(v) for v in meas_min) else None,
            "measured_xyz_max": list(meas_max) if all(math.isfinite(v) for v in meas_max) else None,
            "commanded_xyz_min": list(cmd_min) if all(math.isfinite(v) for v in cmd_min) else None,
            "commanded_xyz_max": list(cmd_max) if all(math.isfinite(v) for v in cmd_max) else None,
            "skipped": skipped,
        }
        with open(OUT_JSON_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {OUT_JSON_PATH}")


if __name__ == "__main__":
    main()
