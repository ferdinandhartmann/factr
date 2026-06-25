#!/usr/bin/env python3

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Folder containing .pkl files to update.
PKL_FOLDER = Path("~/activeinference/factr/process_data/data_to_process/boxlift_1_lead/data").expanduser()

# Topic and fixed payload to add to every episode.
OUTPUT_TOPIC = "/cartesian_admittance_controller/pose_desired_raw"
POSE_KEY = "ee_pose_commanded"
FIXED_POSE_9D: List[float] = [0.0] * 9
POSE_VALUE_MODE: str = "computed_admittance"  # "zero", "pose_command", or "computed_admittance"

ADMITTANCE_MASS = np.array([15.0, 12.0, 12.0, 1.0, 1.0, 0.5])
ADMITTANCE_STIFFNESS = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 20.0])
ADMITTANCE_DAMPING = np.array([80.0, 80.0, 80.0, 25.0, 25.0, 5.0])
EXTERNAL_WRENCH_OFFSET = np.array([14.0, 0.0, -1.2, 0.0, 0.0, 0.0])
EXTERNAL_WRENCH_SCALE = np.ones(6)
CARTESIAN_POSITION_LIMITS_ENABLED = False
CARTESIAN_POSITION_MIN = np.array([0.0, -0.5, 0.028])
CARTESIAN_POSITION_MAX = np.array([0.75, 0.3, 0.70])
SIMULATION_HZ = 1000.0
SIMULATION_DT = 1.0 / SIMULATION_HZ

# Set this to False if you only want to fill files that do not already have the topic.
OVERWRITE_EXISTING_TOPIC: bool = True

OUTPUT: str = "pose_desired_raw_assignments.json"
TARGET_FILE_SUFFIXES: Tuple[str, ...] = (".pkl",)

ROBOT_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
POSE_COMMAND_TOPICS = (
    "/cartesian_admittance_controller/pose_command",
    "/cartesian_impedance_controller/pose_command",
)
EXTERNAL_WRENCH_TOPIC = "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame"
ADMITTANCE_OFFSET_TOPIC = "/cartesian_admittance_controller/admittance_offset"


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


def _rot6d_to_rotmat(rot6d: Any) -> np.ndarray:
    a1 = np.asarray(rot6d[:3], dtype=np.float64)
    a2 = np.asarray(rot6d[3:6], dtype=np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-9)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def _rotmat_to_rot6d(rot: np.ndarray) -> np.ndarray:
    return np.array(
        [rot[0, 0], rot[1, 0], rot[2, 0], rot[0, 1], rot[1, 1], rot[2, 1]],
        dtype=np.float64,
    )


def _rotation_matrix_from_quaternion(q: Any) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64)
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


def _pose_to_pose9(pose: Any) -> np.ndarray:
    pose_arr = np.asarray(pose, dtype=np.float64)
    if pose_arr.shape == (9,):
        return pose_arr
    if pose_arr.shape == (7,):
        rot6d = _rotmat_to_rot6d(_rotation_matrix_from_quaternion(pose_arr[3:7]))
        return np.concatenate((pose_arr[:3], rot6d))
    raise ValueError(f"Unsupported pose length: {pose_arr.size}")


def _topic_series(
    recording: Dict[str, Any],
    topic: Any,
    value_key: str,
    converter: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    topics = (topic,) if isinstance(topic, str) else tuple(topic)
    selected_topic = next(
        (
            candidate
            for candidate in topics
            if candidate in recording.get("data", {}) and candidate in recording.get("timestamps", {})
        ),
        None,
    )
    if selected_topic is None:
        raise KeyError(f"Required topic is missing; checked: {', '.join(topics)}")

    values = []
    timestamps = []
    raw_timestamps = _to_seconds(recording["timestamps"][selected_topic])
    for message, timestamp in zip(recording["data"][selected_topic], raw_timestamps):
        if not isinstance(message, dict) or value_key not in message:
            continue
        value = converter(message[value_key]) if converter else np.asarray(message[value_key], dtype=np.float64)
        if value is None or not np.all(np.isfinite(value)):
            continue
        values.append(value)
        timestamps.append(float(timestamp))

    if not values:
        raise ValueError(f"No valid '{value_key}' samples found on {selected_topic}")
    return np.asarray(timestamps, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _rotation_vector_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-9:
        return np.eye(3)
    axis = rotation_vector / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    return np.concatenate((rotation[:, 0], rotation[:, 1]))


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


def _simulate_admittance(
    measured_timestamps: np.ndarray,
    measured_poses: np.ndarray,
    wrench_timestamps: np.ndarray,
    raw_wrenches: np.ndarray,
    admittance_offset_timestamps: Optional[np.ndarray] = None,
    admittance_offsets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    mass = ADMITTANCE_MASS
    wrench_scale = EXTERNAL_WRENCH_SCALE
    position_min = CARTESIAN_POSITION_MIN
    position_max = CARTESIAN_POSITION_MAX
    limits_enabled = CARTESIAN_POSITION_LIMITS_ENABLED
    if mass.shape != (6,) or np.any(mass <= 0.0):
        raise ValueError("admittance_mass must contain six positive values")
    if wrench_scale.shape != (6,):
        raise ValueError("external_wrench_scale must contain six values")

    start_time = max(measured_timestamps[0], wrench_timestamps[0])
    end_time = min(measured_timestamps[-1], wrench_timestamps[-1])
    if end_time <= start_time:
        raise ValueError("Measured pose and wrench topics do not overlap in time")

    sample_count = int(np.floor((end_time - start_time) * SIMULATION_HZ)) + 1
    simulation_timestamps = start_time + np.arange(sample_count, dtype=np.float64) * SIMULATION_DT
    measured_poses_1khz = _interpolate_series(simulation_timestamps, measured_timestamps, measured_poses)
    wrenches_1khz = _interpolate_series(simulation_timestamps, wrench_timestamps, raw_wrenches)

    initial_rotation = _rot6d_to_rotmat(measured_poses_1khz[0, 3:9])
    equilibrium_roll = np.arctan2(initial_rotation[2, 1], initial_rotation[2, 2])
    equilibrium_pitch = np.arctan2(
        -initial_rotation[2, 0], np.hypot(initial_rotation[2, 1], initial_rotation[2, 2])
    )

    offset = np.zeros(6, dtype=np.float64)
    velocity = np.zeros(6, dtype=np.float64)
    if admittance_offset_timestamps is not None and admittance_offsets is not None:
        fit_count = min(10, len(simulation_timestamps))
        fit_timestamps = simulation_timestamps[:fit_count]
        inferred_offsets = _interpolate_series(
            fit_timestamps, admittance_offset_timestamps, admittance_offsets[:, :6]
        )
        offset = inferred_offsets[0].copy()
        if fit_count >= 2:
            fit_time = fit_timestamps - fit_timestamps[0]
            velocity = np.array(
                [np.polyfit(fit_time, inferred_offsets[:, axis], 1)[0] for axis in range(6)],
                dtype=np.float64,
            )

    simulated = np.empty_like(measured_poses_1khz)

    initial_yaw = np.arctan2(initial_rotation[1, 0], initial_rotation[0, 0])
    initial_equilibrium_rotation = (
        _rotation_vector_to_matrix(np.array([0.0, 0.0, initial_yaw]))
        @ _rotation_vector_to_matrix(np.array([0.0, equilibrium_pitch, 0.0]))
        @ _rotation_vector_to_matrix(np.array([equilibrium_roll, 0.0, 0.0]))
    )
    simulated[0, :3] = measured_poses_1khz[0, :3] + offset[:3]
    simulated[0, 3:9] = _matrix_to_rot6d(_rotation_vector_to_matrix(offset[3:]) @ initial_equilibrium_rotation)

    for index in range(1, len(simulation_timestamps)):
        current_rotation = _rot6d_to_rotmat(measured_poses_1khz[index, 3:9])

        corrected_stiffness = wrench_scale * (-wrenches_1khz[index] + EXTERNAL_WRENCH_OFFSET)
        corrected_base = np.concatenate(
            (current_rotation @ corrected_stiffness[:3], current_rotation @ corrected_stiffness[3:])
        )

        acceleration = (corrected_base - ADMITTANCE_DAMPING * velocity - ADMITTANCE_STIFFNESS * offset) / mass
        velocity += acceleration * SIMULATION_DT
        offset += velocity * SIMULATION_DT

        equilibrium_position = measured_poses_1khz[index, :3]
        desired_position = equilibrium_position + offset[:3]
        if limits_enabled:
            for axis in range(3):
                if desired_position[axis] < position_min[axis]:
                    if equilibrium_position[axis] >= position_min[axis]:
                        desired_position[axis] = equilibrium_position[axis]
                        offset[axis] = 0.0
                        velocity[axis] = 0.0
                    else:
                        desired_position[axis] = position_min[axis]
                elif desired_position[axis] > position_max[axis]:
                    if equilibrium_position[axis] <= position_max[axis]:
                        desired_position[axis] = equilibrium_position[axis]
                        offset[axis] = 0.0
                        velocity[axis] = 0.0
                    else:
                        desired_position[axis] = position_max[axis]

        current_yaw = np.arctan2(current_rotation[1, 0], current_rotation[0, 0])
        equilibrium_rotation = (
            _rotation_vector_to_matrix(np.array([0.0, 0.0, current_yaw]))
            @ _rotation_vector_to_matrix(np.array([0.0, equilibrium_pitch, 0.0]))
            @ _rotation_vector_to_matrix(np.array([equilibrium_roll, 0.0, 0.0]))
        )
        desired_rotation = _rotation_vector_to_matrix(offset[3:]) @ equilibrium_rotation
        simulated[index, :3] = desired_position
        simulated[index, 3:9] = _matrix_to_rot6d(desired_rotation)

    return simulation_timestamps, simulated


def _compute_admittance_pose_entries(
    pkl_data: Dict[str, Any],
    base_ts_list: List[Optional[int]],
) -> List[Dict[str, List[float]]]:
    measured_timestamps, measured_poses = _topic_series(pkl_data, ROBOT_STATE_TOPIC, "ee_pose", _pose_to_pose9)
    command_timestamps, _ = _topic_series(pkl_data, POSE_COMMAND_TOPICS, "ee_pose_commanded", _pose_to_pose9)
    wrench_timestamps, raw_wrenches = _topic_series(pkl_data, EXTERNAL_WRENCH_TOPIC, "external_wrench")

    admittance_offset_timestamps = None
    admittance_offsets = None
    if ADMITTANCE_OFFSET_TOPIC in pkl_data.get("data", {}) and ADMITTANCE_OFFSET_TOPIC in pkl_data.get("timestamps", {}):
        admittance_offset_timestamps, admittance_offsets = _topic_series(pkl_data, ADMITTANCE_OFFSET_TOPIC, "data")

    # Match the pasted script's global time-zero normalization before simulation.
    global_time_zero = min(measured_timestamps[0], command_timestamps[0], wrench_timestamps[0])
    measured_timestamps = measured_timestamps - global_time_zero
    wrench_timestamps = wrench_timestamps - global_time_zero
    if admittance_offset_timestamps is not None:
        admittance_offset_timestamps = admittance_offset_timestamps - global_time_zero

    simulation_timestamps, simulated_poses = _simulate_admittance(
        measured_timestamps,
        measured_poses,
        wrench_timestamps,
        raw_wrenches,
        admittance_offset_timestamps=admittance_offset_timestamps,
        admittance_offsets=admittance_offsets,
    )
    target_timestamps = _to_seconds(base_ts_list) - global_time_zero
    target_poses = _interpolate_series(target_timestamps, simulation_timestamps, simulated_poses)
    return [{POSE_KEY: pose.astype(float).tolist()} for pose in target_poses]


def _copy_pose_command_entries(
    pkl_data: Dict[str, Any],
    base_ts_list: List[Optional[int]],
) -> List[Dict[str, List[float]]]:
    command_timestamps, command_poses = _topic_series(
        pkl_data,
        "/cartesian_admittance_controller/pose_command",
        "ee_pose_commanded",
        _pose_to_pose9,
    )
    target_timestamps = _to_seconds(base_ts_list)
    target_poses = _interpolate_series(target_timestamps, command_timestamps, command_poses)
    return [{POSE_KEY: pose.astype(float).tolist()} for pose in target_poses]


def _build_base_timestamps(pkl_data: Dict[str, Any]) -> List[Optional[int]]:
    timestamps = pkl_data.get("timestamps") if isinstance(pkl_data.get("timestamps"), dict) else {}

    # Prefer robot-state timing because low-dim processing is aligned to robot observations.
    robot_ts = timestamps.get(ROBOT_STATE_TOPIC) if isinstance(timestamps, dict) else None
    if isinstance(robot_ts, list) and robot_ts:
        return list(robot_ts)

    if isinstance(timestamps, dict) and timestamps:
        best_ts = None
        best_len = -1
        for ts_list in timestamps.values():
            if isinstance(ts_list, list) and len(ts_list) > best_len:
                best_ts = ts_list
                best_len = len(ts_list)
        if best_ts is not None:
            return list(best_ts)

    data = pkl_data.get("data") if isinstance(pkl_data.get("data"), dict) else {}
    max_len = 0
    for entries in data.values():
        if isinstance(entries, list):
            max_len = max(max_len, len(entries))
    return [None] * max_len


def _inject_pose_desired_raw(pkl_data: Dict[str, Any]) -> int:
    if "data" not in pkl_data or not isinstance(pkl_data["data"], dict):
        raise ValueError("Missing 'data' dict in pkl structure")

    if OUTPUT_TOPIC in pkl_data["data"] and not OVERWRITE_EXISTING_TOPIC:
        return 0

    base_ts_list = _build_base_timestamps(pkl_data)
    if POSE_VALUE_MODE == "zero":
        pose_entries = [{POSE_KEY: list(FIXED_POSE_9D)} for _ in base_ts_list]
    elif POSE_VALUE_MODE == "pose_command":
        pose_entries = _copy_pose_command_entries(pkl_data, base_ts_list)
    elif POSE_VALUE_MODE == "computed_admittance":
        pose_entries = _compute_admittance_pose_entries(pkl_data, base_ts_list)
    else:
        raise ValueError(f"Unsupported POSE_VALUE_MODE: {POSE_VALUE_MODE!r}")

    # Keep topic data and topic timestamps the same length.
    pkl_data["data"][OUTPUT_TOPIC] = pose_entries
    timestamps = pkl_data.get("timestamps")
    if not isinstance(timestamps, dict):
        timestamps = {}
        pkl_data["timestamps"] = timestamps
    timestamps[OUTPUT_TOPIC] = base_ts_list

    return len(pose_entries)


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
            count = _inject_pose_desired_raw(pkl_data)
            if count > 0 or OVERWRITE_EXISTING_TOPIC:
                _save_pkl(pkl_path, pkl_data)
                successful_updates += 1
                print(f"  success: wrote {count} pose entries")
            else:
                print("  skipped: topic already exists")

            results.append(
                {
                    "file": pkl_path.name,
                    "topic": OUTPUT_TOPIC,
                    "pose_key": POSE_KEY,
                    "pose_value_mode": POSE_VALUE_MODE,
                    "pose_value": list(FIXED_POSE_9D) if POSE_VALUE_MODE == "zero" else POSE_VALUE_MODE,
                    "number_of_pose_entries": count,
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

    print(f"Saved pose assignments to {output_path}")
    print(f"Successfully updated {successful_updates}/{len(target_pkl_paths)} files.")


if __name__ == "__main__":
    main()
