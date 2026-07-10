#!/usr/bin/env python3

import json
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

###### Assigns goal based on list or last pose to closest preset goal.

ROBOT_STATE_TOPIC = "/franka_robot_state_broadcaster/robot_state"
ARRANGEMENT_TOPIC = "/arrangement"
MODE_TOPIC = "/mode"

PKL_FOLDER = Path("~/activeinference/factr/process_data/data_to_process/boxlift_4_follow/data").expanduser()
XLSX_PATH = Path(
    "~/activeinference/factr/process_data/data_to_process/boxlift_1_follow/boxlift_1_follow_goalconfig.xlsx"
).expanduser()

# Choose how each episode goal is assigned:
# - "last_pose": use the final ee_pose and choose the closest preset goal.
# - "list": use GOALS_LIST in sorted .pkl filename order.
# - "xlsx": use XLSX_PATH with Episode Index, Goal, and Arr. ID columns.
GOAL_ASSIGNMENT_MODE: str = "xlsx"

PRESET_PATH: Optional[Path] = None
OUTPUT: str = "goal_assignments.json"
TOP_K: int = 1
W_POS: float = 1.0
W_ROT: float = 1.0


UPDATE_FILES: bool = True

ADD_MODE_TOPIC: bool = True  # /mode topic and 0 for following or 1 for leading, based on folder name.

# Optional manual goal assignment in sorted .pkl filename order.
# Example: ["goal_1", "goal_3", "goal_2"] or [1, 3, 2]

# fourgoals_2_stiff
# GOALS_LIST: Optional[List[Any]] = [1,4,1,3,1,4,1,4,3,2,3,1,2,1,2,4,3,4,4,1,2,1,4,3,1,4,1,3,4,4,3,2,4,4,1,3,1,1,2,3,1,3,3,2,3,1,3,2,2,2,3,3,4,1,3,2,2,2,2,2]

# fourgoals_2_soft
# GOALS_LIST: Optional[List[Any]] = [3,1,3,1,2,2,4,4,3,3,1,2,2,3,2,1,1,1,4,4,3,1,1,4,3,4,4,4,1,3,3,1,1,3,2,4,1,1,2,4,2,1,1,4,3,4,3,3,3,4,2,1,4,3,4,2,2,2,4,1]

# fourgoals_3_stiff
# GOALS_LIST: Optional[List[Any]] = [3,1,4,1,4,2,2,4,4,4,3,3,3,3,1,2,4,1,1,4,1,2,3,3,2,1,1,4,2,1,4,2,4,1,2,1,3,2,2,2,3,3,3,1,4]

# fourgoals_4_stiff
# GOALS_LIST: Optional[List[int]] = [3,1,1,1,4,3,3,2,3,2,4,2,2,1,3,4,1,1,2,2,1,4,4,4,3,1,3,4,3,1,1,2,4,3,3,3,2,1,3,4,2,4,3,2,1,2,2,4,4,1,2,1,2,3,1,3,4,2,4,4]
GOALS_LIST: Optional[List[Any]] = None


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


def _goal_from_ordered_list(path: Path, file_order: List[Path], goals_list: Optional[List[Any]]) -> Optional[Any]:
    if goals_list is None:
        return None
    try:
        idx = file_order.index(path)
    except ValueError as exc:
        raise ValueError(f"File not found in sorted pkl ordering: {path.name}") from exc
    if idx >= len(goals_list):
        raise ValueError(
            f"GOALS_LIST has {len(goals_list)} entries but needs at least {len(file_order)} "
            f"for pkl files. Missing assignment for {path.name}."
        )
    return goals_list[idx]


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"Missing integer value for {field_name}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text:
            as_float = float(text)
            if as_float.is_integer():
                return int(as_float)
    raise ValueError(f"Expected integer value for {field_name}, got {value!r}")


def _xlsx_col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    if not letters:
        raise ValueError(f"Invalid XLSX cell reference: {cell_ref!r}")
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - ord("A") + 1
    return idx - 1


def _read_xlsx_rows_with_openpyxl(path: Path) -> List[List[Any]]:
    try:
        import openpyxl  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("openpyxl is not installed") from exc

    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    worksheet = workbook[workbook.sheetnames[0]]
    return [list(row) for row in worksheet.iter_rows(values_only=True)]


def _read_xlsx_rows_with_zip_xml(path: Path) -> List[List[Any]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for shared_item in root.findall("a:si", ns):
                parts = [node.text or "" for node in shared_item.findall(".//a:t", ns)]
                shared_strings.append("".join(parts))

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: List[List[Any]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", ns):
            values: List[Any] = []
            for cell in row.findall("a:c", ns):
                cell_ref = cell.attrib.get("r", "A1")
                col_idx = _xlsx_col_index(cell_ref)
                while len(values) < col_idx:
                    values.append(None)

                value_node = cell.find("a:v", ns)
                value: Any = None if value_node is None else value_node.text
                cell_type = cell.attrib.get("t")
                if cell_type == "s" and value is not None:
                    value = shared_strings[int(value)]
                elif cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.findall(".//a:t", ns))
                values.append(value)
            rows.append(values)
    return rows


def _read_xlsx_rows(path: Path) -> List[List[Any]]:
    if not path.exists():
        raise FileNotFoundError(f"XLSX file not found: {path}")
    try:
        return _read_xlsx_rows_with_openpyxl(path)
    except RuntimeError:
        return _read_xlsx_rows_with_zip_xml(path)


def _find_required_columns(rows: List[List[Any]], required: List[str]) -> Tuple[int, Dict[str, int]]:
    for row_idx, row in enumerate(rows):
        normalized = {str(value).strip(): idx for idx, value in enumerate(row) if value is not None}
        if all(name in normalized for name in required):
            return row_idx, {name: normalized[name] for name in required}
    raise ValueError(f"Could not find XLSX header row with columns: {required}")


def _load_xlsx_goal_assignments(path: Path) -> Tuple[Dict[int, Dict[str, int]], Dict[int, int]]:
    rows = _read_xlsx_rows(path)
    required_columns = ["Episode Index", "Goal", "Arr. ID"]
    header_idx, columns = _find_required_columns(rows, required_columns)

    records_by_episode: Dict[int, Dict[str, int]] = {}
    raw_goals = set()
    for row in rows[header_idx + 1 :]:
        if len(row) <= max(columns.values()) or row[columns["Episode Index"]] in (None, ""):
            continue

        episode_index = _coerce_int(row[columns["Episode Index"]], "Episode Index")
        raw_goal = _coerce_int(row[columns["Goal"]], "Goal")
        arrangement_id = _coerce_int(row[columns["Arr. ID"]], "Arr. ID")
        if episode_index in records_by_episode:
            raise ValueError(f"Duplicate Episode Index in XLSX: {episode_index}")

        records_by_episode[episode_index] = {
            "episode_index": episode_index,
            "raw_goal": raw_goal,
            "arrangement_id": arrangement_id,
        }
        raw_goals.add(raw_goal)

    if not records_by_episode:
        raise ValueError(f"No goal assignments found in XLSX: {path}")

    goal_remap = {raw_goal: idx + 1 for idx, raw_goal in enumerate(sorted(raw_goals))}
    for record in records_by_episode.values():
        record["dense_goal"] = goal_remap[record["raw_goal"]]

    return records_by_episode, goal_remap


def _episode_index_from_path(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"Could not infer episode index from filename: {path.name}")
    return int(match.group(1))


def _xlsx_record_for_path(path: Path, assignments_by_episode: Dict[int, Dict[str, int]]) -> Dict[str, int]:
    episode_index = _episode_index_from_path(path)
    if episode_index not in assignments_by_episode:
        raise ValueError(f"No XLSX assignment for {path.name} with Episode Index {episode_index}")
    return assignments_by_episode[episode_index]


def _infer_mode_from_folder(folder: Path) -> int:
    # The dataset folder name carries the demonstration role, e.g. boxlift_1_follow.
    for path in [folder, *folder.parents]:
        tokens = [token for token in re.split(r"[^a-zA-Z]+", path.name.lower()) if token]
        if any(token in {"follow", "following"} for token in tokens):
            return 0
        if any(token in {"lead", "leading"} for token in tokens):
            return 1
    raise ValueError(f"Could not infer mode from folder path: {folder}")


def _choose_goal_assignment(
    mode: str,
    pkl_path: Path,
    pkl_order: List[Path],
    goals_list: Optional[List[Any]],
    closest: List[Dict[str, Any]],
) -> Any:
    if mode == "list":
        goal_index = _goal_from_ordered_list(pkl_path, pkl_order, goals_list)
        if goal_index is None:
            raise ValueError("GOAL_ASSIGNMENT_MODE is 'list' but GOALS_LIST is None")
        return goal_index
    if mode == "last_pose":
        return closest[0]["index"]
    if mode == "xlsx":
        raise ValueError("XLSX assignments are handled before _choose_goal_assignment")
    raise ValueError("GOAL_ASSIGNMENT_MODE must be 'last_pose', 'list', or 'xlsx'")


def _format_topic_value(value: Any) -> str:
    return _format_goal_name(value)


def _inject_multiple_topic_entries_after_steps(data_list: List[Any], entries: List[Dict[str, Any]]) -> int:
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

    count = 0
    new_list: List[Any] = []
    for idx, entry in enumerate(data_list):
        new_list.append(entry)
        if isinstance(entry, dict) and "step" in entry:
            step = entry["step"]
            if step in last_index_by_step and last_index_by_step[step] == idx:
                ts = last_timestamp_by_step.get(step)
                if ts is not None:
                    for topic_entry in entries:
                        new_list.append(topic_entry.copy())
                        count += 1

    data_list[:] = new_list
    return count


def _inject_topic_entries_after_steps(data_list: List[Any], field_name: str, value: Any) -> int:
    return _inject_multiple_topic_entries_after_steps(data_list, [{field_name: _format_topic_value(value)}])


def _inject_goal_entries_after_steps(data_list: List[Any], goal_name: str) -> int:
    return _inject_topic_entries_after_steps(data_list, "goal", goal_name)


def _base_timestamps_for_pkl(pkl_data: Dict[str, Any]) -> List[Any]:
    data = pkl_data["data"]
    timestamps = pkl_data.get("timestamps") if isinstance(pkl_data.get("timestamps"), dict) else {}
    if isinstance(timestamps, dict):
        if ROBOT_STATE_TOPIC in timestamps and isinstance(timestamps[ROBOT_STATE_TOPIC], list):
            return timestamps[ROBOT_STATE_TOPIC]
        for v in timestamps.values():
            if isinstance(v, list):
                return v

    max_len = 0
    for entries in data.values():
        if isinstance(entries, list):
            max_len = max(max_len, len(entries))
    return [None] * max_len


def _inject_topic_into_pkl(pkl_data: Dict[str, Any], topic: str, field_name: str, value: Any) -> int:
    if "data" not in pkl_data:
        raise ValueError("Missing 'data' in pkl structure")
    data = pkl_data["data"]
    if isinstance(data, list):
        return _inject_topic_entries_after_steps(data, field_name, value)
    if isinstance(data, dict):
        timestamps = pkl_data.get("timestamps") if isinstance(pkl_data.get("timestamps"), dict) else {}
        base_ts_list = _base_timestamps_for_pkl(pkl_data)

        topic_value = _format_topic_value(value)
        topic_entries = []
        topic_timestamps = []
        for ts in base_ts_list:
            topic_entries.append({field_name: topic_value})
            topic_timestamps.append(ts)

        data[topic] = topic_entries
        if isinstance(timestamps, dict):
            timestamps[topic] = topic_timestamps
        return len(topic_entries)
    raise ValueError("Unsupported 'data' type in pkl structure")


def _inject_goal_into_pkl(pkl_data: Dict[str, Any], goal_name: str) -> int:
    return _inject_topic_into_pkl(pkl_data, "/goal", "goal", goal_name)


def _inject_arrangement_into_pkl(pkl_data: Dict[str, Any], arrangement_id: Any) -> int:
    return _inject_topic_into_pkl(pkl_data, ARRANGEMENT_TOPIC, "arrangement", arrangement_id)


def _inject_mode_into_pkl(pkl_data: Dict[str, Any], mode_label: Any) -> int:
    return _inject_topic_into_pkl(pkl_data, MODE_TOPIC, "mode", mode_label)


def _inject_xlsx_topics_into_pkl(
    pkl_data: Dict[str, Any],
    goal_name: Any,
    arrangement_id: Any,
    mode_label: Optional[int],
) -> Dict[str, int]:
    if "data" not in pkl_data:
        raise ValueError("Missing 'data' in pkl structure")
    if isinstance(pkl_data["data"], list):
        entries = [
            {"goal": _format_topic_value(goal_name)},
            {"arrangement": _format_topic_value(arrangement_id)},
        ]
        if mode_label is not None:
            entries.append({"mode": _format_topic_value(mode_label)})
        total = _inject_multiple_topic_entries_after_steps(
            pkl_data["data"],
            entries,
        )
        counts = {
            "pkl_goal_entries": total // len(entries),
            "pkl_arrangement_entries": total // len(entries),
        }
        if mode_label is not None:
            counts["pkl_mode_entries"] = total // len(entries)
        return counts

    counts = {
        "pkl_goal_entries": _inject_goal_into_pkl(pkl_data, goal_name),
        "pkl_arrangement_entries": _inject_arrangement_into_pkl(pkl_data, arrangement_id),
    }
    if mode_label is not None:
        counts["pkl_mode_entries"] = _inject_mode_into_pkl(pkl_data, mode_label)
    return counts


def _inject_goal_and_arrangement_into_pkl(
    pkl_data: Dict[str, Any], goal_name: Any, arrangement_id: Any
) -> Dict[str, int]:
    return _inject_xlsx_topics_into_pkl(pkl_data, goal_name, arrangement_id, mode_label=None)


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


def _inject_arrangement_into_structured_list(data: Any, arrangement_id: Any) -> int:
    if not isinstance(data, list):
        raise ValueError("Expected a list of timestep entries")
    return _inject_topic_entries_after_steps(data, "arrangement", arrangement_id)


def _inject_mode_into_structured_list(data: Any, mode_label: Any) -> int:
    if not isinstance(data, list):
        raise ValueError("Expected a list of timestep entries")
    return _inject_topic_entries_after_steps(data, "mode", mode_label)


def _inject_xlsx_topics_into_structured_list(
    data: Any,
    goal_name: Any,
    arrangement_id: Any,
    mode_label: Optional[int],
) -> Dict[str, int]:
    if not isinstance(data, list):
        raise ValueError("Expected a list of timestep entries")
    entries = [
        {"goal": _format_topic_value(goal_name)},
        {"arrangement": _format_topic_value(arrangement_id)},
    ]
    if mode_label is not None:
        entries.append({"mode": _format_topic_value(mode_label)})
    total = _inject_multiple_topic_entries_after_steps(
        data,
        entries,
    )
    counts = {
        "goal_entries": total // len(entries),
        "arrangement_entries": total // len(entries),
    }
    if mode_label is not None:
        counts["mode_entries"] = total // len(entries)
    return counts


def _inject_goal_and_arrangement_into_structured_list(data: Any, goal_name: Any, arrangement_id: Any) -> Dict[str, int]:
    return _inject_xlsx_topics_into_structured_list(data, goal_name, arrangement_id, mode_label=None)


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
    xlsx_path = XLSX_PATH
    output_name = OUTPUT
    top_k = TOP_K
    w_pos = W_POS
    w_rot = W_ROT
    update_files = UPDATE_FILES
    assignment_mode = GOAL_ASSIGNMENT_MODE
    goals_list = GOALS_LIST
    mode_label = _infer_mode_from_folder(PKL_FOLDER) if ADD_MODE_TOPIC else None
    xlsx_assignments: Dict[int, Dict[str, int]] = {}
    xlsx_goal_remap: Dict[int, int] = {}

    folder = PKL_FOLDER
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    if ADD_MODE_TOPIC:
        print(f"Adding mode topic: {mode_label} ({'following' if mode_label == 1 else 'leading'})")

    preset_9d = []
    if assignment_mode in {"last_pose", "list"}:
        presets = _load_presets(preset_path)
        if len(presets) == 0:
            raise SystemExit("No presets provided")

        for p in presets:
            if "index" not in p or "pose" not in p:
                raise SystemExit("Each preset must include 'index' and 'pose'")
            pose_9d = _pose_to_9d(p["pose"])
            preset_9d.append({"index": p["index"], "pose": np.array(pose_9d, dtype=np.float64)})
    elif assignment_mode == "xlsx":
        try:
            xlsx_assignments, xlsx_goal_remap = _load_xlsx_goal_assignments(xlsx_path)
        except Exception as exc:
            raise SystemExit(f"Failed to load XLSX assignments: {exc}") from exc
        print(f"Loaded {len(xlsx_assignments)} XLSX assignments from {xlsx_path}")
        print(f"Raw goal to dense goal mapping: {xlsx_goal_remap}")
    else:
        raise SystemExit("GOAL_ASSIGNMENT_MODE must be 'last_pose', 'list', or 'xlsx'")

    all_pkl_paths = sorted(folder.glob("*.pkl"))
    print(f"Found {len(all_pkl_paths)} target .pkl files in {folder}")
    if not all_pkl_paths:
        print("WARNING: No .pkl files found. No files were changed and no assignments file was saved.")
        return

    if assignment_mode == "list" and goals_list is not None and len(goals_list) != len(all_pkl_paths):
        raise SystemExit(f"GOALS_LIST has {len(goals_list)} entries, but {len(all_pkl_paths)} .pkl files were found.")
    if assignment_mode == "xlsx":
        missing = []
        for pkl_path in all_pkl_paths:
            try:
                _xlsx_record_for_path(pkl_path, xlsx_assignments)
            except ValueError as exc:
                missing.append(str(exc))
        if missing:
            raise SystemExit("Missing XLSX assignments:\n" + "\n".join(missing))

    results = []
    successful_updates = 0
    for pkl_path in all_pkl_paths:
        print(f"Processing {pkl_path.name} ...")
        try:
            pkl_data = _load_pkl(pkl_path)
            last_ee_pose = None
            xlsx_record = None

            if assignment_mode == "xlsx":
                xlsx_record = _xlsx_record_for_path(pkl_path, xlsx_assignments)
                goal_index = xlsx_record["dense_goal"]
                closest = None
            else:
                ee_pose = _get_last_ee_pose(pkl_data)
                ee_pose_9d = np.array(_pose_to_9d(ee_pose), dtype=np.float64)
                last_ee_pose = ee_pose_9d.tolist()

                distances = []
                for preset in preset_9d:
                    dist = _distance(ee_pose_9d, preset["pose"], w_pos, w_rot)
                    distances.append({"index": preset["index"], "distance": dist})

                distances.sort(key=lambda x: x["distance"])
                closest = distances[: max(1, top_k)]
                goal_index = _choose_goal_assignment(
                    assignment_mode,
                    pkl_path,
                    all_pkl_paths,
                    goals_list,
                    closest,
                )
                if assignment_mode == "list":
                    closest = [{"index": goal_index, "distance": None, "source": "list"}]
                else:
                    closest[0]["source"] = "last_pose"
            updated_counts = {}

            if update_files:
                if xlsx_record is not None:
                    updated_counts.update(
                        _inject_xlsx_topics_into_pkl(
                            pkl_data,
                            goal_index,
                            xlsx_record["arrangement_id"],
                            mode_label,
                        )
                    )
                else:
                    pkl_count = _inject_goal_into_pkl(pkl_data, goal_index)
                    updated_counts["pkl_goal_entries"] = pkl_count

                for ext in (".json", ".yaml", ".yml"):
                    structured_path = pkl_path.with_suffix(ext)
                    if structured_path.exists():
                        structured_data = _load_structured_file(structured_path)
                        suffix = structured_path.suffix.lstrip(".")
                        if xlsx_record is not None:
                            structured_counts = _inject_xlsx_topics_into_structured_list(
                                structured_data,
                                goal_index,
                                xlsx_record["arrangement_id"],
                                mode_label,
                            )
                            updated_counts[f"{suffix}_goal_entries"] = structured_counts["goal_entries"]
                            updated_counts[f"{suffix}_arrangement_entries"] = structured_counts["arrangement_entries"]
                            if mode_label is not None:
                                updated_counts[f"{suffix}_mode_entries"] = structured_counts["mode_entries"]
                        else:
                            structured_count = _inject_goal_into_structured_list(structured_data, goal_index)
                            updated_counts[f"{suffix}_goal_entries"] = structured_count
                        _save_structured_file(structured_path, structured_data)

                _save_pkl(pkl_path, pkl_data)

            if updated_counts and any(count > 0 for count in updated_counts.values()):
                successful_updates += 1
                if xlsx_record is not None:
                    mode_message = f", and mode {mode_label}" if mode_label is not None else ""
                    print(
                        "  success: assigned goal "
                        f"{_format_goal_name(goal_index)}, arrangement {xlsx_record['arrangement_id']}"
                        f"{mode_message} "
                        f"with {updated_counts}"
                    )
                else:
                    print(f"  success: assigned goal {_format_goal_name(goal_index)} with {updated_counts}")
            elif update_files:
                print(f"  warning: assigned goal {_format_goal_name(goal_index)}, but no goal entries were added")
            else:
                print(f"  dry-run: assigned goal {_format_goal_name(goal_index)}")

            result = {
                "file": pkl_path.name,
                "number_of_added_entries": updated_counts,
                "number_of_added_goal_entries": updated_counts,
            }
            if closest is not None:
                result["closest_goal"] = closest
            if last_ee_pose is not None:
                result["last_ee_pose"] = last_ee_pose
            if xlsx_record is not None:
                result.update(
                    {
                        "episode_index": xlsx_record["episode_index"],
                        "raw_goal": xlsx_record["raw_goal"],
                        "dense_goal": xlsx_record["dense_goal"],
                        "arrangement_id": xlsx_record["arrangement_id"],
                    }
                )
                if mode_label is not None:
                    result["mode"] = mode_label
            results.append(result)
        except Exception as exc:
            print(f"  error: {exc}")
            results.append({"file": pkl_path.name, "error": str(exc)})

    if update_files and successful_updates == 0:
        print("WARNING: No files were successfully updated. Assignments file was not saved.")
        return

    # Compute summary counts from dense XLSX goals, or from primary closest/list goal.
    summary_counts: Dict[str, int] = {}
    for r in results:
        if assignment_mode == "xlsx" and "dense_goal" in r:
            goal_name = _format_goal_name(r["dense_goal"])
            summary_counts[goal_name] = summary_counts.get(goal_name, 0) + 1
        elif "closest_goal" in r and isinstance(r["closest_goal"], list) and len(r["closest_goal"]) > 0:
            index = r["closest_goal"][0].get("index")
            if index is not None:
                goal_name = _format_goal_name(index)
                summary_counts[goal_name] = summary_counts.get(goal_name, 0) + 1

    # Append summary to results and save
    summary: Dict[str, Any] = {"goal_counts": summary_counts}
    if assignment_mode == "xlsx":
        raw_goal_counts: Dict[str, int] = {}
        for record in xlsx_assignments.values():
            raw_goal = str(record["raw_goal"])
            raw_goal_counts[raw_goal] = raw_goal_counts.get(raw_goal, 0) + 1
        summary["raw_goal_counts"] = dict(sorted(raw_goal_counts.items(), key=lambda item: int(item[0])))
        summary["max_raw_goal"] = max(xlsx_goal_remap.keys()) if xlsx_goal_remap else None
        summary["raw_goal_to_dense_goal"] = {str(raw): dense for raw, dense in sorted(xlsx_goal_remap.items())}
    if mode_label is not None:
        summary["mode"] = mode_label
        summary["mode_name"] = "following" if mode_label == 0 else "leading"
    results.append({"summary": summary})

    output_path = folder / output_name
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Print the summary to stdout
    print(f"Saved assignments to {output_path}")
    if update_files:
        print(f"Successfully updated {successful_updates}/{len(all_pkl_paths)} files.")
    print("Summary counts per goal:")
    for name, cnt in summary_counts.items():
        print(f"  {name}: {cnt}")


if __name__ == "__main__":
    main()
