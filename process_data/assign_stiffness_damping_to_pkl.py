#!/usr/bin/env python3

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

# Folder containing .pkl files
PKL_FOLDER = Path("~/activeinference/factr/process_data/data_to_process/boxlift_2_lead/data").expanduser()

# Files need to be renamed to end with "_stiff.pkl" or "_soft.pkl" for this script to process them.

# Configuration variables (replaces command-line arguments)
OUTPUT: str = "stiffness_damping_assignments.json"

# Output topic name for injected gains in dict-style PKL data
OUTPUT_TOPIC = "/cartesian_impedance_gains"

# Fixed gains to inject for every timestep
# FIXED_STIFFNESS: List[float] = [1600.0, 1600.0, 1600.0, 50.0, 50.0, 50.0] # stiff stiffness
# FIXED_DAMPING: List[float] = [40.0, 40.0, 40.0, 4.5, 4.5, 4.5]

# FIXED_STIFFNESS: List[float] = [1000.0, 1000.0, 1000.0, 30.0, 30.0, 30.0]  # high2
# FIXED_DAMPING: List[float] = [25.0, 25.0, 25.0, 2.5, 2.5, 2.5]

FIXED_STIFFNESS: List[float] = [1000.0, 1000.0, 1000.0, 30.0, 30.0, 30.0]  # Boxlift
FIXED_DAMPING: List[float] = [100.0, 100.0, 100.0, 2.5, 2.5, 2.5]

# FIXED_STIFFNESS: List[float] = [1000.0, 1000.0, 1000.0, 30.0, 30.0, 30.0]  # high2 + more damping (fourgoals_4_stiff)
# FIXED_DAMPING: List[float] = [120.0, 120.0, 120.0, 2.5, 2.5, 2.5]

# FIXED_STIFFNESS: List[float] = [300.0, 300.0, 300.0, 15.0, 15.0, 15.0]  # medium stiffness
# FIXED_DAMPING: List[float] = [17.0, 17.0, 17.0, 1.5, 1.5, 1.5]

# FIXED_STIFFNESS: List[float] = [50.0, 50.0, 50.0, 4.0, 4.0, 4.0]  # soft stiffness
# FIXED_DAMPING: List[float] = [4.0, 4.0, 4.0, 0.3, 0.3, 0.3]

# FIXED_STIFFNESS: List[float] = [50.0, 50.0, 50.0, 3.0, 3.0, 3.0]  # soft 2 (maybe fourgoals_2_soft)
# FIXED_DAMPING: List[float] = [4.0, 4.0, 4.0, 0.3, 0.3, 0.3]

# Whether to update data files in place
UPDATE_FILES: bool = True

# Whether to skip insertion for list-style data when gains are missing
SKIP_IF_MISSING: bool = False

# Only process PKL files that end with one of these suffixes.
# Use an empty tuple to process every .pkl file in PKL_FOLDER.
TARGET_FILE_SUFFIXES: Tuple[str, ...] = (".pkl", "_stiff.pkl", "_soft.pkl")


def _load_pkl(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pkl(path: Path, data: Dict[str, Any]) -> None:
    with path.open("wb") as f:
        pickle.dump(data, f)


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


def _extract_gains(entry: Any) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    if not isinstance(entry, dict):
        return None, None
    k = entry.get("k_gains") or entry.get("k") or entry.get("stiffness")
    d = entry.get("d_gains") or entry.get("d") or entry.get("damping")
    if k is None and d is None:
        return None, None
    return (list(k) if k is not None else None, list(d) if d is not None else None)


def _build_base_timestamps(pkl_data: Dict[str, Any]) -> List[Optional[int]]:
    timestamps = pkl_data.get("timestamps") if isinstance(pkl_data.get("timestamps"), dict) else {}
    if isinstance(timestamps, dict) and timestamps:
        # Prefer the longest timestamp list
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


def _inject_gains_into_list(
    data_list: List[Any],
    fixed_k: List[float],
    fixed_d: List[float],
    skip_if_missing: bool,
) -> int:
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
                if skip_if_missing and (fixed_k is None or fixed_d is None):
                    continue
                ts = last_timestamp_by_step.get(step)
                new_list.append(
                    {
                        "timestamp": ts,
                        "topic": OUTPUT_TOPIC,
                        "step": step,
                        "stiffness": fixed_k,
                        "damping": fixed_d,
                    }
                )
                count += 1

    data_list[:] = new_list
    return count


def _inject_gains_into_dict(pkl_data: Dict[str, Any]) -> int:
    if "data" not in pkl_data or not isinstance(pkl_data["data"], dict):
        raise ValueError("Missing 'data' dict in pkl structure")

    data = pkl_data["data"]
    base_ts_list = _build_base_timestamps(pkl_data)
    result_entries: List[Dict[str, Any]] = [
        {"stiffness": FIXED_STIFFNESS, "damping": FIXED_DAMPING} for _ in base_ts_list
    ]

    data[OUTPUT_TOPIC] = result_entries
    timestamps = pkl_data.get("timestamps") if isinstance(pkl_data.get("timestamps"), dict) else {}
    if isinstance(timestamps, dict):
        timestamps[OUTPUT_TOPIC] = base_ts_list
    return len(result_entries)


def _inject_gains_into_structured_list(data: Any) -> int:
    if not isinstance(data, list):
        raise ValueError("Expected a list of timestep entries")
    return _inject_gains_into_list(data, FIXED_STIFFNESS, FIXED_DAMPING, SKIP_IF_MISSING)


def main() -> None:
    folder = PKL_FOLDER
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    all_pkl_paths = sorted(folder.glob("*.pkl"))
    if TARGET_FILE_SUFFIXES:
        target_pkl_paths = [p for p in all_pkl_paths if p.name.endswith(TARGET_FILE_SUFFIXES)]
    else:
        target_pkl_paths = all_pkl_paths

    print(f"Found {len(target_pkl_paths)} target .pkl files in {folder}")
    if not target_pkl_paths:
        print(
            "WARNING: No .pkl files matched TARGET_FILE_SUFFIXES. "
            "No files were changed and no assignments file was saved."
        )
        return

    results = []
    successful_updates = 0
    for pkl_path in target_pkl_paths:
        print(f"Processing {pkl_path.name} ...")
        try:
            pkl_data = _load_pkl(pkl_path)
            updated_counts: Dict[str, int] = {}

            if UPDATE_FILES:
                if isinstance(pkl_data.get("data"), list):
                    count = _inject_gains_into_list(pkl_data["data"], FIXED_STIFFNESS, FIXED_DAMPING, SKIP_IF_MISSING)
                else:
                    count = _inject_gains_into_dict(pkl_data)
                _save_pkl(pkl_path, pkl_data)
                updated_counts["pkl_entries"] = count

                for ext in (".json", ".yaml", ".yml"):
                    structured_path = pkl_path.with_suffix(ext)
                    if structured_path.exists():
                        structured_data = _load_structured_file(structured_path)
                        structured_count = _inject_gains_into_structured_list(structured_data)
                        _save_structured_file(structured_path, structured_data)
                        updated_counts[structured_path.suffix.lstrip(".") + "_entries"] = structured_count

            if updated_counts and any(count > 0 for count in updated_counts.values()):
                successful_updates += 1
                print(f"  success: added/updated {updated_counts}")
            else:
                print("  warning: no gain entries were added or updated")

            results.append(
                {
                    "file": pkl_path.name,
                    "number_of_added_gain_entries": updated_counts,
                    # Record the exact gain values written so the assignment file is self-describing.
                    "gain_topic": OUTPUT_TOPIC,
                    "stiffness": list(FIXED_STIFFNESS),
                    "damping": list(FIXED_DAMPING),
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

    print(f"Saved gain assignments to {output_path}")
    print(f"Successfully updated {successful_updates}/{len(target_pkl_paths)} files.")


if __name__ == "__main__":
    main()
