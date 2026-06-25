#!/usr/bin/env python3

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Source folder containing .pkl files.
PKL_FOLDER = Path("~/activeinference/factr/process_data/data_to_process/boxlift_1_lead/data").expanduser()

# Trimmed copies are written here. Source files are not modified.
OUTPUT_FOLDER = PKL_FOLDER.parent / "boxlift_1_lead_trimmed"

TRIM_SECONDS: float = 1.5
OVERWRITE_EXISTING_FILES: bool = True
OUTPUT_SUMMARY: str = "trim_start_assignments.json"
TARGET_FILE_SUFFIXES: Tuple[str, ...] = (".pkl",)


def _load_pkl(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pkl(path: Path, data: Dict[str, Any]) -> None:
    with path.open("wb") as f:
        pickle.dump(data, f)


def _timestamp_to_seconds(timestamp: Any) -> Optional[float]:
    if timestamp is None:
        return None
    try:
        value = float(timestamp)
    except (TypeError, ValueError):
        return None
    if value > 1e12:
        value *= 1e-9
    return value


def _collect_timestamps(pkl_data: Dict[str, Any]) -> List[float]:
    collected: List[float] = []

    all_timestamps = pkl_data.get("all_timestamps")
    if isinstance(all_timestamps, list):
        collected.extend(ts for ts in (_timestamp_to_seconds(t) for t in all_timestamps) if ts is not None)

    timestamps = pkl_data.get("timestamps")
    if isinstance(timestamps, dict):
        for topic_timestamps in timestamps.values():
            if isinstance(topic_timestamps, list):
                collected.extend(ts for ts in (_timestamp_to_seconds(t) for t in topic_timestamps) if ts is not None)

    data = pkl_data.get("data")
    if isinstance(data, list):
        collected.extend(
            ts
            for ts in (_timestamp_to_seconds(entry.get("timestamp")) for entry in data if isinstance(entry, dict))
            if ts is not None
        )

    if not collected:
        raise ValueError("No timestamps found to compute trim cutoff")
    return collected


def _trim_dict_style_data(pkl_data: Dict[str, Any], cutoff_seconds: float) -> Dict[str, Any]:
    data = pkl_data.get("data")
    timestamps = pkl_data.get("timestamps")
    if not isinstance(data, dict) or not isinstance(timestamps, dict):
        raise ValueError("Expected dict-style PKL with 'data' and 'timestamps' dicts")

    removed_by_topic: Dict[str, int] = {}
    for topic, entries in list(data.items()):
        topic_timestamps = timestamps.get(topic)
        if not isinstance(entries, list) or not isinstance(topic_timestamps, list):
            continue

        keep_indices = [
            idx
            for idx, timestamp in enumerate(topic_timestamps)
            if (_timestamp_to_seconds(timestamp) is not None and _timestamp_to_seconds(timestamp) >= cutoff_seconds)
        ]
        original_count = min(len(entries), len(topic_timestamps))
        data[topic] = [entries[idx] for idx in keep_indices if idx < len(entries)]
        timestamps[topic] = [topic_timestamps[idx] for idx in keep_indices]
        removed_by_topic[topic] = original_count - len(timestamps[topic])

    all_timestamps = pkl_data.get("all_timestamps")
    if isinstance(all_timestamps, list):
        pkl_data["all_timestamps"] = [
            timestamp
            for timestamp in all_timestamps
            if (_timestamp_to_seconds(timestamp) is not None and _timestamp_to_seconds(timestamp) >= cutoff_seconds)
        ]

    pkl_data["_trim_start_info"] = {
        "trim_seconds": TRIM_SECONDS,
        "cutoff_seconds": cutoff_seconds,
        "removed_by_topic": removed_by_topic,
    }
    return pkl_data


def _trim_list_style_data(pkl_data: Dict[str, Any], cutoff_seconds: float) -> Dict[str, Any]:
    data = pkl_data.get("data")
    if not isinstance(data, list):
        raise ValueError("Expected list-style PKL with 'data' list")

    original_count = len(data)
    pkl_data["data"] = [
        entry
        for entry in data
        if isinstance(entry, dict)
        and _timestamp_to_seconds(entry.get("timestamp")) is not None
        and _timestamp_to_seconds(entry.get("timestamp")) >= cutoff_seconds
    ]

    all_timestamps = pkl_data.get("all_timestamps")
    if isinstance(all_timestamps, list):
        pkl_data["all_timestamps"] = [
            timestamp
            for timestamp in all_timestamps
            if (_timestamp_to_seconds(timestamp) is not None and _timestamp_to_seconds(timestamp) >= cutoff_seconds)
        ]

    pkl_data["_trim_start_info"] = {
        "trim_seconds": TRIM_SECONDS,
        "cutoff_seconds": cutoff_seconds,
        "removed_entries": original_count - len(pkl_data["data"]),
    }
    return pkl_data


def _trim_pkl_data(pkl_data: Dict[str, Any]) -> Dict[str, Any]:
    episode_start_seconds = min(_collect_timestamps(pkl_data))
    cutoff_seconds = episode_start_seconds + TRIM_SECONDS

    # Current processed FACTR episodes are dict-style; list-style support is kept for raw exports.
    if isinstance(pkl_data.get("data"), dict):
        return _trim_dict_style_data(pkl_data, cutoff_seconds)
    if isinstance(pkl_data.get("data"), list):
        return _trim_list_style_data(pkl_data, cutoff_seconds)
    raise ValueError("Unsupported PKL structure: missing 'data' dict/list")


def main() -> None:
    source_folder = PKL_FOLDER
    output_folder = OUTPUT_FOLDER
    if not source_folder.exists() or not source_folder.is_dir():
        raise SystemExit(f"Folder not found: {source_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    all_pkl_paths = sorted(source_folder.glob("*.pkl"))
    target_pkl_paths = [p for p in all_pkl_paths if p.name.endswith(TARGET_FILE_SUFFIXES)]

    print(f"Found {len(target_pkl_paths)} target .pkl files in {source_folder}")
    print(f"Writing trimmed copies to {output_folder}")
    if not target_pkl_paths:
        print("WARNING: No .pkl files matched TARGET_FILE_SUFFIXES. No files were copied.")
        return

    results = []
    successful_updates = 0
    for pkl_path in target_pkl_paths:
        output_path = output_folder / pkl_path.name
        print(f"Processing {pkl_path.name} ...")
        try:
            if output_path.exists() and not OVERWRITE_EXISTING_FILES:
                print("  skipped: output file already exists")
                results.append({"file": pkl_path.name, "output": str(output_path), "skipped": True})
                continue

            pkl_data = _load_pkl(pkl_path)
            trimmed_data = _trim_pkl_data(pkl_data)
            _save_pkl(output_path, trimmed_data)

            info = trimmed_data.get("_trim_start_info", {})
            successful_updates += 1
            print(f"  success: copied trimmed file to {output_path.name}")
            results.append(
                {
                    "file": pkl_path.name,
                    "output": str(output_path),
                    "trim_seconds": TRIM_SECONDS,
                    "cutoff_seconds": info.get("cutoff_seconds"),
                    "removed_by_topic": info.get("removed_by_topic"),
                    "removed_entries": info.get("removed_entries"),
                }
            )
        except Exception as exc:
            print(f"  error: {exc}")
            results.append({"file": pkl_path.name, "error": str(exc)})

    summary_path = output_folder / OUTPUT_SUMMARY
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved trim summary to {summary_path}")
    print(f"Successfully wrote {successful_updates}/{len(target_pkl_paths)} trimmed files.")


if __name__ == "__main__":
    main()
