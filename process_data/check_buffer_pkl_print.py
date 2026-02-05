#!/usr/bin/env python3

import pickle
from pathlib import Path
from typing import Any, Dict, List

# Update these
PKL_PATH = Path("/home/ferdinand/activeinference/factr/process_data/training_data/fourgoals_1_newnorm/buf.pkl")
TRAJ_INDEX = 0
STEP_INDEX = 0
SHOW_ALL = True
SKIP_IMAGE_TOPICS = True


def _safe_len(obj: Any) -> int:
    return len(obj) if isinstance(obj, list) else 0


def _format_entry(entry: Any) -> str:
    if isinstance(entry, dict):
        keys = ", ".join(entry.keys())
        return f"dict keys=[{keys}] values={entry}"
    if hasattr(entry, "__dict__"):
        keys = ", ".join(entry.__dict__.keys())
        return f"{type(entry).__name__} attrs=[{keys}]"
    return f"{type(entry).__name__}: {entry}"


def _print_dict_buffer(data: Dict[str, Any]) -> None:
    topics: Dict[str, Any] = data["data"]
    timestamps: Dict[str, List[Any]] = data.get("timestamps", {}) if isinstance(data.get("timestamps"), dict) else {}

    if not isinstance(topics, dict):
        raise SystemExit("Expected data['data'] to be a dict of topics -> list entries")

    print(f"File: {PKL_PATH}")
    print(f"Timestep index: {STEP_INDEX}")
    print("\nTopics layout:")
    for topic, entries in sorted(topics.items()):
        count = _safe_len(entries)
        ts_count = _safe_len(timestamps.get(topic)) if timestamps else 0
        print(f"- {topic}: entries={count}, timestamps={ts_count}")

    print("\nData at timestep:")
    for topic, entries in sorted(topics.items()):
        if SKIP_IMAGE_TOPICS and ("/im" in topic or "image" in topic.lower()):
            print(f"- {topic}: (skipped image topic)")
            continue
        if not isinstance(entries, list) or STEP_INDEX >= len(entries):
            print(f"- {topic}: (no entry at index {STEP_INDEX})")
            continue
        entry = entries[STEP_INDEX]
        ts_list = timestamps.get(topic, []) if isinstance(timestamps, dict) else []
        ts_value = ts_list[STEP_INDEX] if isinstance(ts_list, list) and STEP_INDEX < len(ts_list) else None
        if ts_value is None and isinstance(entry, dict) and "timestamp" in entry:
            ts_value = entry.get("timestamp")
        if isinstance(ts_value, int) and ts_value > 1e12:
            ts_value = ts_value / 1e9
        print(f"- {topic}:")
        print(f"  timestamp: {ts_value}")
        print(f"  entry: {_format_entry(entry)}")


def _print_trajectory_dict(traj: Dict[str, Any]) -> None:
    print("Trajectory keys:")
    for k, v in sorted(traj.items()):
        if isinstance(v, list):
            print(f"- {k}: len={len(v)} type={type(v[0]).__name__ if v else 'empty'}")
        else:
            print(f"- {k}: type={type(v).__name__}")

    if "observations" in traj and "actions" in traj:
        idx = min(STEP_INDEX, len(traj["actions"]) - 1)
        print(f"\nSample step {idx}:")
        print(f"- observation: {_format_entry(traj['observations'][idx])}")
        print(f"- action: {_format_entry(traj['actions'][idx])}")
        if "images" in traj:
            print(f"- images entry type: {type(traj['images'][idx]).__name__}")


def _print_step_details(step: Any) -> None:
    if isinstance(step, tuple):
        print(f"tuple len={len(step)}")
        if len(step) >= 1:
            print(f"- obs: {_format_entry(step[0])}")
        if len(step) >= 2:
            print(f"- action: {_format_entry(step[1])}")
        if len(step) >= 3:
            print(f"- is_first: {_format_entry(step[2])}")
        if len(step) >= 4:
            print(f"- is_last: {_format_entry(step[3])}")
        if len(step) >= 5:
            print(f"- reward: {_format_entry(step[4])}")
        return

    print(_format_entry(step))
    if isinstance(step, dict):
        for key in ("obs", "action", "reward", "done", "is_first", "is_last"):
            if key in step:
                print(f"- {key}: {_format_entry(step[key])}")
        if "next" in step:
            print(f"- next: {'present' if step['next'] is not None else 'None'}")
        if "prev" in step:
            print(f"- prev: {'present' if step['prev'] is not None else 'None'}")
        return

    if hasattr(step, "obs") or hasattr(step, "action"):
        obs = getattr(step, "obs", None)
        action = getattr(step, "action", None)
        reward = getattr(step, "reward", None)
        done = getattr(step, "done", None)
        is_first = getattr(step, "is_first", None)
        is_last = getattr(step, "is_last", None)
        next_step = getattr(step, "next", None)
        prev_step = getattr(step, "prev", None)
        print(f"- obs: {_format_entry(obs)}")
        print(f"- action: {_format_entry(action)}")
        print(f"- reward: {reward}")
        print(f"- done: {done}")
        print(f"- is_first: {is_first}")
        print(f"- is_last: {is_last}")
        print(f"- next: {'present' if next_step is not None else 'None'}")
        print(f"- prev: {'present' if prev_step is not None else 'None'}")
        return


def _print_traj_list(traj_list: List[Any]) -> None:
    print(f"File: {PKL_PATH}")
    print(f"Number of trajectories: {len(traj_list)}")
    if not traj_list:
        return

    if SHOW_ALL:
        for traj_idx, traj in enumerate(traj_list):
            print(f"\n=== Trajectory {traj_idx} ===")
            if isinstance(traj, dict):
                _print_trajectory_dict(traj)
                continue
            if isinstance(traj, list):
                print(f"Trajectory length: {len(traj)}")
                for step_idx, step in enumerate(traj):
                    print(f"\nStep {step_idx}:")
                    _print_step_details(step)
                continue
            print(f"Unknown trajectory element type: {type(traj).__name__}")
        return

    traj_idx = min(TRAJ_INDEX, len(traj_list) - 1)
    traj = traj_list[traj_idx]
    print(f"\nSelected trajectory index: {traj_idx}")

    if isinstance(traj, dict):
        _print_trajectory_dict(traj)
        return

    if isinstance(traj, list):
        print(f"Trajectory length: {len(traj)}")
        step_idx = min(STEP_INDEX, len(traj) - 1)
        step = traj[step_idx]
        print(f"\nSample step {step_idx}:")
        _print_step_details(step)
        return

    print(f"Unknown trajectory element type: {type(traj).__name__}")


def main() -> None:
    if not PKL_PATH.exists():
        raise SystemExit(f"File not found: {PKL_PATH}")

    with PKL_PATH.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "data" in data:
        _print_dict_buffer(data)
        return

    if isinstance(data, list):
        _print_traj_list(data)
        return

    raise SystemExit(
        f"Unexpected pickle structure: {type(data).__name__} (expected dict with 'data' or list of trajectories)"
    )


if __name__ == "__main__":
    main()
