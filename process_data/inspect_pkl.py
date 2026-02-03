#!/usr/bin/env python3
"""
inspect_pkl.py  –  Quick viewer for .pkl files

Usage:
    python inspect_pkl.py /path/to/file.pkl
"""

import pickle
import sys
import numpy as np
from pprint import pprint
from pathlib import Path


def summarize_pkl(path: Path):
    print(f"🔍 Inspecting: {path}")
    if not path.exists():
        print("❌ File not found.")
        return

    with open(path, "rb") as f:
        data = pickle.load(f)

    print("\nTop-level keys:")
    pprint(list(data.keys()))

    print("\nFirst item of each top-level key:")
    for k in data.keys():
        v = data[k]
        if v is None:
            print(f"  {k:40s}  → None")
            continue

        # Lists / tuples-like
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                print(f"  {k:40s}  → empty {type(v).__name__}")
                continue
            first = v[0]
            tname = type(first).__name__
            print(f"  {k:40s}  → first_type={tname}")
            if isinstance(first, dict):
                print("    dict keys:", list(first.keys()))
                pprint(first)
            elif isinstance(first, np.ndarray):
                print(f"    numpy shape={first.shape} dtype={first.dtype}")
                # show small numeric preview
                flat = first.ravel()
                preview = flat[:10].tolist() if flat.size > 0 else []
                print("    preview:", preview)
            else:
                pprint(first)
            continue

        # Dict-like top-level value
        if isinstance(v, dict):
            if len(v) == 0:
                print(f"  {k:40s}  → empty dict")
                continue
            subk = next(iter(v))
            subv = v[subk]
            print(f"  {k:40s}  → dict (first key={subk}) first_value_type={type(subv).__name__}")
            if isinstance(subv, (list, tuple)) and len(subv) > 0:
                pprint(subv[0])
            else:
                pprint(subv)
            continue

        # Numpy or scalar or other
        if isinstance(v, np.ndarray):
            print(f"  {k:40s}  → numpy array shape={v.shape} dtype={v.dtype}")
            flat = v.ravel()
            preview = flat[:10].tolist() if flat.size > 0 else []
            print("    preview:", preview)
        else:
            print(f"  {k:40s}  → {type(v).__name__}:")
            pprint(v)

    if "data" not in data:
        print("\n⚠️  No 'data' key found — printing raw object type:")
        print(type(data))
        return

    print("\nAvailable topics:")
    for k in data["data"].keys():
        v = data["data"][k]
        if not v:
            print(f"  {k:60s}  len=0")
            continue
        print(
            f"  {k:60s}  len={len(v):5d}  sample_type={type(v[0]).__name__}"
        )

    # Show a small sample from the first topic
    first_topic = next(iter(data["data"]))
    first_entry = data["data"][first_topic][0]
    print("\nExample entry from:", first_topic)

    if isinstance(first_entry, dict):
        # Likely a ROS-style dictionary message
        keys = list(first_entry.keys())
        print("  Keys:", keys)
        for k in keys:
            v = first_entry[k]
            if isinstance(v, (list, np.ndarray)):
                print(f"   {k}: len={len(v)}")
            else:
                print(f"   {k}: {v}")
    elif isinstance(first_entry, np.ndarray):
        print("  → numpy array shape:", first_entry.shape)

if __name__ == "__main__":
    # Use command-line argument if provided, otherwise use the default test file
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/ferdinand/activeinference/factr/process_data/data_to_process/bld_soft/data/ep_02.pkl")
    summarize_pkl(p)