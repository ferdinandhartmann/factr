#!/usr/bin/env python3

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


# Update these
PKL_PATH = Path("~/factr_ws/raw_data/fourgoals_1_stiff/20260203/data/ep_01.pkl").expanduser()
JSON_PATH = PKL_PATH.with_name(PKL_PATH.stem + "_converted.json")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": True, "hex": bytes(obj).hex()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> None:
    if not PKL_PATH.exists():
        raise SystemExit(f"File not found: {PKL_PATH}")

    with PKL_PATH.open("rb") as f:
        data = pickle.load(f)

    json_ready = _to_jsonable(data)

    with JSON_PATH.open("w") as f:
        json.dump(json_ready, f, indent=2)

    print(f"Saved JSON to {JSON_PATH}")


if __name__ == "__main__":
    main()
