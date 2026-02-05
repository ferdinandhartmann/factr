from __future__ import annotations

import argparse
from pathlib import Path

from data.buffer_dataset import summarize_buffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a FACTR replay buffer.")
    parser.add_argument(
        "--buffer-path",
        type=str,
        default="/home/ferdinand/activeinference/factr/process_data/training_data/fourgoals_1/buf.pkl",
        help="Path to buf.pkl",
    )
    args = parser.parse_args()

    buffer_path = Path(args.buffer_path)
    summary = summarize_buffer(buffer_path)
    print("Buffer summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
