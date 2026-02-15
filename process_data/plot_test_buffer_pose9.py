#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import matplotlib. Install it in the FACTR environment first.") from exc

try:
    from robobuf import ReplayBuffer as RB
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import robobuf.ReplayBuffer. Activate the FACTR environment first.") from exc


POSE9_DIM_NAMES = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6"]


def _obs_to_dict(obs):
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "to_dict"):
        out = obs.to_dict()
        if isinstance(out, dict):
            return out
    if hasattr(obs, "obs") and isinstance(obs.obs, dict):
        return obs.obs
    raise TypeError(f"Unsupported obs type: {type(obs)}")


def _is_episode_start(step):
    return bool(getattr(step, "first", False) or getattr(step, "is_first", False) or getattr(step, "prev", None) is None)


def load_replay_buffer(buffer_path: Path):
    with buffer_path.open("rb") as f:
        traj_list = pickle.load(f)
    return RB.load_traj_list(traj_list)


def resolve_buffer_path(path_like: Path, default_name: str = "buf_test.pkl") -> Path:
    """
    Accept either:
    - direct path to a .pkl buffer file, or
    - directory containing buffer files.
    """
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Buffer path not found: {path}")

    if path.is_file():
        if path.suffix != ".pkl":
            raise ValueError(f"Expected a .pkl file, got: {path}")
        return path

    if path.is_dir():
        preferred = path / default_name
        if preferred.exists():
            return preferred

        pkl_files = sorted(path.glob("*.pkl"))
        if len(pkl_files) == 1:
            return pkl_files[0]
        if len(pkl_files) == 0:
            raise FileNotFoundError(f"No .pkl file found in directory: {path}")
        available = ", ".join([p.name for p in pkl_files[:8]])
        raise FileNotFoundError(
            f"Multiple .pkl files found in {path}. Pass --buffer-path to a specific file. "
            f"Found: {available}"
        )

    raise ValueError(f"Unsupported path type: {path}")


def collect_pose_series(buffer, max_steps=None):
    measured_pose = []
    commanded_pose = []
    episode_starts = []
    skipped = 0

    total_steps = len(buffer)
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))

    for i in range(total_steps):
        step = buffer[i]
        if _is_episode_start(step):
            episode_starts.append(len(measured_pose))

        try:
            obs_dict = _obs_to_dict(step.obs)
            state = np.asarray(obs_dict["state"], dtype=np.float32).reshape(-1)
            action = np.asarray(step.action, dtype=np.float32).reshape(-1)
        except Exception:
            skipped += 1
            continue

        if state.size < 9 or action.size < 9:
            skipped += 1
            continue

        measured_pose.append(state[:9])
        commanded_pose.append(action[:9])

    if len(measured_pose) == 0:
        raise RuntimeError("No valid samples with >=9D state/action were found in the buffer.")

    return (
        np.asarray(measured_pose, dtype=np.float32),
        np.asarray(commanded_pose, dtype=np.float32),
        np.asarray(episode_starts, dtype=np.int64),
        skipped,
        total_steps,
    )


def plot_pose_comparison(measured, commanded, episode_starts, output_path: Path, title_suffix="", dpi=180):
    fig, axes = plt.subplots(3, 3, figsize=(30, 20), sharex=True)
    axes = axes.reshape(-1)
    x = np.arange(measured.shape[0])

    for dim, ax in enumerate(axes):
        ax.plot(x, measured[:, dim], color="black", linewidth=1.2, label="measured" if dim == 0 else None)
        ax.plot(x, commanded[:, dim], color="#E41A1C", linewidth=1.0, alpha=0.9, label="commanded" if dim == 0 else None)

        if episode_starts.size > 1:
            for start_idx in episode_starts[1:]:
                ax.axvline(start_idx, color="gray", linewidth=0.6, alpha=0.2)

        ax.set_title(POSE9_DIM_NAMES[dim])
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    fig.suptitle(
        f"Measured vs Commanded End-Effector Pose (9D){title_suffix}\n"
        f"samples={measured.shape[0]}",
        fontsize=12,
    )
    axes[-1].set_xlabel("Global buffer step index")
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot measured vs commanded end-effector pose (first 9 dims) for the whole test buffer."
    )
    parser.add_argument(
        "--buffer-path",
        type=Path,
        default=Path(
            "/home/ferdinand/activeinference/factr/process_data/processed_data/fourgoals_1_act/buf_test.pkl"
        ),
        help="Path to buffer .pkl file OR directory containing it.",
    )
    parser.add_argument(
        "--buffer-name",
        type=str,
        default="buf_test.pkl",
        help="When --buffer-path is a directory, preferred filename to resolve.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output PNG path. Default: <buffer_dir>/<buffer_stem>_measured_vs_commanded_pose9.png",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on number of transitions to plot (default: whole buffer).",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Saved figure DPI.")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively.")
    return parser.parse_args()


def main():
    args = parse_args()
    resolved_buffer_path = resolve_buffer_path(args.buffer_path, default_name=args.buffer_name)

    output_path = args.output_path
    if output_path is None:
        output_path = resolved_buffer_path.parent / f"{resolved_buffer_path.stem}_measured_vs_commanded_pose9.png"

    buffer = load_replay_buffer(resolved_buffer_path)
    measured, commanded, episode_starts, skipped, scanned = collect_pose_series(buffer, max_steps=args.max_steps)

    title_suffix = f" | episodes={episode_starts.size}, skipped={skipped}, scanned={scanned}"
    plot_pose_comparison(
        measured=measured,
        commanded=commanded,
        episode_starts=episode_starts,
        output_path=output_path,
        title_suffix=title_suffix,
        dpi=args.dpi,
    )

    print(f"Buffer path: {resolved_buffer_path}")
    print(f"Scanned transitions: {scanned}")
    print(f"Used samples: {measured.shape[0]}")
    print(f"Episode starts detected: {episode_starts.size}")
    print(f"Skipped transitions: {skipped}")
    print(f"Saved plot: {output_path}")

    if args.show:
        preview = plt.imread(output_path)
        fig = plt.figure(figsize=(14, 8))
        plt.imshow(preview)
        plt.axis("off")
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
