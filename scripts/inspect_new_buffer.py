import argparse
import pickle
from collections import Counter

import numpy as np
from robobuf import ReplayBuffer as RB


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect FACTR lowdim buffer for stiffness-conditioned training.")
    parser.add_argument(
        "--buffer-path",
        type=str,
        default="/home/ferdinand/activeinference/factr/process_data/processed_data/fourgoals_1_act/buf_train.pkl",
        help="Path to buf.pkl",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.buffer_path, "rb") as f:
        traj_list = pickle.load(f)
    buffer = RB.load_traj_list(traj_list)

    obs_keys = Counter()
    state_dims = Counter()
    action_dims = Counter()
    episode_lengths = []
    stiffness_signatures = Counter()
    stiffness_label_counts = Counter()
    stiffness_label_episode_counts = Counter()

    current_len = 0
    first_count = 0

    for idx in range(len(buffer)):
        step = buffer[idx]
        obs_dict = step.obs.to_dict() if hasattr(step.obs, "to_dict") else step.obs
        if bool(getattr(step, "first", False) or step.prev is None):
            first_count += 1
            if current_len > 0:
                episode_lengths.append(current_len)
            current_len = 0
            action = np.asarray(step.action, dtype=np.float32).reshape(-1)
            if action.shape[0] >= 15:
                sig = tuple(np.round(action[9:15], 5).tolist())
                stiffness_signatures[sig] += 1
            if "stiffness_label" in obs_dict:
                label = int(np.asarray(obs_dict["stiffness_label"]).reshape(-1)[0])
                stiffness_label_episode_counts[label] += 1

        current_len += 1

        for key in obs_dict.keys():
            obs_keys[key] += 1

        if "state" in obs_dict:
            state_dims[int(np.asarray(obs_dict["state"]).reshape(-1).shape[0])] += 1
        if "stiffness_label" in obs_dict:
            label = int(np.asarray(obs_dict["stiffness_label"]).reshape(-1)[0])
            stiffness_label_counts[label] += 1
        action_dims[int(np.asarray(step.action).reshape(-1).shape[0])] += 1

    if current_len > 0:
        episode_lengths.append(current_len)

    print("=== Buffer Inspection ===")
    print(f"buffer_path: {args.buffer_path}")
    print(f"num_transitions: {len(buffer)}")
    print(f"num_episodes (is_first): {first_count}")
    print(f"obs_keys: {dict(obs_keys)}")
    print(f"state_dims: {dict(state_dims)}")
    print(f"action_dims: {dict(action_dims)}")
    if len(action_dims) == 1 and 9 in action_dims:
        print("action_semantics: commanded EE pose = [x,y,z] + 6D rotation")
    print(
        "episode_length_stats: "
        f"min={int(np.min(episode_lengths))}, max={int(np.max(episode_lengths))}, "
        f"mean={float(np.mean(episode_lengths)):.2f}"
    )

    print("\n=== Stiffness Labels (from obs['stiffness_label']) ===")
    if len(stiffness_label_counts) == 0:
        print("No stiffness_label found in observations.")
    else:
        print(f"transition_label_counts: {dict(sorted(stiffness_label_counts.items()))}")
        print(f"episode_start_label_counts: {dict(sorted(stiffness_label_episode_counts.items()))}")

    print("\n=== Stiffness Signatures (legacy check from action[9:15] at episode start) ===")
    if len(stiffness_signatures) == 0:
        print("No stiffness signatures found (expected when action dim is 9 pose-only).")
    else:
        sorted_sigs = sorted(stiffness_signatures.items(), key=lambda item: float(np.linalg.norm(np.asarray(item[0]))))
        for class_id, (sig, count) in enumerate(sorted_sigs, start=1):
            print(f"label={class_id} count={count} signature={sig}")


if __name__ == "__main__":
    main()
