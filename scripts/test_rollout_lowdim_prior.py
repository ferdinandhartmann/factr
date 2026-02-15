import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Lowdim prior rollout with stiffness-conditioned sampling.")
    parser.add_argument("--model-name", type=str, required=True, help="Checkpoint folder name under factr/checkpoints")
    parser.add_argument("--checkpoint", type=str, default="latest", help="Checkpoint name (latest or ckpt_XXXXXX)")
    parser.add_argument("--buffer-path", type=str, required=True, help="Path to buf.pkl used for quick rollout test")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index in traj list")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of prior samples per timestep")
    parser.add_argument("--stiffness-label", type=int, default=1, help="Stiffness label in {1,2,3}")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--output-dir", type=str, default="factr/scripts/rollout_lowdim_output", help="Output folder")
    return parser.parse_args()


def make_obs_window(states, t, window):
    start = max(0, t - window + 1)
    slice_states = states[start : t + 1]
    if slice_states.shape[0] < window:
        pad = np.repeat(slice_states[:1], repeats=window - slice_states.shape[0], axis=0)
        slice_states = np.concatenate([pad, slice_states], axis=0)
    return slice_states


def load_policy(checkpoints_dir, checkpoint_name, device):
    exp_cfg_path = checkpoints_dir / "rollout" / "exp_config.yaml"
    cfg = OmegaConf.load(exp_cfg_path)

    if "task" in cfg and "cam_indexes" in cfg.task:
        cfg.task.n_cams = len(cfg.task.cam_indexes)
    if "curriculum" in cfg:
        cfg.curriculum.max_step = cfg.max_iterations

    raw_cfg = OmegaConf.to_container(cfg, resolve=False)
    if "hydra" in raw_cfg:
        cfg.pop("hydra", None)
    OmegaConf.resolve(cfg)

    policy = instantiate(cfg.agent).to(device)
    if checkpoint_name == "latest":
        ckpt_path = checkpoints_dir / "rollout" / "latest_ckpt.ckpt"
    else:
        ckpt_path = checkpoints_dir / f"{checkpoint_name}.ckpt"

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy, cfg, ckpt_path


def infer_stiffness_from_episode(episode):
    first_action = np.asarray(episode[0][1], dtype=np.float32).reshape(-1)
    if first_action.shape[0] < 15:
        return 1
    norm_value = float(np.linalg.norm(first_action[9:15]))
    if norm_value < 200:
        return 1
    if norm_value < 1000:
        return 2
    return 3


def main():
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    project_root = Path(__file__).resolve().parents[1]
    checkpoints_dir = project_root / "checkpoints" / args.model_name
    policy, cfg, ckpt_path = load_policy(checkpoints_dir, args.checkpoint, device=device)

    with open(args.buffer_path, "rb") as f:
        traj_list = pickle.load(f)
    episode = traj_list[args.episode_index]

    states = np.stack([np.asarray(step[0]["state"], dtype=np.float32).reshape(-1) for step in episode], axis=0)
    actions = np.stack([np.asarray(step[1], dtype=np.float32).reshape(-1) for step in episode], axis=0)
    true_pose = actions[:, : policy.ac_dim]

    stiffness_label = int(args.stiffness_label)
    if stiffness_label <= 0:
        stiffness_label = infer_stiffness_from_episode(episode)

    num_samples = int(args.num_samples)
    obs_window = int(getattr(policy, "obs_window", cfg.get("obs_window", 8)))
    chunk = int(policy.ac_chunk)

    sampled_chunks = np.zeros((len(states), num_samples, chunk, policy.ac_dim), dtype=np.float32)
    sampled_first_step = np.zeros((num_samples, len(states), policy.ac_dim), dtype=np.float32)

    label_tensor = torch.tensor([stiffness_label], dtype=torch.long, device=device)

    with torch.no_grad():
        for t in range(len(states)):
            window = make_obs_window(states, t=t, window=obs_window)
            obs_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            sample = policy.get_actions_prior(
                imgs={},
                obs=obs_tensor,
                class_labels=label_tensor,
                sample=True,
                num_samples=num_samples,
            )
            sample_np = sample.squeeze(0).cpu().numpy()
            sampled_chunks[t] = sample_np
            sampled_first_step[:, t] = sample_np[:, 0]

    mse_per_sample = np.mean((sampled_first_step - true_pose[None, :, :]) ** 2, axis=(1, 2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model_name}_ep{args.episode_index:03d}_{Path(ckpt_path).stem}.npz"

    np.savez_compressed(
        output_path,
        sampled_chunks=sampled_chunks,
        sampled_first_step=sampled_first_step,
        true_pose=true_pose,
        stiffness_label=stiffness_label,
        mse_per_sample=mse_per_sample,
    )

    print("=== Lowdim Prior Rollout Summary ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"episode_index: {args.episode_index}")
    print(f"stiffness_label: {stiffness_label}")
    print(f"states shape: {states.shape}")
    print(f"sampled_chunks shape: {sampled_chunks.shape}")
    print(f"sampled_first_step shape: {sampled_first_step.shape}")
    print(f"mse_per_sample: {mse_per_sample}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
