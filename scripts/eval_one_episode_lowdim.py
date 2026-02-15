import argparse
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one episode for lowdim stiffness-conditioned model with trajectory sampling."
    )
    parser.add_argument("--model-name", type=str, required=True, help="Checkpoint folder name under factr/checkpoints")
    parser.add_argument("--checkpoint", type=str, default="latest", help="Checkpoint name (latest or ckpt_XXXXXX)")
    parser.add_argument("--buffer-path", type=str, required=True, help="Path to buf.pkl")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index in trajectory list")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of sampled trajectories per step")
    parser.add_argument("--stiffness-label", type=int, default=0, help="1/2/3. Use 0 to auto-read from episode.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--output-dir", type=str, default="/home/ferdinand/activeinference/factr/scripts/eval_lowdim_output", help="Output directory")
    return parser.parse_args()


def make_obs_window(states, t, window):
    start = max(0, t - window + 1)
    states_slice = states[start : t + 1]
    if states_slice.shape[0] < window:
        pad = np.repeat(states_slice[:1], repeats=window - states_slice.shape[0], axis=0)
        states_slice = np.concatenate([pad, states_slice], axis=0)
    return states_slice


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


def _extract_label(obs_dict):
    if "stiffness_label" not in obs_dict:
        return 1
    label = int(np.asarray(obs_dict["stiffness_label"]).reshape(-1)[0])
    return max(1, min(3, label))


def plot_pose_trajectories(save_path, true_pose, sampled_first_step):
    num_samples, time_steps, pose_dim = sampled_first_step.shape
    pred_mean = sampled_first_step.mean(axis=0)
    pred_std = sampled_first_step.std(axis=0)
    sample_colors = plt.cm.tab10(np.linspace(0, 1, max(num_samples, 10)))

    dim_names = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True)
    axes = axes.flatten()
    t = np.arange(time_steps)

    for dim in range(pose_dim):
        ax = axes[dim]
        for sample_idx in range(num_samples):
            label = f"sample_{sample_idx + 1}" if dim == 0 else None
            ax.plot(
                t,
                sampled_first_step[sample_idx, :, dim],
                color=sample_colors[sample_idx],
                alpha=0.45,
                linewidth=0.95,
                label=label,
            )

        ax.plot(t, true_pose[:, dim], color="black", linewidth=2.0, label="ground truth")
        ax.plot(t, pred_mean[:, dim], color="#E41A1C", linewidth=1.8, label="sample mean")
        ax.fill_between(
            t,
            pred_mean[:, dim] - pred_std[:, dim],
            pred_mean[:, dim] + pred_std[:, dim],
            color="#FB9A99",
            alpha=0.25,
            label="±1 std",
        )
        ax.set_title(f"Pose Dim {dim + 1} ({dim_names[dim]})")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 0.995), fontsize=8)
    fig.suptitle(
        f"One-Episode Evaluation: Ground Truth vs {num_samples} Sampled Trajectories",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_metrics(save_path, true_pose, sampled_first_step):
    pred_mean = sampled_first_step.mean(axis=0)
    mse_per_sample = np.mean((sampled_first_step - true_pose[None, :, :]) ** 2, axis=(1, 2))
    mse_per_dim = np.mean((pred_mean - true_pose) ** 2, axis=0)
    position_mse = np.mean((pred_mean[:, :3] - true_pose[:, :3]) ** 2)
    rotation6d_mse = np.mean((pred_mean[:, 3:] - true_pose[:, 3:]) ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    axes[0].bar(np.arange(len(mse_per_sample)), mse_per_sample, color="#6BAED6")
    axes[0].set_title("MSE per sampled trajectory")
    axes[0].set_xlabel("sample index")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.25)

    axes[1].bar(np.arange(1, len(mse_per_dim) + 1), mse_per_dim, color="#FB9A99")
    axes[1].set_title("MSE per pose dimension (sample mean)")
    axes[1].set_xlabel("pose dim")
    axes[1].set_ylabel("MSE")
    axes[1].grid(alpha=0.25)

    abs_err = np.abs(pred_mean - true_pose)
    axes[2].plot(abs_err.mean(axis=1), color="black", linewidth=1.7)
    axes[2].set_title("Mean absolute error over time")
    axes[2].set_xlabel("timestep")
    axes[2].set_ylabel("MAE")
    axes[2].grid(alpha=0.25)

    axes[3].bar(["position(3)", "rotation6d(6)"], [position_mse, rotation6d_mse], color=["#74C476", "#FD8D3C"])
    axes[3].set_title("Position vs rotation MSE (sample mean)")
    axes[3].set_ylabel("MSE")
    axes[3].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"

    project_root = Path(__file__).resolve().parents[1]
    checkpoints_dir = project_root / "checkpoints" / args.model_name
    policy, cfg, ckpt_path = load_policy(checkpoints_dir, args.checkpoint, device=device)

    with open(args.buffer_path, "rb") as f:
        traj_list = pickle.load(f)
    if not (0 <= args.episode_index < len(traj_list)):
        raise IndexError(f"episode-index out of range: {args.episode_index}, valid=[0, {len(traj_list)-1}]")

    episode = traj_list[args.episode_index]
    states = np.stack([np.asarray(step[0]["state"], dtype=np.float32).reshape(-1) for step in episode], axis=0)
    actions = np.stack([np.asarray(step[1], dtype=np.float32).reshape(-1) for step in episode], axis=0)
    true_pose = actions[:, : policy.ac_dim]

    stiffness_label = int(args.stiffness_label)
    if stiffness_label <= 0:
        stiffness_label = _extract_label(episode[0][0])

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
            samples = policy.get_actions_prior(
                imgs={},
                obs=obs_tensor,
                class_labels=label_tensor,
                sample=True,
                num_samples=num_samples,
            )
            sample_np = samples.squeeze(0).cpu().numpy()
            sampled_chunks[t] = sample_np
            sampled_first_step[:, t] = sample_np[:, 0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model_name}_ep{args.episode_index:03d}_{Path(ckpt_path).stem}"
    npz_path = output_dir / f"{tag}.npz"
    fig_pose_path = output_dir / f"{tag}_pose_traj.png"
    fig_metric_path = output_dir / f"{tag}_metrics.png"

    np.savez_compressed(
        npz_path,
        sampled_chunks=sampled_chunks,
        sampled_first_step=sampled_first_step,
        true_pose=true_pose,
        stiffness_label=stiffness_label,
        model_name=args.model_name,
        checkpoint=str(ckpt_path),
    )

    plot_pose_trajectories(fig_pose_path, true_pose=true_pose, sampled_first_step=sampled_first_step)
    plot_metrics(fig_metric_path, true_pose=true_pose, sampled_first_step=sampled_first_step)

    mse_per_sample = np.mean((sampled_first_step - true_pose[None, :, :]) ** 2, axis=(1, 2))
    print("=== One-Episode Lowdim Evaluation ===")
    print(f"checkpoint: {ckpt_path}")
    print(f"buffer: {args.buffer_path}")
    print(f"episode_index: {args.episode_index}")
    print(f"stiffness_label: {stiffness_label}")
    print(f"obs_window: {obs_window}, ac_chunk: {chunk}, num_samples: {num_samples}")
    print(f"states shape: {states.shape}, true_pose shape: {true_pose.shape}")
    print(f"sampled_chunks shape: {sampled_chunks.shape}")
    print(f"mse_per_sample: {np.round(mse_per_sample, 6)}")
    print(f"saved npz: {npz_path}")
    print(f"saved figure: {fig_pose_path}")
    print(f"saved figure: {fig_metric_path}")


if __name__ == "__main__":
    main()
