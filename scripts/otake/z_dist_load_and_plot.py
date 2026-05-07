import argparse
import pickle
from pathlib import Path
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from factr.utils_plot import (
    build_z_categorical_distribution_with_frames_figure,
    build_z_gaussian_variance_with_frames_figure,
)

# Top-level override flag:
# - None: auto-read latent distribution from YAML
# - "gaussian" / "categorical": force plot mode
LATENT_DISTRIBUTION_OVERRIDE = None


def load_and_extract_raw_data(pkl_path: Path):
    """生データのpklから画像、トルク、アクションを抽出する"""
    if not pkl_path.exists():
        print(f"❌ File not found: {pkl_path}")
        return [], [], []

    print(f"Loading raw data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        raw_data = pickle.load(f)

    image_obs, torque_obs, actions = [], [], []
    entries = raw_data["data"] if "data" in raw_data else raw_data

    image_topic = "/realsense/front/im"
    obs_topic = "/franka_robot_state_broadcaster/external_joint_torques"
    possible_action_topics = [
        "/joint_impedance_dynamic_gain_controller/joint_impedance_command",
        "/joint_impedance_command_controller/joint_trajectory",
    ]

    action_topic = next((t for t in possible_action_topics if t in entries), None)
    if action_topic:
        for v in entries[action_topic]:
            if isinstance(v, dict) and "position" in v:
                actions.append(v["position"])

    if image_topic in entries:
        for v in entries[image_topic]:
            if isinstance(v, dict) and "data" in v:
                try:
                    if isinstance(v["data"], bytes):
                        nparr = np.frombuffer(v["data"], np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img = np.frombuffer(v["data"], dtype=np.uint8).reshape((v["height"], v["width"], -1))
                    image_obs.append(img)
                except:
                    pass

    if obs_topic in entries:
        for v in entries[obs_topic]:
            if isinstance(v, dict) and "effort" in v:
                torque_obs.append(v["effort"])

    N = min(len(image_obs), len(torque_obs), len(actions))
    return image_obs[:N], torque_obs[:N], actions[:N]


def visualize_variance_gains_with_frames(ep_name, entropy_ts, entropy_vals, gains_ts, gains_vals, raw_images, output_dir, Hz=30):
    """
    Uncertainty(Variance), Gains, and Key Frames を1つの図にプロットする
    """
    try:
        num_imgs = 8
        target_steps = [50, 100, 150, 200, 250, 300, 350, 400]

        # Figureの設定 (横長にして画像を見やすくする)
        fig = plt.figure(figsize=(22, 12))

        # GridSpecでレイアウトを定義
        # 1段目: Variance, 2段目: Gains, 3段目: Images (比率 1:1:1.5)
        gs = fig.add_gridspec(3, num_imgs, height_ratios=[1.0, 1.0, 1.5], hspace=0.4, wspace=0.1)

        # --- 1段目: Variance (全体) ---
        ax_var = fig.add_subplot(gs[0, :])
        ax_var.plot(entropy_ts * Hz, entropy_vals, color="tab:blue", linewidth=2.0, label="Z-Variance (Mean)")
        ax_var.set_ylabel("Variance", fontsize=14)
        ax_var.set_title(f"Analysis for {ep_name}", fontsize=16, pad=20)
        ax_var.grid(True, alpha=0.3)
        ax_var.set_xlim(0, 450)
        ax_var.legend(loc="upper right")

        # --- 2段目: Gains (全体) ---
        ax_gain = fig.add_subplot(gs[1, :])
        ax_gain.plot(gains_ts * Hz, gains_vals, color="tab:red", linewidth=2.0, label="Compliance Gains [%]")
        ax_gain.set_ylabel("Gains [%]", fontsize=14)
        ax_gain.set_xlabel("Timestep", fontsize=14)
        ax_gain.set_ylim(-5, 105)
        ax_gain.set_xlim(0, 450)
        ax_gain.grid(True, alpha=0.3)
        ax_gain.legend(loc="upper right")

        # --- 3段目: 画像 (target_stepsごとに配置) ---
        # raw_images は load_and_extract_raw_data で取得した [RGB, RGB, ...] のリスト
        for i, step in enumerate(target_steps):
            img_ax = fig.add_subplot(gs[2, i])

            # ステップが画像枚数の範囲内かチェック
            if step < len(raw_images):
                img_ax.imshow(raw_images[step])
                img_ax.set_title(f"step={step}", fontsize=13, pad=10)
            else:
                img_ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                img_ax.set_title(f"step={step} (Out of range)", fontsize=10)

            img_ax.axis("off")

        # 保存
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"analysis_combined_{ep_name}.png"

        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  ✅ Summary plot saved to: {save_path}")

    except Exception as e:
        print(f"  ❌ Failed to create summary plot: {e}")


def _extract_latent_distribution_from_cfg(cfg_obj: Any) -> Optional[str]:
    if not isinstance(cfg_obj, dict):
        return None

    latent = cfg_obj.get("latent_distribution", None)
    if isinstance(latent, str) and latent.strip():
        return latent.strip().lower()

    agent_cfg = cfg_obj.get("agent", None)
    if isinstance(agent_cfg, dict):
        latent = agent_cfg.get("latent_distribution", None)
        if isinstance(latent, str) and latent.strip():
            return latent.strip().lower()

    return None


def resolve_latent_distribution(project_root: Path, model_name: str) -> str:
    if LATENT_DISTRIBUTION_OVERRIDE is not None:
        forced = str(LATENT_DISTRIBUTION_OVERRIDE).strip().lower()
        print(f"ℹ️ Using forced latent_distribution='{forced}' from LATENT_DISTRIBUTION_OVERRIDE.")
        return forced

    candidate_paths = [
        project_root / "checkpoints" / model_name / "rollout" / "agent_config.yaml",
        project_root / "checkpoints" / model_name / "rollout" / "exp_config.yaml",
        project_root / "checkpoints" / model_name / ".hydra" / "config.yaml",
    ]

    for cfg_path in candidate_paths:
        if not cfg_path.exists():
            continue
        try:
            with open(cfg_path, "r") as f:
                cfg_obj = yaml.safe_load(f)
            latent = _extract_latent_distribution_from_cfg(cfg_obj)
            if latent in ("gaussian", "categorical"):
                print(f"ℹ️ Detected latent_distribution='{latent}' from {cfg_path}")
                return latent
        except Exception as exc:
            print(f"⚠️ Failed to read {cfg_path}: {exc}")

    print("⚠️ Could not detect latent_distribution from YAML. Falling back to 'gaussian'.")
    return "gaussian"


def _guess_latent_distribution_from_payload(dists_data: Any) -> Optional[str]:
    if not isinstance(dists_data, dict) or "Prior" not in dists_data:
        return None

    prior = dists_data["Prior"]
    if isinstance(prior, dict):
        if "logits" in prior or "probs" in prior or "probabilities" in prior:
            return "categorical"
        if "std" in prior or "mean" in prior or "mu" in prior:
            return "gaussian"

    if isinstance(prior, (tuple, list)) and len(prior) > 0:
        first = np.asarray(prior[0])
        if first.ndim == 3:
            return "categorical"
        if len(prior) > 1 and np.asarray(prior[1]).ndim == 2:
            return "gaussian"

    return None


def plot_latent_with_frames(z_pkl_path, raw_images, save_dir, latent_distribution):
    """Plot latent statistics with key observation frames."""
    with open(z_pkl_path, "rb") as f:
        dists_data = pickle.load(f)

    payload_guess = _guess_latent_distribution_from_payload(dists_data)
    if payload_guess is not None and payload_guess != latent_distribution:
        print(f"⚠️ Plot mode mismatch: YAML says '{latent_distribution}', but PKL looks like '{payload_guess}'. Using PKL mode.")
        latent_distribution = payload_guess

    if latent_distribution == "categorical":
        fig, stats = build_z_categorical_distribution_with_frames_figure(
            dists_data=dists_data,
            raw_images=raw_images,
        )
    else:
        fig, stats = build_z_gaussian_variance_with_frames_figure(
            dists_data=dists_data,
            raw_images=raw_images,
        )

    stats = dict(stats)
    stats["latent_distribution"] = latent_distribution
    save_path = save_dir / f"analysis_with_frames_{z_pkl_path.stem}.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default="stiff", choices=["soft", "stiff"])
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    TARGET_DIR = PROJECT_ROOT / "result_output" / args.model_name / f"{args.mode}_{args.num_samples}"

    if not TARGET_DIR.exists():
        print(f"❌ Directory not found: {TARGET_DIR}")
        exit()

    pkl_files = sorted(list(TARGET_DIR.glob("z_raw_data_ep_*.pkl")))

    if not pkl_files:
        print(f"❌ No pkl files found in {TARGET_DIR}")
        exit()

    # 生データ(画像)のパス設定
    if args.mode == "stiff":
        RAW_DATA_ROOT = Path("/data/otake/box_lift_up_side/20251218_stiff/eval")
    else:
        RAW_DATA_ROOT = Path("/data/otake/box_lift_up_side/20251217_soft/eval")

    latent_distribution = resolve_latent_distribution(PROJECT_ROOT, args.model_name)
    print(f"🔍 Found {len(pkl_files)} pkl files. Starting analysis...")
    print(f"📌 Plotting latent distribution mode: {latent_distribution}")

    all_stats = []

    for pkl_path in pkl_files:
        print(f"Processing {pkl_path.name}...")

        # 1. 対応する raw_images をロード
        ep_id = pkl_path.stem.split("ep_")[-1]
        raw_pkl_path = RAW_DATA_ROOT / f"ep_{ep_id}.pkl"

        raw_images, _, _ = load_and_extract_raw_data(raw_pkl_path)

        if len(raw_images) == 0:
            print(f"⚠️ Could not find images for ep_{ep_id}, skipping image subplot.")

        # 2. 統合プロットを作成
        stats = plot_latent_with_frames(pkl_path, raw_images, TARGET_DIR, latent_distribution=latent_distribution)
        all_stats.append(stats)

    print("\n" + "=" * 40)
    print(f"📊 Overall Statistics for {args.model_name} ({args.mode})")
    if all_stats and all_stats[0].get("latent_distribution") == "categorical":
        prior_entropy = [float(s["prior_entropy_mean"]) for s in all_stats if "prior_entropy_mean" in s]
        post_entropy = [float(s["post_entropy_mean"]) for s in all_stats if "post_entropy_mean" in s]
        if prior_entropy and post_entropy:
            print(f"Average Prior Entropy:     {np.mean(prior_entropy):.6f}")
            print(f"Average Posterior Entropy: {np.mean(post_entropy):.6f}")
    else:
        all_max = [float(s["post_mean_var_max"]) for s in all_stats if "post_mean_var_max" in s]
        if all_max:
            print(f"Average Max Variance: {np.mean(all_max):.6f}")
            print(f"Global Max Variance:  {np.max(all_max):.6f}")
    print("=" * 40)
    print(f"✅ All plots saved to: {TARGET_DIR}")
