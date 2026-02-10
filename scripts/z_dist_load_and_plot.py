import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import argparse
from omegaconf import OmegaConf
from hydra.utils import instantiate


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


def visualize_variance_gains_with_frames(
    ep_name, entropy_ts, entropy_vals, gains_ts, gains_vals, raw_images, output_dir, Hz=30
):
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


def plot_variance_with_frames(z_pkl_path, raw_images, save_dir):
    """1段目に分散プロット、2段目に画像を配置する"""
    with open(z_pkl_path, "rb") as f:
        dists_data = pickle.load(f)

    post_var = np.maximum(dists_data["Posterior"][1] ** 2, 1e-10)
    prior_var = np.maximum(dists_data["Prior"][1] ** 2, 1e-10)
    T, Dz = post_var.shape
    time_steps = np.arange(T)
    post_mean_var = np.mean(post_var, axis=1)
    prior_mean_var = np.mean(prior_var, axis=1)

    num_imgs = 8
    target_steps = [150, 175, 200, 225, 250, 275, 300, 325]

    # 図の作成
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, num_imgs, height_ratios=[1.5, 1.0], hspace=0.08, wspace=0.1)

    # --- 1段目: Variance Plot ---
    ax_main = fig.add_subplot(gs[0, :])
    for d in range(Dz):
        if d == 0:
            ax_main.plot(time_steps, post_var[:, d], color="red", alpha=0.3, linewidth=0.5, label="Individual (Post)")
            ax_main.plot(
                time_steps, prior_var[:, d], color="blue", alpha=0.3, linewidth=0.5, label="Individual (Prior)"
            )
        else:
            ax_main.plot(time_steps, post_var[:, d], color="red", alpha=0.3, linewidth=0.5)
            ax_main.plot(time_steps, prior_var[:, d], color="blue", alpha=0.3, linewidth=0.5)

    ax_main.plot(time_steps, post_mean_var, color="red", linewidth=2.5, label="Mean Variance (Posterior)")
    ax_main.plot(time_steps, prior_mean_var, color="blue", linewidth=2.5, label="Mean Variance (Prior)")

    # 垂直線を引いて画像との対応を明確にする
    for step in target_steps:
        if step < T:
            ax_main.axvline(x=step, color="gray", linestyle="--", alpha=0.4)

    ax_main.set_yscale("log")
    ax_main.set_title(f"Z-Variance & Observations", fontsize=15)
    ax_main.set_ylabel("Variance (Log Scale)", fontsize=12)
    ax_main.set_xlabel("Timestep", fontsize=12, labelpad=5)
    ax_main.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_main.legend(loc="upper left", frameon=True, fontsize="small", ncol=2)

    # --- 2段目: Images ---
    for i, step in enumerate(target_steps):
        img_ax = fig.add_subplot(gs[1, i])
        if step < len(raw_images):
            img_ax.imshow(raw_images[step])
            img_ax.set_title(f"step={step}", fontsize=12)
        else:
            img_ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        img_ax.axis("off")

    # 保存
    save_path = save_dir / f"analysis_with_frames_{z_pkl_path.stem}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return np.max(post_mean_var), np.min(post_mean_var)


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

    print(f"🔍 Found {len(pkl_files)} pkl files. Starting analysis...")

    all_max, all_min = [], []

    for pkl_path in pkl_files:
        print(f"Processing {pkl_path.name}...")

        # 1. 対応する raw_images をロード
        ep_id = pkl_path.stem.split("ep_")[-1]
        raw_pkl_path = RAW_DATA_ROOT / f"ep_{ep_id}.pkl"

        raw_images, _, _ = load_and_extract_raw_data(raw_pkl_path)

        if len(raw_images) == 0:
            print(f"⚠️ Could not find images for ep_{ep_id}, skipping image subplot.")

        # 2. 統合プロットを作成
        ma, mi = plot_variance_with_frames(pkl_path, raw_images, TARGET_DIR)

        all_max.append(ma)
        all_min.append(mi)

    print("\n" + "=" * 40)
    print(f"📊 Overall Statistics for {args.model_name} ({args.mode})")
    print(f"Average Max Variance: {np.mean(all_max):.6f}")
    print(f"Global Max Variance:  {np.max(all_max):.6f}")
    print("=" * 40)
    print(f"✅ All plots saved to: {TARGET_DIR}")
