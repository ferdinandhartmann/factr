import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import sys

# ==================================================================================
# 1. 設定
# ==================================================================================

JSON_PATH = Path(
    "/home/otake/FACTR-pr/FACTR-project/result_output/test_beta1_DETRtransformer_z16__ac100/PRIOR_CHECK_stiff/ep_51/prior_inference_data.json"
)
SAVE_DIR = JSON_PATH.parent

# 刻み幅の設定 (30ステップごと)
STEP_INTERVAL = 30
Y_LIMIT = (-4, 4)

# ==================================================================================
# 2. メイン処理
# ==================================================================================

if __name__ == "__main__":
    if not JSON_PATH.exists():
        print(f"❌ JSON file not found: {JSON_PATH}")
        sys.exit(1)

    print(f"Loading JSON from {JSON_PATH} ...")
    with open(JSON_PATH, "r") as f:
        json_data = json.load(f)

    episode_name = json_data["episode_name"]
    data = json_data["data"]

    # データをNumpy配列に戻す
    prior_samples = np.array(data["prior_samples_norm"])  # (Time, Samples, Chunk, Dim)
    gt_actions = np.array(data["gt_actions_norm"])  # (Time, Dim)

    total_timesteps = gt_actions.shape[0]
    num_samples = prior_samples.shape[1]
    chunk_size = prior_samples.shape[2]

    print(f"Episode Length: {total_timesteps}, Chunk: {chunk_size}")

    # ★★★ ここでタイムステップを自動生成 ★★★
    # 0, 30, 60, ... とエピソードが終わるまで作成
    target_timesteps = list(range(0, total_timesteps, STEP_INTERVAL))
    print(f"Target Timesteps: {target_timesteps}")

    # ==========================================================================
    # Plot作成
    # ==========================================================================

    fig, axes = plt.subplots(7, 1, figsize=(12, 22), sharex=True)
    fig.suptitle(f"Trajectory Fan-out (Every {STEP_INTERVAL} Steps)\n{episode_name}", fontsize=16, y=0.99)

    # カラーマップの生成 (本数が増えたので、jetで虹色に変化させます)
    colors = cm.jet(np.linspace(0, 1, len(target_timesteps)))

    for d in range(7):
        ax = axes[d]

        # 1. Ground Truth (黒色・点線)
        ax.plot(gt_actions[:, d], color="black", linewidth=2.0, linestyle=":", alpha=0.4, label="GT", zorder=10)

        # 2. 30ステップごとのループ
        for i, t_start in enumerate(target_timesteps):
            # 念のためインデックス超過チェック
            if t_start >= total_timesteps:
                continue

            current_color = colors[i]

            # データ取得
            current_preds = prior_samples[t_start, :, :, d]
            t_range = np.arange(t_start, t_start + chunk_size)

            # サンプルを描画
            for s_idx in range(num_samples):
                # 凡例は「最初のサンプルの、最初の1本」だけ
                if s_idx == 0:
                    label = f"t={t_start}"
                else:
                    label = None

                ax.plot(
                    t_range,
                    current_preds[s_idx],
                    color=current_color,
                    linewidth=1.0,
                    alpha=0.5,  # 本数が増えるので少し薄くしました
                    label=label,
                )

            # 開始点マーカー
            ax.scatter([t_start], [gt_actions[t_start, d]], color=current_color, s=30, edgecolors="black", zorder=5)

        ax.set_ylabel(f"Joint {d + 1}", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Y軸範囲調整
        y_min, y_max = gt_actions[:, d].min(), gt_actions[:, d].max()
        margin = (y_max - y_min) * 0.5
        ax.set_ylim(Y_LIMIT)

        # 凡例の設定 (本数が多いので列数を増やして小さく表示)
        if d == 0:
            # 凡例の列数を自動調整 (最大6列くらい)
            ncols = min(6, len(target_timesteps) // 2 + 1)
            ax.legend(loc="upper right", ncol=ncols, fontsize="x-small", title="Start Time")

    axes[-1].set_xlabel("Timestep", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    save_path = SAVE_DIR / f"fanout_trajectory_{STEP_INTERVAL}step.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"✅ Saved visualization to {save_path}")
