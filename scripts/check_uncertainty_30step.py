import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import sys

# ==================================================================================
# 1. 設定 & パス収集
# ==================================================================================

# ベースとなるディレクトリ
BASE_DIR = Path("/home/otake/FACTR-pr/FACTR-project/result_output/test_beta1_DETRtransformer_z16_ac100_la7")

# 保存先のルート（全ファイルをここに整理して保存します）
SAVE_ROOT = BASE_DIR / "summary_analysis"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# 読み込みたいJSONパスを格納するリスト
json_paths = []

# --- 1. PRIOR_CHECK_stiff (ep_51 ~ ep_61) ---
stiff_dir = BASE_DIR / "PRIOR_CHECK_stiff"
for i in range(51, 62):
    path = stiff_dir / f"ep_{i:02d}" / "prior_inference_data.json"
    json_paths.append(path)

# --- 2. PRIOR_CHECK_soft (ep_51 ~ ep_62) ---
soft_dir = BASE_DIR / "PRIOR_CHECK_soft"
for i in range(51, 63):
    path = soft_dir / f"ep_{i:02d}" / "prior_inference_data.json"
    json_paths.append(path)

print(f"対象ファイル数: {len(json_paths)}")

# パラメータ設定
OUTPUT_JSON_NAME = "variance_analysis_smooth.json"
PLOT_FILENAME = "variance_plot_smooth_mean_only.png"
PLOT_MEAN_FILENAME = "variance_mean_only.png"  # ★追加: 平均のみプロットのファイル名
LOOKAHEAD_INTERVAL = 10
SMOOTHING_ALPHA = 0.1

# ==================================================================================
# 2. 関数定義
# ==================================================================================


def apply_ema(data, alpha):
    """指数移動平均 (Exponential Moving Average)"""
    if alpha >= 1.0:
        return data
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def process_single_episode(json_path, save_root):
    """1つのJSONファイルを処理して保存する関数"""

    if not json_path.exists():
        print(f"⚠️ File not found: {json_path}")
        return

    # モード(stiff/soft) と エピソード名を取得して保存先フォルダを作成
    mode = "stiff" if "stiff" in str(json_path) else "soft"
    ep_name = json_path.parent.name

    # 保存先: summary_analysis/stiff/ep_51/
    current_save_dir = save_root / mode / ep_name
    current_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {mode} - {ep_name} ...")

    # データの読み込み
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    episode_name_in_data = raw_data.get("episode_name", ep_name)
    data_content = raw_data["data"]

    # -----------------------------
    # A. 計算処理
    # -----------------------------
    prior_samples = np.array(data_content["prior_samples_norm"])
    T, S, C, D = prior_samples.shape

    lookahead_indices = list(range(0, C, LOOKAHEAD_INTERVAL))
    if lookahead_indices[-1] != C - 1:
        lookahead_indices.append(C - 1)

    results = {}
    global_max_var = 0.0
    global_max_mean = 0.0

    for h_idx in lookahead_indices:
        data_at_h = prior_samples[:, :, h_idx, :]
        raw_var = np.var(data_at_h, axis=1, ddof=1)
        smooth_var = apply_ema(raw_var, SMOOTHING_ALPHA)

        mean_at_h = smooth_var.mean(axis=1)
        max_at_h = smooth_var.max(axis=1)

        global_max_var = max(global_max_var, smooth_var.max())
        global_max_mean = max(global_max_mean, mean_at_h.max())

        results[h_idx] = {"var": smooth_var, "mean": mean_at_h, "max": max_at_h}

    # -----------------------------
    # B. JSON保存
    # -----------------------------
    json_output = {
        "episode_name": episode_name_in_data,
        "mode": mode,
        "smoothing_alpha": SMOOTHING_ALPHA,
        "lookahead_indices": lookahead_indices,
        "data_by_horizon": {},
    }

    for h_idx, res in results.items():
        json_output["data_by_horizon"][str(h_idx)] = {
            "variance_per_joint": res["var"].tolist(),
            "mean_variance": res["mean"].tolist(),
            "max_joint_variance": res["max"].tolist(),
        }

    with open(current_save_dir / OUTPUT_JSON_NAME, "w") as f:
        json.dump(json_output, f, indent=None)

    # -----------------------------
    # C. 可視化 (全体: 8段)
    # -----------------------------
    timesteps = np.arange(T)
    colors = cm.jet(np.linspace(0, 1, len(lookahead_indices)))

    ylim_joints = (0, global_max_var * 1.1)
    ylim_mean = (0, global_max_mean * 1.1)

    fig, axes = plt.subplots(8, 1, figsize=(12, 28), sharex=True)
    fig.suptitle(f"Variance Analysis ({mode}/{ep_name})", fontsize=16, y=0.99)

    # 1〜7段目: 各関節
    for d in range(7):
        ax = axes[d]
        for i, h_idx in enumerate(lookahead_indices):
            series = results[h_idx]["var"][:, d]
            label = f"t+{h_idx}" if d == 0 else None
            ax.plot(timesteps, series, color=colors[i], linewidth=1.5, alpha=0.8, label=label)

        ax.set_ylabel(f"J{d + 1} Var", fontsize=10)
        ax.set_ylim(ylim_joints)
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(loc="upper right", ncol=len(lookahead_indices) // 2 + 1, fontsize="x-small", title="Horizon")

    # 8段目: Mean Only
    ax8 = axes[7]
    for i, h_idx in enumerate(lookahead_indices):
        mean_series = results[h_idx]["mean"]
        ax8.plot(timesteps, mean_series, color=colors[i], linestyle="-", linewidth=1.5, alpha=0.9)

    ax8.set_ylabel("Mean Variance", fontsize=10, fontweight="bold")
    ax8.set_ylim(ylim_mean)
    ax8.set_xlabel("Timestep", fontsize=12)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    save_path = current_save_dir / PLOT_FILENAME
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved full plot: {save_path}")

    # -----------------------------
    # D. 追加: Meanだけのプロット
    # -----------------------------
    fig_mean, ax_mean = plt.subplots(figsize=(10, 6))

    for i, h_idx in enumerate(lookahead_indices):
        mean_series = results[h_idx]["mean"]
        label = f"t+{h_idx}"  # 凡例用ラベル
        ax_mean.plot(timesteps, mean_series, color=colors[i], linestyle="-", linewidth=1.0, alpha=0.9, label=label)

    ax_mean.set_title(f"Mean Variance Analysis", fontsize=14)
    ax_mean.set_ylabel("Mean Variance", fontsize=12)
    ax_mean.set_xlabel("Timestep", fontsize=12)
    ax_mean.set_ylim(ylim_mean)  # Y軸スケールを統一
    ax_mean.grid(True, alpha=0.3)
    ax_mean.legend(loc="upper right", title="Horizon")

    mean_save_path = current_save_dir / PLOT_MEAN_FILENAME
    plt.savefig(mean_save_path, dpi=150)
    plt.close(fig_mean)
    print(f"✅ Saved mean-only plot: {mean_save_path}")


# ==================================================================================
# 3. メイン実行ループ
# ==================================================================================

if __name__ == "__main__":
    if not json_paths:
        print("❌ No files found to process.")
        sys.exit(1)

    print(f"Start processing {len(json_paths)} files...")

    for path in json_paths:
        process_single_episode(path, SAVE_ROOT)

    print("\n🎉 All processing complete!")
