import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pathlib import Path
import sys
import math

# ==================================================================================
# 1. 設定
# ==================================================================================

INPUT_JSON_PATH = Path(
    "/home/otake/FACTR-pr/FACTR-project/result_output/test_beta1_DETRtransformer_z16__ac100/PRIOR_CHECK_stiff/ep_55/prior_inference_data.json"
)
SAVE_DIR = INPUT_JSON_PATH.parent

OUTPUT_JSON_NAME = "entropy_analysis_smooth_excluded.json"
PLOT_FILENAME = "entropy_plot_smooth_mean_excluded.png"

# Chunk内を何ステップ刻みで確認するか
LOOKAHEAD_INTERVAL = 10

# 平滑化係数
SMOOTHING_ALPHA = 0.1

# ★ 除外したいジョイントのインデックス (0-indexed)
# 例: [6] -> J7 (7番目の関節) を平均計算から除外
# 例: [0, 1] -> J1, J2 を除外
# 空リスト [] にすれば全平均と同じになります
EXCLUDE_JOINTS = []

# ==================================================================================
# 2. 関数定義
# ==================================================================================


def apply_ema(data, alpha):
    """指数移動平均"""
    if alpha >= 1.0:
        return data
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def calculate_entropy(variance):
    """分散 -> 微分エントロピー"""
    eps = 1e-6
    variance = np.maximum(variance, eps)
    term = 2 * np.pi * np.e * variance
    entropy = 0.5 * np.log(term)
    return entropy


# ==================================================================================
# 3. メイン処理
# ==================================================================================

if __name__ == "__main__":
    if not INPUT_JSON_PATH.exists():
        print(f"❌ Input JSON not found: {INPUT_JSON_PATH}")
        sys.exit(1)

    print(f"Loading raw inference data from {INPUT_JSON_PATH} ...")
    with open(INPUT_JSON_PATH, "r") as f:
        raw_data = json.load(f)

    episode_name = raw_data["episode_name"]
    data_content = raw_data["data"]

    # -------------------------------------------------------------------------
    # A. 計算処理
    # -------------------------------------------------------------------------
    prior_samples = np.array(data_content["prior_samples_norm"])

    T, S, C, D = prior_samples.shape
    print(f"Episode: {episode_name}, Time: {T}, Chunk: {C}, Dim: {D}")
    print(f"Applying EMA Smoothing with alpha={SMOOTHING_ALPHA}")
    print(f"Excluding joints from mean calculation: {EXCLUDE_JOINTS} (0-indexed)")

    lookahead_indices = list(range(0, C, LOOKAHEAD_INTERVAL))
    if lookahead_indices[-1] != C - 1:
        lookahead_indices.append(C - 1)

    # 計算対象のインデックスリストを作成
    valid_indices = [d for d in range(D) if d not in EXCLUDE_JOINTS]
    if not valid_indices:
        print("⚠️ Warning: All joints are excluded! Excluded Mean will be NaN/Zero.")

    results = {}

    global_max_val = -np.inf
    global_min_val = np.inf

    # Y軸範囲計算用 (全平均と除外平均の両方をカバーするため)
    global_max_mean_all = -np.inf
    global_min_mean_all = np.inf

    for h_idx in lookahead_indices:
        data_at_h = prior_samples[:, :, h_idx, :]

        # 分散 -> エントロピー -> 平滑化
        raw_var = np.var(data_at_h, axis=1, ddof=1)
        raw_entropy = calculate_entropy(raw_var)
        smooth_entropy = apply_ema(raw_entropy, SMOOTHING_ALPHA)

        # 1. 全平均 (All Joints)
        mean_at_h = smooth_entropy.mean(axis=1)  # (T,)

        # 2. 除外平均 (Excluded Mean)
        if valid_indices:
            mean_excluded_at_h = smooth_entropy[:, valid_indices].mean(axis=1)  # (T,)
        else:
            mean_excluded_at_h = np.zeros_like(mean_at_h)

        # 最大最小更新 (Y軸スケール用)
        global_max_val = max(global_max_val, smooth_entropy.max())
        global_min_val = min(global_min_val, smooth_entropy.min())

        # 平均値の最大最小（全平均と除外平均の両方を見る）
        current_max_mean = max(mean_at_h.max(), mean_excluded_at_h.max())
        current_min_mean = min(mean_at_h.min(), mean_excluded_at_h.min())

        global_max_mean_all = max(global_max_mean_all, current_max_mean)
        global_min_mean_all = min(global_min_mean_all, current_min_mean)

        results[h_idx] = {"entropy": smooth_entropy, "mean": mean_at_h, "mean_excluded": mean_excluded_at_h}

    # -------------------------------------------------------------------------
    # B. JSON保存
    # -------------------------------------------------------------------------
    json_output = {
        "episode_name": episode_name,
        "smoothing_alpha": SMOOTHING_ALPHA,
        "excluded_joints": EXCLUDE_JOINTS,
        "lookahead_indices": lookahead_indices,
        "data_by_horizon": {},
    }

    for h_idx, res in results.items():
        json_output["data_by_horizon"][str(h_idx)] = {
            "entropy_per_joint": res["entropy"].tolist(),
            "mean_entropy": res["mean"].tolist(),
            "mean_entropy_excluded": res["mean_excluded"].tolist(),
        }

    with open(SAVE_DIR / OUTPUT_JSON_NAME, "w") as f:
        json.dump(json_output, f, indent=None)
    print(f"✅ JSON Saved to {OUTPUT_JSON_NAME}")

    # -------------------------------------------------------------------------
    # C. 可視化
    # -------------------------------------------------------------------------
    print("Generating plot...")

    timesteps = np.arange(T)
    colors = cm.jet(np.linspace(0, 1, len(lookahead_indices)))

    def get_ylim(min_v, max_v):
        margin = (max_v - min_v) * 0.1
        if margin == 0:
            margin = 1.0
        return (min_v - margin, max_v + margin)

    ylim_joints = get_ylim(global_min_val, global_max_val)
    ylim_mean = get_ylim(global_min_mean_all, global_max_mean_all)

    fig, axes = plt.subplots(8, 1, figsize=(12, 28), sharex=True)
    fig.suptitle(f"Entropy Analysis: Smoothed (Excluded: {EXCLUDE_JOINTS})\n{episode_name}", fontsize=16, y=0.99)

    # --- 1〜7段目: 各関節 ---
    for d in range(7):
        ax = axes[d]
        # 除外されている関節は背景を少しグレーにして視覚的に区別
        if d in EXCLUDE_JOINTS:
            ax.set_facecolor("#f2f2f2")

        for i, h_idx in enumerate(lookahead_indices):
            series = results[h_idx]["entropy"][:, d]
            label = f"t+{h_idx}" if d == 0 else None
            ax.plot(timesteps, series, color=colors[i], linewidth=1.5, alpha=0.8, label=label)

        # タイトルに除外情報を付記
        title_suffix = " [Excluded]" if d in EXCLUDE_JOINTS else ""
        ax.set_ylabel(f"J{d + 1} Entropy{title_suffix}", fontsize=10)
        ax.set_ylim(ylim_joints)
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(loc="upper right", ncol=len(lookahead_indices) // 2 + 1, fontsize="x-small", title="Horizon")

    # --- 8段目: Mean (実線) & Excluded Mean (破線) ---
    ax8 = axes[7]

    for i, h_idx in enumerate(lookahead_indices):
        mean_series = results[h_idx]["mean"]
        mean_ex_series = results[h_idx]["mean_excluded"]

        # 全平均 (実線) - 少し薄くして比較しやすく
        ax8.plot(timesteps, mean_series, color=colors[i], linestyle="-", linewidth=1.2, alpha=0.5)

        # 除外平均 (破線) - こちらを強調
        ax8.plot(timesteps, mean_ex_series, color=colors[i], linestyle="--", linewidth=1.8, alpha=0.9)

    # 凡例
    custom_lines = [
        Line2D([0], [0], color="blue", lw=2),
        Line2D([0], [0], color="red", lw=2),
        Line2D([0], [0], color="black", linestyle="-", lw=1.2, alpha=0.5),
        Line2D([0], [0], color="black", linestyle="--", lw=1.8, alpha=0.9),
    ]

    excluded_names = ",".join([f"J{j + 1}" for j in EXCLUDE_JOINTS])
    custom_labels = ["Near (t+0)", "Far", "All Joints Mean (Solid)", f"Mean w/o {excluded_names} (Dashed)"]

    ax8.legend(custom_lines, custom_labels, loc="upper left", fontsize="small", ncol=2)

    ax8.set_ylabel("Mean Entropy", fontsize=10, fontweight="bold")
    ax8.set_ylim(ylim_mean)
    ax8.set_xlabel("Timestep", fontsize=12)
    ax8.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    save_path = SAVE_DIR / PLOT_FILENAME
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"✅ Saved plot to {save_path}")
