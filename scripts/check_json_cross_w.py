import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm

# ==================================================================================
# 1. 設定
# ==================================================================================
BASE_DIR = Path("/home/otake/FACTR-pr/FACTR-project/result_output/test_beta1_DETRtransformer_z16_ac100_la7")
SAVE_ROOT = BASE_DIR / "summary_analysis_attention_ratio_stretched"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

TARGET_EPISODES = range(51, 61)

IDX_FORCE = 1
IDX_IMG_START = 2

SMOOTHING_ALPHA = 0.05

# ==================================================================================
# 2. 関数定義
# ==================================================================================


def apply_ema(data, alpha):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def process_attention_ratio_standard_style(pkl_path, save_root):
    if not pkl_path.exists():
        return

    mode = "stiff" if "stiff" in str(pkl_path) else "soft"
    ep_name = pkl_path.parent.name
    current_save_dir = save_root / mode / ep_name
    current_save_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    attn_raw = data["data"].get("cross_attention", None)
    if attn_raw is None:
        return

    ndim = attn_raw.ndim
    if ndim == 6:
        attn_mean = np.mean(attn_raw, axis=(2, 3))
    elif ndim == 5:
        attn_mean = np.mean(attn_raw, axis=2)
    else:
        return

    attn_t_mem = np.mean(attn_mean[:, -1, :, :], axis=1)

    # 1. 比率計算
    val_force = attn_t_mem[:, IDX_FORCE]
    val_vision = np.sum(attn_t_mem[:, IDX_IMG_START:], axis=1)
    total_val = val_force + val_vision + 1e-9

    ratio_force_raw = val_force / total_val
    ratio_force_smooth = apply_ema(ratio_force_raw, SMOOTHING_ALPHA)

    # 2. Min-Max 正規化
    r_min = np.min(ratio_force_smooth)
    r_max = np.max(ratio_force_smooth)

    if (r_max - r_min) < 1e-6:
        ratio_force_stretched = ratio_force_smooth
    else:
        ratio_force_stretched = (ratio_force_smooth - r_min) / (r_max - r_min)

    ratio_vision_stretched = 1.0 - ratio_force_stretched

    # -------------------------------------------------------
    # 3. プロット（標準スタイルに戻す）
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))  # サイズも標準的に

    t_steps = np.arange(len(ratio_force_stretched))

    # 色は維持: Vision=オレンジ, Force=青
    ax.plot(t_steps, ratio_vision_stretched, label="Force", color="#1E90FF", linewidth=2.0)
    ax.plot(t_steps, ratio_force_stretched, label="Vision", color="#FF8C00", linewidth=2.0)

    # 軸設定 (シンプルに)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, len(t_steps))

    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Normalized Attention")
    ax.set_title(f"Cross-Attention vision vs force ratio")

    # グリッドと凡例 (標準位置)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")  # 以前のように右上などに配置

    plt.tight_layout()

    save_path = current_save_dir / "attention_ratio_standard.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ==================================================================================
# 3. 実行
# ==================================================================================

pkl_paths = []
for i in TARGET_EPISODES:
    pkl_paths.append(BASE_DIR / "PRIOR_CHECK_stiff" / f"ep_{i:02d}" / "prior_inference_data.pkl")
    pkl_paths.append(BASE_DIR / "PRIOR_CHECK_soft" / f"ep_{i:02d}" / "prior_inference_data.pkl")

print(f"Processing {len(pkl_paths)} files...")

for path in tqdm(pkl_paths):
    process_attention_ratio_standard_style(path, SAVE_ROOT)

print("\n✅ Done!")
