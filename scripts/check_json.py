import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch
import pickle
import sys

# ==================================================================================
# 1. 設定 & パス
# ==================================================================================

# 読み込むJSONファイルのパス (ユーザー指定)
JSON_PATH = Path(
    "/home/otake/FACTR-pr/FACTR-project/result_output/test_beta1_DETRtransformer_z16__ac100/PRIOR_CHECK_stiff/ep_51/prior_inference_data.json"
)

# 保存先 (JSONと同じ場所に _reconstructed.png として保存)
SAVE_DIR = JSON_PATH.parent

# 参照軌道(Reference)用の元データパス (もし環境になければスキップされます)
RAW_DATA_BASE = Path("/data/otake/box_lift_up_side/20251218_stiff/data")
RAW_DATA_BASE_2 = Path("/data/otake/box_lift_up_side/20251217_soft/data")

NUM_SAMPLES = 10  # JSON保存時と同じ値

# ==================================================================================
# 2. 関数定義
# ==================================================================================


def load_reference_trajectories(data_dir, action_stats, device="cpu"):
    """参照軌道の読み込み (元コードと同じロジック)"""
    if not data_dir.exists():
        return []

    pkl_files = sorted(Path(data_dir).glob("*.pkl"))
    if not pkl_files:
        return []

    ref_actions_norm = []
    # Numpy配列として受け取る想定
    stat_a = action_stats["mean"]
    stat_b = action_stats["std"]

    for pkl_file in tqdm(pkl_files, desc=f"Loading Refs from {data_dir.name}"):
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            action_topic = "/joint_impedance_dynamic_gain_controller/joint_impedance_command"
            if "data" not in data or action_topic not in data["data"]:
                continue

            raw_data = data["data"][action_topic]
            if isinstance(raw_data, dict) and "position" in raw_data:
                actions = raw_data["position"]
            elif isinstance(raw_data, list):
                actions = []
                for v in raw_data:
                    if isinstance(v, dict) and "position" in v:
                        actions.append(v["position"])
            else:
                actions = []

            if len(actions) == 0:
                continue

            actions = np.array(actions)
            # 正規化
            act_norm = (actions - stat_a) / stat_b
            ref_actions_norm.append(act_norm)

        except Exception:
            continue
    return ref_actions_norm


# ==================================================================================
# 3. メイン処理
# ==================================================================================

if __name__ == "__main__":
    if not JSON_PATH.exists():
        print(f"❌ JSON file not found: {JSON_PATH}")
        sys.exit(1)

    print(f"Loading JSON from {JSON_PATH} ...")
    with open(JSON_PATH, "r") as f:
        json_data = json.load(f)

    episode_name = json_data["episode_name"]
    data_content = json_data["data"]

    # --- 1. データの復元 (List -> Numpy) ---
    print("Restoring numpy arrays...")

    # GT Action (Time, Dim)
    true_actions = np.array(data_content["gt_actions_norm"])

    # Prior Mean (Time, Chunk, Dim)
    pred_actions_mean = np.array(data_content["prior_pred_mean_norm"])

    # Prior Samples (Time, Samples, Chunk, Dim)
    samples_arr = np.array(data_content["prior_samples_norm"])

    # Torque (Time, 7) - 今回のプロットには使わないが一応ロード
    torque_obs = np.array(data_content["torque_input_raw"])

    # 統計量 (参照軌道の正規化用)
    stats = data_content["stats"]
    action_stats_dict = {"mean": np.array(stats["action_mean"]), "std": np.array(stats["action_std"])}

    # --- 2. 参照軌道のロード (Optional) ---
    print("Loading references (if available)...")
    refs_solid = load_reference_trajectories(RAW_DATA_BASE, action_stats_dict)
    refs_dotted = load_reference_trajectories(RAW_DATA_BASE_2, action_stats_dict)

    # --- 3. プロットの作成 (元のロジックを再現) ---
    print(f"Generating plot for {episode_name}...")

    t = np.arange(len(true_actions))
    chunk_size_actual = pred_actions_mean.shape[1]

    # 7次元分のサブプロット
    fig1, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    fig1.suptitle(f"[Reconstructed] GT vs Prior Generation - {episode_name}", fontsize=16, y=0.98)

    for d in range(7):
        ax = axes[d]

        # Reference (Gray)
        for ref_act in refs_solid:
            t_ref = np.arange(len(ref_act))
            ax.plot(t_ref, ref_act[:, d], color="gray", linewidth=0.5, alpha=0.3, linestyle="-")
        for ref_act in refs_dotted:
            t_ref = np.arange(len(ref_act))
            ax.plot(t_ref, ref_act[:, d], color="gray", linewidth=0.5, alpha=0.3, linestyle=":")

        # --- 1. Original GT (赤色) ---
        ax.plot(t, true_actions[:, d], color="red", linestyle="--", linewidth=2.0, alpha=0.6, label="Original GT")

        # --- 2. Prior Samples (水色) ---
        # 重いため、少し間引くか透明度を下げるなどの調整をしてもよいが、再現のためそのまま
        if samples_arr is not None:
            for step_i in range(chunk_size_actual):
                # 全サンプル描画すると重いので、最初のサンプルだけ描画するなどの軽量化も可能
                # ここでは元のロジック通り全サンプル描画
                for s_idx in range(samples_arr.shape[1]):  # NumSamples
                    lbl = "Prior Sample" if (step_i == 0 and s_idx == 0) else None

                    # データの形状チェック
                    # samples_arr: (Time, NumSamples, Chunk, Dim)
                    # Plot: X軸= t + step_i, Y軸= samples_arr[:, s_idx, step_i, d]
                    ax.plot(
                        t + step_i, samples_arr[:, s_idx, step_i, d], color="cyan", linewidth=0.5, alpha=0.15, label=lbl
                    )

        # --- 3. Prior Mean (青色) ---
        for step_i in range(chunk_size_actual):
            if step_i == 0:
                ax.plot(
                    t + step_i,
                    pred_actions_mean[:, step_i, d],
                    color="blue",
                    linewidth=1.2,
                    alpha=0.9,
                    label="Prior Mean (Step 0)",
                )
            else:
                ax.plot(t + step_i, pred_actions_mean[:, step_i, d], color="blue", linewidth=0.5, alpha=0.2)

        ax.set_ylabel(f"J{d + 1} (norm)")
        # 範囲は正規化済みデータに合わせて調整
        ax.set_ylim(-4.0, 4.0)
        ax.grid(True, alpha=0.3)
        if d == 0:
            # 凡例の重複削除
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", ncol=2, fontsize="small")

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    # 保存
    save_path = SAVE_DIR / "prior_generation_check_reconstructed.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig1)

    print(f"✅ Saved reconstructed plot to {save_path}")
