import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path

# =========================================================
# 設定: 色とスタイルの定義
# =========================================================
STYLE_CONFIG = {
    # 個別のエピソード (薄い線)
    "prior_indiv": {"color": "tab:blue", "alpha": 0.15, "linewidth": 0.5, "label": "Prior (Indiv)"},
    "posterior_indiv": {"color": "tab:red", "alpha": 0.15, "linewidth": 0.5, "label": "Posterior (Indiv)"},
    # 全エピソードの平均 (太い線)
    "prior_mean": {"color": "blue", "alpha": 1.0, "linewidth": 2.5, "label": "Prior (Mean)"},
    "posterior_mean": {"color": "red", "alpha": 1.0, "linewidth": 2.5, "label": "Posterior (Mean)"},
}


def load_prior_posterior_data(model_dir, tasks):
    """指定された全タスクのpklから PriorとPosteriorの分散データを抽出してまとめる"""
    collected_data = {"prior": [], "posterior": []}

    print(f"Loading data from {model_dir} ...")

    total_files = 0
    for mode_name, dir_name, start_ep, end_ep in tasks:
        target_dir = model_dir / dir_name

        if not target_dir.exists():
            print(f"⚠️ Directory not found: {target_dir}")
            continue

        target_episodes = range(start_ep, end_ep + 1)

        for ep_num in target_episodes:
            # ファイル検索パターン
            patterns = [
                f"z_raw_data_ep_{ep_num}.pkl",
                f"z_raw_data_ep_{ep_num:02d}.pkl",
                f"z_raw_data_ep_{ep_num}_*.pkl",
                f"z_raw_data_ep_{ep_num:02d}_*.pkl",
            ]

            found_files = []
            for pat in patterns:
                found_files.extend(list(target_dir.glob(pat)))
            found_files = sorted(list(set(found_files)))

            if not found_files:
                continue

            # ロード
            pkl_path = found_files[0]
            with open(pkl_path, "rb") as f:
                dists_data = pickle.load(f)

            # Prior Variance
            prior_std = dists_data["Prior"][1]
            prior_var = np.maximum(prior_std**2, 1e-10)

            # Posterior Variance
            post_std = dists_data["Posterior"][1]
            post_var = np.maximum(post_std**2, 1e-10)

            collected_data["prior"].append(prior_var)
            collected_data["posterior"].append(post_var)
            total_files += 1

    print(f"✅ Loaded {total_files} episodes (Prior & Posterior data).")
    return collected_data


def calculate_mean_trajectory(data_list):
    """異なる長さの時系列データの平均を計算する (最短の長さに合わせる)"""
    if not data_list:
        return None

    min_len = min(len(d) for d in data_list)
    truncated_data = [d[:min_len] for d in data_list]

    stacked_data = np.array(truncated_data)
    mean_data = np.mean(stacked_data, axis=0)

    return mean_data


def plot_z_variance_analysis(collected_data, save_dir, model_name):
    """
    Prior vs Posterior の分散をプロット (Linear Scale, Max of Mean Y-limit)
    """
    z_dim = 16

    if not collected_data["prior"]:
        print("❌ No data found.")
        return

    # 次元の確認
    actual_dim = collected_data["prior"][0].shape[1]
    if actual_dim != z_dim:
        z_dim = actual_dim

    # ---------------------------------------------------------
    # ★修正点: 「平均線の最大値」を探索する
    # ---------------------------------------------------------
    print("Calculating max value of MEAN trajectories...")
    global_max_of_mean = 0.0

    for d in range(z_dim):
        # Prior Mean Max
        prior_dim_data = [p_var[:, d] for p_var in collected_data["prior"]]
        p_mean = calculate_mean_trajectory(prior_dim_data)
        if p_mean is not None:
            global_max_of_mean = max(global_max_of_mean, np.max(p_mean))

        # Posterior Mean Max
        post_dim_data = [q_var[:, d] for q_var in collected_data["posterior"]]
        q_mean = calculate_mean_trajectory(post_dim_data)
        if q_mean is not None:
            global_max_of_mean = max(global_max_of_mean, np.max(q_mean))

    # Y軸の最大値を決定 (平均の最大値 + 10%のマージン)
    # ただし最低でも 1.1 は確保する
    y_limit_top = max(global_max_of_mean * 1.1, 1.1)

    print(f"📊 Global Max of MEAN found: {global_max_of_mean:.4f}")
    print(f"   -> Setting Y-axis limit to: {y_limit_top:.4f}")
    # ---------------------------------------------------------

    # キャンバス作成
    fig, axes = plt.subplots(8, 2, figsize=(20, 24), sharex=True)
    fig.suptitle(f"Z-Variance Analysis: Prior vs Posterior with all test Episodes", fontsize=18, y=0.99)

    legend_added = False

    # --- 次元ごとのループ ---
    for d in range(z_dim):
        col = 0 if d < 8 else 1
        row = d % 8
        ax = axes[row, col]

        # ------------------------------------------------
        # 1. Prior (Blue) のプロット
        # ------------------------------------------------
        prior_dim_data = []
        for p_var in collected_data["prior"]:
            ax.plot(p_var[:, d], **STYLE_CONFIG["prior_indiv"])
            prior_dim_data.append(p_var[:, d])

        # 平均線
        prior_mean = calculate_mean_trajectory(prior_dim_data)
        if prior_mean is not None:
            ax.plot(prior_mean, **STYLE_CONFIG["prior_mean"])

        # ------------------------------------------------
        # 2. Posterior (Red) のプロット
        # ------------------------------------------------
        post_dim_data = []
        for q_var in collected_data["posterior"]:
            ax.plot(q_var[:, d], **STYLE_CONFIG["posterior_indiv"])
            post_dim_data.append(q_var[:, d])

        # 平均線
        post_mean = calculate_mean_trajectory(post_dim_data)
        if post_mean is not None:
            ax.plot(post_mean, **STYLE_CONFIG["posterior_mean"])

        # ------------------------------------------------
        # 3. 装飾
        # ------------------------------------------------
        ax.set_title(f"Dim {d}", fontsize=12, pad=5)

        # ★計算した「平均値基準の最大値」を適用
        ax.set_ylim(-0.05, y_limit_top)

        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.set_ylabel("Variance", fontsize=10)

        # 凡例 (Dim 0のみ作成)
        if d == 0 and not legend_added:
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color=STYLE_CONFIG["prior_mean"]["color"], lw=1, label="Prior (Mean)"),
                Line2D([0], [0], color=STYLE_CONFIG["prior_indiv"]["color"], lw=1, alpha=0.5, label="Prior (Indiv)"),
                Line2D([0], [0], color=STYLE_CONFIG["posterior_mean"]["color"], lw=1, label="Posterior (Mean)"),
                Line2D(
                    [0], [0], color=STYLE_CONFIG["posterior_indiv"]["color"], lw=1, alpha=0.5, label="Posterior (Indiv)"
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper left", fontsize="small", ncol=2)
            legend_added = True

    # X軸ラベル
    axes[7, 0].set_xlabel("Timestep", fontsize=12)
    axes[7, 1].set_xlabel("Timestep", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    save_path = save_dir / f"z_variance_prior_vs_posterior_linear_meanmax.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Saved analysis plot to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_DIR = PROJECT_ROOT / "result_output" / args.model_name

    SAVE_DIR = MODEL_DIR / "summary_analysis"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 読み込むタスクの定義
    tasks = [
        ("stiff", f"stiff_{args.num_samples}", 51, 60),
        ("soft", f"soft_{args.num_samples}", 51, 60),
    ]

    # データロード
    data = load_prior_posterior_data(MODEL_DIR, tasks)

    # プロット実行
    plot_z_variance_analysis(data, SAVE_DIR, args.model_name)
