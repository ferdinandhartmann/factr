import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import pickle
from collections import deque
import sys
import warnings
import argparse

# Hydra/OmegaConf 関連
from omegaconf import OmegaConf
from hydra.utils import instantiate

# 先にライブラリをインポートして resolver を登録させる
try:
    import factr.misc
except ImportError:
    pass

# ==========================================
# 警告・インポート関連
# ==========================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# スクリプト側での register_new_resolver は削除しました（ライブラリ側と競合するため）

# ==================================================================================
# 1. PARAMETERS & PATHS
# ==================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model_A", type=str, required=True, help="Baseline model folder")
parser.add_argument("--model_B", type=str, required=True, help="Your model folder")
parser.add_argument("--ckpt_name", type=str, default="ckpt_020000")
args = parser.parse_args()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 10
IMG_CHUNK = 1
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
ROLLOUT_CFG_PATH = PROJECT_ROOT / "process_data/processed_data/1217_mix/rollout_config.yaml"

SAVE_DIR = Path("/home/otake/FACTR-pr/FACTR-project/result_output/base_vs_mine")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

eval_targets = [
    (Path("/data/otake/box_lift_up_side/20251218_stiff/eval"), [f"ep_{i:02d}" for i in range(51, 60)]),
    (Path("/data/otake/box_lift_up_side/20251217_soft/eval"), [f"ep_{i:02d}" for i in range(51, 60)]),
]


# ==================================================================================
# 2. 補助関数
# ==================================================================================
def run_posterior_inference(policy, imgs, obs, target_action, num_samples=10):
    """
    提供されたロジックに基づくPosterior再構成関数。
    """
    with torch.no_grad():
        action_samples = policy.get_actions_pos(imgs, obs, target_action, num_samples=num_samples, sample=True)

    raw_samples_np = action_samples.detach().cpu().numpy()  # (B, S, Chunk, Dim)
    pred_mean_np = raw_samples_np.mean(axis=1)  # (B, Chunk, Dim)

    # 入力GTの整形（デバッグ/リターン用）
    if target_action.dim() == 2:
        B = target_action.shape[0]
        gt_input_np = target_action.view(B, policy.ac_chunk, policy.ac_dim).detach().cpu().numpy()
    else:
        gt_input_np = target_action.detach().cpu().numpy()

    return raw_samples_np, pred_mean_np, gt_input_np


def get_padded_mean(mae_list):
    maxlen = max(len(x) for x in mae_list)
    arr = np.full((len(mae_list), maxlen), np.nan)
    for i, x in enumerate(mae_list):
        arr[i, : len(x)] = x
    return np.nanmean(arr, axis=0)


def load_policy(model_name, ckpt_name, device):
    model_dir = CHECKPOINTS_DIR / model_name
    exp_cfg_path = model_dir / "rollout/exp_config.yaml"
    ckpt_path = model_dir / f"{ckpt_name}.ckpt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1. Configをロード
    cfg = OmegaConf.load(exp_cfg_path)

    # 2. Hydra 固有の不要なキーを削除
    if "hydra" in cfg:
        cfg.pop("hydra", None)

    # 【重要】OmegaConf.resolve(cfg) は削除します
    # 代わりに、個別のプリミティブ値（パスや数値）だけが必要な場合は個別に resolve しますが、
    # 基本的には instantiate に任せるのが正解です。

    if "task" in cfg and "cam_indexes" in cfg.task:
        cfg.task.n_cams = len(cfg.task.cam_indexes)

    # agent の設定を一部書き換える
    cfg.agent.sanity_check_posterior = False

    print(f"Instantiating policy for {model_name}...")

    # 3. instantiate を実行（ここで内部的にリゾルブされます）
    policy = instantiate(cfg.agent)
    policy.to(device)

    # チェックポイントのロード
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)

    # # 重みが正しくロードされたかチェック
    # if len(missing_keys) > 0:
    #     print(f"⚠️ [WARNING] {model_name} の重みが {len(missing_keys)} 個ロードされていません！")
    #     print(f"   例: {missing_keys[:3]}") # 最初の3個だけ表示
    # else:
    #     print(f"✅ {model_name} の重みがすべて正常にロードされました！")

    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy


def load_and_extract_raw_data(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        raw_data = pickle.load(f)

    image_obs, torque_obs, actions = [], [], []
    if "data" not in raw_data:
        return [], [], []

    entries = raw_data["data"]
    image_topic = "/realsense/front/im"
    obs_topic = "/franka_robot_state_broadcaster/external_joint_torques"
    possible_topics = [
        "/joint_impedance_dynamic_gain_controller/joint_impedance_command",
        "/joint_impedance_command_controller/joint_trajectory",
    ]

    action_topic = next((t for t in possible_topics if t in entries), None)
    if action_topic is None:
        return [], [], []

    for v in entries[action_topic]:
        if isinstance(v, dict) and "position" in v:
            actions.append(v["position"])

    if not actions:
        return [], [], []

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

    torque_obs, actions = np.array(torque_obs), np.array(actions)
    N = min(len(image_obs), len(torque_obs), len(actions))
    return image_obs[:N], torque_obs[:N], actions[:N]


def preprocess_image(img, device):
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    img = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((img_tensor - mean) / std).to(device)


# ==================================================================================
# 3. メイン実行
# ==================================================================================
policy_A = load_policy(args.model_A, args.ckpt_name, DEVICE)
policy_B = load_policy(args.model_B, args.ckpt_name, DEVICE)

with open(ROLLOUT_CFG_PATH, "r") as f:
    rollout_config = yaml.safe_load(f)

obs_mean = torch.tensor(rollout_config["norm_stats"]["state"]["mean"]).float().to(DEVICE)
obs_std = torch.tensor(rollout_config["norm_stats"]["state"]["std"]).float().to(DEVICE)
action_mean = torch.tensor(rollout_config["norm_stats"]["action"]["mean"]).float().to(DEVICE)
action_std = torch.tensor(rollout_config["norm_stats"]["action"]["std"]).float().to(DEVICE)

all_mae_A = []
all_mae_B_prior = []
all_mae_B_posterior = []

chunk_size = policy_B.ac_chunk

print("Starting Evaluation using Chunk-based Posterior logic...")
for target_dir, episodes in eval_targets:
    for ep_name in episodes:
        pkl_path = target_dir / f"{ep_name}.pkl"
        if not pkl_path.exists():
            continue

        image_obs, torque_obs, true_actions = load_and_extract_raw_data(pkl_path)
        if not image_obs:
            continue

        # 全体のGTを先に正規化
        true_actions_norm = (torch.from_numpy(true_actions).float().to(DEVICE) - action_mean) / action_std
        gt_np = true_actions_norm.cpu().numpy()

        ep_mae_A, ep_mae_B_prior, ep_mae_B_posterior = [], [], []
        image_history = deque(maxlen=IMG_CHUNK)

        for i in tqdm(range(len(image_obs)), desc=f"{ep_name}", leave=False):
            img_tensor = preprocess_image(image_obs[i], DEVICE)
            if len(image_history) == 0:
                for _ in range(IMG_CHUNK):
                    image_history.append(img_tensor)
            else:
                image_history.append(img_tensor)

            input_img = {"cam0": torch.cat(list(image_history), dim=1)}
            torque_norm = (torch.from_numpy(torque_obs[i]).float().to(DEVICE) - obs_mean) / obs_std
            torque_norm = torque_norm.unsqueeze(0)

            # --- Posterior用のChunk切り出しロジック ---
            current_gt_chunk = true_actions_norm[i : i + chunk_size]
            if current_gt_chunk.shape[0] < chunk_size:
                pad_len = chunk_size - current_gt_chunk.shape[0]
                last_val = current_gt_chunk[-1:].repeat(pad_len, 1)
                current_gt_chunk = torch.cat([current_gt_chunk, last_val], dim=0)
            current_gt_chunk = current_gt_chunk.unsqueeze(0)  # (1, Chunk, Dim)

            with torch.no_grad():
                # 1. Baseline (A) - これは torch.Tensor を返す想定
                pred_A = policy_A.get_actions_base(input_img, torque_norm)

                # 2. Proposed Prior (B) - これも torch.Tensor を返す想定
                pred_B_prior_samples = policy_B.get_actions_prior(
                    input_img, torque_norm, sample=True, num_samples=NUM_SAMPLES
                )
                pred_B_prior_mean = pred_B_prior_samples.mean(axis=1)

                # 3. Proposed Posterior (B) ★この関数は内部ですでに numpy.ndarray を返している
                _, pred_B_pos_mean_np, _ = run_posterior_inference(
                    policy_B, input_img, torque_norm, target_action=current_gt_chunk, num_samples=NUM_SAMPLES
                )

            # --- 修正箇所 ---
            # pred_A と pred_B_prior_mean は Tensor なので .cpu().numpy() が必要
            mae_a = np.abs(pred_A[0, 0].cpu().numpy() - gt_np[i]).mean()
            mae_prior = np.abs(pred_B_prior_mean[0, 0].cpu().numpy() - gt_np[i]).mean()

            # pred_B_pos_mean_np はすでに NumPy なのでそのまま計算
            mae_pos = np.abs(pred_B_pos_mean_np[0, 0] - gt_np[i]).mean()

            ep_mae_A.append(mae_a)
            ep_mae_B_prior.append(mae_prior)
            ep_mae_B_posterior.append(mae_pos)

        all_mae_A.append(ep_mae_A)
        all_mae_B_prior.append(ep_mae_B_prior)
        all_mae_B_posterior.append(ep_mae_B_posterior)

# ==================================================================================
# 4. プロット出力
# ==================================================================================
final_mean_A = np.nanmean(get_padded_mean(all_mae_A))
final_mean_B_prior = np.nanmean(get_padded_mean(all_mae_B_prior))
final_mean_B_posterior = np.nanmean(get_padded_mean(all_mae_B_posterior))

# --- コンソール出力 ---
print("\n" + "=" * 50)
print("📊 FINAL EVALUATION RESULTS (Total Average MAE)")
print("=" * 50)
print(f"Baseline (A)          : {final_mean_A:.6f}")
print(f"Proposed Prior (B)     : {final_mean_B_prior:.6f}")
print(f"Proposed Posterior (B) : {final_mean_B_posterior:.6f}")
print("-" * 50)

# 改善率の計算 (Baseline vs Proposed Prior)
improvement = (1 - (final_mean_B_prior / final_mean_A)) * 100
print(f"🚀 Improvement (Prior vs Baseline): {improvement:.2f}%")
print("=" * 50 + "\n")

plt.figure(figsize=(10, 6))

# 個別エピソード（細い線）
for i, (m_a, m_bp, m_bpos) in enumerate(zip(all_mae_A, all_mae_B_prior, all_mae_B_posterior)):
    plt.plot(m_a, color="blue", alpha=0.15, linewidth=0.8, label="Baseline (Indiv.)" if i == 0 else None)
    plt.plot(m_bp, color="red", alpha=0.15, linewidth=0.8, label="Proposed Prior (Indiv.)" if i == 0 else None)
    plt.plot(m_bpos, color="green", alpha=0.15, linewidth=0.8, label="Proposed Posterior (Indiv.)" if i == 0 else None)

# 平均（太い実線）
plt.plot(get_padded_mean(all_mae_A), color="blue", linewidth=2.0, label="Baseline (Mean)")
plt.plot(get_padded_mean(all_mae_B_prior), color="red", linewidth=2.0, label="Proposed Prior (Mean)")
plt.plot(get_padded_mean(all_mae_B_posterior), color="green", linewidth=2.0, label="Proposed Posterior (Mean)")

plt.yscale("log")

plt.title("Action MAE Comparison: Baseline vs Proposed (Prior & Posterior)", fontsize=14)
plt.xlabel("Timestep", fontsize=12)
plt.ylabel("Mean Absolute Error (Normalized)", fontsize=12)
plt.legend(loc="upper left", ncol=2, fontsize="small")
plt.grid(True, linestyle=":", alpha=0.6)

plt.savefig(SAVE_DIR / "mae_comparison_final.png", dpi=200, bbox_inches="tight")
print(f"📊 Final result saved to: {SAVE_DIR / 'mae_comparison_final.png'}")
