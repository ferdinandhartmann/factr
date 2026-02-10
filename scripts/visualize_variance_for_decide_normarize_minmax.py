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
parser.add_argument("--model_name", type=str, required=True, help="Baseline model folder")
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
    (Path("/data/otake/box_lift_up_side/20251218_stiff/eval"), [f"ep_{i:02d}" for i in range(55, 60)]),
    (Path("/data/otake/box_lift_up_side/20251217_soft/eval"), [f"ep_{i:02d}" for i in range(55, 60)]),
]


# ==================================================================================
# 2. 補助関数
# ==================================================================================
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
policy = load_policy(args.model_name, args.ckpt_name, DEVICE)

with open(ROLLOUT_CFG_PATH, "r") as f:
    rollout_config = yaml.safe_load(f)

obs_mean = torch.tensor(rollout_config["norm_stats"]["state"]["mean"]).float().to(DEVICE)
obs_std = torch.tensor(rollout_config["norm_stats"]["state"]["std"]).float().to(DEVICE)

all_uncertainties = []  # 10エピソード分を格納

print(f"Starting Uncertainty Evaluation (Samples: {NUM_SAMPLES})...")

for target_dir, episodes in eval_targets:
    for ep_name in episodes:
        pkl_path = target_dir / f"{ep_name}.pkl"
        if not pkl_path.exists():
            continue

        image_obs, torque_obs, _ = load_and_extract_raw_data(pkl_path)
        if not image_obs:
            continue

        ep_unc = []
        image_history = deque(maxlen=IMG_CHUNK)

        for i in tqdm(range(len(image_obs)), desc=f"Eval {ep_name}", leave=False):
            img_tensor = preprocess_image(image_obs[i], DEVICE)
            if len(image_history) == 0:
                for _ in range(IMG_CHUNK):
                    image_history.append(img_tensor)
            else:
                image_history.append(img_tensor)

            input_img = {"cam0": torch.cat(list(image_history), dim=1)}
            torque_norm = (torch.from_numpy(torque_obs[i]).float().to(DEVICE) - obs_mean) / obs_std
            torque_norm = torque_norm.unsqueeze(0)

            with torch.no_grad():
                # get_uncertainty_entropy を使用
                # 加重平均を有効にする設定 (unc_weighted=True)
                _, uncertainty = policy.get_uncertainty_entropy(
                    input_img,
                    torque_norm,
                    sample=True,
                    num_samples=NUM_SAMPLES,
                    unc_step_mode=False,
                    unc_weighted=True,
                    w_start=0.1,
                    w_end=0.9,
                )

            # uncertaintyの形状は (B, 1, Dim) なので、全次元の平均をとってスカラーにする
            ep_unc.append(uncertainty.mean().cpu().item())

        all_uncertainties.append(ep_unc)


# ==================================================================================
# 4. 統計計算 & プロット (塗りつぶしなし・個別線と平均のみ)
# ==================================================================================
def get_padded_mean(unc_list):
    # エピソードごとの長さが異なるため、最大長に合わせてパディング
    maxlen = max(len(x) for x in unc_list)
    arr = np.full((len(unc_list), maxlen), np.nan)
    for i, x in enumerate(unc_list):
        arr[i, : len(x)] = x
    # 時間軸ごとの平均を計算（データがない部分は無視）
    return np.nanmean(arr, axis=0)


mean_unc = get_padded_mean(all_uncertainties)

# --- 数値出力 ---
max_val = np.nanmax(mean_unc)
min_val = np.nanmin(mean_unc)
print("-" * 40)
print(f"📊 Uncertainty Analysis Results (10 episodes)")
print(f"   Mean Uncertainty MAX: {max_val:.6f}")
print(f"   Mean Uncertainty MIN: {min_val:.6f}")
print("-" * 40)

# --- グラフ作成 ---
plt.figure(figsize=(12, 7))

# 1. 各エピソードのプロット (細い線 - 個別の分散の推移)
for i, u in enumerate(all_uncertainties):
    label = "Individual Episode Variance" if i == 0 else None
    plt.plot(u, color="blue", alpha=0.2, linewidth=0.8, label=label)

# 2. 全体の平均プロット (太い実線)
plt.plot(mean_unc, color="blue", linewidth=2.5, label="Mean of Variances")

# グラフの装飾
plt.title(f"Action Uncertainty (Weighted Variance) - {args.model_name}", fontsize=14)
plt.xlabel("Timestep", fontsize=12)
plt.ylabel("Uncertainty Value", fontsize=12)
plt.legend(loc="upper right", frameon=True)
plt.grid(True, linestyle=":", alpha=0.6)

# 保存
save_path = SAVE_DIR / f"unc_plot_{args.model_name}.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"📊 Plot saved to: {save_path}")
