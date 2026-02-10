import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf
import yaml
import pickle
from collections import deque
import sys
import warnings
import math
import argparse

# ==========================================
# 警告・インポート関連
# ==========================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

try:
    import factr.misc
except ImportError:
    pass


def register_if_not_exists(name, resolver):
    if not OmegaConf.has_resolver(name):
        try:
            OmegaConf.register_new_resolver(name, resolver)
        except ValueError:
            pass


register_if_not_exists("mult", lambda x, y: x * y)
register_if_not_exists("add", lambda x, y: x + y)
register_if_not_exists("len", lambda x: len(x))
register_if_not_exists("transform", lambda x: x)
register_if_not_exists("hydra", lambda x: None)

try:
    if "factr.misc" in sys.modules:
        import factr.misc

        factr.misc.get_transform_by_name = lambda name: f"dummy_transform_{name}"
        print("✅ Patched factr.misc.get_transform_by_name for inference.")
except Exception as e:
    print(f"⚠️ Failed to patch factr.misc: {e}")

# ==================================================================================
# 1. CONFIG
# ==================================================================================

parser = argparse.ArgumentParser(description="Run Prior Inference Check")  # 修正
parser.add_argument("--model_name", type=str, help="Name of the model (folder name in checkpoints)")
args = parser.parse_args()

model_name = args.model_name
checkpoint = "ckpt_020000"

episode_names = [f"ep_{i:02d}" for i in range(51, 63)]

# パラメータ
img_chunk = 1
GPU_ID = 0
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 10
MODE = "stiff"  # soft or stiff

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / model_name
RAW_DATA_BASE = Path("/data/otake/box_lift_up_side/20251218_stiff/data")
RAW_DATA_BASE_2 = Path("/data/otake/box_lift_up_side/20251217_soft/data")
ROLLOUT_CFG_PATH = PROJECT_ROOT / "process_data/processed_data/1217_mix/rollout_config.yaml"

CKPT_PATH = None
if checkpoint == "latest":
    CKPT_PATH = CHECKPOINTS_DIR / "rollout/latest_ckpt.ckpt"
else:
    CKPT_PATH = CHECKPOINTS_DIR / f"{checkpoint}.ckpt"

if MODE == "stiff":
    TARGET_INFERENCE_DIR = Path("/data/otake/box_lift_up_side/20251218_stiff/eval")
else:
    TARGET_INFERENCE_DIR = Path("/data/otake/box_lift_up_side/20251217_soft/eval")

SAVE_DIR_ROOT = PROJECT_ROOT / "result_output" / model_name
# ディレクトリ名をPRIORに変更
SAVE_DIR = SAVE_DIR_ROOT / f"PRIOR_CHECK_{MODE}"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================================================
# 2. 関数定義 (Prior Inference用 - MLP Logic)
# ==================================================================================


def run_prior_inference(policy, imgs, obs, num_samples=10):
    """
    Contextを入力し、MLP Prior -> Z -> Decoder で生成を行う関数。
    GTアクションは使用しません。
    """

    # 1. Encoder (Context取得)
    _, unc_tensor, ent_tensor = policy.get_uncertainty_entropy(
        imgs, obs, sample=True, num_samples=num_samples
    )  # (B, T, D)

    uncertainty_np = unc_tensor.detach().cpu().numpy()
    entropy_np = ent_tensor.detach().cpu().numpy()

    return uncertainty_np, entropy_np


def load_and_extract_raw_data(pkl_path: Path):
    """生のpklファイルを読み込み、画像・トルク・アクションを抽出"""
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

    action_topic = None
    for t in possible_topics:
        if t in entries:
            action_topic = t
            break

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
                        if img is None:
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img_flat = np.frombuffer(v["data"], dtype=np.uint8)
                        img = img_flat.reshape((v["height"], v["width"], -1))
                    image_obs.append(img)
                except Exception as e:
                    pass

    if obs_topic in entries:
        for v in entries[obs_topic]:
            if isinstance(v, dict) and "effort" in v:
                torque_obs.append(v["effort"])

    torque_obs = np.array(torque_obs)
    actions = np.array(actions)
    N = min(len(image_obs), len(torque_obs), len(actions))

    if N == 0:
        return [], [], []

    return image_obs[:N], torque_obs[:N], actions[:N]


def preprocess_image(img):
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    img = cv2.resize(img, (224, 224))
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.to(DEVICE)


# ==================================================================================
# 3. メイン処理
# ==================================================================================

# --- Config & Model Load ---
ckpt_path = CKPT_PATH if checkpoint != "latest" else CHECKPOINTS_DIR / "rollout/latest_ckpt.ckpt"
exp_cfg_path = CHECKPOINTS_DIR / "rollout/exp_config.yaml"

print(f"Loading config from {exp_cfg_path}")
cfg = OmegaConf.load(exp_cfg_path)

if "task" in cfg:
    if "cam_indexes" in cfg.task:
        cfg.task.n_cams = len(cfg.task.cam_indexes)
if "curriculum" in cfg:
    cfg.curriculum.max_step = cfg.max_iterations

raw_dict = OmegaConf.to_container(cfg, resolve=False)
if "hydra" in raw_dict:
    cfg.pop("hydra", None)
OmegaConf.resolve(cfg)

# Prior ModeではGT入力チェックは不要なので強制的にFalseにするか、警告を出す
print("\n" + "=" * 60)
print("ℹ️  [Prior Inference Mode]")
print("    Generating actions from Context Only (MLP Prior).")
print("=" * 60 + "\n")
cfg.agent.sanity_check_posterior = False  # Prior推論なのでFalse

print("Instantiating policy...")
policy = instantiate(cfg.agent)
policy.to(DEVICE)
policy.eval()

print(f"Loading checkpoint from {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=DEVICE)
state_dict = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
policy.load_state_dict(state_dict, strict=False)
print("✅ Policy loaded!")

# --- 正規化統計量のロード ---
with open(ROLLOUT_CFG_PATH, "r") as f:
    rollout_config = yaml.safe_load(f)

obs_mean = torch.tensor(rollout_config["norm_stats"]["state"]["mean"]).float().to(DEVICE)
obs_std = torch.tensor(rollout_config["norm_stats"]["state"]["std"]).float().to(DEVICE)
action_mean = torch.tensor(rollout_config["norm_stats"]["action"]["mean"]).float().to(DEVICE)
action_std = torch.tensor(rollout_config["norm_stats"]["action"]["std"]).float().to(DEVICE)
action_stats_dict = {"mean": action_mean, "std": action_std}


chunk_size = getattr(policy, "chunk_size", None)
if chunk_size is None:
    chunk_size = policy.ac_query.weight.shape[0]
print(f"Detected Chunk Size: {chunk_size}")

# --- エピソードごとのループ ---
for episode_name in episode_names:
    pkl_path = TARGET_INFERENCE_DIR / f"{episode_name}.pkl"
    if not pkl_path.exists():
        continue

    # データのロード
    image_obs, torque_obs, true_actions = load_and_extract_raw_data(pkl_path)
    if len(image_obs) == 0:
        continue

    # データ保存用のリスト
    uncertainty_list = []  # (Time, 7)
    entropy_list = []  # (Time, 7)

    image_history = deque(maxlen=img_chunk)

    print(f"Running Prior Inference on {episode_name}...")

    # --- Inference Loop ---
    for i in tqdm(range(len(image_obs)), leave=False):
        img = image_obs[i]
        torque = torque_obs[i]

        # 画像前処理
        img_tensor = preprocess_image(img)
        if len(image_history) == 0:
            for _ in range(img_chunk):
                image_history.append(img_tensor)
        else:
            image_history.append(img_tensor)

        input_img_tensor = torch.cat(list(image_history), dim=1)

        # トルク前処理
        torque_tensor = torch.from_numpy(torque).float().to(DEVICE)
        torque_norm = (torque_tensor - obs_mean) / obs_std
        torque_norm = torque_norm.unsqueeze(0)

        with torch.no_grad():
            # Uncertainty, Entropy を取得
            # raw_samplesなどは今回はプロットしないため受け取らない（"_"で捨てる）
            unc_np, ent_np = run_prior_inference(
                policy, {"cam0": input_img_tensor}, torque_norm, num_samples=NUM_SAMPLES
            )

            # unc_np, ent_np の shape は (1, Chunk, 7)
            # 現在時刻(直近)の不確実性を取得するため、Chunkの先頭 [0, 0] を採用
            uncertainty_list.append(unc_np[0, 0])
            entropy_list.append(ent_np[0, 0])

    # --- Plotting ---

    uncertainty_arr = np.array(uncertainty_list)
    entropy_arr = np.array(entropy_list)

    # 時間軸作成
    t = np.arange(len(uncertainty_arr))

    # 配列の長さを合わせる (GTアクションも現在の予測長にクリップ)
    current_torque = torque_obs[: len(t)]
    current_gt_action = true_actions[: len(t)]

    episode_save_dir = SAVE_DIR / episode_name
    episode_save_dir.mkdir(parents=True, exist_ok=True)

    # 共通のカラーマップ (TorqueとGT Actionで色を揃える)
    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    # =========================================================================
    # PNG 1枚目: Uncertainty (7段) + Torque (1段) + GT Action (1段) = 9段
    # =========================================================================
    # figsizeをさらに縦長(27インチ)にして確保
    fig1, axes1 = plt.subplots(9, 1, figsize=(12, 27), sharex=True)
    fig1.suptitle(f"Uncertainty (J1-J7) & Torque & GT - {episode_name}", fontsize=18, y=0.99)

    # 1〜7段目: 各関節のUncertainty
    for d in range(7):
        ax = axes1[d]
        ax.plot(t, uncertainty_arr[:, d], color="blue", linewidth=2)
        ax.set_ylabel(f"J{d + 1}\nStd Dev", fontsize=10, rotation=0, labelpad=20)
        ax.grid(True, alpha=0.3)
        if d % 2 == 0:
            ax.set_facecolor("#f9f9f9")

    # 8段目: 全関節のTorque
    ax_trq = axes1[7]
    for d in range(7):
        ax_trq.plot(t, current_torque[:, d], label=f"J{d + 1}", color=colors[d], alpha=0.7)

    ax_trq.set_ylabel("Torque\n(Nm)", fontsize=10, rotation=0, labelpad=20)
    ax_trq.grid(True, alpha=0.3)
    ax_trq.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", title="Joints")
    ax_trq.set_title("Input Torques (All Joints)", fontsize=10)

    # 9段目: 全関節のGT Action
    ax_gt = axes1[8]
    for d in range(7):
        ax_gt.plot(t, current_gt_action[:, d], label=f"J{d + 1}", color=colors[d], alpha=0.7)

    ax_gt.set_ylabel("GT Action\n(Pos)", fontsize=10, rotation=0, labelpad=20)
    ax_gt.set_xlabel("Timestep", fontsize=14)
    ax_gt.grid(True, alpha=0.3)
    # 凡例は8段目にあるので省略しても良いが、念のため付けるなら
    ax_gt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", title="Joints")
    ax_gt.set_title("Ground Truth Actions (All Joints)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    save_path_unc = episode_save_dir / "9row_uncertainty.png"
    plt.savefig(save_path_unc, dpi=100)
    plt.close(fig1)
    print(f"✅ Saved 9-row Uncertainty plot to {save_path_unc}")

    # =========================================================================
    # PNG 2枚目: Entropy (7段) + Torque (1段) + GT Action (1段) = 9段
    # =========================================================================
    fig2, axes2 = plt.subplots(9, 1, figsize=(12, 27), sharex=True)
    fig2.suptitle(f"Entropy (J1-J7) & Torque & GT - {episode_name}", fontsize=18, y=0.99)

    # 1〜7段目: 各関節のEntropy
    for d in range(7):
        ax = axes2[d]
        ax.plot(t, entropy_arr[:, d], color="purple", linewidth=2)
        ax.set_ylabel(f"J{d + 1}\nEntropy", fontsize=10, rotation=0, labelpad=20)
        ax.grid(True, alpha=0.3)
        if d % 2 == 0:
            ax.set_facecolor("#f9f9f9")

    # 8段目: 全関節のTorque
    ax_trq2 = axes2[7]
    for d in range(7):
        ax_trq2.plot(t, current_torque[:, d], label=f"J{d + 1}", color=colors[d], alpha=0.7)

    ax_trq2.set_ylabel("Torque\n(Nm)", fontsize=10, rotation=0, labelpad=20)
    ax_trq2.grid(True, alpha=0.3)
    ax_trq2.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", title="Joints")
    ax_trq2.set_title("Input Torques (All Joints)", fontsize=10)

    # 9段目: 全関節のGT Action
    ax_gt2 = axes2[8]
    for d in range(7):
        ax_gt2.plot(t, current_gt_action[:, d], label=f"J{d + 1}", color=colors[d], alpha=0.7)

    ax_gt2.set_ylabel("GT Action\n(Pos)", fontsize=10, rotation=0, labelpad=20)
    ax_gt2.set_xlabel("Timestep", fontsize=14)
    ax_gt2.grid(True, alpha=0.3)
    ax_gt2.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", title="Joints")
    ax_gt2.set_title("Ground Truth Actions (All Joints)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    save_path_ent = episode_save_dir / "9row_entropy.png"
    plt.savefig(save_path_ent, dpi=100)
    plt.close(fig2)
    print(f"✅ Saved 9-row Entropy plot to {save_path_ent}")
