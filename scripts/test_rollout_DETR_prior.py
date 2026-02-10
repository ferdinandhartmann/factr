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

episode_names = [f"ep_{i:02d}" for i in range(51, 55)]

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

    if getattr(policy, "factr_baseline", False):
        # print("use base line")
        action_samples = policy.get_actions_base(imgs, obs)
        action_samples = action_samples.unsqueeze(1)  # ダミーの軸追加 (B, 1, Chunk, Dim)

    else:
        action_samples = policy.get_actions_prior(imgs, obs, sample=True, num_samples=num_samples)

    raw_samples_np = action_samples.detach().cpu().numpy()
    pred_mean = raw_samples_np.mean(axis=1)  # (B, Chunk, Dim) num_sampleをつぶしてmean

    return raw_samples_np, pred_mean


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


def load_reference_trajectories(data_dir, action_stats, device):
    pkl_files = sorted(Path(data_dir).glob("*.pkl"))
    if not pkl_files:
        return []

    ref_actions_norm = []
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
            act_tensor = torch.tensor(actions, device=device, dtype=torch.float32)
            act_norm = (act_tensor - stat_a) / stat_b
            ref_actions_norm.append(act_norm.cpu().numpy())

        except Exception:
            continue
    return ref_actions_norm


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

# --- Refs Load ---
refs_solid = load_reference_trajectories(RAW_DATA_BASE, action_stats_dict, DEVICE)
refs_dotted = load_reference_trajectories(RAW_DATA_BASE_2, action_stats_dict, DEVICE)

chunk_size = getattr(policy, "chunk_size", None)
if chunk_size is None:
    chunk_size = policy.ac_query.weight.shape[0]
print(f"Detected Chunk Size: {chunk_size}")

# --- エピソードごとのループ ---
for episode_name in episode_names:
    pkl_path = TARGET_INFERENCE_DIR / f"{episode_name}.pkl"
    if not pkl_path.exists():
        continue

    image_obs, torque_obs, true_actions = load_and_extract_raw_data(pkl_path)

    # GT Normalization
    true_actions_tensor = torch.from_numpy(true_actions).float().to(DEVICE)
    true_actions_norm = (true_actions_tensor - action_mean) / action_std
    true_actions = true_actions_norm.cpu().numpy()

    pred_actions_mean_list = []
    cvae_samples_list = []

    image_history = deque(maxlen=img_chunk)

    print(f"Running Prior Inference on {episode_name}...")
    for i in tqdm(range(len(image_obs)), leave=False):
        img = image_obs[i]
        torque = torque_obs[i]

        img_tensor = preprocess_image(img)
        if len(image_history) == 0:
            for _ in range(img_chunk):
                image_history.append(img_tensor)
        else:
            image_history.append(img_tensor)

        input_img_tensor = torch.cat(list(image_history), dim=1)

        torque_tensor = torch.from_numpy(torque).float().to(DEVICE)
        torque_norm = (torque_tensor - obs_mean) / obs_std
        torque_norm = torque_norm.unsqueeze(0)

        with torch.no_grad():
            # GTは渡さない
            raw_samples, pred_mean = run_prior_inference(
                policy, {"cam0": input_img_tensor}, torque_norm, num_samples=NUM_SAMPLES
            )

            pred_actions_mean_list.append(pred_mean[0])
            cvae_samples_list.append(raw_samples[0])

    # --- Plotting Preparation ---
    pred_actions_mean = np.array(pred_actions_mean_list)  # (Time, Chunk, Dim)

    if len(cvae_samples_list) > 0:
        samples_arr = np.array(cvae_samples_list)  # (Time, Samples, Chunk, Dim)
    else:
        samples_arr = None

    t = np.arange(len(true_actions))

    episode_save_dir = SAVE_DIR / episode_name
    episode_save_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------
    # Plot 1: Prior Generation Check
    # -----------------------------------------------------------
    chunk_size_actual = pred_actions_mean.shape[1]

    fig1, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    fig1.suptitle(f"GT vs Prior Generation", fontsize=16, y=0.98)

    for d in range(7):
        ax = axes[d]

        # Reference
        for i, ref_act in enumerate(refs_dotted):
            t_ref = np.arange(len(ref_act))
            label = "other GT" if i == 0 else None
            ax.plot(t_ref, ref_act[:, d], color="gray", linewidth=0.5, alpha=0.3, linestyle=":", label=label)

        # --- 1. Original GT (赤色) ---
        ax.plot(t, true_actions[:, d], color="red", linestyle="--", linewidth=2.0, alpha=0.6, label="Original GT")

        # --- 2. Prior Samples (水色) ---
        plot_interval = 1

        if samples_arr is not None:
            actual_num_samples = samples_arr.shape[1]
            for step_i in range(chunk_size_actual):
                for s_idx in range(actual_num_samples):
                    lbl = "Prior Sample" if (step_i == 0 and s_idx == 0) else None

                    ax.plot(
                        (t + step_i)[::plot_interval],
                        samples_arr[::plot_interval, s_idx, step_i, d],
                        color="blue",
                        linewidth=0.4,
                        alpha=0.05,
                        label=lbl,
                    )

        # # --- 3. Prior Mean (青色) ---
        # for step_i in range(chunk_size_actual):
        #     if step_i == 0:
        #         ax.plot(t + step_i, pred_actions_mean[:, step_i, d],
        #                 color='blue', linewidth=1.2, alpha=0.9, label="Prior Mean (Step 0)")
        #     else:
        #         ax.plot(t + step_i, pred_actions_mean[:, step_i, d],
        #                 color='blue', linewidth=0.5, alpha=0.2)

        ax.set_ylabel(f"J{d + 1} (norm)")
        ax.set_ylim(-4.0, 4.0)
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(loc="upper left", ncol=2, fontsize="small")

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    save_path_act = episode_save_dir / "prior_generation_check_.png"
    plt.savefig(save_path_act, dpi=150)
    plt.close(fig1)

    print(f"✅ Saved results for {save_path_act}")
