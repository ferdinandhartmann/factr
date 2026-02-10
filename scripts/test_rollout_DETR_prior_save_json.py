import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
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
import json  # 追加

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

parser = argparse.ArgumentParser(description="Run Prior Inference and Save JSON")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model (folder name in checkpoints)")
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
SAVE_DIR = SAVE_DIR_ROOT / f"PRIOR_CHECK_{MODE}"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================================================
# 2. 関数定義
# ==================================================================================


def run_prior_inference(policy, imgs, obs, num_samples=10):
    """
    Contextを入力し、MLP Prior -> Z -> Decoder で生成を行う関数。
    GTアクションは使用しません。
    """
    # 1. Encoder (Context取得)
    action_samples, cross_w = policy.get_actions_prior(imgs, obs, sample=True, num_samples=10, return_weights=True)
    # action_samples: (B, num_samples, Chunk, Dim)

    raw_samples_np = action_samples.detach().cpu().numpy()
    pred_mean = raw_samples_np.mean(axis=1)  # (B, Chunk, Dim)

    return raw_samples_np, pred_mean, cross_w


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

print("\n" + "=" * 60)
print("ℹ️  [Prior Inference Mode]")
print("    Generating actions from Context Only (MLP Prior).")
print("=" * 60 + "\n")
cfg.agent.sanity_check_posterior = False

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
    true_actions_norm_np = true_actions_norm.cpu().numpy()

    pred_actions_mean_list = []
    cvae_samples_list = []
    cross_attn_list = []

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
            raw_samples, pred_mean, cross_w = run_prior_inference(
                policy, {"cam0": input_img_tensor}, torque_norm, num_samples=NUM_SAMPLES
            )
            # raw_samples: (1, num_samples, chunk, dim) -> (num_samples, chunk, dim)
            # pred_mean:   (1, chunk, dim) -> (chunk, dim)
            pred_actions_mean_list.append(pred_mean[0])
            cvae_samples_list.append(raw_samples[0])

            if cross_w is not None:
                cross_w_cpu = cross_w[:, 0].detach().cpu().numpy()
                cross_attn_list.append(cross_w_cpu)

    # --- 保存データの準備 ---

    pred_mean_arr = np.array(pred_actions_mean_list)
    samples_arr = np.array(cvae_samples_list)

    # AttentionリストをNumpy配列化: (Time, Layers, Samples, Heads, Chunk, Memory)
    if len(cross_attn_list) > 0:
        attn_arr = np.array(cross_attn_list)
    else:
        attn_arr = np.array([])

    # 辞書にまとめる
    save_data = {
        "episode_name": episode_name,
        "timesteps": len(true_actions),
        "data": {
            "gt_actions_norm": true_actions_norm_np,  # numpyのまま
            "prior_pred_mean_norm": pred_mean_arr,  # numpyのまま
            "prior_samples_norm": samples_arr,  # numpyのまま
            "torque_input_raw": torque_obs,  # numpyのまま
            # AttentionもNumpyのまま保存できる (容量は大きいがJSONよりマシ)
            # それでもAttentionがGB単位なら、ここだけ別ファイル(.npy)にするのがベスト
            "cross_attention": attn_arr,
            "stats": {
                # ここだけは単一の数値や小さなリストなのでtolistしておくと無難だが、
                # pklならnumpyのままでもOK
                "action_mean": action_mean.cpu().numpy(),
                "action_std": action_std.cpu().numpy(),
                "torque_mean": obs_mean.cpu().numpy(),
                "torque_std": obs_std.cpu().numpy(),
            },
        },
    }

    episode_save_dir = SAVE_DIR / episode_name
    episode_save_dir.mkdir(parents=True, exist_ok=True)

    # .pkl で保存
    pkl_path = episode_save_dir / "prior_inference_data.pkl"
    print(f"Saving Pickle to {pkl_path} ...")

    with open(pkl_path, "wb") as f:  # 'wb' (write binary) であることに注意
        pickle.dump(save_data, f)

    print(f"✅ Saved Pickle for {episode_name}")
