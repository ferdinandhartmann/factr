import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
from pathlib import Path
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf
import pickle
import yaml
import sys
import warnings
import math
import os
import cv2
import argparse
from tqdm import tqdm
import matplotlib.animation as animation
from scipy.stats import norm

# --- インポートの直後に追加 ---
# def seed_everything(seed=42):
#     import random
#     import os
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # マルチGPU用
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# seed_everything(42)

# --- 警告の抑制 ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*")

# --- factrモジュールのインポート試行 ---
try:
    import factr.misc
except ImportError:
    pass


# --- Hydra Resolverの登録 ---
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

        factr.misc.get_transform_by_name = lambda name: (lambda x: x)
        print("✅ Patched factr.misc.get_transform_by_name for inference.")
except Exception as e:
    print(f"⚠️ Failed to patch factr.misc: {e}")

# --- 引数解析 ---
parser = argparse.ArgumentParser(description="Run analysis for a specific model.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model to load")
args = parser.parse_args()

model_name = args.model_name
checkpoint = "ckpt_020000"
episode_names = [f"ep_{i:02d}" for i in range(51, 63)]  # 必要に応じて範囲を変更
target_episodes = episode_names

GPU_ID = 0
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 10
MODE = "stiff"  # soft or stiff

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / model_name

if MODE == "stiff":
    RAW_DATA_BASE = Path("/data/otake/box_lift_up_side/20251218_stiff/eval")
else:
    RAW_DATA_BASE = Path("/data/otake/box_lift_up_side/20251217_soft/eval")

CKPT_PATH = CHECKPOINTS_DIR / f"{checkpoint}.ckpt"
SAVE_DIR_ROOT = PROJECT_ROOT / "result_output" / model_name
SAVE_DIR = SAVE_DIR_ROOT / f"{MODE}_{NUM_SAMPLES}"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# --- 1. 統計データの読み込みと正規化関数 ---


def load_dataset_stats(data_root):
    """
    データルートの親ディレクトリなどから dataset_stats.pkl を探して読み込む
    """
    stats_path = data_root.parent / "dataset_stats.pkl"
    if not stats_path.exists():
        stats_path = data_root / "dataset_stats.pkl"

    if not stats_path.exists():
        print(f"⚠️ Warning: dataset_stats.pkl not found at {stats_path}. Normalization will be skipped.")
        return None

    print(f"✅ Loading dataset stats from {stats_path}")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return stats


def normalize_data(data, stats, key):
    """
    ガウシアン正規化: (x - mean) / std
    """
    if stats is None:
        return data

    if key == "position" and "position" not in stats and "action" in stats:
        stat_key = "action"
    else:
        stat_key = key

    if stat_key not in stats:
        print(f"⚠️ Key '{stat_key}' not found in stats. Skipping normalization.")
        return data

    mean = stats[stat_key]["mean"]
    std = stats[stat_key]["std"]
    eps = 1e-6

    return (data - mean) / (std + eps)


# --- 2. データ読み込み ---
def load_and_extract_raw_data(pkl_path: Path):
    print(f"Loading raw data from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        raw_data = pickle.load(f)

    image_obs, torque_obs, actions = [], [], []

    if "data" not in raw_data:
        entries = raw_data
    else:
        entries = raw_data["data"]

    image_topic = "/realsense/front/im"
    obs_topic = "/franka_robot_state_broadcaster/external_joint_torques"
    possible_action_topics = [
        "/joint_impedance_dynamic_gain_controller/joint_impedance_command",
        "/joint_impedance_command_controller/joint_trajectory",
    ]

    action_topic = None
    for t in possible_action_topics:
        if t in entries:
            action_topic = t
            break

    if action_topic is None:
        print(f"❌ Action topic not found. Keys: {list(entries.keys())}")
        return [], [], []

    for v in entries[action_topic]:
        if isinstance(v, dict) and "position" in v:
            actions.append(v["position"])

    if not actions:
        print("❌ No actions found.")
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

    if len(torque_obs) > 0:
        torque_obs = np.array(torque_obs)
    else:
        print("⚠️ No torque data found. Filling with zeros.")
        torque_obs = np.zeros((len(actions), 7))

    actions = np.array(actions)
    if len(image_obs) == 0:
        print("❌ No images found.")
        return [], [], []

    N = min(len(image_obs), len(torque_obs), len(actions))
    print(
        f"✅ Extracted data lengths - Img:{len(image_obs)}, Trq:{len(torque_obs)}, Act:{len(actions)} -> Synced N={N}"
    )
    if N == 0:
        return [], [], []

    return image_obs[:N], torque_obs[:N], actions[:N]


def load_episode_data_tensor(data_root, ep_name, device, stats=None):
    path = data_root / f"{ep_name}.pkl"
    if not path.exists():
        path = data_root / ep_name / "data.pkl"
        if not path.exists():
            raise FileNotFoundError(f"File not found: {ep_name}.pkl or {ep_name}/data.pkl in {data_root}")

    raw_imgs, raw_torques, raw_actions = load_and_extract_raw_data(path)
    if len(raw_imgs) == 0:
        raise ValueError(f"Failed to extract valid data from {path}")

    processed_imgs = []
    target_size = (224, 224)
    for img in raw_imgs:
        img_resized = cv2.resize(img, target_size)
        processed_imgs.append(img_resized)

    imgs_np = np.array(processed_imgs)
    imgs_tensor = torch.from_numpy(imgs_np).float() / 255.0
    imgs_tensor = imgs_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)

    # 正規化適用
    norm_torques = normalize_data(raw_torques, stats, "joint_torques")
    norm_actions = normalize_data(raw_actions, stats, "position")

    torques_tensor = torch.from_numpy(norm_torques).float().unsqueeze(0).to(device)
    actions_tensor = torch.from_numpy(norm_actions).float().unsqueeze(0).to(device)

    return {"camera": imgs_tensor, "joint_torques": torques_tensor, "position": actions_tensor}


def load_model(ckpt_path, device):
    print(f"Loading model from {ckpt_path}...")
    cfg_path = ckpt_path.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        cfg_path = ckpt_path.parent / "config.yaml"
    if not cfg_path.exists():
        cfg_path = ckpt_path.parent.parent / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found around {ckpt_path}")

    cfg = OmegaConf.load(cfg_path)
    if "params" in cfg and "agent" in cfg.params:
        model_cfg = cfg.params.agent
    elif "agent" in cfg:
        model_cfg = cfg.agent
    elif "model" in cfg:
        model_cfg = cfg.model
    else:
        raise KeyError("Model config key not found.")

    model = instantiate(model_cfg)
    state_dict = torch.load(ckpt_path, map_location=device)

    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("model.", "").replace("agent.", "")
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device).eval()
    return model, cfg


def load_dataset_stats(data_root):
    """
    rollout_config.yaml (または dataset_stats) を探して読み込み、正規化用辞書を返す
    """

    candidates = ["rollout_config.yaml", "dataset_stats.yaml", "dataset_stats.pkl"]

    search_dirs = [
        CHECKPOINTS_DIR / "rollout",
        CHECKPOINTS_DIR,  # チェックポイント直下
        data_root,  # evalフォルダ
        data_root.parent,  # stiff/soft フォルダ
        data_root.parent.parent,  # 実験ルート
    ]

    found_path = None
    for d in search_dirs:
        for fname in candidates:
            p = d / fname
            if p.exists():
                found_path = p
                break
        if found_path:
            break

    if found_path is None:
        print(f"⚠️ Warning: Config with stats not found. Searched for {candidates} in:")
        for d in search_dirs:
            print(f"  - {d}")
        print("Normalization will be skipped.")
        return None

    print(f"✅ Loading dataset stats from {found_path}")

    stats = {}

    # --- YAMLの場合 ---
    if found_path.suffix == ".yaml":
        with open(found_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        if "norm_stats" in yaml_data:
            raw_stats = yaml_data["norm_stats"]

            if "state" in raw_stats:
                stats["joint_torques"] = {
                    "mean": np.array(raw_stats["state"]["mean"]),
                    "std": np.array(raw_stats["state"]["std"]),
                }

            if "action" in raw_stats:
                stats["position"] = {
                    "mean": np.array(raw_stats["action"]["mean"]),
                    "std": np.array(raw_stats["action"]["std"]),
                }
                stats["action"] = stats["position"]
        else:
            print(f"❌ 'norm_stats' key not found in {found_path}")
            print(f"   Keys found: {list(yaml_data.keys())}")
            return None

    elif found_path.suffix == ".pkl":
        with open(found_path, "rb") as f:
            stats = pickle.load(f)

    return stats


@torch.no_grad()
def extract_z_params(policy, data):
    print("Extracting Prior/Posterior parameters...")
    policy.eval()

    device = next(policy.parameters()).device
    chunk_size = policy.ac_chunk

    images = data["camera"]
    effort = data["joint_torques"]
    gt_actions = data["position"]

    T = gt_actions.shape[1]

    prior_mus, prior_stds = [], []
    post_mus, post_stds = [], []

    if T <= chunk_size:
        return {}

    for t in tqdm(range(T - chunk_size), desc="Processing Z Params"):
        # -------------------------------------------------
        # 1. 画像・トルク -> Context取得
        # -------------------------------------------------
        current_img = images[:, t : t + 1]
        if current_img.ndim == 5 and current_img.shape[1] == 1:
            current_img = current_img.squeeze(1)

        current_effort = effort[:, t : t + 1]
        if current_effort.ndim == 3:
            current_effort = current_effort.squeeze(1)

        img_dict = {"cam0": current_img.to(device)}
        effort_tensor = current_effort.to(device)

        c_tokens = policy.tokenize_obs(img_dict, effort_tensor)  # (B, N, D)
        cls_tok, force_tok = policy._extract_cls_force(c_tokens)

        mu_p, logvar_p = policy.prior(cls_tok, force_tok)
        std_p = torch.exp(0.5 * logvar_p)

        # print("prior mean", mu_p)
        # print("prior std", std_p)

        prior_mus.append(mu_p.cpu().numpy().squeeze(0))  # (Dz,)
        prior_stds.append(std_p.cpu().numpy().squeeze(0))  # (Dz,)

        # -------------------------------------------------
        # 3. Posterior Parameters (Transformer)
        # -------------------------------------------------
        target_action_chunk = gt_actions[:, t : t + chunk_size].to(device)

        c_for_posterior = torch.stack([cls_tok, force_tok], dim=1)

        mu_q, logvar_q = policy.posterior(c_for_posterior.detach(), target_action_chunk, gt_only=policy.gt_only)
        std_q = torch.exp(0.5 * logvar_q)

        post_mus.append(mu_q.cpu().numpy().squeeze(0))
        post_stds.append(std_q.cpu().numpy().squeeze(0))

    return {
        "Prior": (np.array(prior_mus), np.array(prior_stds), "blue"),
        "Posterior": (np.array(post_mus), np.array(post_stds), "red"),
    }


def visualize_distributions_video(dists_data, save_path, ep_name):
    if not dists_data:
        print("No data to visualize.")
        return

    print(f"Preparing video layout for {ep_name}...")

    T = dists_data["Prior"][0].shape[0]
    z_dim = dists_data["Prior"][0].shape[1]

    ncols = int(np.ceil(np.sqrt(z_dim)))
    nrows = int(np.ceil(z_dim / ncols))
    figsize = (ncols * 2.5, nrows * 2.5)

    all_stds = np.concatenate([d[1] for d in dists_data.values()])
    x_min, x_max = -5, 5
    x_range = np.linspace(x_min, x_max, 100)
    min_std = all_stds.min()
    y_max = 1.0 / (np.sqrt(2 * np.pi) * min_std) * 1.1
    y_max = min(y_max, 5.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.93, hspace=0.4, wspace=0.3)

    lines = {}
    for i in range(len(axes)):
        ax = axes[i]
        if i >= z_dim:
            ax.axis("off")
            continue

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_title(f"Dim {i}", fontsize=9)
        lines[i] = {}
        for name, (_, _, color) in dists_data.items():
            lw = 2.0 if name == "Posterior" else 1.5
            alpha = 0.9 if name == "Posterior" else 0.6
            (line,) = ax.plot([], [], color=color, label=name, linewidth=lw, alpha=alpha)
            lines[i][name] = line

    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist="Me"), bitrate=1800)

    with writer.saving(fig, str(save_path), dpi=100):
        for t in tqdm(range(T), desc=f"Rendering {ep_name}"):
            fig.suptitle(f"Z Dist: {ep_name} (t={t}/{T})", fontsize=16)

            for i in range(z_dim):
                if i >= len(axes):
                    break
                for name, (mu_seq, std_seq, _) in dists_data.items():
                    mu = mu_seq[t, i]
                    std = std_seq[t, i]
                    y = norm.pdf(x_range, loc=mu, scale=std)
                    lines[i][name].set_data(x_range, y)

            writer.grab_frame()

    plt.close()
    print(f"✅ Saved to {save_path}")


def visualize_z_statistics(dists_data, save_dir, ep_name):
    """静止画（平均と分散の時系列グラフ）を作成する関数"""
    if not dists_data:
        print("No data to visualize.")
        return

    print(f"Creating time-series plots for {ep_name}...")

    # データの取得
    prior_mu, prior_std, prior_color = dists_data["Prior"]
    post_mu, post_std, post_color = dists_data["Posterior"]

    # 分散の計算 (Std^2)
    prior_var = prior_std**2
    post_var = post_std**2

    # Y軸統一のためのMin/Max取得
    all_mu = np.concatenate([prior_mu, post_mu])
    mu_min, mu_max = all_mu.min(), all_mu.max()
    mu_margin = (mu_max - mu_min) * 0.1 if mu_max != mu_min else 1.0
    mu_ylim = (mu_min - mu_margin, mu_max + mu_margin)

    all_var = np.concatenate([prior_var, post_var])
    var_min, var_max = all_var.min(), all_var.max()

    var_margin = (var_max - var_min) * 0.1 if var_max != var_min else 0.1
    var_ylim = (max(0, var_min - var_margin), var_max + var_margin)

    T, z_dim = prior_mu.shape
    time_steps = np.arange(T)

    dims_per_page = 8
    num_pages = int(np.ceil(z_dim / dims_per_page))

    for page in range(num_pages):
        start_dim = page * dims_per_page
        end_dim = min((page + 1) * dims_per_page, z_dim)

        fig, axes = plt.subplots(dims_per_page, 2, figsize=(16, 20), sharex=True)
        if dims_per_page == 1:
            axes = axes.reshape(1, 2)

        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.95, hspace=0.3, wspace=0.2)
        fig.suptitle(f"Z Statistics: (Dims {start_dim}-{end_dim - 1})", fontsize=16)

        for i in range(dims_per_page):
            dim_idx = start_dim + i

            ax_mean = axes[i, 0]
            if dim_idx < z_dim:
                ax_mean.plot(time_steps, prior_mu[:, dim_idx], label="Prior", color=prior_color, alpha=0.7)
                ax_mean.plot(time_steps, post_mu[:, dim_idx], label="Posterior", color=post_color, alpha=0.7)
                ax_mean.set_ylabel(f"Dim {dim_idx}\nMean", fontsize=12)
                ax_mean.grid(True, linestyle="--", alpha=0.5)

                ax_mean.set_ylim(mu_ylim)

                if i == 0:
                    ax_mean.legend(loc="upper right")
            else:
                ax_mean.axis("off")

            ax_var = axes[i, 1]
            if dim_idx < z_dim:
                ax_var.plot(time_steps, prior_var[:, dim_idx], label="Prior", color=prior_color, alpha=0.7)
                ax_var.plot(time_steps, post_var[:, dim_idx], label="Posterior", color=post_color, alpha=0.7)
                ax_var.set_ylabel("Variance", fontsize=12)
                ax_var.grid(True, linestyle="--", alpha=0.5)

                ax_var.set_ylim(var_ylim)
            else:
                ax_var.axis("off")

        if dims_per_page > 0:
            axes[-1, 0].set_xlabel("Time Step", fontsize=14)
            axes[-1, 1].set_xlabel("Time Step", fontsize=14)

        save_path = save_dir / f"z_stat_{ep_name}_page{page + 1}_.png"
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved plot to {save_path}")


# --- 4. Main Execution ---
if __name__ == "__main__":
    # モデルと設定のロード
    model, cfg = load_model(CKPT_PATH, DEVICE)

    # 統計データのロード
    stats = load_dataset_stats(RAW_DATA_BASE)

    # 分析対象のエピソード
    target_episodes = episode_names

    for ep_name in target_episodes:
        print(f"\n🚀 Processing Episode: {ep_name}")

        try:
            data = load_episode_data_tensor(RAW_DATA_BASE, ep_name, DEVICE, stats=stats)
        except Exception as e:
            print(f"⚠️ Skipping {ep_name} due to data error: {e}")
            continue

        # 2. 潜在変数を抽出
        dists_data = extract_z_params(model, data)

        ##pkl保存の追加
        raw_data_path = SAVE_DIR / f"z_raw_data_{ep_name}.pkl"
        with open(raw_data_path, "wb") as f:
            pickle.dump(dists_data, f)
        print(f"✅ Saved raw Z data to {raw_data_path}")
        ####

        # 5. 動画保存
        # save_file = SAVE_DIR / f"z_dist_{MODE}_{ep_name}_.mp4"
        # visualize_distributions_video(dists_data, save_file, ep_name)

        ####グラフの追加
        visualize_z_statistics(dists_data, SAVE_DIR, ep_name)
