#!/usr/bin/env python3
"""Deterministically decode soft mode/stiffness conditions and inspect their posterior PCA."""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = SCRIPT_DIR / "eval_params.yaml"


def build_interpolation_vectors(step: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return alpha and [1-alpha, alpha], including exact hard-label endpoints."""
    if not 0.0 < float(step) <= 1.0:
        raise ValueError(f"interpolation_step must be in (0, 1], got {step}.")
    count = int(round(1.0 / float(step)))
    if not np.isclose(count * float(step), 1.0, atol=1e-7):
        raise ValueError("interpolation_step must divide the interval [0, 1] exactly.")
    alpha = np.linspace(0.0, 1.0, count + 1, dtype=np.float32)
    return alpha, np.stack([1.0 - alpha, alpha], axis=1)


def fit_pca_2d(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit deterministic two-component PCA using SVD and return scores/components/variance."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"PCA expects shape (N, D) with N >= 2, got {values.shape}.")
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    num_components = min(2, vt.shape[0])
    components = vt[:num_components].copy()

    # Fix SVD's arbitrary sign so repeated runs produce directly comparable coordinates.
    for idx in range(num_components):
        pivot = int(np.argmax(np.abs(components[idx])))
        if components[idx, pivot] < 0.0:
            components[idx] *= -1.0
    scores = centered @ components.T
    explained = singular_values[:num_components] ** 2
    total = float(np.square(singular_values).sum())
    ratio = explained / total if total > 0.0 else np.zeros_like(explained)

    if num_components < 2:
        scores = np.pad(scores, ((0, 0), (0, 2 - num_components)))
        components = np.pad(components, ((0, 2 - num_components), (0, 0)))
        ratio = np.pad(ratio, (0, 2 - num_components))
    return scores.astype(np.float32), components.astype(np.float32), ratio.astype(np.float32)


@torch.no_grad()
def posterior_features(model, context_tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return the deterministic posterior representation used by PCA."""
    params = model.posterior(context_tokens[:, :-1], actions)
    latent_distribution = str(getattr(model, "latent_distribution", "gaussian")).lower()
    if latent_distribution == "categorical":
        logits = params["logits"] if isinstance(params, dict) else params
        return F.softmax(logits, dim=-1).flatten(start_dim=1)
    if latent_distribution == "gaussian":
        mean = params["mu"] if isinstance(params, dict) else params[0]
        return mean.flatten(start_dim=1)
    raise ValueError(f"Unsupported latent_distribution={latent_distribution}.")


@torch.no_grad()
def predict_and_reencode(
    model,
    obs: torch.Tensor,
    target_actions: torch.Tensor,
    condition: torch.Tensor,
    arrangement: Optional[torch.Tensor],
    goal_vectors: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Decode the posterior mode/mean so reconstruction comparisons contain no sampling noise.
    predicted = model.get_actions_pos(
        {},
        obs,
        target_actions,
        class_labels=condition,
        arrangement_vectors=arrangement,
        goal_vectors=goal_vectors,
        sample=False,
        num_samples=1,
    )
    if predicted.ndim != 4 or predicted.shape[1] != 1:
        raise ValueError(f"Expected posterior prediction shape (B, 1, T, A), got {tuple(predicted.shape)}.")
    predicted = predicted[:, 0]
    context = model._build_context_tokens(
        obs,
        class_labels=condition,
        arrangement_vectors=arrangement,
        goal_vectors=goal_vectors,
    )
    return predicted, posterior_features(model, context, predicted)


def _load_config(path: Path) -> Dict:
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle)
    merged = dict(raw["shared"])
    merged.update(raw["eval_stiffness_interpolation_pca"])
    return merged


def _paths(cfg: Dict) -> Tuple[Path, Path]:
    project_name = cfg["dataset_name"] if str(cfg["buffer_set_name"]).strip().lower() == "auto" else str(cfg["buffer_set_name"])
    key = f"{cfg['dataset_project_prefix']}{project_name}"
    run_dir = Path.home() / "activeinference" / "factr" / "checkpoints" / key / cfg["run_name"] / "rollout"
    raw_dir = Path.home() / "activeinference" / "factr" / "process_data" / "data_to_process" / cfg["dataset_name"] / "data"
    return run_dir, raw_dir


def _episode_mode_label(episode_path: Path, rollout_cfg: Dict, fallback: int, episode_eval, use_mode: bool) -> int:
    """Use mode 0/1 as class 1/2 when the processed configuration enables mode override."""
    if not use_mode:
        return int(fallback)
    mode_topic = rollout_cfg.get("obs_config", {}).get("mode_topic")
    if not mode_topic:
        return int(fallback)
    with open(episode_path, "rb") as handle:
        raw = pickle.load(handle)
    messages = raw.get("data", {}).get(mode_topic, [])
    if not messages:
        return int(fallback)
    message = messages[0]
    if isinstance(message, dict):
        value = next((message[key] for key in ("mode", "data", "value") if key in message), None)
        if value is None:
            raise ValueError(f"Could not extract mode from {mode_topic} message keys {list(message)}.")
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    else:
        vector = episode_eval._extract_fallback_vector(message)
    mode = int(round(float(vector[0])))
    if mode not in (0, 1):
        raise ValueError(f"Expected {mode_topic} to contain mode 0 or 1, got {mode}.")
    return mode + 1


def _plot_pca(
    scores: np.ndarray,
    predicted_actions: np.ndarray,
    num_steps: int,
    alpha: np.ndarray,
    explained_ratio: np.ndarray,
    plot_stride: int,
    color_map: str,
    ground_truth_label: int,
    title: str,
    output_path: Path,
    show_plot: bool,
) -> None:
    pred_count = len(alpha) * num_steps
    pred_scores = scores[:pred_count].reshape(len(alpha), num_steps, 2)
    gt_scores = scores[pred_count:]
    stride_idx = np.arange(0, num_steps, max(1, int(plot_stride)))
    if stride_idx[-1] != num_steps - 1:
        stride_idx = np.append(stride_idx, num_steps - 1)

    predictions = np.asarray(predicted_actions, dtype=np.float32)
    expected_shape = (len(alpha), num_steps)
    if predictions.ndim != 4 or predictions.shape[:2] != expected_shape:
        raise ValueError(f"Expected predicted_actions shape ({len(alpha)}, {num_steps}, T, A), got {predictions.shape}.")

    fig, (ax, ax_distance) = plt.subplots(1, 2, figsize=(17, 7.5))
    try:
        cmap = plt.get_cmap(color_map)
    except ValueError as exc:
        raise ValueError(f"Unknown Matplotlib color_map={color_map!r}.") from exc
    for idx, weight in enumerate(alpha):
        points = pred_scores[idx, stride_idx]
        color = cmap(float(weight))
        ax.scatter(points[:, 0], points[:, 1], color=color, s=12, alpha=0.55)

    # Each faint line holds episode time fixed and varies only the conditioning vector.
    for time_idx in stride_idx:
        condition_path = pred_scores[:, time_idx]
        ax.plot(condition_path[:, 0], condition_path[:, 1], color="0.35", alpha=0.14, linewidth=0.7, zorder=1)

    gt_points = gt_scores[stride_idx]
    gt_alpha = float(ground_truth_label - 1)
    gt_color = cmap(gt_alpha)
    ax.scatter(
        gt_points[:, 0],
        gt_points[:, 1],
        marker="X",
        color=gt_color,
        edgecolors="black",
        linewidths=0.7,
        s=42,
        alpha=0.85,
        label=f"Ground truth (class {ground_truth_label}, alpha={gt_alpha:.1f})",
        zorder=6,
    )

    centroids = pred_scores.mean(axis=1)
    ax.plot(centroids[:, 0], centroids[:, 1], color="white", linewidth=4.5, zorder=4)
    ax.plot(centroids[:, 0], centroids[:, 1], color="crimson", linewidth=2.0, marker="o", markersize=5, zorder=5, label="Condition centroid")
    for idx, weight in enumerate(alpha):
        ax.annotate(f"{weight:.1f}", centroids[idx], xytext=(5, 4), textcoords="offset points", fontsize=8)

    scalar = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, pad=0.02)
    colorbar.set_label(r"Second condition weight $\alpha$ in $[1-\alpha,\alpha]$")
    ax.set_xlabel(f"PC1 ({100.0 * explained_ratio[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100.0 * explained_ratio[1]:.1f}% variance)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")

    # Compare decoded actions directly, before re-encoding and PCA projection.
    reference = predictions[0:1]
    per_window_l1 = np.mean(np.abs(predictions - reference), axis=(2, 3))
    distance_mean = per_window_l1.mean(axis=1)
    distance_std = per_window_l1.std(axis=1)
    curve_colors = [cmap(float(weight)) for weight in alpha]
    ax_distance.plot(alpha, distance_mean, color="0.2", linewidth=1.5, zorder=2)
    ax_distance.scatter(alpha, distance_mean, c=curve_colors, s=55, edgecolors="black", linewidths=0.5, zorder=3)
    ax_distance.fill_between(
        alpha,
        np.maximum(0.0, distance_mean - distance_std),
        distance_mean + distance_std,
        color="0.4",
        alpha=0.15,
        label="Across-window mean ± std",
    )
    ax_distance.set_xlabel(r"Second condition weight $\alpha$ in $[1-\alpha,\alpha]$")
    ax_distance.set_ylabel(r"Mean action L1 distance from $[1,0]$")
    ax_distance.set_title("Condition effect in normalized action space")
    ax_distance.set_xticks(alpha)
    ax_distance.grid(True, linestyle="--", alpha=0.25)
    ax_distance.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def main() -> None:
    # Keep project-specific dependencies lazy so PCA helpers can be tested in isolation.
    from omegaconf import OmegaConf
    from scripts import eval_single_episode_lowdim as episode_eval

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    script_cfg = _load_config(args.config)
    alpha, conditions = build_interpolation_vectors(float(script_cfg["interpolation_step"]))

    run_dir, raw_episode_dir = _paths(script_cfg)
    exp_config_path = run_dir / "exp_config.yaml"
    checkpoint_path = run_dir / str(script_cfg["checkpoint_name"])
    run_cfg = episode_eval._load_run_cfg(exp_config_path)
    rollout_path = run_dir / "rollout_config.yaml"
    if not rollout_path.exists():
        rollout_path = Path(run_cfg.test_buffer_path).parent / "rollout_config.yaml"
    rollout_cfg = episode_eval._load_rollout_config(rollout_path)

    stiffness_classes = int(OmegaConf.select(run_cfg, "agent.stiffness_classes", default=1))
    use_conditioning = bool(OmegaConf.select(run_cfg, "agent.use_stiffness_conditioning", default=True))
    override_with_mode = bool(OmegaConf.select(run_cfg, "override_stiffness_with_mode", default=False))
    if not use_conditioning or stiffness_classes != 2:
        raise ValueError(f"Interpolation requires a conditioned two-class model; got enabled={use_conditioning}, classes={stiffness_classes}.")

    raw_dirs = episode_eval._build_rollout_raw_dirs(rollout_cfg, raw_episode_dir)
    key = f"{script_cfg['dataset_project_prefix']}{script_cfg['dataset_name'] if str(script_cfg['buffer_set_name']).lower() == 'auto' else script_cfg['buffer_set_name']}"
    configured_episodes = list(script_cfg.get("episode_lists", {}).get(key, []))
    if bool(script_cfg["use_episode_list"]) and configured_episodes:
        selected = episode_eval._resolve_rollout_episodes(configured_episodes, raw_dirs)
    elif str(script_cfg["buffer_set_name"]).strip().lower() == "auto":
        selected = episode_eval._resolve_rollout_episodes(episode_eval._get_required_split_episodes(rollout_cfg, "test"), raw_dirs)
    else:
        files = episode_eval._list_episode_files(raw_episode_dir)
        selected_path = episode_eval._select_episode_file(files, script_cfg["episode_file_name"], int(script_cfg.get("episode_index", 0)))
        selected = [(selected_path.stem, selected_path)]

    device = torch.device(f"cuda:{int(script_cfg['gpu_id'])}" if torch.cuda.is_available() else "cpu")
    model = episode_eval._load_model(run_cfg, checkpoint_path, device)
    state_stats = rollout_cfg.get("norm_stats", {}).get("state")
    action_stats = rollout_cfg.get("norm_stats", {}).get("action")
    action_pose_mode = episode_eval._infer_action_pose_mode(run_cfg, rollout_cfg, action_stats)
    output_override = script_cfg.get("out_dir_override")
    output_dir = Path(output_override).expanduser() if output_override else run_dir.parent / "stiffness_interpolation_pca"
    output_dir.mkdir(parents=True, exist_ok=True)

    for episode_id, episode_path in selected:
        raw_episode = episode_eval._load_raw_episode_to_arrays(episode_path, rollout_cfg)
        states = raw_episode["states"]
        actions = raw_episode["actions"]
        if action_pose_mode == "delta" and len(actions) > 1:
            relative_actions = np.zeros_like(actions)
            relative_actions[1:] = actions[1:] - actions[:-1]
            actions = relative_actions
        include_tracking = bool(OmegaConf.select(run_cfg, "agent.include_tracking_error", default=True))
        if not include_tracking and states.shape[-1] >= 36:
            states = np.concatenate([states[:, :21], states[:, 27:]], axis=-1)
        actual_label = _episode_mode_label(
            episode_path,
            rollout_cfg,
            raw_episode["episode_label"],
            episode_eval,
            use_mode=override_with_mode,
        )
        samples = episode_eval._build_eval_samples_from_raw_episode(
            states,
            actions,
            actual_label,
            int(run_cfg.obs_window),
            int(run_cfg.ac_chunk),
            int(OmegaConf.select(run_cfg, "task.test_buffer.action_index_offset", default=1)),
        )
        obs_norm, _ = episode_eval._ensure_normalized(samples["obs"], state_stats, script_cfg["normalization_mode"], "state")
        actions_norm, _ = episode_eval._ensure_normalized(samples["actions"], action_stats, script_cfg["normalization_mode"], "action")
        obs = torch.from_numpy(obs_norm).float().to(device)
        actions = torch.from_numpy(actions_norm).float().to(device)
        arrangement = None
        if bool(OmegaConf.select(run_cfg, "agent.use_arrangement_conditioning", default=False)):
            if raw_episode["arrangement_vector"] is None:
                raise ValueError(f"Episode {episode_id} has no arrangement vector required by the checkpoint.")
            arrangement = torch.from_numpy(np.repeat(raw_episode["arrangement_vector"][None], len(obs), axis=0)).float().to(device)

        goal_vectors = None
        if bool(getattr(model, "goal_label", False)):
            if raw_episode["goal_vectors"] is None:
                raise ValueError(f"Episode {episode_id} has no goal labels required by the checkpoint.")
            # Samples are indexed by the final timestep of each observation window.
            goal_np = raw_episode["goal_vectors"][samples["steps"]].astype(np.float32)
            goal_vectors = torch.from_numpy(goal_np).float().to(device)

        feature_groups = []
        prediction_groups = []
        for condition_np in conditions:
            condition = torch.from_numpy(np.repeat(condition_np[None], len(obs), axis=0)).float().to(device)
            predictions, features = predict_and_reencode(
                model, obs, actions, condition, arrangement, goal_vectors
            )
            prediction_groups.append(predictions.cpu().numpy())
            feature_groups.append(features.cpu().numpy())

        hard_labels = torch.full((len(obs),), actual_label, dtype=torch.long, device=device)
        gt_context = model._build_context_tokens(
            obs,
            class_labels=hard_labels,
            arrangement_vectors=arrangement,
            goal_vectors=goal_vectors,
        )
        gt_features = posterior_features(model, gt_context, actions).cpu().numpy()
        all_features = np.concatenate(feature_groups + [gt_features], axis=0)
        scores, _, explained = fit_pca_2d(all_features)

        stem = episode_eval._plot_file_stem(episode_id)
        output_path = output_dir / f"{stem}_stiffness_interpolation_pca.png"
        _plot_pca(
            scores,
            np.stack(prediction_groups, axis=0),
            len(obs),
            alpha,
            explained,
            int(script_cfg["plot_stride"]),
            str(script_cfg["color_map"]),
            actual_label,
            f"{episode_id}: deterministic posterior reconstruction PCA",
            output_path,
            bool(script_cfg["show_plot"]),
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
