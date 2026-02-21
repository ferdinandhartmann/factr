from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RPYPlotConfig:
    subtract_pi: bool = True
    subtract_pi_axis: int = 0
    unit: str = "deg"


DEFAULT_RPY_PLOT_CONFIG = RPYPlotConfig()


def make_pose_dim_names(dim: int) -> List[str]:
    default_names = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6"]
    if dim <= len(default_names):
        return default_names[:dim]
    return [f"dim{i + 1}" for i in range(dim)]


def make_obs_dim_names(dim: int) -> List[str]:
    names = [
        "pose_x",
        "pose_y",
        "pose_z",
        "pose_r1",
        "pose_r2",
        "pose_r3",
        "pose_r4",
        "pose_r5",
        "pose_r6",
        "vel_x",
        "vel_y",
        "vel_z",
        "vel_rx",
        "vel_ry",
        "vel_rz",
        "wrench_fx",
        "wrench_fy",
        "wrench_fz",
        "wrench_tx",
        "wrench_ty",
        "wrench_tz",
    ]
    if dim <= len(names):
        return names[:dim]
    return [f"obs_{idx + 1}" for idx in range(dim)]


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if (not np.isfinite(norm)) or norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return vec / norm


def rot6d_to_matrix(rot6: np.ndarray) -> np.ndarray:
    a1 = np.asarray(rot6[:3], dtype=np.float32)
    a2 = np.asarray(rot6[3:6], dtype=np.float32)
    if np.any(~np.isfinite(a1)) or np.any(~np.isfinite(a2)):
        return np.eye(3, dtype=np.float32)
    b1 = _normalize_vec(a1)
    a2_orth = a2 - float(np.dot(b1, a2)) * b1
    b2 = _normalize_vec(a2_orth)
    b3 = np.cross(b1, b2)
    b3 = _normalize_vec(b3)
    return np.stack([b1, b2, b3], axis=1)


def matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    # Intrinsic XYZ (roll, pitch, yaw) extracted from rotation matrix.
    pitch = float(np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        # Gimbal-lock fallback: keep yaw fixed and solve roll from remaining terms.
        roll = float(np.arctan2(-rot[1, 2], rot[1, 1]))
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    if angles.ndim != 2 or angles.shape[0] < 2:
        return angles
    return np.unwrap(angles, axis=0).astype(np.float32)


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    return ((angles + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def align_angles_to_reference(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    if reference.shape != values.shape or reference.ndim != 2:
        return values
    return (reference + wrap_to_pi(values - reference)).astype(np.float32)


def _get_configured_rpy_axis(angles: np.ndarray, cfg: RPYPlotConfig) -> int:
    if angles.ndim != 2 or angles.shape[1] < 3:
        return -1
    axis = int(cfg.subtract_pi_axis)
    if axis < 0 or axis >= angles.shape[1]:
        return -1
    return axis


def get_pi_shift_from_axis_start(angles: np.ndarray, cfg: RPYPlotConfig = DEFAULT_RPY_PLOT_CONFIG) -> float:
    if not bool(cfg.subtract_pi):
        return 0.0
    axis = _get_configured_rpy_axis(angles, cfg)
    if axis < 0:
        return 0.0
    start_val = float(angles[0, axis])
    if not np.isfinite(start_val):
        return 0.0
    if start_val < -2.0:
        return float(np.pi)
    if start_val > 2.0:
        return float(-np.pi)
    return 0.0


def apply_rpy_axis_shift(angles: np.ndarray, shift_value: float, cfg: RPYPlotConfig = DEFAULT_RPY_PLOT_CONFIG) -> np.ndarray:
    axis = _get_configured_rpy_axis(angles, cfg)
    if axis < 0 or abs(float(shift_value)) < 1e-12:
        return angles
    shifted = angles.copy()
    shifted[:, axis] = shifted[:, axis] + float(shift_value)
    return shifted.astype(np.float32)


def get_rpy_plot_unit(cfg: RPYPlotConfig = DEFAULT_RPY_PLOT_CONFIG) -> str:
    unit = str(cfg.unit).strip().lower()
    if unit not in ("rad", "deg"):
        return "rad"
    return unit


def convert_rpy_to_plot_unit(angles_rad: np.ndarray, cfg: RPYPlotConfig = DEFAULT_RPY_PLOT_CONFIG) -> np.ndarray:
    unit = get_rpy_plot_unit(cfg)
    if unit == "deg":
        return np.rad2deg(angles_rad).astype(np.float32)
    return angles_rad.astype(np.float32)


def compute_pose_rpy(pose: np.ndarray) -> np.ndarray:
    if pose.ndim != 2 or pose.shape[1] < 9:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.zeros((pose.shape[0], 3), dtype=np.float32)
    for idx in range(pose.shape[0]):
        rot = rot6d_to_matrix(pose[idx, 3:9])
        out[idx] = matrix_to_rpy(rot)
    return unwrap_angles(out)


def _rotation_geodesic_distance_rad(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = rot_a.T @ rot_b
    trace_rel = float(np.trace(rel))
    cos_theta = np.clip((trace_rel - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_theta))


def compute_pose_geodesic_distance(true_pose: np.ndarray, pred_pose: np.ndarray) -> np.ndarray:
    if true_pose.ndim != 2 or pred_pose.ndim != 2:
        return np.zeros((0,), dtype=np.float32)
    if true_pose.shape[1] < 9 or pred_pose.shape[1] < 9:
        return np.zeros((0,), dtype=np.float32)
    if true_pose.shape[0] != pred_pose.shape[0]:
        return np.zeros((0,), dtype=np.float32)

    geod = np.zeros((true_pose.shape[0],), dtype=np.float32)
    for idx in range(true_pose.shape[0]):
        rot_true = rot6d_to_matrix(true_pose[idx, 3:9])
        rot_pred = rot6d_to_matrix(pred_pose[idx, 3:9])
        geod[idx] = _rotation_geodesic_distance_rad(rot_true, rot_pred)
    return geod


def _valid_rows_from_mask(mask: np.ndarray) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 1:
        return mask_arr > 0
    if mask_arr.ndim >= 2:
        return mask_arr[:, 0] > 0
    return np.zeros((0,), dtype=bool)


def build_pose_comparison_figure(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    mask: np.ndarray,
    title: str,
    measured_values: Optional[np.ndarray] = None,
    dim_names: Optional[Sequence[str]] = None,
    true_label: str = "ground truth",
    pred_label: str = "prediction",
    measured_label: str = "measured_pose",
    rpy_config: RPYPlotConfig = DEFAULT_RPY_PLOT_CONFIG,
):
    valid_rows = _valid_rows_from_mask(mask)
    if valid_rows.shape[0] == 0 or int(np.sum(valid_rows)) < 2:
        return None

    true_valid = np.asarray(true_values, dtype=np.float32)[valid_rows]
    pred_valid = np.asarray(pred_values, dtype=np.float32)[valid_rows]
    if true_valid.shape[0] < 2 or pred_valid.shape[0] < 2:
        return None

    measured_valid = None
    if measured_values is not None:
        measured_valid = np.asarray(measured_values, dtype=np.float32)[valid_rows]

    pose_dim = min(true_valid.shape[1], pred_valid.shape[1])
    if measured_valid is not None:
        pose_dim = min(pose_dim, measured_valid.shape[1])
    if pose_dim <= 0:
        return None

    true_valid = true_valid[:, :pose_dim]
    pred_valid = pred_valid[:, :pose_dim]
    if measured_valid is not None:
        measured_valid = measured_valid[:, :pose_dim]

    time_index = np.arange(true_valid.shape[0], dtype=np.int64)
    names = list(dim_names) if dim_names is not None else make_pose_dim_names(pose_dim)
    if len(names) < pose_dim:
        names = names + [f"dim{i + 1}" for i in range(len(names), pose_dim)]

    n_cols = min(3, pose_dim)
    n_rows = int(np.ceil(pose_dim / n_cols))
    has_orientation = pose_dim >= 9

    if has_orientation:
        fig = plt.figure(figsize=(5 * n_cols, 2.8 * n_rows + 5.2))
        gs = fig.add_gridspec(n_rows + 2, n_cols, height_ratios=[1.0] * n_rows + [1.15, 0.95], hspace=0.35)
        axes = []
        for row in range(n_rows):
            for col in range(n_cols):
                shared = axes[0] if len(axes) > 0 else None
                axes.append(fig.add_subplot(gs[row, col], sharex=shared))
        axes = np.asarray(axes, dtype=object)
        ax_rpy = fig.add_subplot(gs[n_rows, :], sharex=axes[0] if len(axes) > 0 else None)
        ax_geo = fig.add_subplot(gs[n_rows + 1, :], sharex=axes[0] if len(axes) > 0 else None)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.8 * n_rows), sharex=True)
        axes = np.asarray(axes).reshape(-1)
        ax_rpy = None
        ax_geo = None

    for dim in range(pose_dim):
        ax = axes[dim]
        ax.plot(time_index, true_valid[:, dim], color="black", linewidth=1.0, label=true_label if dim == 0 else None)
        ax.plot(time_index, pred_valid[:, dim], color="#E41A1C", linewidth=1.4, alpha=0.95, label=pred_label if dim == 0 else None)
        if measured_valid is not None:
            ax.plot(
                time_index,
                measured_valid[:, dim],
                color="#1f78b4",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=measured_label if dim == 0 else None,
            )
        ax.set_title(names[dim])
        ax.grid(alpha=0.25)

    for ax in axes[pose_dim:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.92))

    if has_orientation and ax_rpy is not None and ax_geo is not None:
        rpy_true = compute_pose_rpy(true_valid[:, :9])
        rpy_pred = compute_pose_rpy(pred_valid[:, :9])
        shift_value = get_pi_shift_from_axis_start(rpy_true, cfg=rpy_config)
        if rpy_true.shape[0] == len(time_index) and rpy_pred.shape[0] == len(time_index):
            rpy_true = apply_rpy_axis_shift(rpy_true, shift_value, cfg=rpy_config)
            rpy_pred = apply_rpy_axis_shift(rpy_pred, shift_value, cfg=rpy_config)
            rpy_pred = align_angles_to_reference(rpy_true, rpy_pred)
            rpy_true_plot = convert_rpy_to_plot_unit(rpy_true, cfg=rpy_config)
            rpy_pred_plot = convert_rpy_to_plot_unit(rpy_pred, cfg=rpy_config)

            angle_names = ["roll", "pitch", "yaw"]
            gt_colors = ["#8B0000", "#006400", "#00008B"]
            pred_colors = ["#FF4D4D", "#33CC66", "#4D79FF"]
            for angle_idx, angle_name in enumerate(angle_names):
                ax_rpy.plot(
                    time_index,
                    rpy_true_plot[:, angle_idx],
                    color=gt_colors[angle_idx],
                    linewidth=1.3,
                    linestyle="--",
                    label=f"{angle_name} gt",
                )
                ax_rpy.plot(
                    time_index,
                    rpy_pred_plot[:, angle_idx],
                    color=pred_colors[angle_idx],
                    linewidth=1.2,
                    alpha=0.9,
                    linestyle="-",
                    label=f"{angle_name} pred",
                )

            if measured_valid is not None and measured_valid.shape[1] >= 9:
                rpy_measured = compute_pose_rpy(measured_valid[:, :9])
                if rpy_measured.shape[0] == len(time_index):
                    rpy_measured = apply_rpy_axis_shift(rpy_measured, shift_value, cfg=rpy_config)
                    rpy_measured = align_angles_to_reference(rpy_true, rpy_measured)
                    rpy_measured_plot = convert_rpy_to_plot_unit(rpy_measured, cfg=rpy_config)
                    meas_colors = ["#FF8A8A", "#7FD18B", "#8EA8FF"]
                    for angle_idx, angle_name in enumerate(angle_names):
                        ax_rpy.plot(
                            time_index,
                            rpy_measured_plot[:, angle_idx],
                            color=meas_colors[angle_idx],
                            linewidth=1.0,
                            alpha=0.95,
                            linestyle=":",
                            label=f"{angle_name} measured",
                        )

        unit = get_rpy_plot_unit(cfg=rpy_config)
        rpy_title = f"Computed RPY from pose ({unit}, unwrapped)"
        if bool(rpy_config.subtract_pi):
            axis_names = ["roll", "pitch", "yaw"]
            axis_idx = int(np.clip(int(rpy_config.subtract_pi_axis), 0, len(axis_names) - 1))
            if shift_value > 0.0:
                rpy_title += f", +pi on {axis_names[axis_idx]}"
            elif shift_value < 0.0:
                rpy_title += f", -pi on {axis_names[axis_idx]}"
            else:
                rpy_title += f", no pi shift on {axis_names[axis_idx]}"
        ax_rpy.set_title(rpy_title)
        ax_rpy.set_xlabel("step")
        ax_rpy.set_ylabel(unit)
        ax_rpy.grid(alpha=0.25)
        ax_rpy.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)

        geod_rad = compute_pose_geodesic_distance(true_valid[:, :9], pred_valid[:, :9])
        if geod_rad.shape[0] == len(time_index):
            ax_geo.plot(
                time_index,
                np.rad2deg(geod_rad),
                color="#1F78B4",
                linewidth=1.4,
                alpha=0.9,
                linestyle="-",
                label="prediction geodesic (deg)",
            )

        if measured_valid is not None and measured_valid.shape[1] >= 9:
            geod_measured_rad = compute_pose_geodesic_distance(true_valid[:, :9], measured_valid[:, :9])
            if geod_measured_rad.shape[0] == len(time_index):
                ax_geo.plot(
                    time_index,
                    np.rad2deg(geod_measured_rad),
                    color="#6A3D9A",
                    linewidth=1.0,
                    alpha=0.85,
                    linestyle="--",
                    label="measured geodesic (deg)",
                )

        ax_geo.set_title("Geodesic Distance (rotation error)")
        ax_geo.set_xlabel("step")
        ax_geo.set_ylabel("deg")
        ax_geo.grid(alpha=0.25)
        ax_geo.legend(loc="upper right", ncol=1, frameon=False, fontsize=8)

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95 if not has_orientation else 0.93])
    return fig


def _prepare_anchor_indices(anchor_steps: int, prediction_stride: int) -> np.ndarray:
    anchor_idx = np.arange(0, anchor_steps, max(1, int(prediction_stride)), dtype=np.int64)
    if anchor_idx.shape[0] == 0:
        return np.asarray([0], dtype=np.int64)
    if anchor_idx[-1] != (anchor_steps - 1):
        anchor_idx = np.concatenate([anchor_idx, np.asarray([anchor_steps - 1], dtype=np.int64)])
    return anchor_idx


def _compute_x_limits(
    source_time_index: np.ndarray,
    horizon_len: int,
    max_plot_steps: Optional[int],
) -> Tuple[int, int]:
    if max_plot_steps is not None:
        x_min = 0
        x_max = max(0, int(max_plot_steps) - 1)
        return x_min, x_max

    x_min = int(source_time_index[0]) if source_time_index.shape[0] > 0 else 0
    x_max = int(source_time_index[-1] + horizon_len - 1) if source_time_index.shape[0] > 0 else max(1, horizon_len - 1)
    if x_max < x_min:
        x_max = x_min
    return x_min, x_max


def _reconstruct_true_by_time(
    true_action_chunks: np.ndarray,
    mask_chunks: np.ndarray,
    source_time_index: np.ndarray,
    pose_dim: int,
    x_min: int,
    x_max: int,
) -> Dict[int, np.ndarray]:
    true_by_time: Dict[int, np.ndarray] = {}
    anchor_steps = int(min(true_action_chunks.shape[0], mask_chunks.shape[0], source_time_index.shape[0]))
    for t in range(anchor_steps):
        valid_h = mask_chunks[t, :, 0] > 0
        if not np.any(valid_h):
            continue
        base_t = int(source_time_index[t])
        for h in np.where(valid_h)[0]:
            time_val = base_t + int(h)
            if time_val < x_min or time_val > x_max:
                continue
            if time_val not in true_by_time:
                true_by_time[time_val] = true_action_chunks[t, h, :pose_dim].astype(np.float32)
    return true_by_time


def build_pose_fan_figure(
    true_action_chunks: np.ndarray,
    pred_action_chunks: np.ndarray,
    mask_chunks: np.ndarray,
    source_time_index: np.ndarray,
    prediction_stride: int,
    measured_pose: Optional[np.ndarray] = None,
    background_actions: Optional[List[np.ndarray]] = None,
    max_plot_steps: Optional[int] = None,
    stiffness_label: Optional[int] = None,
    global_step: Optional[int] = None,
    title: Optional[str] = None,
    plot_ground_truth_h0: bool = True,
    plot_ground_truth_reconstructed: bool = False,
    rpy_config: RPYPlotConfig = DEFAULT_RPY_PLOT_CONFIG,
):
    anchor_steps = int(min(true_action_chunks.shape[0], pred_action_chunks.shape[0], mask_chunks.shape[0], source_time_index.shape[0]))
    if measured_pose is not None:
        anchor_steps = int(min(anchor_steps, measured_pose.shape[0]))
    if anchor_steps < 1:
        return None

    true_action_chunks = np.asarray(true_action_chunks, dtype=np.float32)[:anchor_steps]
    pred_action_chunks = np.asarray(pred_action_chunks, dtype=np.float32)[:anchor_steps]
    mask_chunks = np.asarray(mask_chunks, dtype=np.float32)[:anchor_steps]
    source_time_index = np.asarray(source_time_index, dtype=np.int64)[:anchor_steps]
    measured_pose_arr = np.asarray(measured_pose, dtype=np.float32)[:anchor_steps] if measured_pose is not None else None

    pose_dim = int(min(true_action_chunks.shape[-1], pred_action_chunks.shape[-1]))
    if measured_pose_arr is not None:
        pose_dim = int(min(pose_dim, measured_pose_arr.shape[-1]))
    if pose_dim <= 0:
        return None

    true_action_chunks = true_action_chunks[..., :pose_dim]
    pred_action_chunks = pred_action_chunks[..., :pose_dim]
    mask_chunks = mask_chunks[..., :pose_dim]
    if measured_pose_arr is not None:
        measured_pose_arr = measured_pose_arr[:, :pose_dim]

    n_samples = int(pred_action_chunks.shape[1])
    horizon_len = int(true_action_chunks.shape[1])
    anchor_idx = _prepare_anchor_indices(anchor_steps, prediction_stride)
    anchor_colors = plt.cm.rainbow(np.linspace(0.0, 1.0, max(1, len(anchor_idx))))

    dim_names = make_pose_dim_names(pose_dim)
    n_cols = min(3, pose_dim)
    n_rows = int(np.ceil(pose_dim / n_cols))
    has_orientation = pose_dim >= 9

    if has_orientation:
        fig = plt.figure(figsize=(6 * n_cols, 3 * n_rows + 5.0))
        gs = fig.add_gridspec(n_rows + 2, n_cols, height_ratios=[1.0] * n_rows + [1.05, 0.95], hspace=0.32)
        axes = []
        for row in range(n_rows):
            for col in range(n_cols):
                shared = axes[0] if len(axes) > 0 else None
                axes.append(fig.add_subplot(gs[row, col], sharex=shared))
        axes = np.asarray(axes, dtype=object)
        ax_rpy = fig.add_subplot(gs[n_rows, :], sharex=axes[0] if len(axes) > 0 else None)
        ax_geo = fig.add_subplot(gs[n_rows + 1, :], sharex=axes[0] if len(axes) > 0 else None)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), sharex=True)
        axes = np.asarray(axes).reshape(-1)
        ax_rpy = None
        ax_geo = None

    x_min, x_max = _compute_x_limits(source_time_index, horizon_len, max_plot_steps=max_plot_steps)
    true_by_time = {}
    if plot_ground_truth_reconstructed:
        true_by_time = _reconstruct_true_by_time(
            true_action_chunks=true_action_chunks,
            mask_chunks=mask_chunks,
            source_time_index=source_time_index,
            pose_dim=pose_dim,
            x_min=x_min,
            x_max=x_max,
        )
        true_times = sorted(true_by_time.keys())
    else:
        true_times = []

    for dim in range(pose_dim):
        ax = axes[dim]

        if background_actions:
            max_len = max(0, x_max - x_min + 1)
            for traj in background_actions:
                if traj.ndim != 2 or traj.shape[1] < pose_dim:
                    continue
                use_len = min(int(traj.shape[0]), int(max_len))
                if use_len < 2:
                    continue
                x_vals = x_min + np.arange(use_len)
                ax.plot(
                    x_vals,
                    traj[:use_len, dim],
                    color="#BDBDBD",
                    linewidth=0.6,
                    alpha=0.2,
                )

        if plot_ground_truth_h0:
            ax.plot(
                source_time_index[:anchor_steps],
                true_action_chunks[:anchor_steps, 0, dim],
                color="black",
                linewidth=1.0,
                label="ground_truth" if dim == 0 else None,
                zorder=11,
            )

        if plot_ground_truth_reconstructed and len(true_times) > 0:
            vals = np.asarray([true_by_time[t][dim] for t in true_times], dtype=np.float32)
            ax.plot(
                true_times,
                vals,
                color="black",
                linewidth=1.0,
                label="ground_truth" if dim == 0 and not plot_ground_truth_h0 else None,
                zorder=10,
            )

        if measured_pose_arr is not None:
            ax.plot(
                source_time_index[:anchor_steps],
                measured_pose_arr[:anchor_steps, dim],
                color="#1f78b4",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label="measured_pose" if dim == 0 else None,
            )

        for anchor_pos, t in enumerate(anchor_idx):
            valid_h = mask_chunks[t, :, 0] > 0
            if not np.any(valid_h):
                continue
            c_t = anchor_colors[anchor_pos]
            base_t = int(source_time_index[t])
            horizon_idx = np.where(valid_h)[0]
            x_vals = base_t + horizon_idx
            within = np.logical_and(x_vals >= x_min, x_vals <= x_max)
            if not np.any(within):
                continue
            x_vals = x_vals[within]
            h_idx = horizon_idx[within]
            for s_idx in range(n_samples):
                ax.plot(
                    x_vals,
                    pred_action_chunks[t, s_idx, h_idx, dim],
                    color=c_t,
                    linewidth=0.8,
                    alpha=0.7,
                    label="prior_samples" if (anchor_pos == 0 and s_idx == 0 and dim == 0) else None,
                )

        ax.set_xlim(x_min, x_max)
        ax.set_title(dim_names[dim])
        ax.grid(alpha=0.25)

    for ax in axes[pose_dim:]:
        ax.axis("off")

    if has_orientation and ax_rpy is not None and ax_geo is not None:
        true_pose_first = true_action_chunks[:anchor_steps, 0, :9]
        true_rpy = compute_pose_rpy(true_pose_first)
        shift_value = get_pi_shift_from_axis_start(true_rpy, cfg=rpy_config)

        if true_rpy.shape[0] == anchor_steps:
            true_rpy = apply_rpy_axis_shift(true_rpy, shift_value, cfg=rpy_config)
            true_rpy_plot = convert_rpy_to_plot_unit(true_rpy, cfg=rpy_config)
            angle_names = ["roll", "pitch", "yaw"]
            gt_colors = ["#8B0000", "#006400", "#00008B"]
            meas_colors = ["#FF8A8A", "#7FD18B", "#8EA8FF"]

            for angle_idx, angle_name in enumerate(angle_names):
                ax_rpy.plot(
                    source_time_index[:anchor_steps],
                    true_rpy_plot[:, angle_idx],
                    color=gt_colors[angle_idx],
                    linewidth=1.2,
                    linestyle="--",
                    label=f"{angle_name} gt",
                )

            if measured_pose_arr is not None and measured_pose_arr.shape[1] >= 9:
                meas_pose_first = measured_pose_arr[:anchor_steps, :9]
                meas_rpy = compute_pose_rpy(meas_pose_first)
                if meas_rpy.shape[0] == anchor_steps:
                    meas_rpy = apply_rpy_axis_shift(meas_rpy, shift_value, cfg=rpy_config)
                    meas_rpy = align_angles_to_reference(true_rpy, meas_rpy)
                    meas_rpy_plot = convert_rpy_to_plot_unit(meas_rpy, cfg=rpy_config)
                    for angle_idx, angle_name in enumerate(angle_names):
                        ax_rpy.plot(
                            source_time_index[:anchor_steps],
                            meas_rpy_plot[:, angle_idx],
                            color=meas_colors[angle_idx],
                            linewidth=1.0,
                            linestyle=":",
                            alpha=0.9,
                            label=f"{angle_name} measured",
                        )

            for anchor_pos, t in enumerate(anchor_idx):
                valid_h = mask_chunks[t, :, 0] > 0
                if not np.any(valid_h):
                    continue
                horizon_idx = np.where(valid_h)[0]
                base_t = int(source_time_index[t])
                x_vals = base_t + horizon_idx
                within = np.logical_and(x_vals >= x_min, x_vals <= x_max)
                if not np.any(within):
                    continue
                x_vals = x_vals[within]
                h_idx = horizon_idx[within]
                true_chunk_rpy = compute_pose_rpy(true_action_chunks[t, h_idx, :9])
                true_chunk_rpy = apply_rpy_axis_shift(true_chunk_rpy, shift_value, cfg=rpy_config)
                true_chunk_rpy_plot = convert_rpy_to_plot_unit(true_chunk_rpy, cfg=rpy_config)

                c_t = anchor_colors[anchor_pos]
                for s_idx in range(n_samples):
                    pred_chunk_rpy = compute_pose_rpy(pred_action_chunks[t, s_idx, h_idx, :9])
                    pred_chunk_rpy = apply_rpy_axis_shift(pred_chunk_rpy, shift_value, cfg=rpy_config)
                    pred_chunk_rpy = align_angles_to_reference(true_chunk_rpy, pred_chunk_rpy)
                    pred_chunk_rpy_plot = convert_rpy_to_plot_unit(pred_chunk_rpy, cfg=rpy_config)
                    for angle_idx in range(3):
                        ax_rpy.plot(
                            x_vals,
                            pred_chunk_rpy_plot[:, angle_idx],
                            color=c_t,
                            linewidth=0.7,
                            alpha=0.2,
                            linestyle="-",
                            label="rpy prior samples" if (anchor_pos == 0 and s_idx == 0 and angle_idx == 0) else None,
                        )
                    if s_idx == 0:
                        for angle_idx in range(3):
                            ax_rpy.plot(
                                x_vals,
                                true_chunk_rpy_plot[:, angle_idx],
                                color=gt_colors[angle_idx],
                                linewidth=0.6,
                                alpha=0.35,
                                linestyle="--",
                                label=None,
                            )

        unit = get_rpy_plot_unit(cfg=rpy_config)
        rpy_title = f"RPY Fan ({unit})"
        if bool(rpy_config.subtract_pi):
            axis_names = ["roll", "pitch", "yaw"]
            axis_idx = int(np.clip(int(rpy_config.subtract_pi_axis), 0, len(axis_names) - 1))
            if shift_value > 0.0:
                rpy_title += f", +pi on {axis_names[axis_idx]}"
            elif shift_value < 0.0:
                rpy_title += f", -pi on {axis_names[axis_idx]}"
            else:
                rpy_title += f", no pi shift on {axis_names[axis_idx]}"
        ax_rpy.set_title(rpy_title)
        ax_rpy.set_ylabel(unit)
        ax_rpy.set_xlim(x_min, x_max)
        ax_rpy.grid(alpha=0.25)
        ax_rpy.legend(loc="upper right", ncol=4, frameon=False, fontsize=8)

        if measured_pose_arr is not None and measured_pose_arr.shape[1] >= 9:
            geod_meas_rad = compute_pose_geodesic_distance(true_pose_first, measured_pose_arr[:anchor_steps, :9])
            if geod_meas_rad.shape[0] == anchor_steps:
                ax_geo.plot(
                    source_time_index[:anchor_steps],
                    np.rad2deg(geod_meas_rad),
                    color="#1f78b4",
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.85,
                    label="measured geodesic (h=0)",
                )

        for anchor_pos, t in enumerate(anchor_idx):
            valid_h = mask_chunks[t, :, 0] > 0
            if not np.any(valid_h):
                continue
            horizon_idx = np.where(valid_h)[0]
            base_t = int(source_time_index[t])
            x_vals = base_t + horizon_idx
            within = np.logical_and(x_vals >= x_min, x_vals <= x_max)
            if not np.any(within):
                continue
            x_vals = x_vals[within]
            h_idx = horizon_idx[within]
            true_chunk_pose = true_action_chunks[t, h_idx, :9]
            c_t = anchor_colors[anchor_pos]
            for s_idx in range(n_samples):
                pred_chunk_pose = pred_action_chunks[t, s_idx, h_idx, :9]
                geod_deg = np.rad2deg(compute_pose_geodesic_distance(true_chunk_pose, pred_chunk_pose))
                ax_geo.plot(
                    x_vals,
                    geod_deg,
                    color=c_t,
                    linewidth=0.8,
                    alpha=0.45,
                    linestyle="-",
                    label="geodesic prior samples" if (anchor_pos == 0 and s_idx == 0) else None,
                )

        ax_geo.axhline(0.0, color="black", linewidth=0.8, linestyle=":", alpha=0.9, label="zero error")
        ax_geo.set_title("Geodesic Distance Fan (rotation error)")
        ax_geo.set_xlabel("step")
        ax_geo.set_ylabel("deg")
        ax_geo.set_xlim(x_min, x_max)
        ax_geo.grid(alpha=0.25)
        ax_geo.legend(loc="upper right", ncol=2, frameon=False, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.92))

    if title is None:
        if stiffness_label is not None or global_step is not None:
            stiff_str = "all" if stiffness_label is None else str(stiffness_label)
            step_str = f" | step={global_step}" if global_step is not None else ""
            title = f"Sampled Prior Trajectories vs Ground Truth | stiffness={stiff_str}{step_str}"
        else:
            title = f"Sampled Prior Fan + Measured Pose (full episode, stride={max(1, int(prediction_stride))})"
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.9 if not has_orientation else 0.93])
    return fig


def build_obs_prediction_figure(
    true_obs: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    max_dims: int,
    time_index: Optional[np.ndarray] = None,
    pred_sample: Optional[np.ndarray] = None,
    title: str = "Observation Prediction (mean +- std)",
    rpy_config: RPYPlotConfig = DEFAULT_RPY_PLOT_CONFIG,
):
    n_dims = min(int(max_dims), int(true_obs.shape[1]), int(pred_mean.shape[1]), int(pred_std.shape[1]))
    if n_dims <= 0:
        return None

    if time_index is None:
        time_axis = np.arange(true_obs.shape[0], dtype=np.int64)
    else:
        time_axis = np.asarray(time_index, dtype=np.int64)
        if time_axis.shape[0] != true_obs.shape[0]:
            return None

    if pred_sample is not None:
        pred_sample_arr = np.asarray(pred_sample, dtype=np.float32)
        if pred_sample_arr.shape[0] != true_obs.shape[0] or pred_sample_arr.shape[1] < n_dims:
            pred_sample_arr = None
    else:
        pred_sample_arr = None

    true_arr = np.asarray(true_obs, dtype=np.float32)[:, :n_dims]
    mean_arr = np.asarray(pred_mean, dtype=np.float32)[:, :n_dims]
    std_arr = np.asarray(pred_std, dtype=np.float32)[:, :n_dims]

    dim_names = make_obs_dim_names(int(true_obs.shape[1]))
    n_cols = min(3, n_dims)
    n_rows = int(np.ceil(n_dims / n_cols))
    has_orientation = n_dims >= 9

    if has_orientation:
        fig = plt.figure(figsize=(5 * n_cols, 2.8 * n_rows + 5.2))
        gs = fig.add_gridspec(n_rows + 2, n_cols, height_ratios=[1.0] * n_rows + [1.15, 0.95], hspace=0.35)
        axes = []
        for row in range(n_rows):
            for col in range(n_cols):
                shared = axes[0] if len(axes) > 0 else None
                axes.append(fig.add_subplot(gs[row, col], sharex=shared))
        axes = np.asarray(axes, dtype=object)
        ax_rpy = fig.add_subplot(gs[n_rows, :], sharex=axes[0] if len(axes) > 0 else None)
        ax_geo = fig.add_subplot(gs[n_rows + 1, :], sharex=axes[0] if len(axes) > 0 else None)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.8 * n_rows), sharex=True)
        axes = np.asarray(axes).reshape(-1)
        ax_rpy = None
        ax_geo = None

    for dim in range(n_dims):
        ax = axes[dim]
        lower = mean_arr[:, dim] - std_arr[:, dim]
        upper = mean_arr[:, dim] + std_arr[:, dim]
        ax.plot(time_axis, true_arr[:, dim], color="black", linewidth=1.3, label="target" if dim == 0 else None)
        ax.plot(time_axis, mean_arr[:, dim], color="#E41A1C", linewidth=1.2, label="pred mean" if dim == 0 else None)
        if pred_sample_arr is not None:
            ax.plot(
                time_axis,
                pred_sample_arr[:, dim],
                color="#377EB8",
                linewidth=0.9,
                alpha=0.7,
                linestyle="--",
                label="pred sample" if dim == 0 else None,
            )
        ax.fill_between(time_axis, lower, upper, color="#FB9A99", alpha=0.25, label="mean +- std" if dim == 0 else None)
        ax.set_title(dim_names[dim])
        ax.grid(alpha=0.25)

    for ax in axes[n_dims:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False)

    if has_orientation and ax_rpy is not None and ax_geo is not None:
        rpy_true = compute_pose_rpy(true_arr[:, :9])
        rpy_pred = compute_pose_rpy(mean_arr[:, :9])
        shift_value = get_pi_shift_from_axis_start(rpy_true, cfg=rpy_config)
        if rpy_true.shape[0] == len(time_axis) and rpy_pred.shape[0] == len(time_axis):
            rpy_true = apply_rpy_axis_shift(rpy_true, shift_value, cfg=rpy_config)
            rpy_pred = apply_rpy_axis_shift(rpy_pred, shift_value, cfg=rpy_config)
            rpy_pred = align_angles_to_reference(rpy_true, rpy_pred)
            rpy_true_plot = convert_rpy_to_plot_unit(rpy_true, cfg=rpy_config)
            rpy_pred_plot = convert_rpy_to_plot_unit(rpy_pred, cfg=rpy_config)
            angle_names = ["roll", "pitch", "yaw"]
            gt_colors = ["#8B0000", "#006400", "#00008B"]
            pred_colors = ["#FF4D4D", "#33CC66", "#4D79FF"]
            for angle_idx, angle_name in enumerate(angle_names):
                ax_rpy.plot(
                    time_axis,
                    rpy_true_plot[:, angle_idx],
                    color=gt_colors[angle_idx],
                    linewidth=1.3,
                    linestyle="-",
                    label=f"{angle_name} gt",
                )
                ax_rpy.plot(
                    time_axis,
                    rpy_pred_plot[:, angle_idx],
                    color=pred_colors[angle_idx],
                    linewidth=1.2,
                    alpha=0.9,
                    linestyle=":",
                    label=f"{angle_name} pred",
                )
        unit = get_rpy_plot_unit(cfg=rpy_config)
        rpy_title = f"Computed RPY from pose ({unit}, unwrapped)"
        if bool(rpy_config.subtract_pi):
            axis_names = ["roll", "pitch", "yaw"]
            axis_idx = int(np.clip(int(rpy_config.subtract_pi_axis), 0, len(axis_names) - 1))
            if shift_value > 0.0:
                rpy_title += f", +pi on {axis_names[axis_idx]}"
            elif shift_value < 0.0:
                rpy_title += f", -pi on {axis_names[axis_idx]}"
            else:
                rpy_title += f", no pi shift on {axis_names[axis_idx]}"
        ax_rpy.set_title(rpy_title)
        ax_rpy.set_xlabel("step")
        ax_rpy.set_ylabel(unit)
        ax_rpy.grid(alpha=0.25)
        ax_rpy.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)

        geod_rad = compute_pose_geodesic_distance(true_arr[:, :9], mean_arr[:, :9])
        if geod_rad.shape[0] == len(time_axis):
            geod_deg = np.rad2deg(geod_rad)
            ax_geo.plot(
                time_axis,
                geod_deg,
                color="#1F78B4",
                linewidth=1.4,
                alpha=0.9,
                linestyle="-",
                label="geodesic distance (deg)",
            )
            ax_geo.set_title("Geodesic Distance (rotation error)")
            ax_geo.set_xlabel("step")
            ax_geo.set_ylabel("deg")
            ax_geo.grid(alpha=0.25)
            ax_geo.legend(loc="upper right", ncol=1, frameon=False, fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.95 if not has_orientation else 0.93])
    return fig


def build_tracking_error_figure(tracking_error: np.ndarray, per_dim_mse: np.ndarray):
    if tracking_error.size == 0:
        return None

    l2 = np.linalg.norm(tracking_error, axis=-1)
    pose_dim = int(tracking_error.shape[-1])
    dim_names = make_obs_dim_names(pose_dim)
    mean_abs = np.mean(np.abs(tracking_error), axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(np.arange(l2.shape[0]), l2, color="#377EB8", linewidth=1.5)
    axes[0].set_title("Tracking error L2 per step")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("L2")
    axes[0].grid(alpha=0.25)

    axes[1].bar(np.arange(pose_dim), mean_abs, color="#FB9A99")
    axes[1].set_title("Mean |tracking error| by pose dim")
    axes[1].set_xticks(np.arange(pose_dim))
    axes[1].set_xticklabels(dim_names, rotation=45, ha="right")
    axes[1].grid(alpha=0.25)

    mse_dim = per_dim_mse[:pose_dim]
    axes[2].bar(np.arange(pose_dim), mse_dim, color="#A6CEE3")
    axes[2].set_title("Prediction MSE by pose dim")
    axes[2].set_xticks(np.arange(pose_dim))
    axes[2].set_xticklabels(dim_names, rotation=45, ha="right")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    return fig


def _extract_gaussian_entry_arrays(entry: Any) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(entry, dict):
        mean = entry.get("mean", entry.get("mu", None))
        std = entry.get("std", None)
        if mean is None or std is None:
            raise KeyError("Gaussian entry dict must include mean/mu and std.")
        return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)

    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        return np.asarray(entry[0], dtype=np.float32), np.asarray(entry[1], dtype=np.float32)

    raise TypeError("Unsupported Gaussian entry format. Expected dict or tuple/list.")


def extract_gaussian_prior_posterior(dists_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    if "Prior" not in dists_data or "Posterior" not in dists_data:
        raise KeyError("dists_data must include 'Prior' and 'Posterior'.")

    _, prior_std = _extract_gaussian_entry_arrays(dists_data["Prior"])
    _, post_std = _extract_gaussian_entry_arrays(dists_data["Posterior"])
    prior_var = np.maximum(prior_std**2, 1e-10)
    post_var = np.maximum(post_std**2, 1e-10)
    return prior_var, post_var


def _softmax_last(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_x / np.clip(np.sum(exp_x, axis=-1, keepdims=True), 1e-8, None)


def _extract_categorical_probs_entry(entry: Any) -> np.ndarray:
    if isinstance(entry, dict):
        if "probs" in entry:
            arr = np.asarray(entry["probs"], dtype=np.float32)
        elif "probabilities" in entry:
            arr = np.asarray(entry["probabilities"], dtype=np.float32)
        elif "logits" in entry:
            arr = _softmax_last(np.asarray(entry["logits"], dtype=np.float32))
        else:
            raise KeyError("Categorical entry dict must include probs/probabilities/logits.")
    elif isinstance(entry, (tuple, list)):
        if len(entry) < 1:
            raise ValueError("Categorical tuple/list entry is empty.")
        arr = np.asarray(entry[0], dtype=np.float32)
    else:
        raise TypeError("Unsupported categorical entry format. Expected dict or tuple/list.")

    if arr.ndim != 3:
        raise ValueError(f"Expected categorical array shape (T, num_variables, num_categories), got {arr.shape}.")

    is_prob_like = bool(np.all(arr >= -1e-6) and np.mean(np.abs(np.sum(arr, axis=-1) - 1.0)) < 1e-3)
    probs = arr if is_prob_like else _softmax_last(arr)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / np.clip(np.sum(probs, axis=-1, keepdims=True), 1e-8, None)
    return probs.astype(np.float32)


def extract_categorical_prior_posterior_probs(dists_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    if "Prior" not in dists_data or "Posterior" not in dists_data:
        raise KeyError("dists_data must include 'Prior' and 'Posterior'.")

    prior_probs = _extract_categorical_probs_entry(dists_data["Prior"])
    post_probs = _extract_categorical_probs_entry(dists_data["Posterior"])
    if prior_probs.shape != post_probs.shape:
        time_len = min(int(prior_probs.shape[0]), int(post_probs.shape[0]))
        prior_probs = prior_probs[:time_len]
        post_probs = post_probs[:time_len]
    return prior_probs, post_probs


def build_z_gaussian_variance_with_frames_figure(
    dists_data: Dict[str, Any],
    raw_images: Sequence[np.ndarray],
    target_steps: Optional[Sequence[int]] = None,
):
    prior_var, post_var = extract_gaussian_prior_posterior(dists_data)
    time_len, z_dim = post_var.shape
    time_steps = np.arange(time_len, dtype=np.int64)

    if target_steps is None:
        target_steps = [150, 175, 200, 225, 250, 275, 300, 325]
    target_steps = [int(step) for step in target_steps]
    num_imgs = len(target_steps)

    post_mean_var = np.mean(post_var, axis=1)
    prior_mean_var = np.mean(prior_var, axis=1)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, max(1, num_imgs), height_ratios=[1.5, 1.0], hspace=0.08, wspace=0.1)

    ax_main = fig.add_subplot(gs[0, :])
    for dim in range(z_dim):
        if dim == 0:
            ax_main.plot(time_steps, post_var[:, dim], color="red", alpha=0.3, linewidth=0.5, label="Individual (Post)")
            ax_main.plot(time_steps, prior_var[:, dim], color="blue", alpha=0.3, linewidth=0.5, label="Individual (Prior)")
        else:
            ax_main.plot(time_steps, post_var[:, dim], color="red", alpha=0.3, linewidth=0.5)
            ax_main.plot(time_steps, prior_var[:, dim], color="blue", alpha=0.3, linewidth=0.5)

    ax_main.plot(time_steps, post_mean_var, color="red", linewidth=2.5, label="Mean Variance (Posterior)")
    ax_main.plot(time_steps, prior_mean_var, color="blue", linewidth=2.5, label="Mean Variance (Prior)")
    for step in target_steps:
        if 0 <= step < time_len:
            ax_main.axvline(x=step, color="gray", linestyle="--", alpha=0.4)

    ax_main.set_yscale("log")
    ax_main.set_title("Z-Variance & Observations")
    ax_main.set_ylabel("Variance (Log Scale)")
    ax_main.set_xlabel("Timestep", labelpad=5)
    ax_main.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_main.legend(loc="upper left", frameon=True, fontsize="small", ncol=2)

    for img_idx, step in enumerate(target_steps):
        img_ax = fig.add_subplot(gs[1, img_idx])
        if 0 <= step < len(raw_images):
            img_ax.imshow(raw_images[step])
            img_ax.set_title(f"step={step}", fontsize=12)
        else:
            img_ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        img_ax.axis("off")

    stats = {
        "post_mean_var_max": float(np.max(post_mean_var)),
        "post_mean_var_min": float(np.min(post_mean_var)),
    }
    return fig, stats


def build_z_categorical_distribution_with_frames_figure(
    dists_data: Dict[str, Any],
    raw_images: Sequence[np.ndarray],
    target_steps: Optional[Sequence[int]] = None,
):
    prior_probs, post_probs = extract_categorical_prior_posterior_probs(dists_data)
    time_len, num_vars, num_cats = prior_probs.shape

    if target_steps is None:
        target_steps = [150, 175, 200, 225, 250, 275, 300, 325]
    target_steps = [int(step) for step in target_steps]
    num_imgs = len(target_steps)

    prior_entropy = -np.sum(prior_probs * np.log(np.clip(prior_probs, 1e-8, 1.0)), axis=-1)
    post_entropy = -np.sum(post_probs * np.log(np.clip(post_probs, 1e-8, 1.0)), axis=-1)
    prior_flat = prior_probs.reshape(time_len, num_vars * num_cats)
    post_flat = post_probs.reshape(time_len, num_vars * num_cats)

    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(3, max(1, num_imgs), height_ratios=[1.0, 1.0, 0.9], hspace=0.22, wspace=0.1)

    ax_prior = fig.add_subplot(gs[0, :])
    ax_post = fig.add_subplot(gs[1, :], sharex=ax_prior)

    im_prior = ax_prior.imshow(
        prior_flat.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    im_post = ax_post.imshow(
        post_flat.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    for ax in (ax_prior, ax_post):
        for step in target_steps:
            if 0 <= step < time_len:
                ax.axvline(x=step, color="white", linestyle="--", alpha=0.45, linewidth=0.8)
        for var_idx in range(1, num_vars):
            ax.axhline(y=var_idx * num_cats - 0.5, color="white", linestyle=":", alpha=0.25, linewidth=0.8)
        ax.set_ylabel("var*cat channel")
        ax.grid(False)

    ax_prior.set_title(
        f"Prior categorical probabilities | vars={num_vars}, cats={num_cats}, mean entropy={float(np.mean(prior_entropy)):.3f}"
    )
    ax_post.set_title(f"Posterior categorical probabilities | mean entropy={float(np.mean(post_entropy)):.3f}")
    ax_post.set_xlabel("Timestep")

    fig.colorbar(im_prior, ax=ax_prior, fraction=0.02, pad=0.01, label="probability")
    fig.colorbar(im_post, ax=ax_post, fraction=0.02, pad=0.01, label="probability")

    for img_idx, step in enumerate(target_steps):
        img_ax = fig.add_subplot(gs[2, img_idx])
        if 0 <= step < len(raw_images):
            img_ax.imshow(raw_images[step])
            img_ax.set_title(f"step={step}", fontsize=11)
        else:
            img_ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        img_ax.axis("off")

    stats = {
        "prior_entropy_mean": float(np.mean(prior_entropy)),
        "post_entropy_mean": float(np.mean(post_entropy)),
        "num_variables": int(num_vars),
        "num_categories": int(num_cats),
    }
    return fig, stats
