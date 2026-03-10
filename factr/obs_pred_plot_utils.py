from typing import Optional

import numpy as np

from factr.plot_utils import RPYPlotConfig, build_obs_prediction_figure, build_tracking_error_figure


def build_obs_prediction_plot(
    true_obs: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    max_dims: int,
    time_index: Optional[np.ndarray] = None,
    pred_sample: Optional[np.ndarray] = None,
    title: str = "Observation Prediction (mean +- std)",
    rpy_subtract_pi: bool = True,
    rpy_subtract_pi_axis: int = 0,
    rpy_plot_unit: str = "deg",
):
    rpy_cfg = RPYPlotConfig(
        subtract_pi=bool(rpy_subtract_pi),
        subtract_pi_axis=int(rpy_subtract_pi_axis),
        unit=str(rpy_plot_unit),
    )
    return build_obs_prediction_figure(
        true_obs=true_obs,
        pred_mean=pred_mean,
        pred_std=pred_std,
        max_dims=int(max_dims),
        time_index=time_index,
        pred_sample=pred_sample,
        title=title,
        rpy_config=rpy_cfg,
    )


def collapse_obs_prediction_chunks(
    true_chunks: np.ndarray,
    pred_mean_chunks: np.ndarray,
    pred_std_chunks: np.ndarray,
    pred_sample_chunks: np.ndarray,
    valid_mask: np.ndarray,
    anchor_steps: np.ndarray,
    stride: int = 1,
):
    if true_chunks.shape[0] == 0:
        return None

    selected = np.arange(0, true_chunks.shape[0], max(1, int(stride)), dtype=np.int64)
    if selected.size == 0:
        return None
    if selected[-1] != (true_chunks.shape[0] - 1):
        selected = np.concatenate([selected, np.asarray([true_chunks.shape[0] - 1], dtype=np.int64)])

    obs_dim = int(true_chunks.shape[-1])
    horizon = int(true_chunks.shape[1])
    max_time = int(np.max(anchor_steps[selected]) + horizon - 1)
    if max_time < 0:
        return None

    sum_true = np.zeros((max_time + 1, obs_dim), dtype=np.float32)
    sum_pred_mean = np.zeros((max_time + 1, obs_dim), dtype=np.float32)
    sum_pred_std = np.zeros((max_time + 1, obs_dim), dtype=np.float32)
    sum_pred_sample = np.zeros((max_time + 1, obs_dim), dtype=np.float32)
    counts = np.zeros((max_time + 1,), dtype=np.float32)

    for sample_idx in selected:
        base_t = int(anchor_steps[sample_idx])
        for h_idx in range(horizon):
            if valid_mask[sample_idx, h_idx] <= 0:
                continue
            t_abs = base_t + h_idx
            sum_true[t_abs] += true_chunks[sample_idx, h_idx]
            sum_pred_mean[t_abs] += pred_mean_chunks[sample_idx, h_idx]
            sum_pred_std[t_abs] += pred_std_chunks[sample_idx, h_idx]
            sum_pred_sample[t_abs] += pred_sample_chunks[sample_idx, h_idx]
            counts[t_abs] += 1.0

    valid_time = counts > 0
    if not np.any(valid_time):
        return None

    denom = np.clip(counts[valid_time][:, None], 1e-6, None)
    return {
        "time_index": np.where(valid_time)[0],
        "true": sum_true[valid_time] / denom,
        "pred_mean": sum_pred_mean[valid_time] / denom,
        "pred_std": sum_pred_std[valid_time] / denom,
        "pred_sample": sum_pred_sample[valid_time] / denom,
    }


def build_tracking_error_plot(tracking_error: np.ndarray, per_dim_mse: np.ndarray):
    return build_tracking_error_figure(tracking_error=tracking_error, per_dim_mse=per_dim_mse)
