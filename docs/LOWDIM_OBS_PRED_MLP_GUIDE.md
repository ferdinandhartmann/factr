# Low-Dim Observation Prediction MLP Guide

This guide describes the current low-dimensional observation prediction pipeline.

## 1) Goal

The model predicts the **next 30 observations** (not just one step) from:

- previous observation window
- commanded pose action chunk (30 x 9)
- stiffness class (1-3)
- goal class (1-4, one-hot)

Tracking error is **not** predicted by the model.  
It is computed afterwards from predicted pose and commanded pose:

`tracking_error = commanded_pose - predicted_pose`

## 2) Main Components

- Training entrypoint: `factr/train_obs_model.py`
- Model: `factr/models/lowdim_obs_mlp.py`
- Replay buffer: `factr/replay_buffer.py` (`RobobufReplayBufferObsPredLowdim`)
- Trainer: `factr/trainers/obs_pred.py`
- Eval task: `factr/task_obs_pred.py`

Hydra configs:

- Main config: `factr/cfg/train_obs_pred_lowdim.yaml`
- Agent config: `factr/cfg/agent/obs_mlp_gaussian_lowdim.yaml`
- Task config: `factr/cfg/task/single_franka_obs_pred_lowdim.yaml`
- Trainer config: `factr/cfg/trainer/adamw_cos_obs_pred.yaml`

## 3) Data Flow and Shapes

Per training sample:

- `obs_window`: `(W, 21)` where `W = obs_window`
- `action_chunk`: `(H, 9)` where `H = pred_horizon = 30`
- `target_obs_chunk`: `(H, 21)`
- `target_mask`: `(H,)` (1 for real future step, 0 for padded step near episode end)
- `stiffness_label`: scalar in `{1,2,3}`
- `goal_label`: scalar in `{1,2,3,4}`

Batch shapes:

- `obs_window`: `(B, W, 21)`
- `action_chunk`: `(B, H, 9)`
- `target_obs_chunk`: `(B, H, 21)`
- `target_mask`: `(B, H)`
- `stiffness_label`: `(B,)`
- `goal_label`: `(B,)`

The replay buffer preserves episode boundaries and pads horizon steps at the end of episodes while masking padded targets out of loss/metrics.

## 4) Model Architecture

`LowdimGaussianObsMLP` input feature = concatenation of:

- flattened obs window: `(W * 21)`
- flattened action chunk: `(H * 9)`
- one-hot stiffness: `(3)`
- one-hot goal: `(4)`

Default architecture:

- hidden layers: 3 (allowed range: 2 to 4)
- hidden dim: 384

Output:

- `dist_params`: `(B, H, 21, 2)`
- `mean`: `(B, H, 21)`
- `var`: `(B, H, 21)` via `softplus + min_var`
- `sample`: `(B, H, 21)` from Gaussian reparameterization

## 5) Loss and Metrics

Training loss:

- masked sampled MSE over horizon and observation dims:
  - `loss = MSE(sample, target_obs_chunk)` with `target_mask`

Also logged:

- `mean_mse` (masked MSE of mean prediction)
- `pred_var_mean`
- `tracking_error_l2` (computed post-prediction, masked over horizon)

## 6) Evaluation and W&B

`ObsPredictionTask.eval()` logs:

- `eval/sample_mse`
- `eval/mean_mse`
- `eval/pred_var_mean`
- `eval/tracking_error_l2`

W&B image logging:

- keeps `eval/obs_prediction_plot`
- **tracking error plot is removed from W&B**

## 7) Run

From repository root:

```bash
python factr/train_obs_model.py --config-name train_obs_pred_lowdim
```

Single-episode offline eval script:

```bash
python scripts/eval_single_episode_obs_pred_lowdim.py
```

## 8) Important Config Knobs

In `factr/cfg/train_obs_pred_lowdim.yaml`:

- `obs_window`
- `obs_input_dim` (default 21)
- `obs_target_dim` (default 21)
- `pred_horizon` (default 30)
- `pose_action_dim` (default 9)
- `stiffness_classes` (default 3)
- `goal_classes` (default 4)

In `factr/cfg/agent/obs_mlp_gaussian_lowdim.yaml`:

- `hidden_dim`
- `num_layers` (2-4)
- `dropout`
- `min_var`

## 9) Notes

- This is a clean MLP baseline for multi-step low-dim observation prediction.
- The pipeline stays aligned with the active FACTR style: Hydra configs, trainer/task split, and W&B logging.
