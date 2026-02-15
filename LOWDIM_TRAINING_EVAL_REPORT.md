# Lowdim Stiffness-Conditioned Training and Evaluation Report

## 1) Goal and Scope

This report describes the **current implemented pipeline** for low-dimensional behavior cloning in this repo:

- No image input, no ViT features, no visual checkpoint dependency.
- Model input is lowdim observation history + stiffness class label.
- Model output is commanded EE pose chunk:
  - `9D = [x, y, z] + 6D rotation representation`.
- Training and evaluation are stiffness-conditioned and trajectory-sampling capable.

---

## 2) End-to-End Pipeline (Current)

1. Raw trajectory episodes (`ep_*.pkl`) are processed by `process_data/process_data.py`.
2. A replay buffer (`buf.pkl`) is generated with:
   - `obs["state"]` (27D),
   - `obs["goals"]` (if configured),
   - `obs["stiffness_label"]` (class `1/2/3`),
   - `action` (9D pose command only).
3. `RobobufReplayBufferLowdim` builds training samples:
   - observation window `(W, 27)`,
   - action chunk `(T, 9)`,
   - loss mask `(T, 9)`,
   - stiffness label scalar.
4. `LowdimStiffnessCVAEAgent` trains to predict pose chunks.
5. Training eval logs scalar metrics + W&B plots.
6. One-episode evaluation script samples multiple trajectories per step and plots them.

---

## 3) Processed Data Format

### 3.1 Observation topics (state)

Configured in `process_data/cfg/default.yaml`:

- `/franka_robot_state_broadcaster/robot_state`
- `/cartesian_impedance_controller/ee_velocity`
- `/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame`
- `/cartesian_impedance_controller/tracking_error`

This produces `obs["state"]` with `27` dims total:

- pose: `0:9` (`[xyz + rot6d]`)
- ee velocity: `9:15`
- external wrench: `15:21`
- tracking error: `21:27`

### 3.2 Stiffness label in observation

Configured in `process_data/cfg/default.yaml`:

- `stiffness_label_topic: /cartesian_impedance_gains`
- `stiffness_label_key: stiffness`
- `stiffness_norm_thresholds: [200.0, 1000.0]`

Class mapping in `process_data/process_data.py`:

- class `1` (low): `norm < 200`
- class `2` (medium): `200 <= norm < 1000`
- class `3` (high): `norm >= 1000`

Stored per timestep as `obs["stiffness_label"]`.

### 3.3 Action target

Configured action topic:

- `/cartesian_impedance_controller/pose_command: 9`

So each transition action is only pose command (not stiffness gains).

### 3.4 Buffer transition structure

Generated in `process_data/utils_data_process.py`:

- `obs`:
  - `"state"` (`float32`, 27D),
  - optional `"goals"`,
  - optional `"stiffness_label"` (int).
- `action`: `float32`, 9D.
- `reward`: terminal sparse (`True` at episode end).
- `is_first`: `True` at episode start.

---

## 4) Training Dataset Construction

Implemented in `factr/replay_buffer.py` (`RobobufReplayBufferLowdim`):

- Splits replay buffer into episodes using `step.first`/`step.prev`.
- Splits train/test at **episode level** (`n_test_ratio=0.10` by config).
- For each timestep `t`:
  - Builds left-padded obs window `(obs_window, 27)` (default `8`).
  - Builds future action chunk `(ac_chunk, 9)` (default `30`).
  - Builds mask `(ac_chunk, 9)` with zeros after episode end.
  - Attaches episode stiffness label (1..3).

Returned sample format:

- `({}, obs_tensor)` where `obs_tensor` shape = `(W, 27)`
- `action_tensor` shape = `(T, 9)`
- `mask_tensor` shape = `(T, 9)`
- `label_tensor` shape = `()` long

Batch format used by trainer:

- `(imgs, obs), actions, mask, labels`
- `imgs` is empty dict `{}` in lowdim mode.

---

## 5) Model: LowdimStiffnessCVAEAgent

Implemented in `factr/models/lowdim_action_transformer.py`.

### 5.1 Inputs

- `obs`: `(B, W, 27)`
- `class_labels`: `(B,)` with classes in `{1,2,3}` or `{0,1,2}` (internally normalized).
- Training additionally uses target actions and masks.

### 5.2 Tokenization (grouped, one token per group)

Per sample, model builds **6 tokens**:

1. `CLS` token
2. Pose token from `state[:, :, 0:9]`
3. Velocity token from `state[:, :, 9:15]`
4. Wrench token from `state[:, :, 15:21]`
5. Tracking token from `state[:, :, 21:27]`
6. Stiffness class embedding token

Each group uses flattened observation window (`W * group_dim`) through an MLP encoder.

### 5.3 Latent context for z

`z_context_mode` is currently `cls_all_obs` (from config), so latent context uses:

- `CLS token` built from all observation-group tokens
- `Pose token`
- `Velocity token`
- `Wrench token`
- `Tracking-error token`

Other modes exist (`all_tokens`, `attn_pool`) but are not default.

### 5.4 CVAE path

- Prior: `p(z|context)`
- Posterior: `q(z|context, action_chunk)`
- Decoder: transformer decoder over action queries to output chunk `(B, ac_chunk, 9)`.

### 5.5 Loss

- Reconstruction: masked `L1` over predicted vs target action chunk.
- KL: diagonal Gaussian KL `KL(q || p)`.
- Total: `recon + beta * KL` (`beta=1.0` default, optional free-bits).

### 5.6 Outputs

Training `forward()` returns:

- `total_loss`
- `l1_loss`
- `kl`

Inference helpers:

- `get_actions_base(...)` -> deterministic chunk from prior mean.
- `get_actions_prior(..., num_samples=N)` -> sampled chunks `(B, N, T, 9)`.
- `get_actions_pos(...)` -> posterior-conditioned samples.

---

## 6) Training Execution

### 6.1 Entry points

- Shell launcher: `scripts/train_bc.sh`
- Python entry: `factr/train_bc_policy.py`
- Config: `factr/cfg/train_bc_lowdim.yaml`

### 6.2 Lowdim/no-image behavior

`scripts/train_bc.sh` now defaults to:

- `--config-name train_bc_lowdim`
- `task=single_franka_lowdim`
- no ViT restore path
- no image feature dependency

### 6.3 Device behavior

`train_bc_policy.py` selects CUDA only if usable; otherwise falls back to CPU.
`task.py` enables DataLoader `pin_memory` only if CUDA is truly usable.
`trainers/base.py` contains a CPU safety guard for optimizer CUDA graph checks.

### 6.4 Key training config defaults

From `factr/cfg/train_bc_lowdim.yaml`:

- `batch_size: 64`
- `ac_chunk: 30`
- `obs_window: 8`
- `lr: 2e-4`
- `max_iterations: 20000`
- `eval_freq: 500`
- `save_freq: 2000`

Optimizer/schedule (`factr/cfg/trainer/adamw_cos_lowdim.yaml`):

- `AdamW` with `weight_decay=0.01`
- cosine schedule with warmup (`250` steps).

---

## 7) Evaluation During Training (W&B + Console)

Implemented in `factr/task.py` (`BCTask.eval`):

- Validation reconstruction loss (`eval/task_loss`)
- Action L2 (`eval/action_l2`)
- Sign mismatch metric (`eval/action_lsig`)
- Per pose-dimension L2: `eval/pose_dim{i}_l2` (`i=1..9`)
- W&B plot: chunk-step MSE (`eval/plot_chunk_mse`)
- W&B plot: pose-dim MSE bar (`eval/plot_dim_mse`)
- W&B plot: example trajectory true vs pred (`eval/plot_example_dim1`)
- Optional classification accuracy if logits are present.

Prediction source in eval:

- `get_actions_prior` (default path for this model),
- first sample from sampled trajectories is used for batch-level eval metrics.

---

## 8) One-Episode Offline Evaluation (Trajectory Sampling + Plots)

Script: `scripts/eval_one_episode_lowdim.py`

What it does:

1. Loads trained checkpoint and config.
2. Loads one episode from `buf.pkl`.
3. For each timestep:
   - builds obs window `(W, 27)`,
   - samples `N` prior trajectories (`N=10` default),
   - stores full sampled chunk and first-step action.
4. Saves:
   - `.npz` raw arrays,
   - `_pose_traj.png` with all sampled trajectories plotted,
   - `_metrics.png` summary errors.

Saved arrays include:

- `sampled_chunks`: `(T_episode, N, ac_chunk, 9)`
- `sampled_first_step`: `(N, T_episode, 9)`
- `true_pose`: `(T_episode, 9)`
- `stiffness_label`, checkpoint metadata.

Plot behavior:

- All `N` sampled trajectories are shown explicitly (colored `sample_1..sample_N`).
- Ground truth and sample mean are overlaid.
- ±1 std band is shown around mean.

---

## 9) Quick Buffer Inspection

Script: `scripts/inspect_new_buffer.py`

Checks:

- transition and episode counts,
- observation keys,
- state/action dimensionality,
- episode length stats,
- stiffness label distribution from `obs["stiffness_label"]`,
- legacy stiffness signature check from `action[9:15]` (expected empty for 9D pose-only actions).

---

## 10) Inputs and Outputs Summary

### 10.1 Training input tensors (per batch)

- `obs`: `(B, 8, 27)` (default)
- `actions`: `(B, 30, 9)` (default)
- `mask`: `(B, 30, 9)`
- `labels`: `(B,)` in stiffness classes
- `imgs`: `{}` (unused in lowdim mode)

### 10.2 Training model outputs

- `total_loss` (scalar)
- `l1_loss` (scalar)
- `kl` (scalar)

### 10.3 Inference outputs

- `get_actions_base`: `(B, 30, 9)`
- `get_actions_prior` with `num_samples=N`: `(B, N, 30, 9)`

---

## 11) Important Notes

1. Action semantics are EE pose command (not joint angles, not stiffness gains).
2. Training path is lowdim-only and does not require visual feature `.pth`.
3. Buffer is expected pre-normalized for training; no extra training-side normalization is applied.
4. Stiffness label is provided as observation context and used via embedding token.

---

## 12) Minimal Run Commands

From `factr/`:

```bash
conda activate factr
python -m pip install -e .
```

Regenerate buffer:

```bash
python process_data/process_data.py
```

Inspect buffer:

```bash
python scripts/inspect_new_buffer.py --buffer-path process_data/processed_data/fourgoals_1_act/buf.pkl
```

Train (lowdim, no image):

```bash
./scripts/train_bc.sh
```

One-episode sampled-trajectory evaluation:

```bash
python scripts/eval_one_episode_lowdim.py \
  --model-name <checkpoint_folder_name> \
  --checkpoint latest \
  --buffer-path process_data/processed_data/fourgoals_1_act/buf.pkl \
  --episode-index 0 \
  --num-samples 10 \
  --stiffness-label 0
```
