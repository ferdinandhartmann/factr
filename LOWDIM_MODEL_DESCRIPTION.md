# Low-Dim CVAE Current Implementation (Code-Accurate)

This document explains how the current low-dimensional CVAE policy training works in this repository.
It is based on the active code path used by:

```bash
cd /home/ferdinand/activeinference/factr
python factr/train_bc_policy.py --config-name train_bc_lowdim
```

It describes what the code does now (not a planned design).

## 1) Active config snapshot

Main configs:
- `factr/cfg/train_bc_lowdim.yaml`
- `factr/cfg/task/single_franka_lowdim.yaml`
- `factr/cfg/agent/transformer_lowdim.yaml`
- `factr/cfg/trainer/adamw_cos_lowdim.yaml`

Current default values:
- Observation dim: `27`
- Action dim: `9` (commanded end-effector pose)
- Observation window: `obs_window=8`
- Action chunk horizon: `ac_chunk=30`
- Latent dim: `d_z=16`
- Stiffness classes: `3`
- KL weight: `beta=0.02`
- Free bits: `0.5`
- Eval fan-plot sampling:
  - max eval anchors: `300`
  - anchor stride: `5`
  - sampled trajectories per anchor: `10`

Dataset paths (current defaults):
- Train: `/home/ferdinand/activeinference/factr/process_data/processed_data/fourgoals_1_act/buf_train.pkl`
- Test: `/home/ferdinand/activeinference/factr/process_data/processed_data/fourgoals_1_act/buf_test.pkl`

## 2) Entry point and training control flow

Training entry:
- `factr/train_bc_policy.py`

High-level loop:
1. Initialize run state and WandB (`misc.init_job` in `factr/misc.py`).
2. Instantiate agent (`LowdimStiffnessCVAEAgent`), trainer (`BehaviorCloning`), and task (`BCTask`) via Hydra.
3. Build train/test dataloaders from `RobobufReplayBufferLowdim`.
4. Iterate until `max_iterations`:
   - `training_step` forward
   - backprop
   - optimizer step
   - scheduler step
   - periodic eval (`eval_freq`)
   - periodic checkpoint save (`save_freq`)

Key files:
- `factr/train_bc_policy.py`
- `factr/trainers/bc.py`
- `factr/trainers/base.py`

## 3) Data pipeline (low-dim replay buffer)

Low-dim dataset class:
- `RobobufReplayBufferLowdim` in `factr/replay_buffer.py`

### 3.1 Episode reconstruction

The replay buffer is converted into episodes using:
- `step.first` or `step.is_first` or `step.prev is None` as episode start.

This ensures observation windows and action chunks are built inside episode boundaries.

### 3.2 Observation extraction

For each step:
- Reads `obs["state"]` as a flat vector.
- Optional `goals` concatenation exists but is currently disabled (`include_goals: False`).

Current expected state dimension:
- `obs_dim=27`

### 3.3 Stiffness label extraction

Per episode label is inferred from first step using first available key:
- `stiffness_label`, else `stiffness_class`, else `stiffness`

Behavior:
- Missing label falls back to `1`.
- Internally normalized to class range (1..3 in dataset-level handling, then 0-based inside model embedding).

### 3.4 Observation window and target chunk creation

For each valid anchor time `t` in an episode:
- Build observation window of length `obs_window`.
- Left-pad with earliest available state if not enough history.
- Build target action chunk of length `ac_chunk` starting at `t + action_index_offset`.
- If episode ends before full horizon:
  - repeat last valid action for padding
  - set mask to `0` for padded positions

Current alignment:
- `action_index_offset=1` (predict next-step onward)

### 3.5 Returned sample format

Each dataset sample returns:

```python
({}, obs_window), action_chunk, action_mask, label
```

Shapes:
- `obs_window`: `(W, 27)` where `W=obs_window`
- `action_chunk`: `(T, 9)` where `T=ac_chunk`
- `action_mask`: `(T, 9)` (1 for valid, 0 for padded)
- `label`: scalar stiffness class id

`imgs` is `{}` in this low-dim setup.

## 4) Model architecture

Model:
- `LowdimStiffnessCVAEAgent` in `factr/models/lowdim_action_transformer.py`

This is a conditional VAE with:
- Prior network (MLP) for `p(z | context)`
- Posterior network (Transformer) for `q(z | context, target_actions)`
- Transformer decoder to generate action chunks from `z` and encoded context

### 4.1 Input contracts

`forward(imgs, obs, ac_flat, mask_flat, class_labels=...)` expects:
- `obs`: `(B, W, 27)`
- `ac_flat`: `(B, T*9)` or `(B, T, 9)`
- `mask_flat`: same flattening convention as actions
- `class_labels`: `(B,)`

### 4.2 Observation tokenization with windowed grouped features

The 27D state is split into fixed groups:
- pose: `0:9`
- velocity: `9:15`
- wrench: `15:21`
- tracking: `21:27`

For each group:
1. Flatten over time window (`W * group_dim`).
2. Encode with a small MLP into one token (`token_dim`).

Produces 4 data tokens:
- `pose_token`, `vel_token`, `wrench_token`, `track_token`

Stiffness conditioning:
- Label goes through `nn.Embedding(stiffness_classes, token_dim)` -> `stiffness_token`

CLS token:
- Built from concatenated data tokens through `cls_from_obs` MLP

Context sequence:
- `[cls, pose, vel, wrench, track, stiffness]` -> `(B, 6, token_dim)`
- Add learned positional token bias
- Pass through Transformer encoder (`context_encoder`)
- LayerNorm (`context_norm`)

### 4.3 Prior network (MLP)

`z_context` is built from encoded context tokens based on `z_context_mode`.
Current default:
- `z_context_mode=cls_all_obs`
- Prior input is concatenation `[cls, pose, vel, wrench, track]`

Prior outputs:
- `mu_p`: `(B, d_z)`
- `logvar_p`: `(B, d_z)`

Distribution:
- `p(z|c) = N(mu_p, diag(exp(logvar_p)))`

### 4.4 Posterior network (Transformer)

Posterior module:
- `_PosteriorTransformer`

Inputs:
- encoded context tokens
- ground-truth action chunk `(B, T, 9)`

Process:
1. Embed each action step into token_dim.
2. Add action positional embedding.
3. Prepend a learned posterior CLS token.
4. Run Transformer encoder over:
   - `[post_cls, context_tokens, action_tokens]`
5. Take `post_cls` output and project to:
   - `mu_q`, `logvar_q`

Distribution:
- `q(z|a,c) = N(mu_q, diag(exp(logvar_q)))`

### 4.5 Latent sampling (reparameterization)

Training uses posterior sample:
- `eps ~ N(0, I)`
- `z = mu_q + exp(0.5 * logvar_q) * eps`

For prior sampling APIs, `z` is sampled from prior in the same form when `sample=True`.

### 4.6 Action decoder

Decoder is TransformerDecoder-based:
1. Map latent `z` to token bias (`z_to_token`).
2. Start from learned action queries (`ac_chunk` queries).
3. Add latent bias to each query.
4. Decode with cross-attention on context tokens.
5. Project each decoded query to 9D action.

Output action chunk:
- `(B, T, 9)` (or `(B, num_samples, T, 9)` in sampled prior/posterior helper APIs)

## 5) Losses and optimization

Inside `LowdimStiffnessCVAEAgent.forward`:

### 5.1 Reconstruction loss

Masked L1:
- `recon = L1(pred_actions, target_actions)` elementwise
- Multiply by `mask`
- Normalize by total valid mask count

This prevents padded tail steps from affecting training.

### 5.2 KL loss

KL between two diagonal Gaussians:
- `KL(q||p)` with closed-form expression

Optional free-bits:
- If `free_bits` is set, KL per sample is clamped from below by `free_bits * d_z`

### 5.3 Total loss

- `total_loss = recon + beta * kl`

Current defaults:
- `beta=0.02`
- `free_bits=0.5`

### 5.4 Extra diagnostics computed by model

Returned in loss dict:
- `l1_loss`
- `kl`
- `prior_std_mean`
- `posterior_std_mean`
- `prior_entropy`
- `posterior_entropy`
- `total_loss`

## 6) Trainer behavior and logging

Trainer class:
- `BehaviorCloning` in `factr/trainers/bc.py`

Per step:
1. Move tensors to device.
2. Flatten actions/masks for model API.
3. Call model and get `loss_dict`.
4. Backprop only on `total_loss`.
5. Log training metrics via `BaseTrainer.log`.

Logging conventions (`BaseTrainer`):
- train keys prefixed `train/`
- eval keys prefixed `eval/`
- train smoothing uses running mean (`TRAIN_LOG_FREQ=100`)
- train grad norm is additionally logged every 20 global steps in `train_bc_policy.py`

## 7) Evaluation behavior (current)

Evaluation code:
- `BCTask.eval` in `factr/task.py`

### 7.1 Scalars

Computed/logged during eval:
- `eval/task_loss` (posterior reconstruction L1 from forward pass)
- `eval/prior_l1` (deterministic prior prediction L1, `sample=False`)
- `eval/posterior_kl`
- `eval/action_l2`
- `eval/action_lsig` (sign mismatch ratio)
- `eval/prior_std_mean`
- `eval/posterior_std_mean`
- `eval/prior_entropy`
- `eval/posterior_entropy`
- Per-dim MSE: `eval/joint{i}_l2`
- Per-horizon MSE: `eval/chunk_step_{k}_mse`

### 7.2 Plots to WandB

If matplotlib is available:
- `eval/prediction_example`
  - single deterministic chunk prediction vs ground truth
- `eval/error_summary`
  - chunk-step MSE and per-dimension MSE
- `eval/prior_fan_all`
  - sampled prior trajectories vs ground truth over many anchors
- `eval/prior_fan_stiffness_1`, `_2`, `_3`
  - same fan plot split by stiffness class

Fan-plot sampling setup (from task config):
- collect up to `eval_plot_max_steps` anchors
- keep every `eval_plot_prediction_stride`-th eval sample
- sample `eval_plot_num_samples` trajectories from prior at each anchor

Current implementation does not use an explicit sampling-temperature parameter.

## 8) Inference helper methods in model

Implemented helper APIs:
- `get_actions_base(...)`
  - deterministic decoding from prior mean
- `get_actions_prior(..., sample=True/False, num_samples=N)`
  - sample from prior (or use mean) and decode actions
- `get_actions_pos(..., sample=True/False, num_samples=N)`
  - sample from posterior using target actions and decode
- `get_uncertainty_entropy(...)`
  - sample prior trajectories and compute std-based uncertainty summary

## 9) Shape summary

For default config:
- Input obs to model: `(B, 8, 27)`
- Context tokens after encoding: `(B, 6, token_dim)`
- Prior/posterior params: `(B, 16)`
- Decoded action chunk: `(B, 30, 9)`
- Eval sampled trajectories: `(B, 10, 30, 9)` for fan plot generation

## 10) Practical run and sanity checks

Run command:

```bash
cd /home/ferdinand/activeinference/factr
python factr/train_bc_policy.py --config-name train_bc_lowdim
```

Good quick checks:
1. Startup prints correct train/test buffer paths and eval fan-plot config.
2. `Posterior L1` and `Prior L1` decrease over eval calls.
3. `KL` stays positive and stable (not exploding, not collapsing to zero too early).
4. Fan plots show plausible sampled trajectories and stiffness-conditioned differences.
5. WandB receives scalar metrics and image plots at each eval step.
