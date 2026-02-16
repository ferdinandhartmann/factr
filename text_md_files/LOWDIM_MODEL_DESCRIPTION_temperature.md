# Low-Dim Stiffness CVAE Model: Detailed Description

This document describes how the current low-dimensional model works end-to-end in this repository, based on the current code path used by:

- `python factr/train_bc_policy.py --config-name train_bc_lowdim`

It focuses on the implemented pipeline (data -> model -> loss -> eval/plots), not a conceptual prototype.

---

## 1) Goal and Current Setup

The model learns offline behavior cloning with a CVAE structure to predict future commanded end-effector pose chunks while modeling trajectory ambiguity.

Current setup:
- Observation per timestep: `27D` low-dim state
  - `pose(9) + ee_velocity(6) + external_wrench(6) + tracking_error(6)`
- Additional conditioning: stiffness label in `{1,2,3}`
- Action per timestep: commanded EE pose `9D`
- Training target: action chunk of length `ac_chunk` (default `30`)
- Observation input to model: sliding window of length `obs_window` (default `8`)
- CVAE latent: `z` (default dimension `d_z=32`)

Main config files:
- `factr/cfg/train_bc_lowdim.yaml`
- `factr/cfg/task/single_franka_lowdim.yaml`
- `factr/cfg/agent/transformer_lowdim.yaml`
- `factr/cfg/trainer/adamw_cos_lowdim.yaml`

---

## 2) Data Pipeline

### 2.1 Buffer source

Training and test data are loaded from robobuf pickles:
- Train: `.../buf_train.pkl`
- Test: `.../buf_test.pkl`

Configured in:
- `factr/cfg/train_bc_lowdim.yaml`

### 2.2 Dataset class

Low-dim dataset class:
- `factr.replay_buffer.RobobufReplayBufferLowdim`
- File: `factr/replay_buffer.py`

What it does:
1. Loads robobuf transitions.
2. Splits transitions into episodes using `is_first`/`prev`.
3. Extracts:
   - observation state vector (`obs["state"]`)
   - stiffness label (`obs["stiffness_label"]` or fallback keys)
   - action vector (first 9 dims)
4. Builds sliding observation windows of shape `(W, 27)`.
5. Builds action chunks of shape `(T, 9)` with mask `(T, 9)`.
6. Pads incomplete tail chunks by repeating last valid action and masking invalid steps.

Returned sample format:
```python
((imgs_dict, obs_window), action_chunk, action_mask, label)
```

For low-dim mode:
- `imgs_dict` is empty `{}`.
- `obs_window` is `(W, 27)`.
- `action_chunk` is `(T, 9)`.
- `action_mask` is `(T, 9)`.
- `label` is scalar class index (1..3 in data; normalized internally in model).

---

## 3) Training Entry and Control Flow

Entry script:
- `factr/train_bc_policy.py`

High-level flow:
1. Hydra builds config.
2. `misc.init_job(cfg)` handles WandB new-run vs resume behavior.
3. Instantiate model, trainer, task/dataloaders.
4. Training loop:
   - forward
   - backward
   - optimizer step
   - schedule step
   - periodic eval and checkpointing

Important details implemented:
- New runs are created by default (`resume: false`), even if prior run files exist.
- If `resume: true` and checkpoint exists, trainer loads checkpoint and sets global step.
- Additional terminal prints show active train/test buffer and eval-plot config.

---

## 4) Model Architecture

Model class:
- `factr.models.lowdim_action_transformer.LowdimStiffnessCVAEAgent`
- File: `factr/models/lowdim_action_transformer.py`

### 4.1 Inputs to model `forward`

`forward(imgs, obs, ac_flat, mask_flat, class_labels=...)`

Shapes:
- `obs`: `(B, W, 27)`
- `ac_flat`: `(B, T*9)` or `(B, T, 9)` (internally reshaped)
- `mask_flat`: same flattening convention as `ac_flat`
- `class_labels`: `(B,)`

### 4.2 Observation tokenization with windowed grouped features

The model splits the 27D state into fixed slices:
- pose: `0:9`
- velocity: `9:15`
- wrench: `15:21`
- tracking: `21:27`

For each group:
1. Flatten across window (`W * group_dim`)
2. Encode with a small MLP to one token of size `token_dim`

This produces four group tokens:
- `pose_token`, `vel_token`, `wrench_token`, `track_token`

Stiffness token:
- `class_labels` -> normalized to `0..(stiffness_classes-1)`
- Embedded via `nn.Embedding` to `stiffness_token`

CLS token:
- Built from concatenated observation group tokens via another MLP

Context token sequence:
- `[cls, pose, vel, wrench, track, stiffness]` -> `(B, 6, token_dim)`
- Learned positional token bias added
- Transformer encoder processes this sequence

### 4.3 CVAE prior

Prior path:
- Build `z_context` from encoded context tokens (mode-controlled; default `cls_all_obs`)
- Pass through prior MLP backbone
- Output:
  - `mu_p`: `(B, d_z)`
  - `logvar_p`: `(B, d_z)`

Distribution:
- `p(z|c) = N(mu_p, diag(exp(logvar_p)))`

### 4.4 CVAE posterior (Transformer-based)

Posterior module:
- `_PosteriorTransformer`

Inputs:
- `context_tokens` (detached for posterior path)
- ground-truth action chunk `(B, T, 9)`

Process:
1. Actions embedded into token space.
2. Action positional embeddings added.
3. `POST_CLS` token prepended.
4. Transformer encoder runs over `[POST_CLS, context_tokens, action_tokens]`.
5. `POST_CLS` output projected to:
   - `mu_q`: `(B, d_z)`
   - `logvar_q`: `(B, d_z)`

Distribution:
- `q(z|x,c) = N(mu_q, diag(exp(logvar_q)))`

### 4.5 Reparameterization

Training sample:
- `eps ~ N(0, I)`
- `z = mu_q + exp(0.5*logvar_q) * eps`

### 4.6 Action decoder

Decoder is TransformerDecoder-based:
1. Convert `z` to token bias via `z_to_token`.
2. Start from learned action queries (`ac_chunk` queries).
3. Add `z` bias to each query.
4. Decode against context tokens (cross-attention).
5. Linear head outputs `(B, T, 9)`.

This produces predicted action chunk.

---

## 5) Losses and Diagnostics

Primary training loss:
- Reconstruction:
  - masked L1 over predicted vs target actions
  - normalized by valid mask sum
- KL term:
  - `KL(q||p)` for diagonal Gaussians
  - optional free-bits clamp (`free_bits`)
- Total:
  - `total_loss = recon + beta * kl`

Additional diagnostics computed and logged:
- `prior_std_mean`
- `posterior_std_mean`
- `prior_entropy`
- `posterior_entropy`

Interpretation:
- Posterior typically sharper (lower entropy/std) than prior.
- Prior should keep uncertainty where futures are ambiguous.

---

## 6) Inference and Sampling Behavior

Implemented action APIs:
- `get_actions_base(...)`: deterministic baseline (uses prior mean as latent condition)
- `get_actions_prior(...)`: sample from prior or use prior mean
- `get_actions_pos(...)`: sample from posterior (with target action provided)

Sampling temperature:
- `sample_temperature` multiplies latent noise:
  - `z = mu + temperature * sigma * eps`
- temperature > 1 increases diversity.
- Used in eval fan plotting to visualize multi-modal behavior.

---

## 7) Evaluation Pipeline

Evaluation runs in `BCTask.eval` (`factr/task.py`) and logs:

### 7.1 Scalar metrics
- `eval/task_loss` (posterior reconstruction L1)
- `eval/prior_l1` (deterministic prior rollout quality)
- `eval/posterior_kl`
- `eval/action_l2`
- `eval/action_lsig` (sign mismatch ratio)
- per-dim errors: `eval/joint{i}_l2`
- per-horizon errors: `eval/chunk_step_{k}_mse`
- entropy/std diagnostics:
  - `eval/prior_std_mean`
  - `eval/posterior_std_mean`
  - `eval/prior_entropy`
  - `eval/posterior_entropy`
- diversity metric:
  - `eval/prior_sample_action_std_mean`

### 7.2 Plot generation

Configured by:
- `eval_plot_max_steps` (default 300)
- `eval_plot_prediction_stride` (default 5)
- `eval_plot_num_samples` (default 10)
- `eval_plot_sample_temperature` (default 1.5)

Generated plots:
1. `eval/prediction_example`
   - Single deterministic true vs predicted chunk.
2. `eval/error_summary`
   - Horizon MSE curve + per-dim MSE bars.
3. `eval/prior_fan_all`
   - Fan plot from sampled prior trajectories.
4. `eval/prior_fan_stiffness_{label}`
   - Same fan plot split by stiffness label.

Fan plot logic:
- Collect eval anchor steps every `stride`.
- For each anchor, sample `num_samples` trajectories from prior.
- Overlay sampled trajectories against ground truth.
- This is used for trajectory-selection use cases and uncertainty understanding.

---

## 8) WandB and Terminal Behavior

### 8.1 WandB run lifecycle

Handled by `factr/misc.py`:
- Creates a new run by default.
- Resumes only when configured (`resume: true`) with compatible run state.
- Avoids accidentally reusing stale run IDs when `exp_name` changes.

### 8.2 Terminal logging

At startup:
- device, buffers, batch size, chunk/window sizes
- eval plot parameters

At each eval:
- Posterior/Prior L1
- KL, Action L2, sign error
- prior/post std and entropy
- plot sampling config and number of sampled eval anchors
- sampled diversity summary line

---

## 9) Key Config Parameters and Their Effect

### Data/task
- `buffer_path`, `test_buffer_path`: train/test split quality.
- `obs_window`: temporal context length.
- `ac_chunk`: prediction horizon length.
- `action_index_offset`: observation-to-action alignment.

### Model
- `d_z`: latent capacity.
- `beta`: KL weight (trade-off between reconstruction and latent regularization).
- `free_bits`: prevents trivial KL collapse.
- `token_dim`, `hidden_dim`, layer counts: capacity/compute trade-off.
- `z_context_mode`: what context summary prior uses.

### Eval/sampling
- `eval_plot_num_samples`: number of sampled trajectories in fan plot.
- `eval_plot_sample_temperature`: latent sampling spread multiplier.

---

## 10) Practical Readout for “Is It Working?”

Healthy signs:
1. `eval/task_loss` and `eval/prior_l1` both decrease.
2. Prior/posterior gap stays small.
3. `eval/posterior_kl` remains positive (not zero collapse, not unstable explosion).
4. Fan plots show plausible spread:
   - tight in deterministic regions
   - wider around branching regions
5. Stiffness-specific fan plots differ meaningfully.
6. `eval/prior_sample_action_std_mean` is non-trivial and controlled by temperature.

If sampled trajectories are too similar:
- increase `eval_plot_sample_temperature` for visualization,
- and/or improve latent usage via `beta`, `free_bits`, and `d_z` tuning.



How I tracked it:

  - Searched for unique tokens (sample_temperature, eval_plot_sample_temperature,
    prior_sample_action_std_mean, plot_temp, Sample diversity) across the repo.
  - Backed up the affected files first.
  - Removed only the matching blocks in those files:
      - factr/models/lowdim_action_transformer.py
      - factr/cfg/task/single_franka_lowdim.yaml
      - LOWDIM_TRAINING_DIAGNOSTICS_GUIDE.md
  - Re-ran search to confirm those tokens were gone.
  - Created patches/reapply_sampling_diversity.patch from backup-vs-current diffs.
      - z = mu + T * sigma * eps
  - Effect:
      - T = 1.0: normal learned stochasticity.
      - T > 1.0: wider spread, more diverse but can become less realistic.
      - T < 1.0: tighter, more conservative, less diversity.
  - In your code, it was only used for sampled trajectory generation (fan-plot/eval sampling path),