# Plan: Switch FACTR BC+CVAE To Low-Dim EE State + Stiffness + Obs-Window

This document is a detailed, actionable plan to change the **training I/O** from:

- **Old**: images + external joint torque token (ViT-based `TransformerAgent`)
- **New**: purely low-dimensional observations:
  - end-effector pose (9D = 3D position + 6D rotation)
  - end-effector velocity (6D)
  - external end-effector wrench (6D)
  - controller tracking error (6D)
  - plus a **stiffness label** in `{1,2,3}`
  - with an **observation window** of `W` timesteps
  - actions are **commanded end-effector pose** (9D), predicted as an `ac_chunk` sequence

The goal is to keep the **same CVAE behavior**:
- Prior `p(z|c)` depends only on the observation context `c`.
- Posterior `q(z|x,c)` depends on observation context `c` and ground-truth actions `x` (training only).
- Sample `z` via reparameterization.
- Decode action trajectories conditioned on `(c, z)`.
- Train with `recon + beta * KL(q||p)` and masked losses.

This repo already contains a low-dim pipeline (`RobobufReplayBufferLowdim` + `LowdimStiffnessCVAEAgent`). The plan below uses that path because it is the most direct way to get:
- `obs_window` support,
- stiffness conditioning,
- 9D pose-command actions.

If you need the posterior to be a Transformer (like the original explanation), there is a dedicated step to implement that too.

---

## Target Interface (Lock This First)

### Observation at timestep `t`

We define a per-step state vector `s_t ∈ R^27` with a fixed ordering:

1. **EE pose**: `ee_pose_t ∈ R^9`
   - `ee_pos_t ∈ R^3`
   - `ee_rot6d_t ∈ R^6` (whatever 6D representation your controller logs; keep consistent everywhere)
2. **EE velocity**: `ee_vel_t ∈ R^6`
3. **External wrench**: `ee_wrench_t ∈ R^6` (in stiffness frame, per your topic)
4. **Tracking error**: `track_err_t ∈ R^6`

Concatenate:

```
s_t = [ee_pose(9), ee_vel(6), ee_wrench(6), track_err(6)]  # total 27
```

### Observation window

Each training sample provides a window ending at `t`:

```
o_t = [s_{t-W+1}, ..., s_t]  # shape (W, 27)
```

Rules:
- Windows must **not cross episode boundaries**.
- For early timesteps in an episode, left-pad by repeating the first available state (or drop those samples; choose one policy and keep it consistent).

### Stiffness

Stiffness is a categorical label:

```
label ∈ {1,2,3}
```

We will treat stiffness as:
- a separate input (`class_labels`) passed into the model, and/or
- an embedded token concatenated to the observation context tokens.

### Action at timestep `t`

Actions are commanded end-effector pose:

```
a_t ∈ R^9
```

Training predicts an action chunk:

```
A_t = [a_{t+offset}, ..., a_{t+offset+T-1}]   # shape (T, 9), T = ac_chunk
```

Typical alignment:
- `offset=1` so `obs[t] -> actions starting at t+1`.

---

## Phase 0: Confirm Where These Signals Live In The Buffer

Before coding, do a quick inventory:

1. What ROS topics in your recorded data correspond to:
   - measured EE pose (9D)
   - commanded EE pose (9D)
   - EE velocity (6D)
   - tracking error (6D)
   - external wrench (6D)
   - stiffness gains / stiffness label
2. Confirm each topic’s message structure:
   - is the payload under a key (e.g., `"ee_velocity"`) or directly under `"data"`?
3. Confirm the **rotation 6D representation** used in the dataset and ensure it is consistent between:
   - observed pose rotation (6D)
   - commanded pose rotation (6D)

Output of this phase should be a single table of:
`signal -> topic -> key -> dim`.

---

## Phase 1: Buffer Generation (process_data) Produces `state(27)` + `stiffness_label`

Files involved:
- `process_data/cfg/default.yaml`
- `process_data/process_data.py`
- `process_data/utils_data_process.py` (already writes `obs["state"]` and `obs["stiffness_label"]`)

### 1.1 Update `process_data/cfg/default.yaml`

Goal: `obs_topics` should cover exactly the 4 observation groups that form the 27D state.

Recommended topic set (adjust topic names to your dataset):
- pose (9D): measured EE pose topic (or a topic already giving 9D pose)
- `.../ee_velocity` (6D)
- `.../external_wrench_in_stiffness_frame` (6D)
- `.../tracking_error` (6D)

Keep the action config as commanded pose:
- `action_config: /cartesian_impedance_controller/pose_command: 9`

Enable stiffness label extraction:
- `stiffness_label_topic: /cartesian_impedance_gains`
- `stiffness_label_key: stiffness`
- thresholds mapping norm -> class `{1,2,3}`

### 1.2 Ensure `process_data/process_data.py` extracts 27D state in a stable order

The buffer builder (`generate_robobuf`) expects trajectories like:
- `traj["states"]` shape `(N, 27)`
- `traj["actions"]` shape `(N, >=9)` (first 9 used if you keep extra dims)
- optional `traj["stiffness_label"]` shape `(N, 1)` or `(N,)`

Concrete tasks:

1. Make sure the state vector is built in the exact order:
   - `[pose(9), ee_vel(6), wrench(6), tracking(6)]`
2. If your pose topic is not already 9D, add a spec in `state_topic_specs`:
   - e.g. `"/cartesian_impedance_controller/ee_pose": {"keys": [...], "dim": 9, ...}`
3. Keep action extraction to commanded pose 9D:
   - `action_topic_specs` already supports `/cartesian_impedance_controller/pose_command` with key `ee_pose_commanded`.
4. Verify that `stiffness_labels` are computed and stored in each trajectory:
   - `traj["stiffness_label"] = stiffness_labels[:, None]`
5. Regenerate the buffer(s) (`buf_train.pkl`, `buf_test.pkl` if split enabled).

### 1.3 Normalization policy

`process_data/process_data.py` already has grouped normalization helpers designed for this exact 27D structure:
- EE position: Gaussian (optional clip)
- EE orientation 6D: identity
- EE velocity: Gaussian (optional clip)
- tracking error: Gaussian (optional clip)
- wrench: Gaussian (optional clip)

Concrete tasks:
- Set `pose_topic` in `process_data/cfg/default.yaml` to the topic that provides the 9D measured pose.
- Check that grouped normalization is being used (no "falling back to gaussian norm" warnings).
- Confirm that action normalization treats orientation as identity and position as Gaussian.

---

## Phase 2: Dataset/Dataloader Emits `(B, W, 27)` Obs Windows + `(B, T, 9)` Action Chunks + Labels

Files involved:
- `factr/replay_buffer.py` (`RobobufReplayBufferLowdim`)
- `factr/cfg/task/single_franka_lowdim.yaml`
- `factr/task.py` (`BCTask`)

### 2.1 Use `RobobufReplayBufferLowdim`

This dataset already does what you want:
- Splits the robobuf transitions into episodes using `is_first` / `.prev`.
- Builds sliding windows of length `obs_window=W`.
- Builds action chunks of length `ac_chunk=T`.
- Creates a `loss_mask` for padded actions at the end of an episode.
- Outputs a stiffness label per sample.

Key fields to set:
- `obs_window: W`
- `obs_dim: 27`
- `pose_action_dim: 9`
- `action_index_offset: 1` (recommended alignment)
- `stiffness_classes: 3`

Important: label source
- Preferred: `obs["stiffness_label"]` stored explicitly in the buffer (process_data already supports this).
- Fallback: signature inference from action dims `[9:15]` exists, but explicit labels are safer.

### 2.2 Update task config (`factr/cfg/task/single_franka_lowdim.yaml`)

Concrete tasks:
- Ensure `cam_indexes: []` and `n_cams` resolves to 0.
- Set `obs_dim: 27`, `ac_dim: 9`.
- Point `train_buffer.buffer_path` and `test_buffer.buffer_path` to your new buffers.

### 2.3 Ensure training loop passes labels through

No changes needed if you use:
- `factr/trainers/bc.py` which calls `self.model(..., class_labels=labels)`.

---

## Phase 3: Model Architecture For Low-Dim + Window + Stiffness (Keep CVAE)

You have two implementation options.

### Option A (Recommended): Use `LowdimStiffnessCVAEAgent` (Already Exists)

File:
- `factr/models/lowdim_action_transformer.py`

This model already:
- Takes `obs` shaped `(B, W, 27)`.
- Builds context tokens from 4 groups:
  - pose token from flattened `W x 9`
  - vel token from flattened `W x 6`
  - wrench token from flattened `W x 6`
  - tracking token from flattened `W x 6`
- Adds a stiffness embedding token.
- Runs a Transformer encoder over the context tokens.
- CVAE:
  - Prior MLP `p(z|context)` -> `(mu_p, logvar_p)`
  - Posterior MLP `q(z|context, actions)` -> `(mu_q, logvar_q)`
  - Reparameterization `z = mu_q + sigma_q * eps`
- Decoder:
  - Transformer decoder conditioned on context tokens
  - Injects `z` into every action query (biasing queries)
  - Predicts action chunk `(B, T, 9)`
- Loss:
  - masked L1 recon + `beta * KL(q||p)`

Concrete tasks:
- Confirm the `state_slices` match your 27D ordering:
  - pose slice `0:9`
  - vel slice `9:15`
  - wrench slice `15:21`
  - tracking slice `21:27`
- Confirm the actions are exactly 9D.
- Tune `obs_window`, `token_dim`, `d_z`, `beta`, `free_bits`.

### Option B (Match The Original Explanation): Make Posterior A Transformer Encoder

Rationale:
- Original design used a Transformer posterior because it conditions on high-dimensional ground-truth data and can learn richer relationships.

Concrete changes (conceptual):

1. Keep the low-dim **context tokenization** (pose/vel/wrench/track/stiffness tokens).
2. Implement a posterior module similar to `factr/models/action_transformer.py:PosteriorNet`:
   - Inputs:
     - `context_tokens` (detached) shaped `(B, N_ctx, D)`
     - action tokens shaped `(B, T, D)` from embedding the ground-truth actions
   - Prepend a learned `POST_CLS` token.
   - Run a Transformer encoder over `[POST_CLS, context_tokens, action_tokens]`.
   - Project the `POST_CLS` output to `(mu_q, logvar_q)`.
3. Keep the prior as an MLP over a pooled context summary (mean/max/cls token).

Notes:
- Detach the context tokens into the posterior if you want to preserve the CVAE training dynamics used in the original code (prevents posterior gradients from reshaping the observation embedding space in ways the prior cannot match).

---

## Phase 4: Hydra Config Changes (Wire Everything Together)

Files involved:
- `factr/cfg/agent/transformer_lowdim.yaml`
- `factr/cfg/task/single_franka_lowdim.yaml`
- `factr/cfg/train_bc_lowdim.yaml`

Concrete tasks:

1. Set training defaults to lowdim:
   - agent: `transformer_lowdim`
   - task: `single_franka_lowdim`
2. Set key sizes:
   - `task.obs_dim=27`
   - `task.ac_dim=9`
   - `obs_window=W`
   - `ac_chunk=T`
3. CVAE params:
   - `d_z` (start 16 or 32; increase if multi-modality is high)
   - `beta` (start small if KL dominates early)
   - optional `free_bits`
4. Stiffness:
   - `stiffness_classes=3`
   - ensure labels are passed into agent forward

---

## Phase 5: Training + Sanity Checks

### 5.1 Dataset sanity checks (must pass before training)

Write or run an inspection script that prints:
- `state` shape and ordering assumptions (check mean/std per group)
- action dim is 9
- label distribution `{1,2,3}`
- episode length distribution
- window sampling does not cross boundaries

### 5.2 One-batch forward/backward check

Acceptance criteria:
- `obs` batch is `(B, W, 27)`
- `actions` batch is `(B, T, 9)`
- `labels` is `(B,)` with values in `{1,2,3}`
- forward returns `{"total_loss", "l1_loss", "kl"}`
- loss finite (no NaNs/inf), gradients finite

### 5.3 Run training

Use the low-dim Hydra config:
- `factr/cfg/train_bc_lowdim.yaml`

Log:
- recon / KL
- posterior vs prior sampling behavior (optional: variance over time, by stiffness)

---

## Phase 6: Evaluation and Rollout Visualization

Files involved:
- `factr/task.py` (`BCTask.eval`)
- `scripts/test_rollout_lowdim_prior.py` (sampling visualization)
- `scripts/eval_one_episode_lowdim.py` (episode-level check)

Concrete tasks:
- Ensure `BCTask` calls `get_actions_prior(..., class_labels=labels)` (already the pattern).
- Plot results grouped by stiffness label.
- For diversity at branch points:
  - sample multiple `z` from the prior and overlay predicted trajectories (fan plot)

---

## Minimal Change Checklist (If You Want The Fastest Path)

If you want the fastest working result with the desired I/O:

1. Generate a buffer where `obs["state"]` is exactly 27D in the specified order and `obs["stiffness_label"]` exists.
2. Use:
   - `RobobufReplayBufferLowdim` for the dataset (windowed obs, action chunks, labels)
   - `LowdimStiffnessCVAEAgent` for the model (stiffness token + obs-window support)
3. Train via `train_bc_lowdim.yaml` and verify sampling diversity by stiffness.

