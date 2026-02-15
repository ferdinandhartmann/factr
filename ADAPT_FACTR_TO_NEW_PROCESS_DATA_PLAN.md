# FACTR Adaptation Plan: Stiffness-Conditioned Obs Window -> Commanded Pose Chunk

## Objective (updated)
- Input to the model: observation window + stiffness label (`1`, `2`, `3`).
- Output from the model: commanded pose chunk only (`/cartesian_impedance_controller/pose_command`, 9D per step), conditioned on the given stiffness label.
- Preserve current sampling behavior for multi-trajectory generation:
  - `get_actions_prior(..., num_samples=N)` -> `(B, N, ac_chunk, 9)`
  - `get_actions_pos(..., num_samples=N)` -> `(B, N, ac_chunk, 9)`

## Target Data Contract (from current `rollout_config.yaml`)
- State input remains low-dimensional:
  - ee pose (9), ee velocity (6), external wrench (6), tracking error (6) -> total `27`.
- Stiffness label input:
  - discrete class id in `{1,2,3}` (mapped to index `{0,1,2}` in code).
  - if missing in buffer, derive once per episode from metadata or from impedance values.
- Supervision target:
  - use only action slice `[0:9]` (commanded pose).
  - do not predict gains directly in this model.

## Hard Constraints from This Update
1. Do not add normalization/denormalization inside FACTR training/inference for this path.
2. Assume `buf.pkl` is already normalized and pass values through unchanged.
3. Keep `z` path compatible with current design where only selected context tokens feed prior/posterior.

## Phase 1: Dataset Validation + Stiffness Label Audit
1. Update inspection script to print:
   - obs keys, state/action dims, episode boundaries
   - availability/source of stiffness label per episode
2. Confirm label generation rule:
   - preferred: read explicit label if present
   - fallback: map from episode metadata (`soft/medium/stiff` -> `1/2/3`)
3. Verify target extraction:
   - commanded pose is exactly first 9 action dims for every step

## Phase 2: Replay Buffer Adapter (No Extra Normalization)
Planned file: `factr/factr/replay_buffer_lowdim.py` (or class in `factr/factr/replay_buffer.py`).

Requirements:
1. Model input sample:
   - `obs_window`: `(W, 27)` (or tokenized version)
   - `stiffness_label`: scalar class id
2. Target:
   - `pose_chunk`: `(H, 9)`
   - `loss_mask`: `(H, 9)`
3. Episode boundary safety:
   - no cross-episode windows/chunks
   - pad only within episode
4. No normalization code in this adapter (no mean/std, no grouped stats transforms).

## Phase 3: Model Architecture Update (Stiffness-Conditioned CVAE)
Planned file: `factr/factr/models/lowdim_action_transformer.py`.

Core behavior:
1. Encode observation context + stiffness label embedding.
2. Prior `p(z|context,label)` and posterior `q(z|context,label,pose_chunk)`.
3. Decode to pose chunk only `(H, 9)`.
4. Train with reconstruction + KL (free-bits optional).

## Token Design Decision (your question: flatten vs grouped tokens)
### Recommended default
- Use one token per low-dim group (not one giant flattened token):
  - `ee_pose_token`, `ee_vel_token`, `wrench_token`, `tracking_error_token`, plus `stiffness_token`.
- Add one `CLS` token for aggregation.
- Feed tokens into a small transformer encoder, then use:
  - `CLS` as global summary
  - selected tokens for `z` networks

Why this is best here:
1. Better structure than flattening everything early.
2. Very low compute overhead (only a few tokens, unlike image tokens).
3. Easier ablation and interpretability for which modality informs `z`.

### `z` conditioning plan (explicit)
- Baseline (current style): `z` uses `[CLS, wrench_token]`.
- Ablation A: `z` uses `[CLS, wrench, ee_pose, ee_vel, tracking_error, stiffness]`.
- Ablation B: learned attention pooling over all context tokens to produce a compact `z`-context vector.
- Choose final by validation metrics (below). This is likely better than flattening raw tokens before attention.

## Phase 4: Hydra Config Changes
1. `factr/factr/cfg/task/single_franka_lowdim.yaml`
   - `obs_dim: 27`
   - `ac_dim: 9` (pose only)
   - include stiffness label handling in dataset config
2. `factr/factr/cfg/agent/transformer_lowdim.yaml`
   - add stiffness embedding config (`num_classes=3`, embed dim)
   - add `z_context_mode` (`cls_force`, `all_tokens`, `attn_pool`)
3. `factr/factr/cfg/train_bc_lowdim.yaml`
   - point to new buffer path
   - keep existing seed/logging style

Suggested start:
- `obs_window=8`, `ac_chunk=30`, `batch_size=64`, `num_samples=10`

## Phase 5: Rollout Script for Multi-Trajectory Sampling
Planned file: `factr/scripts/test_rollout_lowdim_prior.py`.

Behavior:
1. Build rolling observation window.
2. Provide stiffness label (`1/2/3`) at each query.
3. Sample pose chunks:
   - `samples = model.get_actions_prior(obs_window, stiffness_label, num_samples=N)`
   - output `(1, N, H, 9)`
4. Produce multiple candidate trajectories from the sampled chunks.
5. Execute/score with receding horizon using first action per chunk.

## Phase 6: Validation and Selection Criteria
1. Shape checks:
   - input window `(B,W,27)`, label `(B,)`, output `(B,N,H,9)`
2. Training checks:
   - stable recon + KL, no NaN/Inf
3. Sampling checks:
   - diversity across samples
   - consistency with label-conditioned behavior
4. Ablation decision for `z` context:
   - compare `cls_force` vs `all_tokens` vs `attn_pool`
   - pick best by validation pose error + rollout behavior

## Risks and Mitigations
- Risk: stiffness label quality is noisy or inconsistent.
  - Mitigation: fix deterministic label mapping at dataset build time and log distribution.
- Risk: removing gain prediction may reduce controllability in downstream controller.
  - Mitigation: keep stiffness label fixed externally per rollout and evaluate closed-loop behavior.
- Risk: `z` underuses non-force signals with `CLS+force` only.
  - Mitigation: run the planned ablation and move to `all_tokens` or `attn_pool` if it improves rollout error.

## Execution Order Summary
1. Validate buffer + stiffness label source.
2. Implement low-dim adapter (obs window + label -> pose chunk target) with no normalization step.
3. Implement stiffness-conditioned low-dim CVAE transformer.
4. Add configs (`ac_dim=9`) and train smoke test.
5. Add rollout sampler for multi-trajectory generation.
6. Run `z` token ablations and lock final context design.
