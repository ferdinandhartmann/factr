# Low-Dim ACT/CVAE Training Flow (Detailed)

This document explains exactly how low-dimensional training works in this repo, from buffer samples to loss to inference calls.

> Naming note: the active low-dim model in this repository is implemented as a CVAE-style transformer policy (`LowdimStiffnessCVAEAgent`) rather than the original image-token ACT pipeline. In practice, your training flow is “low-dim ACT-like decoder + CVAE latent”.

---

## 1) Config Composition: What actually gets instantiated

Using `train_bc_lowdim` composes:

- **Agent**: `factr.models.lowdim_action_transformer.LowdimStiffnessCVAEAgent`
- **Task**: `factr.task.BCTask`
- **Trainer**: `factr.trainers.bc.BehaviorCloning`
- **Optimizer/Scheduler**: AdamW + cosine scheduler (from diffusers helper)

Key default hyperparameters:
- `obs_dim=27`
- `ac_dim=9`
- `obs_window=8`
- `ac_chunk=30`
- `d_z=32`
- `beta=0.01`

---

## 2) Data semantics for each sample (critical)

`RobobufReplayBufferLowdim` constructs each sample as:

- `obs_tensor`: shape `(obs_window, obs_dim)` = `(8, 27)`
- `action_tensor`: shape `(ac_chunk, ac_dim)` = `(30, 9)`
- `mask_tensor`: shape `(30, 9)` with 1 for valid and 0 for padded horizons
- `label_tensor`: integer stiffness label (internally 1..K then normalized in model)

### Observation grouping expected by model
The model assumes `obs_dim=27` and splits each timestep into:
- pose: 9 dims
- velocity: 6 dims
- wrench/force: 6 dims
- tracking: 6 dims

This grouped structure is hard-checked by assertions in model init/prepare.

---

## 3) Training loop flow (step-by-step)

For each global step in `train_bc_policy.py`:

1. Read a batch from `task.train_loader`.
2. `BehaviorCloning.training_step` moves tensors to device.
3. Actions and masks are flattened to `(B, ac_chunk * ac_dim)` before model call.
4. Model reshapes back to `(B, ac_chunk, ac_dim)` internally.
5. Model computes `total_loss = recon_l1 + beta * KL`.
6. Backprop + optimizer step.
7. Periodic scheduler step, eval, and checkpoint save.

---

## 4) Inside `LowdimStiffnessCVAEAgent`

## 4.1 Context token construction

Given `obs` with shape `(B, 8, 27)`:

1. Slice into 4 groups (pose, vel, wrench, tracking).
2. Flatten each group over time and pass through per-group encoders:
   - `pose_encoder`, `vel_encoder`, `wrench_encoder`, `track_encoder`.
3. Convert stiffness class index using `nn.Embedding`.
4. Build a learned CLS-like token from concatenated 4 encoded groups (`cls_from_obs`).
5. Stack tokens in this order:
   - `[cls, pose, vel, wrench, track, stiffness]` (6 tokens total).
6. Add learnable positional token parameters.
7. Pass through transformer encoder + layer norm.

This produces context tokens used by both prior/posterior and decoder.

## 4.2 Latent context selection (`z_context_mode`)

A compact vector is created from context tokens for latent inference.
Modes:
- `cls_all_obs` (default): concat `[cls, pose, vel, wrench, track]`
- `cls_force`: concat `[cls, wrench]`
- `all_tokens`: flatten all 6 tokens
- `attn_pool`: attention-pool all tokens then concat with cls

## 4.3 Prior and Posterior

- Prior network computes `mu_p, logvar_p = p(z|context)`.
- Posterior network computes `mu_q, logvar_q = q(z|context, target_actions)`.
  - target actions are flattened then encoded before fusion.

During training, latent sampling uses posterior reparameterization:
- `z = mu_q + exp(0.5 * logvar_q) * eps`

## 4.4 Action decoding

1. Latent `z` is projected to token space (`z_to_token`).
2. Learned action query embeddings of length `ac_chunk` are repeated per batch.
3. Each query receives latent bias (inject z into every action query).
4. Transformer decoder cross-attends from queries to context memory.
5. Linear `action_head` outputs 9-dim pose command per horizon step.

Final prediction shape: `(B, 30, 9)`.

## 4.5 Losses

- **Reconstruction loss**: masked L1 between predicted and target action chunks.
- **KL divergence**: diagonal Gaussian KL from posterior to prior.
- Optional **free bits** floor on KL.

Total:

`total_loss = recon_l1 + beta * kl`

---

## 5) Eval-time behavior

`BCTask.eval` computes both posterior and prior diagnostics:

- Calls forward pass (posterior path) for `l1_loss` and `kl`.
- Calls prediction path (usually prior mean for deterministic eval) for action metrics.
- Logs:
  - prior/posterior L1,
  - action L2,
  - sign mismatch,
  - per-dim pose MSE,
  - per-chunk-step MSE,
  - sampled trajectory fan plots (10 prior samples) by stiffness class.

---

## 6) Inference APIs and when to use each

`LowdimStiffnessCVAEAgent` provides several APIs:

- `get_actions_base(...)`: prior mean decode (deterministic baseline output).
- `get_actions_prior(..., sample=True/False, num_samples=N)`: sample or mean from prior.
- `get_actions_pos(...)`: sample from posterior using provided target action (diagnostics/analysis).
- `get_uncertainty_entropy(...)`: returns sampled-action-based uncertainty summary (std over samples).

For deployment-like rollout, use `get_actions_prior` with either deterministic (`sample=False`) or stochastic (`sample=True`) behavior.

---

## 7) Common pitfalls and debugging checklist

1. **Shape mismatch errors**
   - Confirm buffer truly contains state dim 27 and pose action dim 9.
2. **Label mismatch / class shift**
   - Check whether labels are 1-based or 0-based; model normalizes but malformed labels still hurt.
3. **Poor prior rollouts with good posterior loss**
   - Indicates prior network underfitting latent mapping; inspect KL scale (`beta`) and context mode.
4. **Unstable training**
   - Verify action normalization assumptions in preprocessing and rollout config.
5. **Noisy fan plots**
   - Increase samples and inspect uncertainty function; check if this is expected multimodality.

