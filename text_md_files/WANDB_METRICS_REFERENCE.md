# W&B Metrics Reference (FACTR)

This document explains every metric that can show up in Weights & Biases (W&B) logs for the FACTR BC/CVAE training pipeline.

Primary scope (active path):
- Entrypoint: `factr/factr/train_bc_policy.py`
- Trainer: `factr/factr/trainers/bc.py::BehaviorCloning`
- Task/eval: `factr/factr/task.py::BCTask`
- Low-dim agent: `factr/factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent`

Notes:
- **Step axis** in W&B is `global_step` (optimizer update count), not epochs.
- Most `train/*` scalars are logged via `BaseTrainer.log(...)` with a **running mean window of 100 steps** and emitted every 100 steps. See `factr/factr/trainers/base.py`.
- Some scalars (notably `train/grad_norm` and `train/scale`) are logged directly with `wandb.log(...)` and are **not** running means.

---

## 1) Train Scalars (`train/*`)

### `train/total_loss`
What it is:
- The scalar the optimizer minimizes.

Low-dim CVAE definition:
- `total_loss = l1_loss + beta * kl`
- Computed in `LowdimStiffnessCVAEAgent.forward(...)`.

How to read it:
- Should decrease early and then plateau.
- Absolute value depends heavily on `beta`, `free_bits`, and action scaling/normalization.

Common pitfall:
- If `free_bits` clamps KL, `total_loss` can be dominated by a nearly constant `beta * kl_floor`.

### `train/l1_loss`
What it is:
- Masked L1 reconstruction error over the action chunk (mean absolute error per action element).

Definition:
- `abs(pred - target)`, masked, summed, divided by number of valid (mask==1) elements.

How to read it:
- This is the main “behavior cloning accuracy” term during training.
- It should steadily drop and then flatten.

### `train/l2_loss`
What it is:
- Masked L2 reconstruction error over the action chunk (mean squared error per action element).

How to read it:
- Useful as a secondary metric even if training optimizes L1.
- If it plateaus high while L1 improves, you may have occasional larger errors (outliers).

### `train/kl`
What it is:
- KL divergence between posterior and prior latents:
  - `KL(q(z | context, actions) || p(z | context))`

Where:
- Low-dim: `LowdimStiffnessCVAEAgent.forward(...)`.
- Image-based CVAE transformer: `factr/factr/models/action_transformer.py` (if you train that agent).

How to read it (important):
- If `free_bits` is enabled, KL is clamped:
  - `kl := max(kl, free_bits * d_z)`
  - This can make `train/kl` stick near a constant floor and **hide** the true KL.
- If KL goes to ~0 with `free_bits: null`, it often means the latent is being ignored (posterior collapse), especially if sampling diversity disappears.

### `train/prior_std_mean` (low-dim CVAE only)
What it is:
- Mean of the prior standard deviation across batch and latent dims:
  - `mean(exp(0.5 * logvar_p))`

How to read it:
- Very small values can indicate the prior is becoming overly confident (low diversity).
- Very large values can indicate the model is inflating variance (can happen if KL pressure is mis-tuned).

### `train/posterior_std_mean` (low-dim CVAE only)
What it is:
- Mean of the posterior standard deviation:
  - `mean(exp(0.5 * logvar_q))`

How to read it:
- Posterior often becomes sharper than prior (lower std) because it sees ground-truth actions.
- If posterior and prior std become nearly identical and KL becomes tiny, you may be in a collapsed-latent regime.

### `train/prior_entropy` (low-dim CVAE only)
What it is:
- Differential entropy (in nats) of the diagonal-Gaussian prior, averaged over the batch.

Implementation detail:
- Computed from `logvar_p` using the closed-form Gaussian entropy.

How to read it:
- Tracks similar information to `prior_std_mean`, but aggregated across all latent dims.

### `train/posterior_entropy` (low-dim CVAE only)
What it is:
- Differential entropy of the diagonal-Gaussian posterior, averaged over the batch.

How to read it:
- Often lower than the prior (posterior is more certain because it sees actions).

### `train/lr`
What it is:
- Learning rate of the optimizer (first param group).

Where:
- Logged by `BehaviorCloning.training_step(...)` from `trainer.lr`.

How to read it:
- With cosine scheduling, it should smoothly decay from `lr` to `eta_min`.

### `train/grad_norm`
What it is:
- Global L2 norm of gradients across model parameters for the current step.

Where:
- Computed and logged in `factr/factr/train_bc_policy.py`.

How to read it:
- Occasional spikes can be normal.
- Repeated large spikes or blow-ups often correlate with instability (bad LR, exploding activations, NaNs).

### `train/scale` (vision curriculum only)
What it is:
- Curriculum “scale” value used for blur/downsampling when `BaseAgent.tokenize_obs(...)` applies curriculum.

Where:
- Logged in `factr/factr/agent.py`.

How to read it:
- Only appears for agents using `BaseAgent` with curriculum enabled (not the low-dim-only agent).

---

## 2) Eval Scalars (`eval/*`)

Eval is run in `BCTask.eval(...)` (`factr/factr/task.py`) every `eval_freq` steps.

### `eval/task_loss`
What it is:
- Mean **posterior** reconstruction L1 over the test set.

In the console printout this corresponds to:
- `Posterior L1: ...`

Notes:
- This uses the model forward path (posterior-conditioned during training-style forward), not the prior sampler.

### `eval/prior_l1`
What it is:
- Mean L1 error when predicting actions using the **prior mean** (deterministic):
  - `get_actions_prior(..., sample=False, num_samples=1)`

In the console printout this corresponds to:
- `Prior L1: ...`

How to read it:
- This is the most important number if your rollout uses the prior.
- Goal: `eval/prior_l1` close to `eval/task_loss` (small prior-vs-posterior gap).

### `eval/posterior_kl`
What it is:
- Mean KL reported by the model forward pass on the test set.

How to read it:
- Interpret with the same `free_bits` caveat as `train/kl`.

### `eval/action_l2`
What it is:
- Mean squared error per action element (masked), computed from **prior predictions**.

How to read it:
- A stricter penalty on larger errors than L1.
- If you care about squared error, watch this alongside `eval/prior_l1`.

### `eval/action_lsig`
What it is:
- Fraction of action elements whose sign differs from ground truth (masked), using prior predictions.

How to read it:
- Useful when sign is meaningful (e.g., directionality).
- Lower is better; values are in `[0, 1]`.

### `eval/prior_std_mean`, `eval/posterior_std_mean`
What they are:
- Same as the corresponding train metrics, but averaged over the eval batches.

How to read them:
- Helpful for diagnosing diversity collapse (std too small) or variance inflation (std too large).

### `eval/prior_entropy`, `eval/posterior_entropy`
What they are:
- Same as train entropies, averaged over eval batches.

### `eval/classification_accuracy` (only if present)
What it is:
- If the model returns logits and the task computes accuracy, the mean accuracy is logged here.

When it appears:
- Only if `accuracy_list` is populated in `BCTask.eval(...)` (model must provide classification outputs).

---

## 3) Eval Images (`eval/*` images)

These are logged by `BCTask.eval(...)` using `wandb.Image(...)`.

### `eval/prior_fan_stiffness_1`, `eval/prior_fan_stiffness_2`, `eval/prior_fan_stiffness_3`
What they are:
- “Fan” plots: many sampled prior trajectories (10 by default) compared against ground truth.

How to read them:
- Tight fan: model predicts confidently / deterministically.
- Wide fan: model expresses uncertainty / multi-modality.
- For good behavior, the fan should be plausible and centered around the ground-truth trend.

Why you might see “missing stiffness” images:
- If eval plot selection didn’t include any anchors for a given stiffness label within the configured global timeline.

Controls:
- `eval_plot_max_steps`, `eval_plot_prediction_stride`, `eval_plot_num_samples` in `factr/factr/cfg/task/single_franka_lowdim.yaml`.

---

## 4) Metrics That Exist in Code But May Be Disabled

Some W&B logs in `BCTask.eval(...)` are currently commented out (or gated).
If enabled, you may also see:
- `eval/joint{i}_l2`: per-action-dimension MSE summary.
- `eval/chunk_step_{k}_mse`: MSE by prediction horizon step.
- `eval/prediction_example`: a single example plot (true vs predicted).
- `eval/error_summary`: summary plot (error by horizon + per-dim).
- `eval/prior_fan_all`: fan plot without stiffness filtering.

---

## 5) Quick “What Matters” Checklist (Prior Rollouts)

If rollouts sample from the prior:
- Primary: `eval/prior_l1` (should be close to `eval/task_loss`).
- Secondary: `eval/action_l2` and the `eval/prior_fan_stiffness_*` plots.
- Diagnostics: `eval/posterior_kl`, `eval/prior_std_mean`, `eval/prior_entropy` to detect latent collapse or variance hacks.

