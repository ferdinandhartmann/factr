# Low-Dim BC (CVAE) Training + Eval Review (FACTR)

This note reviews the *current* low-dimensional behavior cloning (stiffness-conditioned CVAE) training/eval pipeline in `factr`,
and sanity-checks the YAML parameters used by the default run:

```bash
cd /home/ferdinand/activeinference/factr
python factr/train_bc_policy.py
```

Scope:
- Training entrypoint: `factr/train_bc_policy.py`
- Hydra config stack: `factr/cfg/train_bc_lowdim.yaml` and its defaults:
  - `factr/cfg/agent/transformer_lowdim.yaml`
  - `factr/cfg/task/single_franka_lowdim.yaml`
  - `factr/cfg/trainer/adamw_cos_lowdim.yaml`

This is *not* about the `world_model/` training configs.

---

## 1) What training actually does (code-accurate)

Training loop: `factr/train_bc_policy.py`
- Instantiates:
  - model/agent: `factr.models.lowdim_action_transformer.LowdimStiffnessCVAEAgent`
  - trainer: `factr.trainers.bc.BehaviorCloning`
  - task/eval: `factr.task.BCTask`
- Builds datasets:
  - `RobobufReplayBufferLowdim` (train + test) from `buf_train.pkl` / `buf_test.pkl`
- Builds dataloaders:
  - train loader uses `IterableWrapper` (samples *with replacement* forever)
  - test loader iterates deterministically over the test dataset (`drop_last=True`)
- Runs `max_iterations` optimizer steps:
  - forward: `loss_dict = model(...)`
  - backward: `loss = mean(loss_dict["total_loss"])`
  - `AdamW` step
  - cosine LR schedule step (every step if `schedule_freq=1`)

Important detail: **iteration count is not epoch count**.
Because the train loader samples with replacement, `max_iterations` is simply "number of optimizer updates".

---

## 2) Data + alignment (what one sample means)

Dataset class: `factr/replay_buffer.py::RobobufReplayBufferLowdim`

For each episode and each anchor time `t_idx`:
- Observation window:
  - shape `(obs_window, obs_dim)` = `(8, 27)`
  - left-pads the earliest state if `t_idx` is near the start of an episode
- Target action chunk:
  - shape `(ac_chunk, ac_dim)` = `(30, 9)`
  - starts at index `t_idx + action_index_offset`
  - pads beyond episode end by repeating the last available action
  - mask is `1` for valid steps and `0` for padded tail steps

Critical YAML knob:
- `action_index_offset` (currently `1` in `single_franka_lowdim.yaml`)
  - If this is wrong, you usually see a systematic "one-step lag" or consistently worse `prior_l1`.
  - If you ever suspect misalignment, this is the first parameter to re-check.

---

## 3) Losses (what each logged number means)

Model: `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent`

Per-batch losses:
- Reconstruction: masked L1 over the action chunk
- KL: `KL(q(z|context, actions) || p(z|context))` (diagonal Gaussians)
- Total: `total_loss = recon + beta * kl`

Your config makes two things easy to misread:

### 3.1 Free-bits forces a KL floor

YAML: `transformer_lowdim.yaml`
- `d_z: 32`
- `free_bits: 0.5`

Code clamps the KL per-sample from below by:
- `free_bits * d_z = 0.5 * 32 = 16.0`

So if the "true" KL wants to go below 16, the logged `kl` will sit around **~16** anyway.
That's exactly what your run shows.

Interpretation:
- A flat `train/kl ~ 16` is *expected* here, and does **not** necessarily mean "the model found a perfect KL equilibrium".
- To assess latent collapse in this setup, look at `prior_std_mean`, `posterior_std_mean`, and `prior_l1` vs `task_loss` more than raw `kl`.

### 3.2 Total loss has a hard lower bound

With free-bits and beta:
- `beta: 0.02`
- minimum KL term in total loss is `beta * 16.0 = 0.32`

So `train/total_loss` cannot go much below `~0.32` even if reconstruction becomes very small.
In this run, the more informative curve is `train/l1_loss`, not `train/total_loss`.

---

## 4) Evaluation (what happens every `eval_freq` steps)

Eval entry: `factr/task.py::BCTask.eval`

Every `eval_freq` steps the script does:
- sets model to eval mode
- loops through the *entire* test loader
- computes:
  - posterior reconstruction (`eval/task_loss` in W&B)
  - deterministic prior error (`eval/prior_l1`) using `get_actions_prior(..., sample=False)`
  - action L2 and sign-mismatch (`eval/action_l2`, `eval/action_lsig`)
  - posterior KL and prior/posterior std + entropy
- optionally logs fan plots (this is the expensive part)

The plot selection logic uses a *global timeline* across episodes:
- `eval_plot_max_steps` is a cap in "global time index", not "number of anchors"
- with `eval_plot_max_steps=400` and average episode length ~500, plots often come from the very first episode only
- that explains logs like `plot_label_counts: 1:16,2:0,3:0` even when the dataset has all labels

If you want fan plots to actually cover all stiffness classes, you need either:
- a larger `eval_plot_max_steps` (more episodes included), or
- a different sampling strategy (code change) that selects anchors per class/episode.

---

## 5) Dataset stats for `fourgoals_1_act` (current default paths)

Using the current default paths in `train_bc_lowdim.yaml`:
- Train buffer: `process_data/processed_data/fourgoals_1_act/buf_train.pkl`
- Test buffer: `process_data/processed_data/fourgoals_1_act/buf_test.pkl`

Computed from `RobobufReplayBufferLowdim`:
- Train samples: `55,546`
- Test samples: `6,404`
- Train episodes: `107` (min/mean/max samples per episode: `244 / 519 / 1241`)
- Test episodes: `12` (min/mean/max samples per episode: `414 / 534 / 755`)
- Train label counts: `{1: 16319, 2: 19951, 3: 19276}` (reasonably balanced)
- Test label counts: `{1: 3276, 2: 2206, 3: 922}` (label 3 is underrepresented)

What `max_iterations=6000` means here:
- "steps per epoch (approx)" = `55546 / 64 ~ 868`
- `6000` steps is about `6.9` passes worth of gradient updates (but sampling is with replacement).

---

## 6) Are the YAML parameters "good"?

They are internally consistent (shapes match the model) and your run looks stable.
The main caveat is that a few parameters strongly affect interpretation and runtime.

### 6.1 Training control (`train_bc_lowdim.yaml`)

- `batch_size: 64`
  - Good default for this dataset size and model size.
  - If you have GPU headroom, `128` can make gradients smoother; if you change it, watch if `l1_loss` slows (might need a small LR tweak).
- `lr: 3e-4` with AdamW + cosine schedule
  - Looks reasonable (loss decreases smoothly in your log).
  - LR decays quite a lot by step 3000; if prior metrics plateau early, consider a longer run or a less aggressive decay schedule.
- `max_iterations: 6000`
  - Fine for a first run (roughly ~7 passes worth of updates).
  - If your goal is *strong prior performance* (small `prior_l1` gap), you often need more steps than you think.
- `eval_freq: 1500`
  - Reasonable given eval includes plotting (expensive).
  - If you disable or reduce plotting cost, you can afford more frequent eval (e.g. 500) to catch regressions earlier.

### 6.2 Data/task (`single_franka_lowdim.yaml`)

- `obs_dim: 27`, `ac_dim: 9`, `obs_window: 8`, `ac_chunk: 30`
  - All consistent with the model's hard-coded expectations.
  - Horizon 30 is non-trivial; expect per-step error to grow with k (later chunk steps are harder).
- `action_index_offset: 1`
  - Plausible for "action stored on the *next* state" buffers; *must* be validated if metrics look suspicious.
  - If `prior_l1` is stubbornly bad while `task_loss` is good, test offset alignment.
- `eval_plot_max_steps: 400`, `eval_plot_prediction_stride: 25`, `eval_plot_num_samples: 10`
  - Good for keeping eval relatively light.
  - Not good for stiffness coverage in fan plots: it commonly samples only the earliest episode(s).

### 6.3 Model (`transformer_lowdim.yaml`)

- `token_dim: 128`, `hidden_dim: 256`, `encoder/decoder/posterior_layers: 2`, `nhead: 8`, `dropout: 0.1`
  - Sensible "small" transformer that should train quickly and not overfit instantly.
- `d_z: 32`
  - Reasonable latent capacity for multi-modality.
- `free_bits: 0.5` (with `d_z=32`)
  - Forces KL to be at least ~16; this prevents collapse but makes `kl` less informative.
  - Whether it's "good" depends on your goal:
    - If you care about a *strong prior* for sampling/planning, enforcing latent usage can help.
    - If you just want best deterministic prediction, it can be stronger than necessary.
- `beta: 0.02`
  - With free-bits, this makes the KL contribution roughly `~0.32` most of the time.
  - If `prior_l1` does not close the gap, increasing `beta` slightly can help prior match (at potential recon cost).

### 6.4 Optimizer/schedule (`adamw_cos_lowdim.yaml`)

- AdamW `weight_decay: 1e-4`
  - Good default.
- Cosine schedule `T_max = max_iterations`, `eta_min = 1e-6`
  - Fine for convergence, but it does "cool down" a lot by the second half of training.
  - If learning stalls early, consider a longer `max_iterations` rather than raising LR late.

---

## 7) What "good training" should look like (for this config)

Trends you want:
- `train/l1_loss` decreases quickly early, then gradually flattens.
- `eval/task_loss` decreases and tracks the train trend (no big generalization gap).
- `eval/prior_l1` decreases and stays close to `eval/task_loss`.
  - The size of the *prior-vs-posterior* gap is a key "is the prior usable?" signal.
- `prior_std_mean >= posterior_std_mean` most of the time.
- `train/grad_norm` stays bounded (no repeated blow-ups).

Trends you *should not* over-interpret here:
- `train/kl` hovering near ~16 is expected (free-bits clamp).
- `train/total_loss` plateauing around ~0.32 + small recon is expected.

---

## 8) Notes on your current run output (why it looks the way it does)

From the snippet you posted:
- `train/l1_loss` falls from ~0.20 (step 100) to ~0.036 (step 3000): good.
- `train/kl` sits near ~16: expected because `free_bits=0.5` and `d_z=32`.
- `train/lr` decays smoothly (cosine schedule): expected.
- Eval shows `Posterior L1` improving and `Prior L1` improving more slowly: normal, but watch the gap.
- `plot_label_counts` showing only one label is consistent with `eval_plot_max_steps=400` selecting from the earliest episode(s) only.

---

## 9) Concrete suggestions (what to change next)

These are prioritized to be low-risk and high signal.

### 9.1 Make eval plots actually cover all stiffness labels

Goal: avoid `plot_label_counts` showing only one label.

Change in `factr/cfg/task/single_franka_lowdim.yaml`:
- Increase `eval_plot_max_steps` so the global timeline includes multiple episodes.
- Increase `eval_plot_prediction_stride` to keep the number of anchors manageable.

Example starting point:
```yaml
eval_plot_max_steps: 6000
eval_plot_prediction_stride: 50
eval_plot_num_samples: 10
```

Why:
- Your test episodes are ~500 steps on average.
- `eval_plot_max_steps=400` often includes only the first episode(s), so plots do not represent the whole test distribution.

### 9.2 Train longer if you care about prior quality

If the goal is a usable prior sampler (small `prior_l1` gap):
- Increase `max_iterations` (often the simplest improvement).

Example:
```yaml
max_iterations: 20000
eval_freq: 2000
save_freq: 5000
```

Why:
- Prior matching typically lags posterior reconstruction.
- `6000` updates is fine for a first run, but not always enough for the prior to fully catch up.

### 9.3 Revisit `free_bits` if you want KL to be informative (or want stronger prior matching pressure)

Current:
- `d_z=32`, `free_bits=0.5` means KL is clamped to ~16.0 for loss/logging whenever the unclamped KL is lower.

Options:
- If you want the KL curve to reflect reality: set `free_bits: null` and watch for collapse.
- If you want a softer floor: try `free_bits: 0.25` (floor ~8.0) or `0.1` (floor ~3.2).

Why:
- With a high free-bits floor, the KL term can become close to constant and less useful for debugging.

### 9.4 Address test-set label imbalance (especially if you compare models)

Your current test label counts are skewed toward label 1 and against label 3.

Suggestions:
- Log eval metrics per label (prior L1 and action MSE per stiffness label).
- If you are regenerating buffers, try to keep test episodes per label roughly balanced.

Why:
- A single aggregate scalar can hide regressions on the rare label (label 3 here).

### 9.5 Sanity-check alignment if the prior gap refuses to close

If `eval/task_loss` looks good but `eval/prior_l1` stays much worse:
- Verify `action_index_offset` by trying `0` vs `1` (one run each) and comparing prior metrics.

Why:
- One-step misalignment often looks like "everything is kind of close but never good".

---

## 10) Should you use L2 instead of L1 for reconstruction?

Right now training uses masked L1 (`train/l1_loss`) and eval reports both:
- L1-style errors (`eval/task_loss`, `eval/prior_l1`)
- an L2-style error for prior predictions (`eval/action_l2`)

### When L2 (MSE) is better

L2 is often a better choice if:
- Your action noise is roughly Gaussian.
- You care a lot about reducing larger deviations (big mistakes should be punished strongly).
- Action dimensions are normalized to comparable scales (otherwise one dimension can dominate the loss).

### When L1 is better

L1 is often a better choice if:
- You have occasional outliers (sensor spikes, rare bad transitions, mislabeled stiffness).
- You want robustness and stable early training.
- You care about median absolute error more than squared error.

### A practical recommendation for this setup

If you are unsure, try these in order:
1. Keep L1 for training, but track how `eval/action_l2` behaves as you tune the model.
2. If L1 improves but `eval/action_l2` plateaus at an unsatisfying value, try switching reconstruction to L2.
3. If L2 becomes unstable or overly sensitive to rare spikes, use Huber (SmoothL1) as a compromise.

Important note:
- If you switch to L2, strongly consider normalizing action dimensions (or weighting dimensions) so translation/rotation components do not fight each other.
