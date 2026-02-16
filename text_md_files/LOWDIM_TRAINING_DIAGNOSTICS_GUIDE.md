# Low-Dim CVAE Training Diagnostics Guide

This guide explains how to read the losses/plots for the low-dimensional stiffness-conditioned CVAE training setup and how key YAML parameters influence behavior.

Relevant configs:
- `factr/cfg/train_bc_lowdim.yaml`
- `factr/cfg/task/single_franka_lowdim.yaml`
- `factr/cfg/agent/transformer_lowdim.yaml`
- `factr/cfg/trainer/adamw_cos_lowdim.yaml`

---

## 1) What gets logged

### Train scalars
- `train/total_loss`
- `train/l1_loss`
- `train/kl`
- `train/prior_std_mean`
- `train/posterior_std_mean`
- `train/prior_entropy`
- `train/posterior_entropy`
- `train/grad_norm`
- `train/lr`

### Eval scalars
- `eval/task_loss` (posterior reconstruction L1)
- `eval/prior_l1` (prior rollout L1 with deterministic prior mean)
- `eval/posterior_kl`
- `eval/action_l2`
- `eval/action_lsig`
- `eval/prior_std_mean`
- `eval/posterior_std_mean`
- `eval/prior_entropy`
- `eval/posterior_entropy`
- `eval/joint{i}_l2` (per output dimension)
- `eval/chunk_step_{k}_mse` (error over prediction horizon)

### Eval images
- `eval/prediction_example`: single-step true vs predicted chunk plot.
- `eval/error_summary`: chunk-step MSE + per-dim MSE.
- `eval/prior_fan_all`: fan plot from sampled latent trajectories.
- `eval/prior_fan_stiffness_1`, `..._2`, `..._3`: fan plots split by stiffness label.

Fan plotting uses task config defaults:
- `eval_plot_max_steps: 300`
- `eval_plot_prediction_stride: 5`
- `eval_plot_num_samples: 10`

---

## 2) What “good” looks like

### Loss trends
- `train/l1_loss` and `eval/task_loss` decrease smoothly, then flatten.
- `eval/prior_l1` tracks close to `eval/task_loss` (small gap).
- `train/kl`/`eval/posterior_kl` stays positive and stable (not exactly zero, not exploding).
- `train/grad_norm` stays bounded (no repeated spikes to very large values).

Interpretation:
- Small prior/posterior L1 gap means prior can reproduce posterior-conditioned behavior well.
- Non-zero KL means latent `z` remains active.

### Entropy/std trends
- `posterior_std_mean` and `posterior_entropy` are usually lower than prior.
- `prior_std_mean` and `prior_entropy` should not collapse to near-zero everywhere.

Interpretation:
- Posterior is conditioned on GT actions and is expected to be sharper.
- Prior should keep some uncertainty for multi-modal futures (branching points).

### Error-by-horizon shape
- `eval/chunk_step_1_mse` is lowest.
- Later chunk steps usually increase gradually.
- Strong monotonic blow-up indicates unstable long-horizon dynamics.

### Fan plots (sampled prior trajectories)
- `eval/prior_fan_all`: sampled trajectories should be plausible and centered near ground truth trend.
- Near easy deterministic regions: fan should be tight.
- Near branching points: fan should widen.
- Stiffness-specific fan plots should show behavior differences by label, not identical patterns.

---

## 3) Warning patterns and likely causes

### A) KL collapses to ~0 very early
Symptoms:
- `train/kl -> 0` quickly.
- Prior/posterior entropies both very low.
- Fan plots overly narrow and deterministic.

Typical fixes:
- Increase `beta` gradually from a very small value to target.
- Set `free_bits` (for example `0.5` to `1.0`) to prevent trivial KL minimization.
- Increase `d_z` if task is clearly multi-modal.

### B) KL explodes or oscillates hard
Symptoms:
- `train/kl` unstable, `total_loss` noisy, training slows or diverges.

Typical fixes:
- Lower `beta`.
- Lower `lr`.
- Increase `batch_size` if memory allows.

### C) Prior worse than posterior by a large margin
Symptoms:
- `eval/prior_l1` much larger than `eval/task_loss`.
- Fan plots are broad but miss the GT manifold.

Typical fixes:
- Increase model capacity (`token_dim`, `hidden_dim`, layers).
- Increase training length.
- Check observation/action alignment (`action_index_offset`).
- Ensure train/test distribution match and labels are correct.

### D) Per-dim errors dominated by specific dimensions
Symptoms:
- One or two `eval/joint{i}_l2` much larger than others.

Typical fixes:
- Verify state/action ordering and normalization.
- Check if a dimension’s scale is inconsistent.

---

## 4) How YAML parameters influence behavior

## Data/Task params (`single_franka_lowdim.yaml`, `train_bc_lowdim.yaml`)
- `buffer_path`, `test_buffer_path`:
  - Most important for trustworthy eval.
  - Train/test must be distinct for unbiased metrics.
- `obs_window`:
  - Larger window gives more temporal context.
  - Too small: poor disambiguation; too large: harder optimization.
- `ac_chunk`:
  - Larger chunk increases long-horizon difficulty.
  - Often raises later-step MSE.
- `action_index_offset`:
  - Controls observation-to-action causality.
  - Wrong value causes systematic prediction lag/mismatch.
- `eval_plot_max_steps`, `eval_plot_prediction_stride`, `eval_plot_num_samples`:
  - Control coverage/detail of fan plots.
  - Larger values improve diagnostics but increase eval runtime.

## Model params (`transformer_lowdim.yaml`)
- `d_z`:
  - Latent capacity for multi-modality.
  - Too low: underfit branching; too high: harder KL balance.
- `beta`:
  - KL weight.
  - Higher beta enforces prior/posterior match, but can hurt recon if too high.
- `free_bits`:
  - Prevents KL collapse by enforcing minimum KL per latent dim.
- `token_dim`, `hidden_dim`:
  - Capacity of encoders/decoder.
  - Larger values fit complex patterns better but may overfit or slow training.
- `encoder_layers`, `posterior_layers`, `decoder_layers`, `nhead`:
  - Depth/attention capacity.
  - More layers can improve modeling but increase training cost and sensitivity.
- `z_context_mode`:
  - Which context summary the prior uses.
  - `cls_all_obs` is a robust default for mixed state features.

## Optimization params (`train_bc_lowdim.yaml`, `adamw_cos_lowdim.yaml`)
- `lr`:
  - Main stability/speed knob.
  - Too high: noisy or diverging KL/recon.
- `batch_size`:
  - Larger batch gives smoother gradients and usually steadier KL.
- `max_iterations`:
  - Must be long enough for prior to catch posterior behavior.
- Cosine schedule (`T_max`, `eta_min`):
  - Slower late-phase learning; helps smooth convergence.

---

## 5) Practical acceptance checklist

Use this quick checklist after a run:

1. `eval/task_loss` and `eval/prior_l1` both trend down and are close.
2. `eval/posterior_kl` remains > 0 and stable.
3. `eval/action_l2` decreases and plateaus.
4. `eval/chunk_step_k_mse` increases gradually with horizon (not exploding).
5. `eval/prior_fan_all` shows plausible spread; no random drift.
6. `eval/prior_fan_stiffness_{1,2,3}` show meaningful stiffness-conditioned differences.
7. No repeated spikes or blow-ups in `train/grad_norm`.

If all seven look reasonable, the model is usually training correctly and the latent stochastic planning behavior is meaningful.
