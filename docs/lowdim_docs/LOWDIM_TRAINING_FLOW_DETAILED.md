# FACTR Low-Dim BC/CVAE Training Flow (Detailed)

This document explains the **active FACTR pipeline** used in this repository: low-dimensional behavior cloning with a CVAE policy.
It is intended for readers with medium robotics/ML background who want to understand both the high-level flow and the concrete Python class/function boundaries.

---

## 1) Active entrypoint and config wiring

The training run starts at:

- `factr/train_bc_policy.py` (`train_bc`) as Hydra entrypoint.

Hydra resolves defaults from `factr/cfg/train_bc_lowdim.yaml`:

- `agent: transformer_lowdim` → `factr.models.lowdim_action_transformer.LowdimStiffnessCVAEAgent`
- `task: single_franka_lowdim` → `factr.task.BCTask`
- `trainer: adamw_cos_lowdim` → `factr.trainers.bc.BehaviorCloning`

So the control plane is:

1. `train_bc_policy.py` orchestrates run setup and loop.
2. `BehaviorCloning.training_step(...)` adapts batches to model-call format.
3. `LowdimStiffnessCVAEAgent.forward(...)` computes reconstruction + KL losses.
4. `BCTask.eval(...)` evaluates posterior/prior metrics and optional sampled rollouts.

This is the concrete object graph for low-dim action training (the currently active path).

---

## 2) Step-by-step runtime flow

## 2.1 Bootstrapping (`train_bc_policy.py`)

At startup, `train_bc` performs:

- seed fixing via `torch_fix_seed` (Python / Lightning / CuDNN deterministic mode),
- run directory creation and rollout config export,
- Hydra `instantiate(...)` of agent, trainer, and task,
- optional checkpoint restore and optional feature restore path handling,
- creation of an infinite-ish iterator over `task.train_loader`.

Then each iteration:

1. Pull batch from dataloader (and recycle iterator on `StopIteration`).
2. Optionally apply GPU image transform (usually irrelevant in pure low-dim mode because image dict is empty).
3. `trainer.optim.zero_grad()`.
4. `loss = trainer.training_step(batch, global_step)`.
5. `loss.backward()` and optimizer step.
6. Logging (`wandb` + terminal payload), scheduler stepping, eval calls, and periodic checkpointing.

The important insight: **the trainer mediates between task batch format and model interface**, while the model owns the actual objective.

---

## 2.2 Task and dataloading (`task.py` + `replay_buffer.py`)

`BCTask` builds two loaders:

- Train loader from `train_buffer` with iterable wrapper for random endless sampling.
- Test loader from `test_buffer` deterministic order for eval.

For low-dim training, both buffers are `RobobufReplayBufferLowdim`, which:

- loads robobuf transitions,
- reconstructs episode boundaries,
- builds observation windows with front-padding inside each episode,
- builds fixed-length action chunks with tail padding + loss mask,
- extracts a stiffness class label per episode,
- stores metadata (`episode_id`, `episode_step`, etc.) for plotting.

This is critical for alignment:

- **Observation tensor shape**: `(B, W, obs_dim)`
- **Action tensor shape**: `(B, T, ac_dim)`
- **Mask tensor shape**: `(B, T, ac_dim)`

The mask ensures padded timesteps do not dominate losses.

---

## 2.3 Training step adapter (`trainers/bc.py`)

`BehaviorCloning.training_step(...)` receives one batch from task:

- `(imgs, obs), actions, mask, labels`

Then it:

- moves tensors to device,
- flattens action/mask into `(B, T*D)` as `ac_flat`, `mask_flat`,
- calls model: `self.model(imgs, obs, ac_flat, mask_flat, class_labels=labels)`,
- expects dictionary with `total_loss`, and optional metrics (`l1_loss`, `kl`, entropies, std, etc.),
- logs all scalars.

This adapter pattern lets multiple policy architectures share one trainer contract as long as they return the same loss dictionary keys.

---

## 2.4 Model forward mechanics (`LowdimStiffnessCVAEAgent`)

### 2.4.1 Context construction

Input observation window `(B, W, 27)` is split into four semantic groups:

- pose: dims `[0:9]`
- velocity: dims `[9:15]`
- wrench: dims `[15:21]`
- tracking: dims `[21:27]`

Each group is flattened across window and encoded independently. A stiffness embedding token is added from class label. A synthetic CLS-like token is built from concatenated group encodings.

Final context token sequence has 6 tokens:

1. cls
2. pose
3. velocity
4. wrench
5. tracking
6. stiffness

Then a transformer encoder refines these context tokens.

### 2.4.2 Prior and posterior

- **Prior** `p(z|context)` is either fixed (standard normal / uniform categorical) or learned from context.
- **Posterior** `q(z|context, action_chunk)` is implemented by `_PosteriorTransformer`, which encodes a sequence containing:
  - posterior CLS token,
  - context tokens,
  - action tokens + positional embedding.

### 2.4.3 Latent distributions

Two latent families are supported:

- `gaussian` (reparameterized normal latent),
- `categorical` (Gumbel-softmax / straight-through one-hot latent, optionally projected to decoder latent width).

### 2.4.4 Decoder and output

Decoder uses learned action queries of length `ac_chunk`, with `z` injected as bias into query tokens, and cross-attends to context memory. Output is `(B, T, ac_dim)` action chunk.

### 2.4.5 Loss

- Reconstruction: masked L1 (`l1_loss`) and masked L2 metric.
- KL term: Gaussian KL or categorical KL with optional free-bits and KL balancing.
- Total: `total_loss = recon_l1 + beta * kl`.

This design keeps the latent variable meaningful while preserving rollout accuracy.

---

## 3) Evaluation loop behavior (`BCTask.eval`)

`BCTask.eval(...)` runs full test loader and computes:

- posterior L1 and KL from model forward,
- prior rollout L1/L2/sign error,
- prior/posterior entropy and std summaries,
- sample diversity over multiple latent samples,
- sweep score combining prior-L1 + penalties for low diversity/low KL.

If plotting is enabled, it also:

- samples multiple action hypotheses per state,
- reconstructs episode timeline via metadata,
- logs fan charts split by stiffness classes.

Hence eval is not only “single loss”; it is designed to monitor collapse modes (low KL / low diversity).

---

## 4) How key classes are related (mental dependency graph)

```text
train_bc_policy.py::train_bc
  ├─ instantiate(cfg.agent) -> LowdimStiffnessCVAEAgent
  ├─ instantiate(cfg.trainer) -> BehaviorCloning
  └─ instantiate(cfg.task) -> BCTask
        ├─ train_buffer/test_buffer -> RobobufReplayBufferLowdim
        └─ train_loader/test_loader

loop:
  batch <- BCTask.train_loader
  loss <- BehaviorCloning.training_step(batch)
            └─ LowdimStiffnessCVAEAgent.forward(...)
  eval <- BCTask.eval(trainer)
            └─ model.get_actions_prior / get_actions_base
```

If you hold this graph in your head, reading individual source files gets much easier.

---

## 5) How to inspect model structure quickly

Use one of these practical methods:

1. **Hydra config first**
   - read `cfg/agent/transformer_lowdim.yaml` to know constructor args.
2. **Class constructor second**
   - open `models/lowdim_action_transformer.py` and read `__init__` top-to-bottom.
3. **Forward pass third**
   - read `forward(...)` and helper calls in call order:
     - `_reshape_actions` → `_build_context_tokens` → `_build_z_context` → `_prior` / `posterior` → `_decode_actions`.
4. **Runtime print**
   - in a Python shell, instantiate with same config and print(model).
5. **Parameter inventory**
   - iterate `named_parameters()` and group by prefix (`pose_encoder`, `context_encoder`, `posterior`, `decoder`, etc.).

A robust pattern for medium-level debugging:

- verify shapes after each helper in forward,
- verify mask sum is non-zero,
- monitor KL + sample diversity together during training.

---

## 6) Practical “what to change where” guide

- Want different sequence lengths? Update `ac_chunk` / `obs_window` in config first, then validate replay-buffer alignment.
- Want different latent behavior? Start in `transformer_lowdim.yaml` (`latent_distribution`, `d_z`, `beta`, `free_bits`, `kl_balance_alpha`).
- Want better robustness? Tune `BCTask` eval diversity knobs and watch `eval/sample_diversity` + `eval/posterior_kl` together.

Prefer config edits before code edits whenever possible.

---

## 7) Summary

The active low-dim training pipeline is a clear three-layer system:

- **data/task layer** prepares aligned `(obs_window, action_chunk, mask, label)` samples,
- **trainer layer** standardizes optimization/logging contract,
- **CVAE model layer** learns conditional chunk prediction with a structured latent variable.

Understanding those boundaries is the shortest path to productive experimentation in FACTR.

---

## 8) Full tensor lifecycle walkthrough (with concrete shapes)

This section is intentionally explicit so you can mentally simulate one minibatch.

Assume low-dim default configuration:

- `batch_size = B`
- `obs_window = W = 8`
- `ac_chunk = T = 30`
- `obs_dim = 27`
- `ac_dim = 9`

### 8.1 From replay buffer to trainer

`RobobufReplayBufferLowdim.__getitem__` returns:

- `imgs`: `{}` in low-dim path (kept for interface compatibility)
- `obs_tensor`: `(W, 27)`
- `action_tensor`: `(T, 9)`
- `mask_tensor`: `(T, 9)`
- `label_tensor`: scalar stiffness class

The dataloader collates to:

- `obs`: `(B, W, 27)`
- `actions`: `(B, T, 9)`
- `mask`: `(B, T, 9)`
- `labels`: `(B,)`

### 8.2 Trainer flattening step

Inside `BehaviorCloning.training_step`:

- `ac_flat = actions.reshape(B, T*9)`
- `mask_flat = mask.reshape(B, T*9)`

These are passed to model forward for API compatibility with other policy types, then reshaped back inside low-dim model.

### 8.3 Lowdim model internal reshape and split

In `LowdimStiffnessCVAEAgent.forward`:

1. `_reshape_actions(ac_flat)` returns `(B, T, 9)`.
2. `_build_context_tokens(obs, class_labels)`:
   - split obs into 4 groups and flatten over `W`,
   - each group encoder outputs `(B, token_dim)`,
   - stack into 6 tokens `(B, 6, token_dim)` including cls and stiffness,
   - transformer encoder keeps `(B, 6, token_dim)`.

### 8.4 Prior/posterior parameterization

- `_build_z_context` makes context vector `(B, context_dim)` depending on mode.
- `_prior` returns either:
  - Gaussian: `mu/logvar` each `(B, z_dim)`
  - Categorical: logits `(B, n_var, n_cat)`
- `posterior(context, target_actions)` returns same family-shaped parameters.

### 8.5 Decoding

- latent sample `z` becomes `(B, d_z)` (after optional categorical projection),
- action queries are `(B, T, token_dim)`,
- decoder output is `(B, T, token_dim)`,
- `action_head` projects to `(B, T, 9)`.

### 8.6 Loss masking

Both L1 and L2 are computed elementwise over `(B, T, 9)` and multiplied by mask.
Padding timesteps at trajectory tail carry `mask=0`, so they do not contribute.

---

## 9) Config field deep explanation (what each knob really affects)

Below are the fields most people tune first and what they change in practice.

### 9.1 Structural/time knobs

- `obs_window`: how much recent history context tokens summarize.
- `ac_chunk`: decoding horizon; larger chunk gives longer open-loop prediction but harder optimization.

### 9.2 Latent knobs

- `latent_distribution`: choose Gaussian vs categorical latent family.
- `d_z`: latent bottleneck width used by decoder.
- `beta`: reconstruction-vs-latent regularization tradeoff.
- `free_bits`: optional KL floor; can prevent “KL to zero” but may hide true KL.
- `kl_balance_alpha`: asymmetric KL gradient balance between prior and posterior terms.

### 9.3 Capacity knobs

- `token_dim`: token width everywhere (encoders/decoder interfaces).
- `hidden_dim`: feedforward width in transformer/prior MLP paths.
- `encoder_layers`, `posterior_layers`, `decoder_layers`: depth allocation across context/posterior/decoder subproblems.

### 9.4 Sampling/diagnostics knobs in task

- `eval_diversity_num_samples`: more samples improve diversity estimate reliability.
- `sweep_target_min_diversity`: lower bound for diversity target in sweep score.
- `sweep_target_min_kl`: lower bound for KL target in sweep score.

When tuning, start with **one axis at a time** (e.g., `beta` only), otherwise interpretation becomes noisy.

---

## 10) Checkpoint artifacts and what they are for

During training startup, `train_bc_policy.py` creates a rollout folder inside the run dir and stores:

- `agent_config.yaml`: stripped inference-facing agent config,
- `exp_config.yaml`: full experiment config,
- optional copied `rollout_config.yaml` from buffer directory.

This is very useful for deployment reproducibility:

- training config provenance,
- inference config without accidental restore-path dependencies,
- easier script-level rollout reproducibility.

---

## 11) Detailed debugging playbook (symptom → likely cause)

### Symptom A: KL is almost zero from early training and stays flat

Likely causes:

- `beta` too high for current decoder capacity,
- prior/posterior too easy to match,
- decoder bypassing latent signal.

Actions:

- reduce `beta`,
- check sample diversity in eval,
- inspect whether prior/posterior entropy are both very high (uninformative latent).

### Symptom B: Prior L1 bad but posterior L1 good

Likely causes:

- posterior learns quickly from target actions,
- prior underfit to context.

Actions:

- increase prior/context capacity (`hidden_dim`, context mode),
- evaluate stiffness-label quality,
- verify train/test distribution mismatch in buffers.

### Symptom C: Action fan plots collapse to nearly one trajectory

Likely causes:

- latent collapse, categorical temperature too low/high mismatch,
- over-regularized prior.

Actions:

- inspect `eval/sample_diversity`,
- tune `categorical_temperature`, `beta`, and KL-balance.

### Symptom D: Sudden spikes or NaNs

Likely causes:

- corrupted data sample,
- invalid label ranges,
- aggressive learning rate for current depth.

Actions:

- run a batch-level shape/range sanity print,
- verify masks are non-zero,
- reduce LR and monitor gradient norm.

---

## 12) Recommended reading order for new contributors

If you are onboarding and want minimum confusion, read in this exact order:

1. `cfg/train_bc_lowdim.yaml`
2. `cfg/task/single_franka_lowdim.yaml`
3. `replay_buffer.py` (`RobobufReplayBufferLowdim` only)
4. `models/lowdim_action_transformer.py` (`__init__`, then `forward`, then inference methods)
5. `trainers/bc.py`
6. `task.py` (`BCTask.eval` metrics)
7. `train_bc_policy.py`

This order follows data first → model second → training orchestration last.

---

## 13) Direct code-reference index (jump table)

Use these symbol references to jump directly into the implementation while reading this doc.

### Training orchestration

- `factr/train_bc_policy.py::train_bc`
- `factr/train_bc_policy.py::torch_fix_seed`
- `factr/train_bc_policy.py::_grad_l2_norm`

### Training step contract

- `factr/trainers/bc.py::BehaviorCloning.training_step`
- `factr/trainers/base.py::BaseTrainer.log`
- `factr/trainers/base.py::BaseTrainer.consume_wandb_payload`

### Data construction and episode alignment

- `factr/replay_buffer.py::RobobufReplayBufferLowdim.__init__`
- `factr/replay_buffer.py::RobobufReplayBufferLowdim._build_episodes`
- `factr/replay_buffer.py::RobobufReplayBufferLowdim._append_episode_samples`
- `factr/replay_buffer.py::RobobufReplayBufferLowdim.__getitem__`
- `factr/replay_buffer.py::RobobufReplayBufferLowdim.get_sample_metadata`

### Eval pipeline

- `factr/task.py::BCTask.eval`
- `factr/task.py::BCTask._predict_actions`
- `factr/task.py::BCTask._sample_actions_for_plot`
- `factr/task.py::_compute_sample_diversity`
- `factr/task.py::_select_episode_plot_candidates`

### Low-dim CVAE model internals

- `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent.__init__`
- `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent.forward`
- `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent._build_context_tokens`
- `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent._build_z_context`
- `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent._prior`
- `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent._compute_kl`
- `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent._decode_actions`
- `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent.get_actions_prior`
- `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent.get_actions_pos`
- `factr/models/lowdim_action_transformer.py::_PosteriorTransformer.forward`

### Config wiring references

- `factr/cfg/train_bc_lowdim.yaml`
- `factr/cfg/agent/transformer_lowdim.yaml`
- `factr/cfg/task/single_franka_lowdim.yaml`
- `factr/cfg/trainer/adamw_cos_lowdim.yaml`

---

## 14) Trace-by-trace code reading recipe (with exact function order)

When you want to explain one gradient step from source code, follow this exact function chain:

1. `train_bc_policy.py::train_bc`
2. `BehaviorCloning.training_step`
3. `LowdimStiffnessCVAEAgent.forward`
4. `LowdimStiffnessCVAEAgent._build_context_tokens`
5. `LowdimStiffnessCVAEAgent._prior`
6. `_PosteriorTransformer.forward`
7. `LowdimStiffnessCVAEAgent._decode_actions`
8. `LowdimStiffnessCVAEAgent._compute_kl`
9. return to `BehaviorCloning.training_step` for logging
10. back to `train_bc_policy.py::train_bc` for optimizer/scheduler/eval/checkpoint

For eval-time prior sampling path, replace steps 3-8 with:

- `BCTask._predict_actions` → `LowdimStiffnessCVAEAgent.get_actions_prior` → `LowdimStiffnessCVAEAgent._sample_latent_batch` → `LowdimStiffnessCVAEAgent._decode_actions`.

This gives you a precise path for walkthroughs, code reviews, and debugging sessions.
