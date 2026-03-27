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
