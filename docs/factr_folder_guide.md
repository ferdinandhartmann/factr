# FACTR Folder Guide (Architecture + Responsibilities)

This document explains the `factr/` folder as a **system**, not just file-by-file. It is intended for readers with medium robotics/ML background who want to quickly understand where data enters, how training happens, and which classes are important.

---

## 1) Top-Level Mental Model

At a high level, the `factr/` package is organized into five layers:

1. **Entry point / orchestration**: starts a job, parses Hydra config, builds modules, runs loop.
2. **Data layer**: loads buffers and creates training/eval samples.
3. **Model layer**: policy networks (vision transformer ACT/CVAE variants, low-dim CVAE variant).
4. **Trainer layer**: optimizer/scheduler handling + one-step loss computation.
5. **Task/eval layer**: validation loops, metrics, plots, and W&B logging.

The canonical training entrypoint is `factr/train_bc_policy.py`.

---

## 2) Folder and File Responsibilities

## `factr/train_bc_policy.py` (very important)

**Role**: Orchestrates behavior-cloning training.

**Key responsibilities**:
- Initializes run directory and W&B resume/new run state via `misc.init_job`.
- Seeds randomness (`torch_fix_seed`).
- Writes rollout-time configs (`rollout/agent_config.yaml`, `exp_config.yaml`, `rollout_config.yaml`) for downstream deployment/evaluation tooling.
- Instantiates `agent`, `trainer`, and `task` via Hydra.
- Iterates the training loop:
  - fetch batch,
  - optional GPU transform,
  - `training_step`,
  - backward + optimizer step,
  - scheduler step,
  - periodic eval and checkpointing.

If you want to change global training behavior (save/eval cadence, startup loading, batch path), this is the first file to inspect.

---

## `factr/misc.py` (important)

**Role**: training-job utilities and global state.

**Key items**:
- Hydra resolvers (`len`, `add`, `mult`, `transform`, etc.).
- `GLOBAL_STEP`: globally shared step used by curriculum scheduling.
- `init_job(cfg)`: creates or resumes W&B run and stores experiment metadata.
- Requeue/signal checkpoint helper (`set_checkpoint_handler`).

If scheduler behavior or run-resume behavior seems strange, inspect here.

---

## `factr/replay_buffer.py` (very important)

**Role**: converts stored trajectory buffer into supervised samples `(obs, action_chunk, mask, label)`.

Contains multiple dataset variants:

- `ReplayBuffer`: simple generic in-memory format.
- `RobobufReplayBuffer`: image-based robobuf loader.
- `RobobufReplayBufferLowdim`: low-dimensional stiffness-conditioned loader (used by low-dim ACT/CVAE training).

### Why `RobobufReplayBufferLowdim` matters

It performs the dataset semantics for low-dim training:
- Splits transition list into episodes.
- Builds observation windows (`obs_window`, usually 8).
- Extracts only pose action part (`pose_action_dim`, usually 9).
- Builds fixed-length future chunks (`ac_chunk`, usually 30) with padding mask.
- Infers/normalizes stiffness labels.

This class effectively defines “what one training sample means.”

---

## `factr/task.py` (very important)

**Role**: dataloaders + evaluation protocol.

Contains:
- `_build_data_loader`: wraps dataset into train/eval DataLoader.
- `DefaultTask`: basic validation loop.
- `BCTask`: richer BC eval with detailed metrics/plots.

`BCTask.eval` computes and logs:
- prior L1 and posterior L1,
- KL (if model returns it),
- action L2,
- sign mismatch metric (`lsig`),
- per-dimension pose MSE,
- chunk-step MSE,
- sampled trajectory fan plots per stiffness label.

If your model trains but “looks wrong” in rollout quality, this is the place to inspect metrics and plotting logic.

---

## `factr/trainers/` (important)

- `base.py`: generic trainer machinery (device placement, optimizer/scheduler creation, checkpoint save/load, logging wrappers).
- `bc.py`: BC-specific `training_step`, including moving tensors to device and invoking model forward.
- `utils.py`: config-driven optimizer and scheduler constructors.

Most optimization-policy behavior is controlled by this layer.

---

## `factr/models/` (very important)

Main policy implementations and building blocks live here:

- `lowdim_action_transformer.py`: low-dimensional stiffness-conditioned CVAE policy (`LowdimStiffnessCVAEAgent`) used in low-dim setup.
- `action_transformer.py`: ACT-like transformer/CVAE architecture for higher-dimensional and/or visual-token settings.
- `action_distributions.py`: action distribution parameterization helpers (Gaussian / mixture style setups).
- `vit.py`: visual encoder wrappers.
- `base.py`, `classification.py`: shared building blocks.

For lowdim use-cases, the key model class is `LowdimStiffnessCVAEAgent`.

---

## `factr/cfg/` (very important)

Hydra config tree controlling training composition:
- `train_bc_lowdim.yaml`: low-dim experiment root config.
- `agent/transformer_lowdim.yaml`: model hyperparameters.
- `task/single_franka_lowdim.yaml`: dataset + eval settings.
- `trainer/adamw_cos_lowdim.yaml`: optimizer/scheduler selection.

Think of `cfg` as the “wiring diagram” for which classes get instantiated.

---

## 3) Important vs Medium-Important Classes

## Critical (understand these first)

- `LowdimStiffnessCVAEAgent` (`factr/models/lowdim_action_transformer.py`)
- `RobobufReplayBufferLowdim` (`factr/replay_buffer.py`)
- `BCTask` (`factr/task.py`)
- `BehaviorCloning` (`factr/trainers/bc.py`)
- `BaseTrainer` (`factr/trainers/base.py`)
- training entrypoint in `factr/train_bc_policy.py`

## Medium-important (frequently touched when debugging/extending)

- `DefaultTask` and plotting helper functions (`factr/task.py`)
- `optim_builder` / `schedule_builder` (`factr/trainers/utils.py`)
- `misc.init_job` and Hydra resolvers (`factr/misc.py`)
- `BaseAgent` tokenization logic (mainly for visual pipelines, in `factr/agent.py`)

---

## 4) End-to-End Dependency Graph (Conceptual)

1. `train_bc_policy.py` reads config.
2. Hydra instantiates:
   - model (`LowdimStiffnessCVAEAgent`),
   - trainer (`BehaviorCloning`),
   - task (`BCTask` with lowdim buffers).
3. `RobobufReplayBufferLowdim` emits training tuples.
4. `BehaviorCloning.training_step` flattens action/mask and calls model.
5. Model computes CVAE losses (recon + KL).
6. `BaseTrainer` handles optimization/checkpoint.
7. `BCTask.eval` runs posterior/prior metrics and logs plots.

---

## 5) Practical Extension Points

- Change observation/action semantics: `RobobufReplayBufferLowdim` extraction methods.
- Change latent/context architecture: `LowdimStiffnessCVAEAgent` internals.
- Change evaluation metrics: `BCTask.eval`.
- Change optimizer/schedule: trainer config + `trainers/utils.py`.
- Change experiment composition: Hydra config files in `factr/cfg`.

