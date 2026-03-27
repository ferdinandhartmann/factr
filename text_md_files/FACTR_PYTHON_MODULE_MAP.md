# FACTR Python Module Map (Focused on Active Low-Dim Pipeline)

This file gives a practical map of the `factr/` Python package so you can quickly find where responsibilities live.

---

## 1) Top-level training scripts

- `factr/train_bc_policy.py`
  - Main low-dim BC/CVAE training entrypoint.
  - Creates agent/trainer/task via Hydra and runs training loop.

- `factr/train_obs_model.py`
  - Observation prediction training entrypoint for low-dim Gaussian MLP path.

---

## 2) Task modules (data + eval behavior)

- `factr/task.py`
  - `DefaultTask`: common loader/eval base.
  - `BCTask`: low-dim action policy evaluation, prior/posterior metrics, diversity metrics, fan plots.

- `factr/task_obs_pred.py`
  - `ObsPredictionTask`: evaluation for obs prediction and goal inference diagnostics.

In practice, task classes own dataloader creation and eval-time analytics.

---

## 3) Replay buffers and sample construction

- `factr/replay_buffer.py`
  - `RobobufReplayBufferLowdim`: action-policy samples `(obs_window, action_chunk, mask, stiffness_label)`.
  - `RobobufReplayBufferObsPredLowdim`: obs-pred samples `(obs_window, action_chunk, target_obs, target_mask, stiffness, goal)`.
  - Older image-centric replay classes also exist (`RobobufReplayBuffer`, etc.).

This module is where episode alignment and padding policy are enforced.

---

## 4) Model modules

- `factr/models/lowdim_action_transformer.py`
  - `LowdimStiffnessCVAEAgent` (active low-dim BC/CVAE policy).

- `factr/models/lowdim_obs_mlp.py`
  - `LowdimGaussianObsMLP` (obs prediction + goal posterior logic).

- `factr/models/action_transformer.py`
  - `TransformerAgent` + ACT/CVAE components for image/token based policy path.

- `factr/models/vit.py`
  - Vision transformer backbones for image-feature pipelines.

- `factr/models/action_distributions.py`
  - Distribution heads for policy outputs (deterministic, Gaussian, GMM).

- `factr/models/classification.py`
  - Classification helper head.

- `factr/models/base.py`
  - Minimal model base utilities.

---

## 5) Agent abstraction layer

- `factr/agent.py`
  - `BaseAgent`: multimodal tokenization framework for image + obs streams.
  - `MLPAgent`: baseline policy using flattened tokens + MLP.

Even in low-dim-only experiments, this file is useful context for legacy and image-based variants.

---

## 6) Trainer modules

- `factr/trainers/base.py`
  - `BaseTrainer` and running metric helpers.

- `factr/trainers/bc.py`
  - `BehaviorCloning`: action-policy training-step adapter and logger.

- `factr/trainers/obs_pred.py`
  - `GaussianObsPredictionTrainer`: obs-pred training-step adapter.

- `factr/trainers/utils.py`
  - optimizer and LR scheduler builders.

---

## 7) Config modules (`factr/cfg/...`)

Hydra configs provide the wiring between classes:

- `train_bc_lowdim.yaml` + `agent/transformer_lowdim.yaml` + `task/single_franka_lowdim.yaml` + `trainer/adamw_cos_lowdim.yaml`
- `train_obs_pred_lowdim.yaml` + `agent/obs_mlp_gaussian_lowdim.yaml` + `task/single_franka_obs_pred_lowdim.yaml`

A very effective workflow is: **find behavior in config first, then inspect target class**.

---

## 8) Plot/utility modules

- `factr/plot_utils.py`, `factr/obs_pred_plot_utils.py`
  - plotting primitives reused by task eval code.

- `factr/transforms.py`
  - data/image augmentation pipelines.

- `factr/misc.py`, `factr/utils.py`
  - common utility functions, logging helpers, scheduling helpers.

---

## 9) Quick “where do I edit?” index

- Change low-dim policy architecture: `models/lowdim_action_transformer.py`
- Change low-dim data slicing/windowing/chunking: `replay_buffer.py`
- Change action training/eval logic: `trainers/bc.py`, `task.py`
- Change obs prediction model: `models/lowdim_obs_mlp.py`
- Change obs prediction eval/plots/goal diagnostics: `task_obs_pred.py`
- Change experiment wiring/hyperparams: `cfg/*.yaml`

---

## 10) Relationship summary

For low-dim action training:

`train_bc_policy.py` → `BehaviorCloning` + `BCTask` + `RobobufReplayBufferLowdim` + `LowdimStiffnessCVAEAgent`

For low-dim observation prediction:

`train_obs_model.py` → `GaussianObsPredictionTrainer` + `ObsPredictionTask` + `RobobufReplayBufferObsPredLowdim` + `LowdimGaussianObsMLP`

Keeping these two pipelines conceptually separate will save you a lot of confusion when navigating the repository.
