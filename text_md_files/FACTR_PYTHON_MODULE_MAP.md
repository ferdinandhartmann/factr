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

---

## 11) Important classes index by priority

If you only have a few hours, this priority list helps.

### Tier 1 (must know for active low-dim BC/CVAE)

1. `factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent`
2. `factr/replay_buffer.py::RobobufReplayBufferLowdim`
3. `factr/task.py::BCTask`
4. `factr/trainers/bc.py::BehaviorCloning`
5. `factr/train_bc_policy.py::train_bc`

### Tier 2 (important supporting modules)

1. `factr/cfg/train_bc_lowdim.yaml`
2. `factr/cfg/agent/transformer_lowdim.yaml`
3. `factr/cfg/task/single_franka_lowdim.yaml`
4. `factr/cfg/trainer/adamw_cos_lowdim.yaml`
5. `factr/task.py` plotting/metrics helper functions

### Tier 3 (obs-pred / analysis path)

1. `factr/models/lowdim_obs_mlp.py::LowdimGaussianObsMLP`
2. `factr/replay_buffer.py::RobobufReplayBufferObsPredLowdim`
3. `factr/task_obs_pred.py::ObsPredictionTask`
4. `factr/trainers/obs_pred.py::GaussianObsPredictionTrainer`
5. `factr/train_obs_model.py`

---

## 12) Python file responsibilities (more granular)

### Entrypoints

- `train_bc_policy.py`: run lifecycle, checkpoint/eval cadence, logging pipeline.
- `train_obs_model.py`: equivalent lifecycle for observation prediction.

### Data pipeline

- `replay_buffer.py`: all sequence slicing, episode boundaries, mask creation, class label extraction.
- `task.py` / `task_obs_pred.py`: dataloader wrappers + validation/evaluation/plot orchestration.

### Model pipeline

- `models/lowdim_action_transformer.py`: context modeling + latent modeling + action decoding.
- `models/lowdim_obs_mlp.py`: conditional Gaussian prediction + Bayesian-style goal posterior.
- `models/action_transformer.py`: image/token transformer CVAE baseline path.

### Optimization pipeline

- `trainers/base.py`: optimizer/scheduler/checkpoint utilities.
- `trainers/bc.py`: action-policy objective call and logging keys.
- `trainers/obs_pred.py`: obs-pred objective call and logging keys.

### Utilities and plotting

- `plot_utils.py`, `obs_pred_plot_utils.py`: reusable figure builders.
- `utils.py`, `misc.py`: helper utilities and run-level global state.
- `transforms.py`: augmentation logic (more relevant for image paths).

---

## 13) Typical experiment workflows (practical recipes)

### Workflow A: Improve low-dim action quality

1. tune `cfg/agent/transformer_lowdim.yaml` (`beta`, `latent_distribution`, depth/width),
2. confirm replay alignment in `RobobufReplayBufferLowdim`,
3. watch `eval/prior_l1`, `eval/posterior_kl`, `eval/sample_diversity`,
4. inspect fan plots by stiffness label from `BCTask.eval`.

### Workflow B: Improve uncertainty/diversity behavior

1. increase sampling diagnostics (`eval_diversity_num_samples`),
2. tune categorical latent temperature and KL balance,
3. compare prior entropy vs posterior entropy trends,
4. verify multi-sample outputs in rollout scripts.

### Workflow C: Improve observation prediction + goal inference

1. tune `lowdim_obs_mlp` hidden depth/width and `nll_loss_weight`,
2. ensure target mask correctness in obs-pred buffer,
3. monitor goal log-likelihood margin and posterior entropy,
4. inspect per-goal probability trajectories in `ObsPredictionTask` plots.

---

## 14) Where model structure is easiest to inspect

For low-dim action policy:

1. inspect yaml constructor args in `cfg/agent/transformer_lowdim.yaml`,
2. map to submodules in `LowdimStiffnessCVAEAgent.__init__`,
3. verify flow in `forward`,
4. inspect inference methods (`get_actions_prior`, `get_actions_pos`) for rollout behavior.

For obs prediction:

1. inspect `cfg/agent/obs_mlp_gaussian_lowdim.yaml`,
2. map to `LowdimGaussianObsMLP.__init__`,
3. inspect `forward` and `infer_goals` for objective and posterior math.

This strategy keeps config, model architecture, and runtime behavior tightly connected during debugging.
