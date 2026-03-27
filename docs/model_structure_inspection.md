# How to Inspect FACTR Model Structure (for Low-Dim and Beyond)

This guide shows practical ways to inspect and verify model structure from this repository.

---

## 1) Start from Hydra wiring (what class is active)

Before inspecting layers, confirm which class is instantiated:

1. Open experiment config (for lowdim: `factr/cfg/train_bc_lowdim.yaml`).
2. Follow defaults:
   - `agent: transformer_lowdim`
   - `task: single_franka_lowdim`
   - `trainer: adamw_cos_lowdim`
3. Open `factr/cfg/agent/transformer_lowdim.yaml` and read `_target_`.

For lowdim setup, `_target_` points to `LowdimStiffnessCVAEAgent`.

---

## 2) Read model class anatomy directly

Open:
- `factr/models/lowdim_action_transformer.py`

Read in this order:

1. `__init__`:
   - expected dimensions,
   - token/group encoders,
   - prior/posterior heads,
   - decoder and action head.
2. `_build_context_tokens`:
   - how observation fields are converted to tokens.
3. `_prior` / `_posterior`:
   - latent Gaussian parameterization.
4. `_decode_actions`:
   - transformer decoder prediction path.
5. `forward`:
   - complete training computation and loss return dictionary.
6. `get_actions_*` methods:
   - inference variants.

---

## 3) Runtime introspection commands (recommended)

From repository root, run one-off Python snippets.

### 3.1 Print full module tree

```bash
python - <<'PY'
import hydra
from omegaconf import OmegaConf
from hydra import initialize, compose

with initialize(version_base=None, config_path='factr/cfg'):
    cfg = compose(config_name='train_bc_lowdim')
agent = hydra.utils.instantiate(cfg.agent)
print(agent)
print('\nTotal params:', sum(p.numel() for p in agent.parameters()))
print('Trainable params:', sum(p.numel() for p in agent.parameters() if p.requires_grad))
PY
```

### 3.2 Print named submodules and parameter shapes

```bash
python - <<'PY'
import hydra
from hydra import initialize, compose

with initialize(version_base=None, config_path='factr/cfg'):
    cfg = compose(config_name='train_bc_lowdim')
agent = hydra.utils.instantiate(cfg.agent)

print('=== Named Modules ===')
for n, _ in agent.named_modules():
    if n:
        print(n)

print('\n=== Key Parameter Shapes ===')
for n, p in agent.named_parameters():
    if any(k in n for k in ['encoder', 'decoder', 'prior', 'posterior', 'action_head', 'action_queries', 'stiffness_embed']):
        print(f'{n:60s} {tuple(p.shape)}')
PY
```

### 3.3 Sanity-check forward pass shapes

```bash
python - <<'PY'
import torch, hydra
from hydra import initialize, compose

with initialize(version_base=None, config_path='factr/cfg'):
    cfg = compose(config_name='train_bc_lowdim')
model = hydra.utils.instantiate(cfg.agent)

B = 4
obs = torch.randn(B, cfg.obs_window, cfg.task.obs_dim)
a = torch.randn(B, cfg.ac_chunk, cfg.task.ac_dim)
m = torch.ones(B, cfg.ac_chunk, cfg.task.ac_dim)
labels = torch.ones(B, dtype=torch.long)

out = model({}, obs, a.reshape(B,-1), m.reshape(B,-1), class_labels=labels)
print({k: (v.shape if hasattr(v, 'shape') else type(v)) for k,v in out.items()})
print('total_loss:', float(out['total_loss']))
PY
```

---

## 4) How to inspect data-model compatibility

Most training failures come from data mismatch, not model code.

Inspect these together:
- `factr/cfg/task/single_franka_lowdim.yaml` (declared dims and buffer class)
- `factr/replay_buffer.py::RobobufReplayBufferLowdim` (actual sample builder)
- `factr/models/lowdim_action_transformer.py` (hard shape expectations)

Checklist:
- `obs_dim` in config equals state vector length in processed buffer.
- `ac_dim` equals pose-only command dimensions extracted from action.
- `obs_window` and `ac_chunk` match both task config and model expectations.
- stiffness label encoding (0/1-based) is handled as expected.

---

## 5) Where model-structure differences appear in eval

If you swap architecture settings and want to observe impact quickly:
- Monitor `eval/posterior_kl` (latent usage signal).
- Compare `eval/action_l2` and `eval/pose_dim*_l2`.
- Inspect `eval/plot_example_gt_vs_pred_stiffness_*` fan plots for calibration of multimodal prior samples.

These are logged by `BCTask.eval`.

---

## 6) Suggested workflow for medium-level users

1. Confirm class wiring from Hydra `_target_`.
2. Print instantiated model and param count.
3. Run one synthetic forward pass and verify output dict.
4. Inspect one real batch from buffer and print tensor shapes.
5. Start short training run (`max_iterations=1000`) and check eval plots.

Following this sequence usually surfaces mistakes early and saves long debugging cycles.

