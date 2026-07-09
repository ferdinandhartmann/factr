<INSTRUCTIONS>

Current project status:
- The actively used pipeline is the FACTR low-dimensional BC/CVAE policy path.
- `factr/world_model/` exists, but the world model is currently **not used** in the active workflow unless explicitly requested.

Dont make test files.

FACTR quick map (active path):
- Training entrypoint: `factr/factr/train_bc_policy.py` (Hydra default config: `train_bc_lowdim.yaml`)
- Main training config: `factr/factr/cfg/train_bc_lowdim.yaml`
- Task/data config: `factr/factr/cfg/task/single_franka_lowdim.yaml`
- Agent config: `factr/factr/cfg/agent/transformer_lowdim.yaml`
- Optimizer/schedule config: `factr/factr/cfg/trainer/adamw_cos_lowdim.yaml`
- Replay buffer class: `factr/factr/replay_buffer.py::RobobufReplayBufferLowdim`
- Low-dim model: `factr/factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent`

FACTR data/shape notes:
- Keep shapes explicit and consistent: observation windows `(B, W, obs_dim)`, action chunks `(B, T, ac_dim)`.
- Current low-dim defaults in configs/docs are centered around `obs_dim=27`, `ac_dim=9`, windowed observations, and chunked action prediction.
- Preserve episode boundaries and alignment assumptions in replay-buffer code.

FACTR commands (reference):
- Process data (from `factr/`): `python process_data/process_data.py`
- Train low-dim BC/CVAE (from `factr/`): `python factr/train_bc_policy.py --config-name train_bc_lowdim`
- Use `factr/scripts/` utilities for rollout/analysis when possible instead of one-off scripts.

Style / hygiene
- Prefer small, testable modules; avoid giant training scripts.
- Keep shapes explicit (`(B,T,...)`) and assert early.
- Seed everything in training scripts (Python/NumPy/Torch).
- Don’t change FACTR or DreamerV2 internals unless necessary; add adapters first.
- Prefer config-level changes (Hydra/YAML) over hard-coded constants.
- When you implement something, add a few comments that descibe what is happening and comments at important steps. but dont add to many, just a few.n 

When editing:
- Scope changes tightly to the active pipeline unless asked otherwise.
- If a task mentions world models, confirm whether to use `factr/world_model/` first because it is not part of the default current workflow.
</INSTRUCTIONS>

How to respond:
- Only ask for additional follow-up at the end when it is genuinely important and helpful.
