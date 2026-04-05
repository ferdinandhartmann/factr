# FACTR Model Dimensions, Tensor Shapes, and Structural Reference

This document is a direct shape-and-structure reference for the main models used in this repository’s low-dimensional workflows.

It complements the narrative docs by focusing on:

- exact input/output shapes,
- latent shapes,
- token counts,
- important dimensional assumptions,
- per-model module structure.

---

## 1) Global notation used in this file

- `B`: batch size
- `W`: observation window length (default low-dim: `W=8`)
- `T`: action chunk / prediction horizon (default low-dim: `T=30`)
- `D_obs`: observation dimension (default low-dim action policy: `27`)
- `D_ac`: action dimension (default low-dim action policy: `9`)
- `D_tok`: token dimension (model/config dependent)
- `D_z`: decoder latent width
- `G`: number of goal classes (default obs-pred: `4`)
- `S`: number of latent/action samples during inference

---

## 2) `LowdimStiffnessCVAEAgent` (`factr/models/lowdim_action_transformer.py`)

## 2.1 High-level contract

### Inputs

- `obs`: `(B, W, 27)`
- `ac_flat`: `(B, T*9)` or action tensor reshape-compatible to `(B,T,9)`
- `mask_flat`: `(B, T*9)` reshape-compatible to `(B,T,9)`
- `class_labels`: `(B,)` stiffness labels

### Outputs from `forward`

Dictionary keys (core):

- `total_loss`: scalar
- `l1_loss`: scalar
- `l2_loss`: scalar
- `kl`: scalar
- optional latent metrics (`prior_std_mean`, `posterior_std_mean`, entropies)

---

## 2.2 Fixed dimensional assumptions (current implementation)

The class currently enforces:

- `obs_dim == 27`
- `ac_dim == 9`

State split across 27 dims:

- pose: `9`
- velocity: `6`
- wrench: `6`
- tracking: `6`

Total context tokens before decoding: `6`

1. cls token from obs groups
2. pose token
3. velocity token
4. wrench token
5. tracking token
6. stiffness embedding token

So context tensor shape after encoding is:

- `context_tokens = (B, 6, D_tok)`

---

## 2.3 Latent families and dimensions

### Gaussian latent mode

- prior/posterior params:
  - `mu`: `(B, D_z)`
  - `logvar`: `(B, D_z)`
- sampled `z`: `(B, D_z)`

### Categorical latent mode

- logits:
  - `(B, categorical_num_variables, categorical_num_categories)`
- flattened latent sample dimension before optional projection:
  - `D_flat = categorical_num_variables * categorical_num_categories`
- if `D_flat != D_z`, projection maps `D_flat -> D_z`

---

## 2.4 Decoder-side shapes

- learned action queries: `(T, D_tok)` as embedding table
- expanded queries per batch: `(B, T, D_tok)`
- z-token bias: `(B, 1, D_tok)` then broadcast to `(B,T,D_tok)`
- decoder output: `(B, T, D_tok)`
- action head output: `(B, T, 9)`

---

## 2.5 Inference API shapes

- `get_actions_base(...)` → `(B, T, 9)`
- `get_actions_prior(..., num_samples=S)` → `(B, S, T, 9)`
- `get_actions_pos(..., num_samples=S)` → `(B, S, T, 9)`
- `get_uncertainty_entropy(...)`:
  - final action: `(B, T, 9)`
  - uncertainty: `(B, T, 9)` or weighted-step mode `(B,1,9)`

---

## 2.6 Structural blocks summary

- group encoders (`pose_encoder`, `vel_encoder`, `wrench_encoder`, `track_encoder`)
- stiffness embedding
- obs-CLS constructor MLP (`cls_from_obs`)
- context transformer encoder
- prior network (optional fixed prior)
- posterior transformer (`_PosteriorTransformer`)
- latent projection / token bridge (`z_to_token`)
- transformer decoder + linear action head

---

## 3) `_PosteriorTransformer` (`factr/models/lowdim_action_transformer.py`)

### Inputs

- `context_tokens`: `(B, N_ctx, D_tok)` where `N_ctx=6`
- `target_actions`: `(B, T, D_ac)` where `D_ac=9`

### Internal sequence

- action tokens: `(B, T, D_tok)`
- add learned action positional embeddings
- prepend posterior-CLS token
- concat sequence length = `1 + N_ctx + T`

### Outputs

- Gaussian mode: `{mu, logvar}` each `(B, D_z)`
- Categorical mode: `{logits}` `(B, N_var, N_cat)`

---

## 4) `LowdimGaussianObsMLP` (`factr/models/lowdim_obs_mlp.py`)

## 4.1 High-level contract

### Inputs

- `obs_window`: `(B, W, D_in_obs)`
- `action_chunk`: `(B, T, D_pose_action)`
- `stiffness_labels`: `(B,)`
- `goal_labels`: `(B,)`
- optional `target_obs`: `(B, T, D_pred_obs)`
- optional `target_mask`: `(B, T)`

Typical defaults in config:

- `W=8`
- `T=30`
- `D_in_obs=21`
- `D_pred_obs=21`
- `D_pose_action=9`
- `stiffness_classes=3`
- `goal_classes=4`

---

## 4.2 Feature vector dimension

Flattened feature fed to backbone MLP:

`(W*D_in_obs) + (T*D_pose_action) + stiffness_classes + goal_classes`

With defaults:

- `8*21 + 30*9 + 3 + 4 = 168 + 270 + 7 = 445`

So MLP input is `(B, 445)` under default obs-pred config.

---

## 4.3 Distribution output shapes

Backbone final linear outputs:

- `(B, T*D_pred_obs*2)`

Reshaped to:

- `(B, T, D_pred_obs, 2)`

Interpreted as:

- `mean`: `(B, T, D_pred_obs)`
- `var`: `(B, T, D_pred_obs)` (softplus + floor)
- `std`: `(B, T, D_pred_obs)`
- sample: `(B, T, D_pred_obs)` for single sample

---

## 4.4 Goal-inference expansion shapes

`infer_goals` evaluates all goals by expanding batch:

- expanded obs/action/stiffness length: `B*G`
- per-goal outputs reshaped back to:
  - `mean`: `(B, G, T, D_pred_obs)`
  - `std`: `(B, G, T, D_pred_obs)`
  - `log_likelihood_per_goal`: `(B, G)`
  - `goal_posterior`: `(B, G)`
  - `goal_posterior_over_time`: `(B, T, G)`

---

## 4.5 Structural blocks summary

- feature builder (flatten obs/action + one-hot labels)
- MLP backbone (`num_layers` hidden GELU blocks)
- Gaussian head (mean/variance packing)
- training objective composition (mean MSE + weighted NLL)
- Bayesian-style goal posterior utilities

---

## 5) `TransformerAgent` (`factr/models/action_transformer.py`) — comparison reference

This model is more image/token oriented and inherits `BaseAgent`.

## 5.1 Typical shapes

- tokenized context from `BaseAgent.tokenize_obs`: `(B, N_tokens, D_tok)`
- action queries: `(T, D_tok)` embedding table
- prior/posterior Gaussian params: `(B, D_z)`
- decoded actions: `(B, T, D_ac)`
- sampled prior actions: `(B, S, T, D_ac)`

## 5.2 Structural blocks summary

- `BaseAgent` tokenization stack
- `_ACT` encoder-decoder transformer core
- `PriorNet` and `PosteriorNet` Gaussian latent modules
- latent-to-token projection and action projection head

---

## 6) `BaseAgent` / `MLPAgent` (`factr/agent.py`)

## 6.1 `BaseAgent` tokenization dimensions

- image embedding per camera/time returns token sequences
- optional obs integration strategies:
  - `add_token`: append one obs token → token count increases by 1
  - `pad_img_tokens`: concatenate obs feature to each image token channel

Final token tensor shape typically:

- `(B, N_tokens, D_tok)`

or flattened:

- `(B, N_tokens*D_tok)` when requested.

## 6.2 `MLPAgent` output dimensions

- flattened tokens pass through shared MLP trunk
- policy head returns action distribution over flattened action dims
- `ac_chunk` and action dimensionality are dictated by policy head config

---

## 7) Buffer/task interfaces (shape contracts)

## 7.1 `RobobufReplayBufferLowdim.__getitem__`

Returns:

- `({}, obs_window)` where `obs_window: (W,27)`
- `action_chunk: (T,9)`
- `mask: (T,9)`
- `label: ()`

Loader collation gives:

- obs `(B,W,27)`, actions `(B,T,9)`, mask `(B,T,9)`, labels `(B,)`

## 7.2 `RobobufReplayBufferObsPredLowdim.__getitem__`

Returns:

- `obs_window: (W, D_in_obs)`
- `action_chunk: (T, D_pose_action)`
- `target_chunk: (T, D_pred_obs)`
- `target_mask: (T,)`
- stiffness label scalar
- goal label scalar

---

## 8) Quick default-dimension snapshot table

| Component | Default key dims |
|---|---|
| Low-dim BC agent input | obs `(B,8,27)`, actions `(B,30,9)` |
| Low-dim BC context tokens | `(B,6,D_tok)` |
| Low-dim BC decoder output | `(B,30,9)` |
| Obs-pred input | obs `(B,8,21)`, action `(B,30,9)` |
| Obs-pred mean/std output | `(B,30,21)` |
| Goal posterior | `(B,4)` |

---

## 9) Code locations for this dimension sheet

- `factr/models/lowdim_action_transformer.py`
- `factr/models/lowdim_obs_mlp.py`
- `factr/models/action_transformer.py`
- `factr/agent.py`
- `factr/replay_buffer.py`
- `factr/cfg/agent/transformer_lowdim.yaml`
- `factr/cfg/agent/obs_mlp_gaussian_lowdim.yaml`
- `factr/cfg/task/single_franka_lowdim.yaml`
- `factr/cfg/task/single_franka_obs_pred_lowdim.yaml`

Use this file as a shape reference before changing configs or model code.
