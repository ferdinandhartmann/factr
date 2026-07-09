# ACT Implementation in This Repository (High-Level + Easy Shape Guide)

This document explains how the ACT-style transformer policy is implemented in this codebase, how learning works, and where to read the source.

It is intentionally high-level and practical.

---

## 1) Where ACT lives in this repo

Primary implementation file:

- `factr/models/action_transformer.py`

Main classes/functions to know:

- `_ACT` (core encoder/decoder transformer)
- `_TransformerEncoderLayer`, `_TransformerDecoderLayer`
- `_TransformerEncoder`, `_TransformerDecoder`
- `PriorNet`, `PosteriorNet`
- `TransformerAgent` (full policy model that wraps ACT + CVAE)
- `reparameterize`, `kl_diag_gaussians`

Related abstractions:

- `factr/agent.py::BaseAgent` (tokenization from image/obs inputs)
- `factr/trainers/bc.py::BehaviorCloning.training_step` (training call contract)

---

## 2) What “ACT” means here

In this repo, ACT-like behavior means:

1. Build context tokens from observations.
2. Use learned query embeddings for each action step in the chunk.
3. Decode all action steps in parallel with transformer decoder cross-attending to context memory.

So instead of autoregressive one-step-at-a-time prediction, the model predicts an **action chunk** of length `T` in one forward pass.

---

## 3) High-level data flow

For `TransformerAgent.forward(...)` in `action_transformer.py`:

1. `BaseAgent.tokenize_obs(imgs, obs)` produces context tokens.
2. Prior network estimates `p(z|context)` from selected context tokens.
3. Posterior network estimates `q(z|context, target_actions)` during training.
4. Sample latent `z` with reparameterization trick.
5. Convert latent to token (`z_to_token`) and inject into ACT model.
6. ACT decoder predicts action-token sequence for all `T` steps.
7. Linear projection maps decoder tokens to action dimensions.
8. Compute reconstruction loss + KL loss.

This is a CVAE policy where ACT is the sequence modeling backbone.

---

## 4) Shape walkthrough (typical)

Notation:

- `B`: batch size
- `N`: number of context tokens
- `D`: token dimension (e.g., 512)
- `T`: action chunk length
- `A`: action dim
- `Z`: latent dim

### 4.1 Context tokens

From `tokenize_obs`:

- `c_tokens`: `(B, N, D)`

### 4.2 Prior / posterior

- `mu_p`, `logvar_p`: `(B, Z)` from `PriorNet`
- `mu_q`, `logvar_q`: `(B, Z)` from `PosteriorNet`
- sampled `z`: `(B, Z)`
- `z_token = z_to_token(z)`: `(B, D)`

### 4.3 ACT decoder input/output

- learned queries (`ac_query.weight`): `(T, D)`
- decoder output tokens: `(B, T, D)`
- final actions via projection: `(B, T, A)`

### 4.4 Loss tensors

- target actions reshaped from flat to `(B, T, A)`
- reconstruction loss: elementwise L1/L2 over `(B, T, A)` with mask
- KL term: scalar per sample from Gaussian KL
- total: `recon + beta * KL`

---

## 5) Core classes explained simply

## 5.1 `_ACT`

Think of `_ACT` as the “sequence engine.”

- Encoder reads context tokens.
- Decoder uses learned action queries to ask the encoder memory: “what action should happen at each step?”
- Returns one token per action step.

Important method:

- `_ACT.forward(input_tokens, query_enc, z_token=None, ...)`

If `z_token` is provided, it is concatenated into memory (conditioning decoder on latent intent).

## 5.2 `PriorNet`

Maps compact context summary to latent Gaussian parameters:

- input: selected context features (in code: typically CLS + FORCE tokens)
- output: `(mu_p, logvar_p)`

## 5.3 `PosteriorNet`

Reads context + true target actions and predicts posterior latent params:

- output: `(mu_q, logvar_q)`

This gives a training signal for latent variables.

## 5.4 `TransformerAgent`

Top-level policy class combining:

- tokenization (`BaseAgent`)
- ACT backbone (`_ACT`)
- prior/posterior latent modules
- action projection and loss computation

For most practical understanding of ACT behavior in this repo, read `TransformerAgent.forward` first, then drill down.

---

## 6) Learning objective (calculation reference)

At a high level:

- Reconstruction term: make predicted chunk match target chunk.
- KL term: keep posterior close to prior.

Formula style:

- `L_total = L_recon + beta * KL(q(z|x,c) || p(z|c))`

In this code:

- recon is masked L1 (and logged L2)
- KL from `kl_diag_gaussians(...)`
- optional free-bits clamp to avoid trivial KL collapse

Why this helps:

- posterior can use target actions to learn useful latent signal,
- prior learns to approximate that latent from context alone,
- at inference, model samples from prior and still produces meaningful action chunks.

---

## 7) Inference behavior

Main APIs in `TransformerAgent`:

- `get_actions_base(...)`: deterministic base prediction
- `get_actions_prior(..., num_samples=S)`: sample `S` latents from prior and decode multiple action candidates
- `get_actions_pos(...)`: posterior-conditioned reconstruction given target action
- `get_uncertainty_entropy(...)`: uncertainty estimate from sample spread

Typical sampled shape:

- `(B, S, T, A)`

This is used for diversity analysis and uncertainty-aware downstream logic.

---

## 8) ACT vs low-dim custom transformer in this repo

- `action_transformer.py::TransformerAgent` is ACT-style and built around `BaseAgent` tokenization (often image/obs pipelines).
- `lowdim_action_transformer.py::LowdimStiffnessCVAEAgent` is a specialized low-dim-only CVAE transformer with explicit state-group tokens.

Both are chunked-action CVAE policies, but they differ in how context tokens are built and how latent is injected.

---

## 9) Recommended reading order (easy)

1. `factr/models/action_transformer.py::TransformerAgent.forward`
2. `factr/models/action_transformer.py::_ACT.forward`
3. `factr/models/action_transformer.py::PriorNet.forward`
4. `factr/models/action_transformer.py::PosteriorNet.forward`
5. `factr/models/action_transformer.py::reparameterize`
6. `factr/models/action_transformer.py::kl_diag_gaussians`
7. `factr/agent.py::BaseAgent.tokenize_obs`

If you follow this order, you can understand both model architecture and loss in under one reading pass.

---

## 10) One-paragraph intuition

ACT in this repo is a transformer chunk-policy: observation context becomes memory, learned queries represent each future action step, and the decoder predicts the whole action chunk at once. A CVAE latent sits on top so the model can represent multi-modal futures; during training it learns from posterior latents (using target actions), and during inference it generates from prior latents (using only context). This is why the model can be both accurate and diverse when tuned correctly.
