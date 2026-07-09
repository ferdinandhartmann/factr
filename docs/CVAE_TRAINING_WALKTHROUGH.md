# FACTR CVAE Training Walkthrough (commit `1f20153`)

This note explains, end-to-end, how the **CVAE-enabled Transformer policy** trains in FACTR: what the inputs are, how the **prior** and **posterior** produce a latent `z`, how `z` is injected into the Transformer decoder, and how the losses are computed.

Scope:
- Architecture: `factr/models/action_transformer.py`
- Training loop: `factr/train_bc_policy.py`, `factr/trainers/bc.py`
- Data batches: `factr/replay_buffer.py` (and the task dataloaders in `factr/task.py`)

The explanation matches the code behavior in commit `1f20153` (message: "more cleanup"). If you are on a newer commit, small implementation details may differ (for example, which observation tokens are used as the CVAE context).

---

## 1) What We Are Training

We train an offline **behavior cloning** policy that predicts an **action chunk** of length `ac_chunk`:

- Input: observation at time `t` (images + low-dim `obs`)
- Output: `ac_chunk` actions: `a[t], a[t+1], ..., a[t+ac_chunk-1]`

To make predictions **diverse** when the future is multi-modal (branching trajectories), we use a **Conditional VAE (CVAE)**:

- Prior `p(z | c)` uses only the observation context `c`
- Posterior `q(z | x, c)` also sees the ground-truth actions `x` (during training only)
- Sample `z` from the posterior during training, decode actions from `(c, z)`
- Regularize with `KL(q || p)` so that, at inference, we can sample `z ~ p(z|c)` and still get plausible trajectories

---

## 2) The Training Batch (Inputs)

During training, the dataloader yields a batch shaped like:

```
((imgs, obs), actions, mask, labels)
```

Where:
- `imgs`: `dict[str, Tensor]`, e.g. `{"cam0": ..., "cam1": ...}`
  - Typically `imgs["cam0"]` is `(B, C, H, W)`
  - Some buffer settings can produce `(B, T_img, C, H, W)`; the embedding code supports both
- `obs`: `(B, odim)` low-dimensional observation vector (in this codepath it is appended as a token)
- `actions`: `(B, ac_chunk, ac_dim)` ground-truth action chunk
- `mask`: `(B, ac_chunk, ac_dim)` float mask (1.0 = valid, 0.0 = padding)
- `labels`: `(B,)` optional labels (e.g. stiffness class); **not used** by `TransformerAgent` in this file

In `factr/trainers/bc.py`, `actions` and `mask` are flattened to:
- `ac_flat`: `(B, ac_chunk * ac_dim)`
- `mask_flat`: `(B, ac_chunk * ac_dim)`

And then the agent is called as:

```
loss_dict = agent(imgs, obs, ac_flat, mask_flat, class_labels=labels)
```

---

## 3) Observation Tokenization ("Memory")

File: `factr/agent.py` (base class `BaseAgent`)

The observation is turned into a sequence of tokens:

1. **Image tokens** via a visual backbone (e.g., ViT):
   - `img_tokens = features(imgs["cam0"], ...)`
   - Shape: `(B, N_img, D)`

2. **Low-dim obs token** (because `use_obs: add_token` in config):
   - `obs_token = Linear(obs)` -> `(B, D)`
   - Unsqueeze to `(B, 1, D)` and append to the token list

3. Post-processing (projection / normalization / dropout):
   - `post_proc(tokens)` keeps shape `(B, N, D)`

The resulting token sequence is what the CVAE code (and the Transformer policy) uses as the observation representation:

```
c_tokens = tokenize_obs(imgs, obs)   # (B, N, D)
```

In `factr/models/action_transformer.py`, you will also see the term **`memory`** inside the Transformer implementation. Concretely:
- `c_tokens` are the *input* observation tokens.
- `_ACT.encoder(...)` produces `memory`, the *encoded* observation tokens used by the decoder for cross-attention.

---

## 4) CVAE Context `c` (What Prior/Posterior Condition On)

File: `factr/models/action_transformer.py`

In commit `1f20153`, the CVAE context is built from two special tokens:

- `cls_tok = c_tokens[:, cls_index]`
  - Usually the vision CLS token (config default: `cls_index: 0`)
- `force_tok = c_tokens[:, force_index]`
  - Usually the appended low-dim observation token (config default: `force_index: -1`)

Each token has shape `(B, D)`, where `D = token_dim` (commonly 512).

These are the inputs to the prior, and (stacked) become the context tokens for the posterior:

```
c_for_posterior = stack([cls_tok, force_tok], dim=1)   # (B, 2, D)
```

---

## 5) Prior Network `p(z|c)` (MLP)

File: `factr/models/action_transformer.py`, class `PriorNet`

The prior sees only the observation context and outputs a diagonal Gaussian over `z`:

1. Concatenate context summary:
   - `u = concat([cls_tok, force_tok], dim=-1)` -> shape `(B, 2D)`
   - With `D=512`, this is `(B, 1024)` (this matches the explanation you provided)

2. MLP:
   - `LayerNorm(2D)`
   - `Linear(2D -> hdim) + GELU`
   - `Linear(hdim -> hdim) + GELU`

3. Output heads:
   - `mu_p = Linear(hdim -> d_z)` -> `(B, d_z)`
   - `logvar_p = Linear(hdim -> d_z)` -> `(B, d_z)`
   - `logvar_p` is initialized with a negative bias (default `-3.0`) to start with modest variance

Interpretation:

```
p(z|c) = Normal(mu_p, diag(exp(logvar_p)))
```

---

## 6) Posterior Network `q(z|x,c)` (Transformer Encoder)

File: `factr/models/action_transformer.py`, class `PosteriorNet`

The posterior conditions on:
- Context tokens derived from the observation (`c_for_posterior`)
- Ground-truth action chunk `actions` (this is what makes it "posterior")

Key steps:

1. Embed actions into token space:
   - Input: `actions` is `(B, T, ac_dim)` where `T = ac_chunk`
   - `x_tokens = Linear(ac_dim -> D)(actions)` -> `(B, T, D)`

2. Build an encoder input sequence:
   - A learned posterior CLS token `POST_CLS` is prepended
   - The final sequence is:
     - `[POST_CLS, c_tokens, x_tokens]`
   - Sequence length: `1 + N_c + T` where `N_c=2` in this commit

3. Positional encoding:
   - The action tokens `x_tokens` receive sinusoidal positional encodings
   - The context tokens and `POST_CLS` use zero positional encodings

4. Transformer encoder:
   - A small stack of self-attention layers (default: `num_layers=3`)
   - Output: `(1 + N_c + T, B, D)`

5. Posterior summary + heads:
   - Take only the first output token (the transformed `POST_CLS`) as the summary:
     - `h = out[0]` -> `(B, D)`
   - Project to Gaussian parameters:
     - `mu_q = Linear(D -> d_z)` -> `(B, d_z)`
     - `logvar_q = Linear(D -> d_z)` -> `(B, d_z)`

Important detail (gradient flow):
- In `TransformerAgent.forward`, the context tokens passed into the posterior are **detached**:
  - `mu_q, logvar_q = posterior(c_for_posterior.detach(), actions, ...)`
- This means posterior gradients do **not** backprop into the observation tokenizer / image backbone through the context tokens.

This matches the practical intent of many CVAE implementations:
- The posterior is allowed to use ground-truth actions to infer `z`,
- but it should not reshape the observation embedding space in ways that the prior (which never sees actions) cannot match.

---

## 7) Reparameterization Trick (Sampling `z`)

File: `factr/models/action_transformer.py`, function `reparameterize`

We sample `z` with the reparameterization trick:

```
eps ~ Normal(0, I)
z = mu_q + exp(0.5 * logvar_q) * eps
```

Shapes:
- `mu_q`, `logvar_q`, `z`, `eps`: `(B, d_z)`

The explanation you gave ("split into mean and log-variance") corresponds to viewing `[mu, logvar]` together as a `2*d_z`-dimensional output. In this code, `mu` and `logvar` are produced by two separate linear layers, but the math is identical.

Example:
- If you set `d_z = 16`, then `mu` is 16-dim and `logvar` is 16-dim (often visualized together as 32 dims).

---

## 8) Trajectory Decoder (Transformer With `z` Injection)

File: `factr/models/action_transformer.py`, class `_ACT` and `TransformerAgent.forward`

The core policy network is an encoder-decoder Transformer:

1. Encode observation tokens:
   - `_ACT.encoder` processes `c_tokens` -> `memory`
   - `memory` has shape `(N, B, D)` (note the transpose inside `_ACT`)

2. Inject `z`:
   - `z_token = Linear(d_z -> D)(z)` -> `(B, D)`
   - `_ACT.forward` prepends `z_token` to the encoder memory:
     - `memory = concat([z_token, memory], dim=0)`
   - This makes `z` available to the decoder via cross-attention

3. Decode action chunk from fixed queries:
   - There is one learned query embedding per action step:
     - `ac_query`: `(T, D)` where `T = ac_chunk`
   - The decoder cross-attends these queries to `memory` and produces:
     - `action_tokens`: `(B, T, D)`

4. Project to action space:
   - `actions_hat = Linear(D -> ac_dim)(action_tokens)` -> `(B, T, ac_dim)`

Because `z` is stochastic, the model can produce **multiple plausible action chunks** for the same observation by sampling multiple `z`.

---

## 9) Losses (Outputs During Training)

File: `factr/models/action_transformer.py`, method `TransformerAgent.forward`

The forward pass returns a dictionary:

```
{
  "total_loss": recon + beta * kl,
  "l1_loss": recon,
  "kl": kl,
}
```

### 9.1 Reconstruction loss (masked L1)

- Compute elementwise L1 on flattened predictions:
  - `all_l1 = |ac_flat_hat - ac_flat|` -> `(B, ac_chunk * ac_dim)`
- Apply mask:
  - `all_l1 * mask_flat`
- Reduce:
  - In this commit the code sums over batch and feature dimensions.
  - (If you want an average per valid element instead, normalize by `mask_flat.sum()`.)

### 9.2 KL divergence `KL(q||p)`

File: `factr/models/action_transformer.py`, function `kl_diag_gaussians`

For diagonal Gaussians:

```
KL( N(mu_q, sig_q^2) || N(mu_p, sig_p^2) )
```

Computed per batch element and then averaged.

Optional stabilization:
- `free_bits` can clamp the KL to a minimum per dimension (a common anti-collapse trick).

### 9.3 Total loss

```
loss = recon + beta * kl
```

`beta` controls how strongly the prior is forced to match the posterior.

---

## 10) Expected Behavior (Why This Helps Branching Trajectories)

This matches the intuition in your explanation:

- The **posterior** is conditioned on ground-truth actions, so it tends to become sharp:
  - `logvar_q` often becomes very negative (small variance)
- The **prior** sees only the observation context:
  - When the future is essentially determined, the prior can match the posterior (low variance)
  - At **branch points** (multiple plausible futures), the best the prior can do is spread probability mass:
    - `logvar_p` grows (variance increases) in the relevant `z` dimensions

At inference time, sampling `z ~ p(z|c)` yields diverse but still realistic rollouts.

---

## 11) How Sampling Works (Inference Helpers)

File: `factr/models/action_transformer.py`

The agent provides three key action-generation methods:

- `get_actions_base(...)`:
  - Deterministic baseline (no CVAE, no `z`)
  - Useful as an ablation

- `get_actions_prior(..., num_samples=S, sample=True)`:
  - Compute `(mu_p, logvar_p)` from the prior
  - Sample `S` different `z` values per batch element
  - Decode `S` action chunks:
    - Output shape `(B, S, ac_chunk, ac_dim)`

- `get_actions_pos(..., target_action, num_samples=S)`:
  - Uses the posterior (needs ground truth actions), mainly for analysis/visualization

---

## 12) Configuration Knobs That Matter Most

Hydra config (typical):
- `factr/cfg/agent/transformer_vit.yaml`

Key parameters:
- `d_z`:
  - Latent dimension of `z`
  - If you want the "32 split into 16+16" behavior from your explanation, set `d_z=16`
  - For harder / more multi-modal tasks, increasing `d_z` (e.g., 32) can help
- `beta`:
  - KL weight; lower values make training easier but weaken the match between prior and posterior
- `free_bits`:
  - Anti-collapse; can help when KL goes to ~0 too early

Transformer capacity:
- `transformer_kwargs.num_encoder_layers`, `transformer_kwargs.num_decoder_layers`
- `transformer_kwargs.dim_feedforward`

---

## 13) One-Pass Pseudocode (Training Forward)

This is the training forward pass in compact form:

```python
# batch: ((imgs, obs), actions, mask, labels)
ac_flat   = actions.reshape(B, -1)
mask_flat = mask.reshape(B, -1)

c_tokens = tokenize_obs(imgs, obs)             # (B, N, D)
cls_tok  = c_tokens[:, cls_index]              # (B, D)
obs_tok  = c_tokens[:, force_index]            # (B, D)

mu_p, logvar_p = prior(cls_tok, obs_tok)       # (B, d_z), (B, d_z)
mu_q, logvar_q = posterior(
  stack([cls_tok, obs_tok], dim=1).detach(),   # (B, 2, D), detached
  actions,                                     # (B, T, ac_dim)
)                                              # (B, d_z), (B, d_z)

z = mu_q + exp(0.5 * logvar_q) * randn_like(mu_q)  # (B, d_z)
z_token = z_to_token(z)                            # (B, D)

actions_hat = decode_transformer(c_tokens, z_token)  # (B, T, ac_dim)

recon = sum(abs(actions_hat.reshape(B,-1) - ac_flat) * mask_flat)
kl = KL_diag_gaussians(mu_q, logvar_q, mu_p, logvar_p).mean()
loss = recon + beta * kl
```

