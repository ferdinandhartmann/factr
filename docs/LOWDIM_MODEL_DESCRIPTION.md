# Low-Dimensional Categorical CVAE Policy

This document describes the architecture and default configuration currently used by the active FACTR low-dimensional behavior-cloning pipeline. It is a description of the implementation, not a proposed design.

The active entry point is:

```bash
cd /home/ferdinand/activeinference/factr
python factr/train_bc_policy.py --config-name train_bc_lowdim
```

The main implementation and configuration files are:

- `factr/models/lowdim_action_transformer.py`
- `factr/cfg/train_bc_lowdim.yaml`
- `factr/cfg/agent/transformer_lowdim.yaml`
- `factr/cfg/task/single_franka_lowdim.yaml`
- `factr/cfg/trainer/adamw_cos_lowdim.yaml`
- `factr/replay_buffer.py`
- `factr/trainers/bc.py`

The world model under `factr/world_model/` is not used by this pipeline.

## 1. Exact active configuration

The following values are the resolved defaults of `train_bc_lowdim.yaml` and its included configs.

### Data and tensor sizes

| Setting | Current value | Meaning |
|---|---:|---|
| `dataset_name` | `boxlift_124_relative_perdim` | Processed dataset directory |
| `obs_dim` before feature removal | 36 | Full stored low-dimensional state width |
| `include_velocity` | `true` | Keep the 6D velocity group |
| `include_tracking_error` | `false` | Remove the 6D tracking-error group |
| Effective model `obs_dim` | 30 | `36 - 6` tracking dimensions |
| `obs_window` | 4 | Number of observations in each history window |
| `ac_dim` | 9 | Action pose width: position 3 + rotation representation 6 |
| `ac_chunk` | 20 | Number of predicted action steps |
| `action_index_offset` | 1 | Preprocessing/alignment offset for the predicted chunk |
| `action_chunk_mode` | `relative` | Use precomputed normalized relative pose chunks |
| `batch_size` | 180 | Training batch size |

Therefore, the principal batched shapes are:

```text
observations       x: (B, 4, 30)
target actions     a: (B, 20, 9)
action mask        m: (B, 20, 9)
mode/stiffness     s: (B,)
goal group         g: (B, 3)
predicted actions  a_hat: (B, 20, 9)
```

### Conditioning

| Setting | Current value | Effect |
|---|---:|---|
| `use_stiffness_conditioning` | `true` | Add a mode/stiffness context token |
| `override_stiffness_with_mode` | `true` | Use binary dataset `mode` instead of physical stiffness |
| Resolved `stiffness_classes` | 2 | Mode 0 and mode 1 become model labels 1 and 2 |
| `goal_label` | `true` | Condition on one of three grouped goal labels |
| `use_arrangement_conditioning` | `false` | Do not use the 9D arrangement condition |
| `use_adaptive_layer_norm` | `true` | Apply goal-conditioned scale and shift after context normalization |
| `use_stiffness_goal_adaln_gate` | `true` | Gate goal conditioning according to mode |
| `goal_adaln_gate_min` | 0.02 | Mode 0 retains 2% of the goal vector; mode 1 retains 100% |
| `use_cls_token` | `false` | The context encoder has no CLS token |
| `posterior_include_command` | `true` | Include the commanded-pose token in posterior inference |

### Transformer and latent sizes

| Setting | Current value |
|---|---:|
| Token width `token_dim` | 128 |
| Feed-forward/prior width `hidden_dim` | 256 |
| Attention heads `nhead` | 4 |
| Context encoder layers | 2 |
| Posterior encoder layers | 2 |
| Action decoder layers | 4 |
| Dropout | 0.1 |
| Transformer activation | GELU |
| Latent family | categorical |
| Categorical variables `V` | 4 |
| Categories per variable `K` | 3 |
| Flattened categorical width `V*K` | 12 |
| Decoder latent width `d_z` | 12 |
| Gumbel-Softmax temperature | 1.0 |
| Straight-through hard samples | `true` |
| Fixed prior | `true` |

The categorical sample is a concatenation of four 3-way one-hot variables. It is 12D, exactly matching `d_z=12`, so `categorical_latent_proj` is an identity operation and does not compress the categorical code.

### Loss and optimization

| Setting | Current value |
|---|---:|
| `variable_beta` | `true` |
| `variable_beta_upper` | 0.1 |
| `variable_beta_lower` | 0.018 |
| Scalar `beta` | 0.035, inactive while `variable_beta=true` |
| `free_bits` | `null`, disabled |
| `kl_balance_alpha` | 0.75 |
| Diversity loss | disabled |
| Optimizer | AdamW |
| Learning rate | 0.00035 |
| Weight decay | 0.0001 |
| Scheduler | CosineAnnealingLR |
| Scheduler `T_max` | 10,000 iterations |
| Scheduler minimum LR | 0.000001 |
| Training iterations | 10,000 |
| Seed | 42 |

## 2. End-to-end architecture

At training time the model implements a categorical conditional VAE:

```text
windowed state x ──> grouped state encoders ──> context Transformer ──> context C
mode label s ──────> mode token ────────────────┘          │
goal group g ──────> gated adaptive LayerNorm ─────────────┘

C including command + target chunk a ──> posterior Transformer ──> q(z | x,s,g,a)
                                                            │
                                                   hard Gumbel sample
                                                            │
                                                            z

fixed uniform categorical prior ─────────────────────────> p(z)

[projected z token, full context C] + learned action queries
                         └──────────> Transformer decoder ──> action chunk a_hat
```

The target action chunk is available only to the posterior during training or posterior diagnostics. Prior inference does not require ground-truth future actions.

## 3. Dataset construction and alignment

### 3.1 Episode boundaries

`RobobufReplayBufferLowdim` reconstructs episodes from the flat robobuf sequence. A step begins a new episode if any of the following is true:

- `step.first`
- `step.is_first`
- `step.prev is None`

Observation windows never cross these episode boundaries.

### 3.2 Effective state layout

The stored state is configured as 36D. Velocity is retained and tracking error is removed by `lowdim_filter_state_features`, giving the model this compact 30D layout:

| Slice | Width | Meaning |
|---|---:|---|
| `0:9` | 9 | Measured pose |
| `9:15` | 6 | Velocity |
| `15:21` | 6 | Wrench |
| `21:30` | 9 | Commanded pose |

The model constructs these slices programmatically and raises an error if their total width does not equal the effective observation dimension. With tracking enabled, an additional 6D tracking group would occur before the command, but that group is absent in the active configuration.

### 3.3 Observation window

For an anchor step `t`, the replay buffer takes up to four states ending at `t`:

```text
x_t = [state_(t-3), state_(t-2), state_(t-1), state_t]
```

At the start of an episode, the earliest available state is repeated on the left until the window has length four. The resulting sample has shape `(4, 30)`.

### 3.4 Relative action target and mask

`action_chunk_mode=relative` means the replay buffer expects every processed step to already store:

- an action array with exact shape `(20, 9)`;
- `obs["action_mask"]` with exact shape `(20,)`;
- processing metadata reporting relative pose mode;
- `ac_chunk=20` and `action_index_offset=1`;
- `relative_chunk_normalized=true`;
- `relative_chunk_anchor=current_command`;
- dedicated action-normalization metadata.

The relative chunk is anchored to the current commanded pose and is read directly from `step.action`. It is not rebuilt online from future steps. Dataset processing must therefore have used the same horizon and offset as training.

The 1D time mask is repeated over all nine action dimensions, producing `(20, 9)`. Invalid padded tail positions do not contribute to reconstruction loss.

### 3.5 Mode and goal conditions

Because `override_stiffness_with_mode=true`, the first step of every episode must contain binary `obs["mode"]`:

| Dataset value | Model-facing label | Meaning used by evaluation |
|---:|---:|---|
| 0 | 1 | follow |
| 1 | 2 | leading |

The API boundary uses 1-based scalar labels. Inside the model they are converted to 2D one-hot vectors.

The goal value is read from `obs["goals"]` or `obs["goal"]`. A scalar goal ID from 1 through 9 is grouped into a 3D one-hot vector:

| Goal IDs | Goal vector |
|---|---|
| 1, 2, 3 | `[1, 0, 0]` |
| 4, 5, 6 | `[0, 1, 0]` |
| 7, 8, 9 | `[0, 0, 1]` |

The condition stored for a sample is taken at the anchor step `t`.

### 3.6 DataLoader output

With the active goal conditioning and no arrangement conditioning, one batch is:

```python
((imgs, obs), actions, mask, labels, goal_vectors)
```

where `imgs` is an empty dictionary and the remaining shapes are:

```text
obs:          (B, 4, 30)
actions:      (B, 20, 9)
mask:         (B, 20, 9)
labels:       (B,)
goal_vectors: (B, 3)
```

The trainer flattens actions and masks to `(B, 180)` for the model API. The model immediately reshapes them back to `(B, 20, 9)`.

## 4. Context tokenization

Each state group is flattened across the complete observation window and converted into one token. For a group of width `G`, the encoder is:

```text
LayerNorm(4*G)
Linear(4*G, 128)
GELU
Linear(128, 128)
```

The active encoders and inputs are:

| Token | Flattened input | Encoder dimensions |
|---|---:|---|
| Pose | `4*9 = 36` | `LN(36) -> Linear(36,128) -> GELU -> Linear(128,128)` |
| Velocity | `4*6 = 24` | `LN(24) -> Linear(24,128) -> GELU -> Linear(128,128)` |
| Wrench | `4*6 = 24` | `LN(24) -> Linear(24,128) -> GELU -> Linear(128,128)` |
| Mode | 2D one-hot | bias-free `Linear(2,128)` |
| Command | `4*9 = 36` | `LN(36) -> Linear(36,128) -> GELU -> Linear(128,128)` |

There is no tracking token, goal token, arrangement token, or CLS token in the active configuration. Goal conditioning is applied by adaptive LayerNorm instead of being appended as a token.

Tokens are stacked in this exact order:

```text
[pose, velocity, wrench, mode, command]
```

This produces `(B, 5, 128)`. A learned positional tensor of shape `(1, 5, 128)` is added so that the Transformer can distinguish token roles.

## 5. Context Transformer and adaptive LayerNorm

### 5.1 Context Transformer

The five tokens pass through a two-layer `nn.TransformerEncoder`. Each layer uses:

- model width 128;
- 4 attention heads, so each head has width 32;
- feed-forward width 256;
- GELU activation;
- dropout 0.1;
- batch-first layout.

The encoder output is then passed through `LayerNorm(128)`, producing normalized context tokens `C_norm` with shape `(B, 5, 128)`.

### 5.2 Mode-gated goal condition

The goal vector is scaled before it reaches the adaptive modulation network. For two mode classes, `_stiffness_goal_gate` linearly assigns gate strengths from `goal_adaln_gate_min` to 1:

```text
mode 0 / label 1 / follow:  gate = 0.02
mode 1 / label 2 / leading: gate = 1.00
```

Thus:

```text
g_gated = gate(s) * g
```

The follow mode still receives a small goal-dependent signal; it is not exactly goal-independent.

### 5.3 Adaptive modulation

The gated 3D goal vector is mapped to one scale and one shift vector:

```text
Linear(3, 128)
GELU
Linear(128, 256)
split -> scale (B,128), shift (B,128)
```

The final linear layer is initialized with all-zero weights and bias. Consequently, training begins with `scale=0` and `shift=0`, exactly reproducing the ordinary normalized context. Goal modulation is learned gradually.

The same feature-wise scale and shift are broadcast across all five context tokens:

```text
C = C_norm * (1 + scale[:, None, :]) + shift[:, None, :]
```

This implementation is an adaptive modulation after the context Transformer's final LayerNorm. It does not replace every internal LayerNorm inside the Transformer layers.

## 6. Decoder context versus latent context

The commanded-pose token is treated specially by the prior, while the active posterior includes it:

- the action decoder receives all five context tokens;
- the posterior receives all five context tokens because `posterior_include_command=true`;
- the prior receives only the first four tokens;
- the command token is always last and is removed with `context_tokens[:, :-1]` only when building prior context.

Therefore:

```text
full decoder context: [pose, velocity, wrench, mode, command] -> (B,5,128)
posterior context:    [pose, velocity, wrench, mode, command] -> (B,5,128)
prior context:        [pose, velocity, wrench, mode]          -> (B,4,128)
flattened prior z_context:                                   -> (B,512)
```

Although `z_context_mode` remains an accepted constructor argument, `_build_z_context` currently always flattens all supplied no-command tokens. The configured string does not select a different behavior in this implementation.

## 7. Fixed categorical prior

The active prior is fixed and independent of context:

```text
p(z) = product over v=1..4 of Categorical(uniform over 3 categories)
```

Its logits have shape `(B, 4, 3)` and are all zero. Each of the four variables has probability vector `[1/3, 1/3, 1/3]`.

There are `3^4 = 81` possible joint hard codes. The prior entropy summed over all variables is:

```text
H[p] = 4 * ln(3) ~= 4.394 nats
```

Because `fixed_prior=true`, the learned prior MLP and prior-logit head are not constructed. The flattened context is still formed by the current forward and inference paths, but it does not affect the fixed prior parameters.

## 8. Posterior Transformer

The posterior represents:

```text
q(z | state context, mode, goal modulation, target action chunk)
```

It receives all five context tokens, including the command token, and the ground-truth target chunk `(B, 20, 9)`.

Each target action step is embedded with `Linear(9,128)` and receives a learned action-position embedding from `Embedding(20,128)`. A learned posterior CLS token is prepended. The posterior sequence is:

```text
[posterior_CLS, pose, velocity, wrench, mode, command, action_0, ..., action_19]
```

Its shape is `(B, 26, 128)`: one posterior CLS token, five context tokens, and twenty action tokens.

This sequence passes through a two-layer Transformer encoder with width 128, four attention heads, feed-forward width 256, GELU, and dropout 0.1. A final `LayerNorm(128)` is applied. Only the output at the posterior CLS position is used.

A linear head maps that summary to 12 logits, reshaped to:

```text
posterior logits: (B, 4 variables, 3 categories)
```

No Gaussian mean or log-variance heads are active in the current categorical configuration.

## 9. Categorical latent sampling

### 9.1 Training sample

During training, the model samples from posterior logits using:

```python
F.gumbel_softmax(logits_q, tau=1.0, hard=True, dim=-1)
```

The forward value is a hard one-hot choice for each of the four variables. The backward pass uses the straight-through soft Gumbel-Softmax gradient estimator. The four one-hot vectors are flattened from `(B,4,3)` to `(B,12)`.

### 9.2 Latent projection

Because the categorical width and `d_z` are both 12, the categorical projection is an identity operation:

```text
categorical code (B,12) -> Identity -> decoder z (B,12)
```

That 12D vector is then mapped by `Linear(12,128)` to the decoder's latent token.

### 9.3 Prior and posterior sampling helpers

For `sample=True`, inference uses categorical probabilities and `torch.multinomial` independently for every latent variable. It returns hard one-hot codes; Gumbel-Softmax is used only by the training forward path.

For `sample=False`, inference takes `argmax` in every variable. With the fixed uniform prior all logits tie, so deterministic prior inference selects category 0 for all four variables. Consequently, `get_actions_base` always decodes the same fixed prior code for a given context.

There is no sampling-temperature multiplier in the current inference helpers. `categorical_temperature=1.0` controls training-time Gumbel-Softmax only.

## 10. Action Transformer decoder

The projected latent is injected as a dedicated memory token:

```text
z (B,12) -> Linear(12,128) -> z_token (B,1,128)
memory = [z_token, pose, velocity, wrench, mode, command]
memory shape = (B,6,128)
```

The decoder target is not an autoregressive action sequence. It is a bank of 20 learned query embeddings:

```text
action_queries = Embedding(20,128) -> (B,20,128)
```

A four-layer `nn.TransformerDecoder` lets all 20 queries attend to the six memory tokens. Each decoder layer has width 128, four heads, feed-forward width 256, GELU, and dropout 0.1. There is no causal target mask, so all output queries are decoded in parallel and can self-attend to one another.

Finally:

```text
decoded queries (B,20,128) -> Linear(128,9) -> predicted actions (B,20,9)
```

The output remains in the processed dataset's normalized relative-action representation. Conversion back to absolute poses and denormalization are evaluation/rollout concerns, not part of the model forward pass.

## 11. Training objective

### 11.1 Masked L1 reconstruction

The reconstruction term is the mean absolute error over valid scalar action entries:

```text
L_recon = sum(|a_hat - a| * m) / max(sum(m), 1)
```

Because the time mask is repeated across action dimensions, a padded step contributes zero loss in all nine dimensions.

### 11.2 Categorical KL divergence

For each sample, the model computes the categorical KL and sums over all four latent variables and all categories:

```text
KL(q || p) = sum_v sum_k q(v,k) * [log q(v,k) - log p(v,k)]
```

The active prior is uniform, so this term penalizes posterior distributions that depart from uniform categorical usage. `free_bits=null`, so the per-sample KL is not clamped.

### 11.3 KL balancing and its fixed-prior consequence

With `kl_balance_alpha=0.75`, the implementation constructs:

```text
KL_balanced = 0.75 * KL(stopgrad(q) || p)
            + 0.25 * KL(q || stopgrad(p))
```

Both terms have the same numerical KL value, so the reported forward value equals the ordinary `KL(q || p)`. However, the prior is fixed and has no trainable parameters. The first term therefore carries no gradient, while the posterior receives only the second term's 0.25-scaled KL gradient. This is an important consequence of using `fixed_prior=true` together with `kl_balance_alpha=0.75`.

### 11.4 Mode-dependent beta

The model maps one-based labels linearly from `variable_beta_upper` for the lowest class to `variable_beta_lower` for the highest class. With two classes:

```text
mode 0 / label 1 / follow:  beta = 0.1
mode 1 / label 2 / leading: beta = 0.018
```

The batch KL contribution is:

```text
L_KL = mean_i(beta_i * KL_balanced_i)
```

The configured scalar `beta=0.035` is ignored while `variable_beta=true`.

### 11.5 Total loss

The active objective is:

```text
L_total = L_recon + L_KL
```

The optional diversity objective is configured but disabled with `diversity_enable=false`, so it does not alter the current loss.

## 12. Returned diagnostics

The model forward method returns:

| Key | Meaning in the active configuration |
|---|---|
| `total_loss` | Masked L1 plus weighted KL contribution |
| `l1_loss` | Masked reconstruction L1 |
| `kl` | Unweighted mean categorical KL |
| `kl_loss` | Mean per-sample beta-weighted KL |
| `beta_mean` | Mean active beta in the batch |
| `prior_entropy` | Sum of categorical entropy across four prior variables, averaged over batch |
| `posterior_entropy` | Sum of categorical entropy across four posterior variables, averaged over batch |
| `prior_std_mean` | `None` for categorical latents |
| `posterior_std_mean` | `None` for categorical latents |

## 13. Inference APIs

The model exposes three principal action methods:

### `get_actions_base(...)`

- Builds the conditioned context.
- Uses the deterministic prior mode.
- Returns `(B,20,9)`.
- Its `sample` argument is discarded.
- With the uniform fixed prior, the latent is the all-category-zero joint code.

### `get_actions_prior(..., sample=True, num_samples=N)`

- Samples `N` categorical codes from the prior, or repeats the deterministic modes when `sample=False`.
- Decodes every sample against the same context.
- Returns `(B,N,20,9)`.
- With the uniform fixed prior, all 81 joint hard codes are possible under stochastic sampling.

### `get_actions_pos(..., target_action=..., sample=True, num_samples=N)`

- Infers posterior logits using the supplied target action chunk.
- Samples or takes posterior modes.
- Returns `(B,N,20,9)`.
- This is a reconstruction/analysis path because it requires future target actions.

## 14. Shape trace for the exact defaults

The complete active forward pass has the following shapes:

```text
Input observation window                          (B, 4, 30)

Flattened pose history                            (B, 36)
Flattened velocity history                        (B, 24)
Flattened wrench history                          (B, 24)
Mode one-hot                                      (B, 2)
Flattened command history                         (B, 36)

Raw context token stack                           (B, 5, 128)
Context Transformer output                        (B, 5, 128)
Goal-conditioned adaptive output                  (B, 5, 128)

Full posterior context                            (B, 5, 128)
No-command prior context                          (B, 4, 128)
Flattened prior context                           (B, 512)
Fixed prior logits                                (B, 4, 3)

Target action chunk                               (B, 20, 9)
Embedded posterior action tokens                  (B, 20, 128)
Posterior sequence: 1 + 5 + 20 tokens             (B, 26, 128)
Posterior logits                                  (B, 4, 3)

Hard straight-through categorical sample          (B, 4, 3)
Flattened categorical sample                      (B, 12)
Identity-projected decoder latent                 (B, 12)
Latent memory token                               (B, 1, 128)

Decoder memory: latent + five context tokens      (B, 6, 128)
Learned action queries                            (B, 20, 128)
Decoded action chunk                              (B, 20, 9)
```

For prior sampling with `N` trajectories, the final output is `(B,N,20,9)`.

## 15. Training and evaluation schedule

The active run uses:

- training logging every 100 iterations;
- scalar evaluation every 500 iterations;
- evaluation plots every 2,500 iterations;
- up to 1,200 evaluation anchors;
- an evaluation anchor stride of 25;
- 39 sampled prior trajectories per evaluated condition.

Goal-conditioned evaluation divides the requested 39 samples equally across the three goal groups, producing 13 samples per group when that grouped evaluation helper is used.

The training script seeds Python and PyTorch/PyTorch Lightning with seed 42, enables deterministic cuDNN behavior, performs one optimizer step per iteration, and advances the cosine scheduler every iteration.

## 16. Configuration-dependent alternatives that are not active

The implementation also supports Gaussian latents, learned context-dependent priors, arrangement tokens or arrangement-based adaptive modulation, a tracking-error token, a CLS context token, fixed scalar beta, free bits, and a diversity term. None of those alternatives describes the exact active defaults above.

In particular, the current model must not be described as:

- a Gaussian CVAE with `mu` and `logvar`;
- a learned conditional prior `p(z | context)`;
- a six-token context containing a CLS token;
- an observation model with `(B,8,27)` inputs;
- a 30-step action-chunk model;
- a model that embeds the goal as a normal context token;
- a model with an inference sampling-temperature multiplier.

Those descriptions correspond to older configurations or inactive branches, not the current `train_bc_lowdim` defaults.
