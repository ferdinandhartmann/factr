# FACTR Low-Dim Component Deep Dives and Architecture Comparison

This companion document focuses on the exact modules you asked for:

- `factr/models/lowdim_action_transformer.py`
- `factr/agent.py`
- `factr/task_obs_pred.py`
- `factr/models/lowdim_obs_mlp.py` (interpreting your “low_dims_mlp” request)
- comparison: low-dim action transformer vs `factr/models/action_transformer.py`

The style here is deliberately explanatory (not only bullets), so you can read it linearly like a mini technical walkthrough.

---

## 1) `lowdim_action_transformer.py`: the active low-dim CVAE policy core

The heart of the current low-dim policy is `LowdimStiffnessCVAEAgent`.

Conceptually, it does three things:

1. **Encodes context from observation windows and stiffness label** into semantic tokens.
2. **Learns a latent variable `z`** with a CVAE objective (prior vs posterior).
3. **Decodes an action chunk** using transformer decoding conditioned on context + latent.

### Important classes/functions

### `_PosteriorTransformer`

This is a dedicated posterior encoder for `q(z | context, action_chunk)`.
It converts ground-truth target actions into action tokens, appends positional embeddings, prepends a posterior-CLS token, and transformer-encodes the concatenated sequence.
The final summary token is projected to:

- `(mu, logvar)` for Gaussian latent, or
- categorical logits `(num_variables, num_categories)`.

You can think of it as the “teacher pathway” that reads the true future chunk during training.

### `LowdimStiffnessCVAEAgent._build_context_tokens`

This method is particularly important because it hardcodes the semantic decomposition of observation dimensions:

- pose / velocity / wrench / tracking each get their own encoder.
- stiffness class gets an embedding token.
- CLS-like token summarizes all observation group tokens.

Those 6 tokens become the context memory for prior and decoder.

### `LowdimStiffnessCVAEAgent._prior` and `_compute_kl`

- `_prior` defines `p(z|context)` (learned or fixed).
- `_compute_kl` computes KL with optional KL balancing (`kl_balance_alpha`) and free-bits.

If your training ever shows posterior collapse (KL near zero, diversity collapse), this area is where you should focus first.

### `LowdimStiffnessCVAEAgent._decode_actions`

`z` is first converted to token space and added as a query bias to learned action query embeddings.
Then the transformer decoder cross-attends to context tokens and outputs one token per action step, finally projected to action dimensions.

This “query + latent bias” approach keeps decoding parallel and chunk-oriented.

### Inference APIs

- `get_actions_base(...)`: deterministic decode from prior mode.
- `get_actions_prior(...)`: sample/mode from prior, supports multi-sample output `(B,S,T,D)`.
- `get_actions_pos(...)`: sample from posterior with a target action.
- `get_uncertainty_entropy(...)`: derives uncertainty proxy from multi-sample spread.

These methods are used by evaluation scripts and tasks to compute fan plots/diversity.

---

## 2) `agent.py`: shared image+obs tokenization framework (historical/general)

`agent.py` provides generic agent scaffolding used strongly by image-based transformer policies, and less by the purely low-dim one.
Still, it is an important architectural layer in FACTR.

### `BaseAgent`

`BaseAgent` defines a reusable tokenization pipeline:

- visual feature extraction per camera,
- optional observation injection strategy (`add_token` or `pad_img_tokens`),
- optional curriculum operators in pixel or latent space (blur/downsample),
- post-processing normalization/projection.

This class also centralizes camera-sharing behavior and early-fusion behavior.

In short: `BaseAgent` is the framework for assembling multimodal tokens before policy heads.

### `MLPAgent`

`MLPAgent` inherits `BaseAgent`, flattens tokenized observations, runs an MLP, then routes to a policy distribution head.
It is a simpler baseline architecture compared to transformer CVAE variants.

Even if you do not currently train with `MLPAgent`, reading it helps understand the expected policy interface:

- `forward(...)` returns training loss.
- `get_actions(...)` returns sampled/deterministic actions.

---

## 3) `task_obs_pred.py`: evaluation-heavy task logic for observation prediction

`ObsPredictionTask` is the task wrapper for low-dim observation prediction training/eval.
It has richer evaluation than a minimal dataloader wrapper.

At runtime:

- it builds train/test dataloaders,
- runs evaluation over `LowdimGaussianObsMLP` outputs,
- computes prediction metrics (`sample_mse`, `mean_mse`, `nll_per_elem`, variance, tracking error),
- optionally runs goal inference metrics if model supports it,
- builds diagnostic visualizations (fan plots, collapsed timeline plots, goal-by-goal plots).

A notable design choice is that `task_obs_pred.py` keeps a lot of plotting/metric logic close to the task boundary rather than embedding it inside the model. This makes model code cleaner and keeps experiment diagnostics configurable at task level.

---

## 4) `lowdim_obs_mlp.py` (your “low_dims_mlp”): probabilistic obs predictor + goal inference

`LowdimGaussianObsMLP` predicts future observations conditioned on:

- past observation window,
- planned action chunk,
- stiffness class,
- goal class.

### Core architecture

It is an MLP that outputs both mean and variance for each timestep/dimension in the prediction horizon.
So output is a Gaussian parameterization, not only point estimate.

### Why this is useful

It serves dual roles:

1. **Observation prediction** (forward model-ish within low-dim space).
2. **Goal inference** via likelihood over goal-conditioned predictions.

### Important methods

- `_build_features(...)`: shape validation + flatten + one-hot conditioning.
- `_predict_distribution(...)`: returns mean/var/std tensors.
- `forward(...)`: training objective = deterministic fit + weighted NLL.
- `infer_goals(...)`: evaluates all candidate goals, computes posterior over goals from observation likelihood.
- `compute_goal_posterior_over_time(...)`: temporal posterior evolution (great for diagnostics).

If you need explainable goal uncertainty from low-dim trajectories, this class is where most of that capability lives.

---

## 5) Comparison: `lowdim_action_transformer.py` vs `action_transformer.py`

Both are CVAE-flavored action models, but they differ in data modality assumptions and internal structure.

## 5.1 Shared ideas

Both models use:

- conditional latent variable `z` with prior/posterior,
- chunked action decoding,
- reconstruction + KL objective,
- multi-sample prior decoding for diversity/uncertainty analysis.

So conceptually they live in the same family.

## 5.2 Key differences

### Input modality and token source

- `TransformerAgent` in `action_transformer.py` inherits `BaseAgent`, so it expects image/obs tokenization pipeline and often uses camera features.
- `LowdimStiffnessCVAEAgent` is pure low-dim and **does not** depend on visual backbone features.

### Context semantics

- Low-dim variant explicitly partitions 27-dim state into pose/vel/wrench/tracking groups and adds stiffness token.
- Original action transformer treats context tokens as outputs from a generic tokenization process and extracts `[CLS, FORCE]` style features for prior.

### Latent family support

- `TransformerAgent` is Gaussian CVAE.
- `LowdimStiffnessCVAEAgent` supports both Gaussian and categorical latent families.

### Decoder coupling

- In `TransformerAgent`, `z` is injected as an extra memory token through `_ACT` mechanism.
- In low-dim model, `z` biases action queries before decoder.

Both are valid designs; low-dim implementation is more explicit about structured state groups.

### Strict shape contracts

- Low-dim model enforces `obs_dim=27`, `ac_dim=9` (as currently implemented) and raises early on mismatch.
- Original transformer is more general around tokenized visual inputs.

### Practical implication

For the active low-dim workflow, `LowdimStiffnessCVAEAgent` is better aligned and easier to reason about because it directly encodes the state semantics used in your processed dataset.

---

## 6) Medium-importance classes worth knowing

Even if not always edited, these classes are useful to understand the full flow:

- `factr.trainers.bc.BehaviorCloning`: training step contract and logging adapter.
- `factr.task.BCTask`: evaluation metrics, diversity checks, fan-plot orchestration.
- `factr.replay_buffer.RobobufReplayBufferLowdim`: episode-aware sample builder for action policy.
- `factr.replay_buffer.RobobufReplayBufferObsPredLowdim`: episode-aware sample builder for obs prediction.
- `factr.trainers.obs_pred.GaussianObsPredictionTrainer`: trainer adapter for obs prediction model.

If you understand these plus the core model classes, you can trace almost any low-dim experiment end-to-end.

---

## 7) How to see model structure from code and runtime

A pragmatic workflow:

1. **Read config yaml first** to know the exact constructor kwargs.
2. **Open `__init__` of model class** and map each submodule (encoders, prior, posterior, decoder, heads).
3. **Read `forward(...)` in call-order** and annotate tensor shapes.
4. **At runtime print model + parameter names**.

Minimal runtime snippet:

```python
import hydra
from omegaconf import OmegaConf

cfg = OmegaConf.load('factr/cfg/train_bc_lowdim.yaml')
model = hydra.utils.instantiate(cfg.agent)
print(model)
for n, p in model.named_parameters():
    if p.requires_grad:
        print(n, tuple(p.shape))
```

And for behavior sanity:

- inspect `total_loss`, `l1_loss`, `kl`, entropy stats,
- inspect prior sample diversity in eval,
- verify that masks and chunk lengths match expected config.

---

## 8) Final note

If your experiments remain in low-dim BC/CVAE, prioritize documentation and modifications around:

- `lowdim_action_transformer.py`
- `replay_buffer.py` low-dim classes
- `task.py` and `trainers/bc.py`

If you shift into obs prediction or goal inference, then additionally prioritize:

- `lowdim_obs_mlp.py`
- `task_obs_pred.py`
- obs-pred task/trainer configs.

That split mirrors the actual active workflows in this codebase.

---

## 9) Method-by-method detail: `lowdim_action_transformer.py`

This section is a denser reference for everyday development.

### Helper math functions

- `_kl_diag_gaussians(...)`: diagonal Gaussian KL per sample.
- `_kl_categorical(...)`: KL for categorical latent variables over `(num_variables, num_categories)` axes.
- `_reparameterize(...)`: Gaussian reparameterization trick.
- `_categorical_entropy(...)`: entropy metric for monitoring latent uncertainty.

### `_PosteriorTransformer` internals

- `action_embed`: projects action dim (`ac_dim=9`) to token dim.
- `action_pos_embed`: learned position embeddings for each chunk step.
- `post_cls`: learned summary token for posterior extraction.
- `encoder`: TransformerEncoder over `[post_cls | context_tokens | action_tokens]`.
- output head:
  - Gaussian mode: `mu`, `logvar`
  - categorical mode: flattened logits reshaped to `(B, n_var, n_cat)`.

### `LowdimStiffnessCVAEAgent` internal subsystems

1. **State grouping and tokenization**
   - `state_slices` maps semantic state sub-vectors.
   - per-group encoder MLPs convert flattened windows to token vectors.
2. **Context encoder**
   - refines `[cls, pose, vel, wrench, track, stiffness]` tokens.
3. **Prior network**
   - either fixed distribution or learned from `z_context` summary.
4. **Posterior network**
   - `_PosteriorTransformer` reading context + target action chunk.
5. **Latent bridge**
   - optional categorical latent projection, then `z_to_token`.
6. **Action decoder**
   - learned query embeddings + transformer decoder + linear action head.

### Most important private methods by purpose

- Input validation: `_prepare_obs`, `_normalize_labels`, `_reshape_actions`
- Context build: `_build_context_tokens`, `_build_z_context`
- Latent logic: `_prior`, `_compute_kl`, `_sample_train_latent`, `_sample_latent_batch`, `_prepare_decoder_latent`
- Decode: `_decode_actions`
- Monitoring: `_latent_metrics`

### Public training/inference interface

- `forward`: returns a rich loss dict used directly by trainer.
- `get_actions_base`: deterministic base output (often baseline-like behavior).
- `get_actions_prior`: multi-sample prior predictions for rollout uncertainty/diversity.
- `get_actions_pos`: posterior-conditioned reconstruction path.
- `get_uncertainty_entropy`: uncertainty estimate from action sample dispersion.

---

## 10) Method-by-method detail: `agent.py`

Although not the main low-dim active model file, `agent.py` defines the reusable language many models inherit.

### `_BatchNorm1DHelper`

Convenience wrapper allowing batch-norm use on both `(B,D)` and `(B,T,D)` tensors by transposing when needed.

### `BaseAgent` constructor concepts

- camera feature sharing (`share_cam_features`) vs per-camera copies,
- observation token strategy (`use_obs`) deciding how state enters token stream,
- optional projection to `token_dim`,
- feature norm choice (`batch_norm`, `layer_norm`, identity).

### `tokenize_obs`

This method does more than tokenization:

- computes curriculum scale over global steps,
- optionally applies blur/downsample in pixel or latent spaces,
- applies image dropout,
- appends or concatenates observation-derived representation,
- runs final post-processing block.

If token statistics look wrong in visual pipelines, start debugging here.

### `embed`

Handles camera/time packing rules:

- early-fusion path,
- per-time-step embedding path,
- shared vs per-camera encoders,
- final token concatenation across cameras.

### `MLPAgent`

Adds shared MLP trunk and policy head on top of flattened tokens.
Useful as a strong baseline when transformer complexity is unnecessary.

---

## 11) Method-by-method detail: `task_obs_pred.py`

`ObsPredictionTask.eval` is long because it is both evaluator and diagnostics engine.

### What happens each eval batch

1. Move tensors to device.
2. Run model forward for prediction losses.
3. Compute tracking error (`compute_tracking_error`) and aggregate masked norms.
4. If supported, run `infer_goals` for goal posterior metrics.
5. Collect plotting candidates with episode metadata for timeline-consistent figures.

### Metrics families collected

- prediction fit: sample MSE, mean MSE, NLL-per-element,
- uncertainty stats: predicted variance,
- control-relevance: tracking error norm,
- goal inference quality:
  - accuracy,
  - true-goal log-likelihood,
  - normalized log-likelihood per valid element,
  - margin vs best competing goal,
  - posterior entropy,
  - posterior probability normalization error.

### Plotting pipeline highlights

- selects candidates at configurable stride/max-step,
- builds trajectory fan plots for predicted observations,
- collapses chunks into timeline summaries with uncertainty bands,
- optionally builds goal-by-goal comparison figures.

This makes `task_obs_pred.py` a great source when designing experiment diagnostics.

---

## 12) Method-by-method detail: `lowdim_obs_mlp.py`

### Data contract

Inputs:

- `obs_window`: `(B,W,input_obs_dim)`
- `action_chunk`: `(B,H,pose_action_dim)`
- class labels: stiffness and goal

Outputs:

- Gaussian params for each horizon step and observation dimension,
- optional training losses,
- optional inferred goal posterior.

### Core internal steps

1. normalize labels to robust class indices,
2. one-hot encode class conditions,
3. flatten and concatenate all condition vectors,
4. MLP predicts `[mean, raw_var]` pairs,
5. softplus variance floor for stable std,
6. sample via reparameterized Gaussian distribution.

### Training loss detail

Loss mixes:

- deterministic fit (`mean_mse`) for strong point prediction,
- small weighted NLL to keep variance head calibrated.

This helps avoid pathological “variance floor collapse” while preserving mean accuracy.

### Goal inference detail

`infer_goals` evaluates every goal class, computes goal-conditioned likelihoods, then combines with optional prior to produce posterior distributions.
It also exposes posterior-over-time, which is particularly valuable for episode-level interpretability.

---

## 13) Extended comparison table (quick decision aid)

| Axis | `LowdimStiffnessCVAEAgent` | `TransformerAgent` (`action_transformer.py`) |
|---|---|---|
| Primary modality | low-dim state windows | image/obs token pipeline |
| Base class | `nn.Module` (standalone) | `BaseAgent` inheritance |
| Context tokens | explicit semantic state groups + stiffness | generic tokenized context |
| Latent family | Gaussian + Categorical | Gaussian |
| z injection | query bias in decoder target | extra memory token in ACT |
| Hard shape assumptions | explicit (`obs_dim=27`, `ac_dim=9`) | broader tokenized settings |
| Active workflow fit | highest for current repo path | legacy/alternate path |

### Practical takeaway

If your dataset is already low-dim and semantically structured (pose/velocity/wrench/tracking), the lowdim model minimizes abstraction mismatch and tends to be easier to debug end-to-end.
