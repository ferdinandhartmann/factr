# World Model (RSSM) for FACTR Buffers

This folder contains a **low‑dimensional RSSM** world model trained on FACTR replay buffers (`buf.pkl`).  
It mirrors Dreamer’s world‑model structure but is **offline** and **state‑based** (no images).

## What’s included

- **Data loader**: `data/buffer_dataset.py`
  - Reads `robobuf` `ReplayBuffer` from `buf.pkl`
  - Splits into episodes using `is_first`
  - Builds fixed‑length sequences
  - Uses Dreamer-style alignment: `L` actions with `L+1` observations so `action[t]` predicts `obs[t+1]`
  - Optional normalization using `rollout_config.yaml`
- **Model**: `models/rssm.py`
  - Continuous stochastic latent (`Normal`)
  - GRU deterministic state
  - Prior and posterior networks
  - MLP decoder for low‑dim obs
  - Variable observation window (`obs_window`)
- **Training**: `train.py`
  - Hydra‑driven config
  - WandB logging
  - Checkpointing to `checkpoints/<run_name>/` (where `run_name = logging.name`)
  - Each run folder also stores `config.yaml`, `train_logging.yaml`, and (if provided) `rollout_config.yaml`
- **Evaluation**: `eval_rollout.py`
  - Posterior vs prior MSE
  - Optional open‑loop error plots
- **Buffer inspection**: `inspect_buffer.py`

## Quick start

1) **Inspect the buffer**

```bash
python factr/world_model/inspect_buffer.py --buffer-path /path/to/buf.pkl
```

2) **Train**

```bash
python factr/world_model/train.py
```

3) **Evaluate + plot**

```bash
python factr/world_model/eval_rollout.py eval.checkpoint_path=checkpoints/<run_name>/rssm_step_10000.pt
```

4) **Plot a single sequence (true vs predicted)**

```bash
python factr/world_model/plot_sequence.py --checkpoint-path checkpoints/<run_name>/rssm_step_10000.pt --seq-index 0 --seq-len 128
```

## Config layout (Hydra)

- `cfg/train_defualt.yaml` → data + model + training + logging in one file (buffer path, sizes, steps, lr, `logging.name`, etc.)
- `cfg/eval.yaml` → evaluation + plotting
- `inspect_buffer.py` has CLI args (no YAML)
- `plot_sequence.py` has CLI args (no YAML)

## Observation window

Set `model.obs_window` to control how many recent observations are concatenated for the posterior:

```bash
python factr/world_model/train.py model.obs_window=4
```

If `obs_window=1` (default), it uses only the current observation.

## WandB

WandB is on by default. Disable with:

```bash
python factr/world_model/train.py logging.enabled=false
```

## Plots

`eval_rollout.py` can save per‑dimension error curves:

- `eval.plot.enabled`: toggle plots
- `eval.plot.output_dir`: folder name
- `eval.plot.max_dims`: number of dims to plot

Example:

```bash
python factr/world_model/eval_rollout.py eval.plot.max_dims=8
```

Plots are saved into the Hydra run directory (e.g. `outputs/YYYY-MM-DD/HH-MM-SS/plots/`).

For single‑sequence plots you can override `--dims` to pick specific dims:

```bash
python factr/world_model/plot_sequence.py --dims 0 3 5
```

## How the model works (medium detail)

This implementation follows Dreamer‑style world modeling but is adapted to **offline, low‑dimensional** FACTR data. The flow is:

```mermaid
flowchart TD
  A[ReplayBuffer buf.pkl] --> B[Sequence dataset]
  B --> C[Observation o_t]
  C -->|optional| D[VAE encoder]
  D --> E[Latent z_vae]
  C -->|if no VAE| F[Obs embed]
  E --> G[RSSM posterior q(z_t|h_t,o_t)]
  F --> G
  G --> H[Sample z_t]
  H --> I[GRU h_t]
  I --> J[Prior p(z_t|h_t)]
  H --> K[Decoder p(o_t|h_t,z_t)]
  K --> L[Recon loss]
  G --> M[KL loss]
  D --> N[VAE decoder]
  N --> O[VAE recon loss]
  D --> P[VAE KL loss]
```

If Mermaid isn’t supported in your viewer, here’s a cleaner ASCII version:

```
buf.pkl
  |
  v
sequence dataset  --->  o_t  ----------------------------------------------+
                      |                                                   |
                      | (optional VAE)                                    |
                      v                                                   |
                 VAE encoder --> z_vae --> RSSM posterior q(z_t|h_t,o_t)   |
                      |                     |                             |
                      |                     v                             |
                      |                  sample z_t                        |
                      |                     |                             |
                      |                     v                             |
                      |                  GRU state h_t --> prior p(z_t|h_t)|
                      |                     |                             |
                      |                     v                             |
                      +--> VAE decoder <-- RSSM decoder p(o_t|h_t,z_t) ----+
                           |                  |
                           v                  v
                       VAE recon loss     RSSM recon loss
                       VAE KL loss        RSSM KL loss
```

1) **Buffer → sequences**
   - `data/buffer_dataset.py` loads `buf.pkl` and builds fixed‑length `(obs, action)` sequences.
   - Episodes are split using `is_first` to avoid crossing boundaries.
   - Optional normalization uses `rollout_config.yaml` statistics.

2) **(Optional) VAE encoder/decoder**
   - `models/vae.py` encodes each observation into a latent vector and reconstructs it back.
   - In training, if `model.use_vae=true`, the RSSM operates on **VAE latents**, while a reconstruction + KL loss is added for the VAE.
   - The VAE is a simple MLP encoder/decoder (no images here).

3) **RSSM dynamics**
   - `models/rssm.py` runs a deterministic GRU state `h_t` and a stochastic Gaussian state `z_t`.
   - The **prior** predicts `p(z_t | h_t)` from the GRU state.
   - The **posterior** predicts `q(z_t | h_t, embed(o_t))` using the current observation (or VAE latent).
   - The decoder reconstructs the observation (or latent) from `[h_t, z_t]`.

4) **Losses**
   - RSSM: reconstruction MSE + KL(q || p) with optional free‑nats.
   - VAE (if enabled): reconstruction MSE + latent KL.
   - All losses are logged to WandB in `train.py`.

5) **Evaluation & plots**
   - `eval_rollout.py` compares posterior recon vs. prior rollout error.
   - `plot_sequence.py` visualizes true vs predicted trajectories for a single sequence.

Where it happens in code:
 - Data loading/sequence creation: `data/buffer_dataset.py`
 - VAE encoder/decoder + KL: `models/vae.py`
 - RSSM prior/posterior/decoder: `models/rssm.py`
 - Loss composition + training loop: `train.py`
 - Evaluation + plotting: `eval_rollout.py`, `plot_sequence.py`

## Notes

- The current buffer uses low‑dim `state` (+ optional `goals`).
- A small MLP VAE is enabled by default (`cfg/model/rssm.yaml`), encoding obs into a latent used by the RSSM.
  - Disable for training: `model.use_vae=false`
  - Disable for plotting: `python factr/world_model/plot_sequence.py --no-vae`
- The current buffer is already Gaussian‑normalized; leave `data.normalize_obs=false` and `data.normalize_action=false` unless you intentionally want to re‑normalize.


### Training Settings

  - seq_len=64: long enough to capture dynamics but not so long that training becomes unstable or slow.
  - batch_size=64: good gradient signal for low‑dim data; still fits most GPUs.
  - lr=3e-4: standard for Adam with RSSM‑style models; higher often destabilizes KL.
  - free_nats=1.0: prevents KL collapse early so the stochastic state actually carries info.
  - stoch_dim=32, deter_dim=128: enough capacity for a 27‑dim state without overfitting.
  - use_vae + latent_dim=16: compresses obs to a smoother latent, making RSSM dynamics easier to learn.
