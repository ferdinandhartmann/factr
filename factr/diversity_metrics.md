# Full Training Loss (Lowdim BC / CVAE)

This matches the implemented loss in:
- `factr/trainers/bc.py` (`BehaviorCloning.training_step`)
- `factr/models/lowdim_action_transformer.py` (`LowdimStiffnessCVAEAgent.forward`)

## Final Objective

With diversity enabled:

```text
L_train = L_base - lambda(t) * D_comb
L_base  = L_recon + beta * L_KL
D_comb  = w_pos * D_pos + w_ori * D_ori
lambda(t) = diversity_weight * min(1, (t+1)/diversity_warmup_steps)
```

With diversity disabled:

```text
L_train = L_base = L_recon + beta * L_KL
```

## Symbols

| Symbol | Meaning |
|---|---|
| `t` | Global training step |
| `beta` | KL weight (`beta`) |
| `w_pos` | Position diversity weight (`diversity_weight_position`) |
| `w_ori` | Orientation diversity weight (`diversity_weight_orientation`) |
| `D_pos` | Endpoint position diversity |
| `D_ori` | Endpoint orientation diversity (radians) |

## Base CVAE Terms

### 1) Reconstruction term (masked L1)

```text
L_recon = sum( abs(a_hat - a) * m ) / sum(m)
```

- `a_hat`: predicted action chunk
- `a`: target action chunk
- `m`: action mask

### 2) KL term

`L_KL` is KL(posterior || prior), with optional KL-balance and free-bits in code.

Gaussian latent:

```text
KL(q||p) = 0.5 * sum_k[
  log(sigma_p_k^2 / sigma_q_k^2)
  + (sigma_q_k^2 + (mu_q_k - mu_p_k)^2)/sigma_p_k^2
  - 1
]
```

Categorical latent:

```text
KL(q||p) = sum_{v,c} q[v,c] * (log q[v,c] - log p[v,c])
```

## Diversity Terms

Diversity uses `S = diversity_num_samples` prior samples per batch item and only the endpoint (last timestep) of each sampled chunk.

Action layout (`ac_dim=9`):
- `0:3` -> endpoint position `[x, y, z]`
- `3:9` -> endpoint orientation in 6D rotation representation

### 1) Endpoint position diversity

```text
d_pos(i,j) = || p_i - p_j ||_2
D_pos = mean of d_pos(i,j) over upper-triangular sample pairs and batch
```

### 2) Endpoint orientation diversity

Convert each 6D endpoint orientation to rotation matrix `R`.

```text
d_ori(i,j) = acos( clip((trace(R_i^T R_j)-1)/2, -1, 1) )
D_ori = mean of d_ori(i,j) over upper-triangular sample pairs and batch
```

## One Training Step

1. Model forward returns base loss:
   - `base_total_loss = recon + beta * kl`
2. If diversity is enabled:
   - sample prior actions
   - compute `D_pos`, `D_ori`, `D_comb`
   - compute `lambda(t)`
   - set `total_loss = base_total_loss - lambda(t) * D_comb`
3. Backprop uses `total_loss`

## Interpretation

- Larger `beta` => stronger latent regularization.
- Larger `diversity_weight` => stronger reward for endpoint spread.
- `diversity_warmup_steps` ramps diversity pressure gradually.
- `w_pos` vs `w_ori` controls position-vs-orientation emphasis.
