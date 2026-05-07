# Diversity Metrics (Lowdim BC Training)

This training setup uses only endpoint diversity terms from sampled prior actions.

Action layout (`ac_dim=9`):
- `0:3` -> endpoint position: `[x, y, z]`
- `3:9` -> endpoint orientation in 6D rotation representation: `[r1, r2, r3, r4, r5, r6]`

## 1) Endpoint Position Diversity

For each batch item, sample `S` action chunks and take the final-step endpoint position of each sample.
For every pair `(i, j)`:

`d_pos(i,j) = || p_i - p_j ||_2`

The logged metric is the average of `d_pos(i,j)` over all upper-triangular sample pairs and batch elements.

Meaning:
- Units are meters (same as `x,y,z` units in the dataset/action space).
- If the metric is `0.5`, sampled endpoints differ by about `0.5 m` on average.

## 2) Endpoint Orientation Diversity

For each sample endpoint, convert 6D rotation to a rotation matrix `R`.
For every pair `(i, j)`, compute geodesic angle:

`d_ori(i,j) = acos( clamp((trace(R_i^T R_j)-1)/2, -1, 1) )`

The logged orientation diversity is the average of `d_ori(i,j)` over all upper-triangular sample pairs and batch elements.

Meaning:
- Units are radians in `train/endpoint_orientation_diversity_rad`.
- `train/endpoint_orientation_diversity_deg` is the same value converted to degrees.
- If orientation diversity is `0.5`, that is `0.5 rad ~= 28.65 deg` average orientation gap.

## Combined Train Diversity Term

`diversity_combined = w_pos * endpoint_position_diversity + w_ori * endpoint_orientation_diversity_rad`

`diversity_loss_term = diversity_coeff * diversity_combined`

Training loss:

`total_loss = base_total_loss - diversity_loss_term`

So larger endpoint spread (position/orientation) lowers the training loss.
