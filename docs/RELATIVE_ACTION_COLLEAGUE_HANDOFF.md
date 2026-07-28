# Low-Dimensional Transformer: Relative-Action Implementation Handoff

## Review result

The active low-dimensional CVAE/transformer already supports fixed-anchor relative action chunks through the data loader, training evaluation, the main single-episode evaluator, and the shared plotting utilities. The transformer itself does not need a relative-action-specific architecture change: it receives and predicts normalized tensors with shape `(B, T, 9)` regardless of whether their physical meaning is absolute or relative.

The important work is at the boundaries:

1. `process_data` must construct each target pose relative to the commanded pose at the observation time.
2. Inference must denormalize the model output and reconstruct absolute commands before sending anything to the robot.
3. Evaluation must build its ground-truth targets with the same anchor and normalization as training.
4. Plotting must reconstruct both predictions and targets to absolute poses before comparing them with measured or commanded trajectories.

The intended meaning of one training sample is:

```text
observation window: states[t-W+1 : t+1]                 shape (W, obs_dim)
anchor:             commanded_pose[t]                  shape (9,)
target element k:   absolute_command[t + offset + k]   shape (9,)
relative chunk:     target element k relative to anchor
model target:       normalized relative chunk          shape (T, 9)
mask:               valid target positions             shape (T, 9) after loading
```

With the current configuration, `action_index_offset=1`, so chunk element `k=0` predicts the command at `t+1`.

## Exact relative-pose convention

Each pose is `[x, y, z, rot6d_1, ..., rot6d_6]`. `rot6d` contains the first two columns of a rotation matrix.

All elements in a chunk use the same anchor. This is not a cumulative delta representation.

```text
p_rel[k] = p_target[k] - p_command[t]
R_rel[k] = R_command[t]^T @ R_target[k]
```

Reconstruction is:

```text
p_absolute[k] = p_command[t] + p_rel[k]
R_absolute[k] = R_command[t] @ R_rel[k]
```

Do not subtract or add the six rotation values directly. Convert `rot6d` to a valid rotation matrix, compose the matrices, and convert the result back to `rot6d`.

Repository helpers:

- `factr/utils_plot.py::relative_chunk_from_absolute`
- `factr/utils_plot.py::relative_chunk_to_absolute`
- `factr/utils_plot.py::relative_chunk_to_absolute_torch`
- `factr/utils_plot.py::pose_chunks_for_plot`

## What is already implemented

### Data processing

`process_data/process_data.py` currently does the following for `action_pose_mode: relative`:

- Reads the synchronized absolute commanded pose as the raw action.
- Requires the 9D commanded pose to also be present in the observation state.
- Splits episodes before fitting relative-action normalization statistics.
- Builds a precomputed `(N, T, 9)` action tensor per episode.
- Uses `commanded_pose[t]` as the shared anchor for every target in the chunk at time `t`.
- Uses `action_index_offset` and `ac_chunk` to select future absolute commands.
- Pads past the episode end by repeating the final action and records a validity mask.
- Fits per-dimension Gaussian action statistics using only valid targets in the training split.
- Normalizes train and test relative chunks with the same training statistics.
- Stores each `(T, 9)` chunk as the transition action and stores the `(T,)` validity mask in `obs["action_mask"]`.
- Writes the representation contract into `rollout_config.yaml`.

The relevant functions are:

- `_build_relative_chunks(...)`
- `_normalize_relative_chunks(...)`
- `generate_robobuf(...)` in `process_data/utils_data_process.py`

Expected metadata:

```yaml
processing_config:
  action_pose_mode: relative
  ac_chunk: 20
  action_index_offset: 1
  relative_chunk_anchor: current_command
  relative_chunk_normalized: true
  relative_rotation: anchor_inverse_times_target
```

### Replay buffer and training

`factr/replay_buffer.py::RobobufReplayBufferLowdim` loads the precomputed chunk directly in relative mode. It intentionally does not rebuild or renormalize it. It also rejects:

- an absolute or legacy-delta dataset used with `action_chunk_mode: relative`;
- a missing relative-action normalization record;
- a mismatched `ac_chunk` or `action_index_offset`;
- a missing action mask;
- an anchor convention other than `current_command`.

It returns:

```text
obs:     (W, obs_dim)
actions: (T, 9)
mask:    (T, 9)
```

`factr/models/lowdim_action_transformer.py::LowdimStiffnessCVAEAgent` is representation-agnostic. The posterior embeds the normalized target actions, the decoder produces `(B, T, 9)`, and the masked reconstruction loss operates in normalized relative-action space. No relative-to-absolute conversion should be added inside this model.

### Training-time evaluation and plotting

`factr/task.py::BCTask` already:

- denormalizes predicted and true action chunks with the action statistics;
- extracts the current commanded pose from the denormalized observation;
- uses that commanded pose, rather than measured EE pose, as the relative anchor;
- reconstructs predictions and targets to absolute poses before trajectory plots;
- reconstructs sampled trajectories before absolute-space trajectory-variance and goal-distance calculations.

`factr/cfg/task/single_franka_lowdim.yaml` sets:

```yaml
eval_plot_pose_mode: ${action_chunk_mode}
```

This keeps training plots aligned with the training representation.

### Offline single-episode evaluation

The main evaluator, `scripts/eval_single_episode_lowdim.py`, already performs the correct relative-mode sequence:

1. Load and synchronize raw absolute actions and raw observations.
2. Build absolute future chunks using the checkpoint's `W`, `T`, and action offset.
3. Extract `commanded_pose[t]` from the unnormalized observation.
4. Convert absolute target chunks to fixed-anchor relative chunks.
5. Normalize observations and relative actions with the checkpoint dataset statistics.
6. Run the model and calculate normalized-space loss metrics.
7. Denormalize predictions and targets.
8. Reconstruct both to absolute pose chunks for plots.

`scripts/eval_single_episode_thesis.py` follows the same convention for its thesis-style plots. `scripts/eval_z_distr.py` also constructs relative posterior targets before latent-distribution analysis.

## Changes or checks the colleague still needs to make

### 1. Keep processing and training configuration identical

These values form one contract and must match exactly:

```yaml
action_pose_mode/action_chunk_mode: relative
ac_chunk: T
action_index_offset: offset
obs_window: W
pose_action_dim/ac_dim: 9
```

Current repository configuration needs attention:

- `process_data/cfg/default.yaml` currently writes `boxlift_12s4_relative_perdim_36` with `ac_chunk: 36`.
- `factr/cfg/train_bc_lowdim.yaml` currently trains from `boxlift_124_relative_perdim` with `ac_chunk: 20`.
- The existing `boxlift_124_relative_perdim` dataset metadata correctly reports `ac_chunk: 20`.
- The existing `boxlift_124_relative_perdim_36` dataset metadata correctly reports `ac_chunk: 36`.

Therefore the currently selected training dataset and `T=20` agree. However, if the newly processed `T=36` dataset is selected for training, `factr/cfg/train_bc_lowdim.yaml` must also use `ac_chunk: 36` and the exact new dataset name. The replay-buffer check will otherwise fail, as intended. Also verify whether `boxlift_12s4...` versus `boxlift_124...` is deliberate; these are different names.

Recommended improvement: define `ac_chunk` and `action_index_offset` once or validate them at startup against `rollout_config.yaml`, instead of manually duplicating them across processing and training YAML files.

### 2. Decide whether observation statistics may see the test split

Relative-action statistics are correctly fitted on valid training targets only. Observation statistics are currently fitted using `all_states_for_norm`, which contains both train and test episodes before the split buffers are written.

For a strictly leakage-free evaluation, change processing so that state normalization statistics are fitted on training episodes only and then applied unchanged to both train and test states. This is not a relative-action math error, but it matters for clean experimental reporting.

Implementation outline:

```text
split episodes
collect train state arrays
fit state statistics on train state arrays only
apply those statistics to every train and test state array
fit relative-action statistics on valid train chunk elements only
apply those statistics to every train and test relative chunk
```

The current state-normalization helpers both fit and mutate their input arrays, so this cleanup is easiest if fitting and applying are separated into two explicit operations.

### 3. Add the relative decode to real robot inference

The reviewed repository contains offline evaluators, but no clearly active low-dimensional real-robot execution loop. Any deployment/ROS inference code must add the following boundary logic.

```python
# obs_raw: (W, obs_dim), in the exact training feature order
# command_anchor: (9,), extracted before normalizing obs_raw
obs_norm = apply_state_normalization(obs_raw, state_stats)

with torch.inference_mode():
    pred_norm = model.get_actions_prior(
        {},
        obs_norm[None],
        class_labels=labels,
        arrangement_vectors=arrangement_vectors,
        goal_vectors=goal_vectors,
        sample=True,
        num_samples=1,
    )[0, 0]  # (T, 9)

pred_relative = apply_action_denormalization(pred_norm, action_stats)
pred_absolute = relative_chunk_to_absolute(
    pred_relative[None], command_anchor[None]
)[0]  # (T, 9)
```

Then execute only the configured leading portion of `pred_absolute`, observe again, obtain a new commanded-pose anchor, and replan. Never send normalized outputs or relative rotations directly to the controller.

Deployment checks:

- Use the latest controller commanded pose as the anchor, not measured robot pose.
- Extract the anchor from raw/denormalized data, never from normalized observation values.
- Keep frame, units, quaternion-to-matrix conversion, and `rot6d` column order identical to processing.
- Load `state` and `action` statistics from the checkpoint run's copied `rollout_config.yaml`.
- Check checkpoint `obs_window`, `ac_chunk`, feature flags, goal/mode conditioning, and label indexing.
- Re-anchor every replanning cycle. Do not carry the old anchor into the next prediction.
- If action chunks overlap, define explicitly how many leading commands are executed before replanning.

### 4. Report both normalized and physical-space evaluation metrics

The current loss, prior L1, action L2, and sign metrics in `factr/task.py` and the main offline evaluator are computed in normalized relative-action space. That is useful for matching the training objective, but it is not directly interpretable in metres or degrees.

For model comparison, retain the normalized metrics and add physical metrics after denormalization and absolute reconstruction:

- position MAE/RMSE in metres;
- endpoint position error in metres;
- rotation geodesic error in degrees or radians;
- optionally horizon-weighted error;
- metrics calculated only where the action mask is valid.

For relative mode, reconstruct both ground truth and predictions with the same per-sample commanded anchor before calculating absolute-space errors. Position error happens to be invariant to adding the same anchor, but absolute reconstruction is still necessary for correct orientation evaluation and for a consistent implementation.

### 5. Update or avoid incompatible evaluation utilities

The relative-aware paths are:

- `factr/task.py`
- `scripts/eval_single_episode_lowdim.py`
- `scripts/eval_single_episode_thesis.py`
- the relative-target portion of `scripts/eval_z_distr.py`

The following utilities should not be assumed to support precomputed relative chunks without review:

- `scripts/eval_single_episode_lowdim_simple.py` denormalizes and plots action values directly and does not reconstruct fixed-anchor relative chunks.
- `scripts/eval_stiffness_interpolation_pca.py` has explicit handling for legacy `delta` actions but no equivalent fixed-anchor `relative` reconstruction in the reviewed path.
- `process_data/check_buffer_plot.py` and `process_data/plot_buffer_episode.py` were written around per-transition action vectors. A relative dataset stores `(T, 9)` per transition, so these tools need chunk-aware selection, masking, denormalization, and reconstruction.
- Older scripts under `scripts/otake/` target other/older policy paths and should not be used as the reference low-dimensional deployment implementation.

For any updated plotter, the required order is:

```text
stored/model action
-> denormalize with norm_stats.action
-> reconstruct with commanded_pose[t]
-> apply action mask
-> convert rotations to RPY only for display
-> plot against absolute commanded/measured poses
```

Do not plot normalized relative rotation components as if they were Euler angles.

### 6. Make raw-episode extraction explicit for both controllers

The main evaluator has impedance/admittance topic fallback aliases, but some explicit extraction tables list only impedance topics and rely on generic dictionary flattening for configured admittance topics. Processing has explicit specs for both.

For robustness, mirror the processing topic specifications in evaluation for:

- `/cartesian_impedance_controller/ee_velocity`
- `/cartesian_admittance_controller/ee_velocity`
- both controller `tracking_error` topics;
- both controller `pose_command` topics.

This avoids depending on dictionary insertion order or accidentally concatenating additional array-valued message fields.

## Processing implementation recipe

The colleague can use this as the exact processing sequence.

1. Synchronize all required observation, action, goal, mode, and stiffness topics within each episode.
2. Preserve episode boundaries.
3. Store raw state arrays and raw absolute commanded action arrays as `(N, obs_dim)` and `(N, 9)`.
4. Split by episode into train and test sets before fitting statistics.
5. For every time `t`, construct target indices:

   ```text
   idx[t, k] = t + action_index_offset + k
   valid[t, k] = idx[t, k] < episode_length
   ```

6. Clip invalid indices only to obtain safe padding values; keep `valid=0` for them.
7. Extract `anchor[t] = commanded_pose[t]` from the raw state.
8. Convert every selected absolute target to the fixed-anchor relative representation.
9. Fit one mean and standard deviation per action dimension using only elements where `valid=1` in training episodes.
10. Clamp very small standard deviations to a safe epsilon.
11. Apply the same action transform to train, test, valid, and padded values; the mask ensures padding contributes no loss.
12. Normalize state features with training-set statistics if strict evaluation isolation is required.
13. Store the chunk, mask, labels, and state in the replay buffer.
14. Write all representation and normalization metadata to `rollout_config.yaml`.

Required assertions:

```text
states.shape == (N, obs_dim)
absolute_actions.shape == (N, 9)
commanded_poses.shape == (N, 9)
relative_chunks.shape == (N, T, 9)
action_mask.shape == (N, T)
all valid mask values are 0 or 1
training ac_chunk == processed ac_chunk
training offset == processed offset
```

## Evaluation and plotting recipe

For an offline raw episode:

1. Read raw absolute observations and absolute commanded actions.
2. Build observation windows and absolute future action chunks with the same offset as training.
3. Save the raw commanded pose at each observation time as `anchor`.
4. Convert absolute target chunks to relative chunks.
5. Normalize observation windows and relative targets.
6. Run posterior evaluation when ground-truth actions are supplied; run prior inference for deployable prediction quality.
7. Calculate training-space metrics on normalized tensors with the mask.
8. Denormalize true and predicted actions.
9. Reconstruct both with the same anchor.
10. Calculate physical-space metrics and build plots using only valid chunk elements.

For trajectory-fan plots with sampled predictions shaped `(B, S, T, 9)`, expand anchors as `(B, 1, 9)` before reconstruction so every sample and every horizon element uses the same per-window commanded pose.

## Acceptance checklist

Before giving a relative-action run to others, verify:

- [ ] `rollout_config.yaml` says `action_pose_mode: relative`.
- [ ] Dataset, train config, model, and evaluator all use the same `T`, `W`, and offset.
- [ ] State feature ordering and optional velocity/tracking-error removal match the checkpoint.
- [ ] Relative action statistics were fitted only on valid training chunk elements.
- [ ] The anchor is the raw commanded pose at time `t`.
- [ ] Every chunk element is relative to that same anchor.
- [ ] Rotation uses `R_anchor.T @ R_target` and inverse composition uses `R_anchor @ R_relative`.
- [ ] Inference denormalizes before reconstructing.
- [ ] Robot execution receives absolute poses only.
- [ ] Evaluation reconstructs both prediction and target before physical metrics and plots.
- [ ] Padding masks are used in every loss and metric.
- [ ] A reconstruction sanity check recovers the original absolute target chunk within numerical tolerance.

## Verification status of this review

The code paths and existing relative-action tests were inspected. The repository already contains checks for shared-anchor construction, train-only valid-target action normalization, replay-buffer loading, and NumPy/Torch reconstruction agreement. The tests could not be executed in the current shell because `pytest` is not installed (`pytest: command not found`).
