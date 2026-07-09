# Relative-Mode Inference

This guide describes the low-dimensional policy mode:

```yaml
action_chunk_mode: relative
```

## Meaning

For an observation at time `t`, the policy predicts an action chunk with shape `(T, 9)`:

```text
[relative_x, relative_y, relative_z, relative_rot6d_1, ..., relative_rot6d_6]
```

Every pose in the chunk is relative to the same current commanded pose at time `t`. The poses are **not** relative to the previous chunk element.

With `action_index_offset: 1`, element `k` represents the absolute command at:

```text
target_time = t + 1 + k
```

## Required anchor

Use the robot's current **commanded EE pose**, not its measured EE pose, as the anchor:

```text
anchor_pose = current_commanded_pose[t]  # shape (9,)
```

The pose uses XYZ followed by the first two columns of the rotation matrix (`rot6d`). It must use the same convention and frame as the training dataset.

## Model inputs

Build the observation window with shape `(B, W, obs_dim)` and normalize it using `norm_stats.state` from the training run's `rollout_config.yaml`.

The model returns normalized relative actions with shape `(B, T, 9)`. Do not treat these values as metres or valid rotation columns yet.

## Decode predictions

First denormalize the complete action chunk using `norm_stats.action`:

```text
relative_action = normalized_prediction * action_std + action_mean
```

Then convert each relative pose to an absolute command.

Position:

```text
absolute_xyz[k] = anchor_xyz + relative_xyz[k]
```

Rotation:

```text
R_anchor      = rot6d_to_matrix(anchor_rot6d)
R_relative[k] = rot6d_to_matrix(relative_rot6d[k])
R_absolute[k] = R_anchor @ R_relative[k]
absolute_rot6d[k] = first_two_columns(R_absolute[k])
```

The inverse convention used during training is:

```text
R_relative[k] = R_anchor.T @ R_target[k]
```

The repository implementation is `relative_chunk_to_absolute()` in `factr/utils_plot.py`.

## Execution

For receding-horizon inference:

1. Read the latest observation window and current commanded pose.
2. Normalize the observation window.
3. Predict one normalized `(T, 9)` chunk.
4. Denormalize the chunk with the relative-action statistics.
5. Convert the whole chunk to absolute poses using one shared commanded-pose anchor.
6. Execute the desired number of leading absolute commands.
7. Observe again, obtain a new current commanded-pose anchor, and predict a new chunk.

Never cumulatively sum the elements of a `relative` chunk. Cumulative reconstruction belongs to `delta` mode.

## Compatibility checks

Before inference, verify the training `rollout_config.yaml` contains:

```yaml
processing_config:
  action_pose_mode: relative
  relative_chunk_anchor: current_command
  relative_chunk_normalized: true
  relative_rotation: anchor_inverse_times_target
  ac_chunk: 20
  action_index_offset: 1
```

The inference code must use the checkpoint's `obs_window`, `ac_chunk`, observation ordering, tracking-error setting, stiffness/arrangement conditioning, and normalization statistics. A mismatch can produce plausible-looking but incorrect trajectories.

## Difference from delta mode

```text
relative: target[k] is relative to command[t] for every k
delta:    target[k] is relative to target[k-1]
```

For `relative`, use fixed-anchor addition and rotation composition. For `delta`, use cumulative reconstruction starting from the current commanded pose.
