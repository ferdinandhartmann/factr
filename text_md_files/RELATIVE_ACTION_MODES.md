# Low-Dimensional Action Modes

Set the action representation once in `factr/cfg/train_bc_lowdim.yaml`:

```yaml
action_chunk_mode: relative_chunks  # absolute, relative_timesteps, relative_chunks
```

The same value is used for training, evaluation, and plotting.

## Modes

- `absolute`: Every action is an absolute commanded 9D EE pose.
- `relative_timesteps`: Each action is relative to the action immediately before it. An absolute trajectory is recovered with a cumulative sum.
- `relative_chunks`: Every action in a chunk is relative to the **current commanded pose at time `t`**. All `T` actions use this same anchor; they are not relative to each other.

For `relative_chunks`, position is encoded as:

```text
relative_position[k] = target_position[k] - command_position[t]
```

Orientation uses geometric relative rotation:

```text
relative_rotation[k] = command_rotation[t]^T @ target_rotation[k]
```

Evaluation and plotting reverse these operations to recover absolute commanded poses.

## Dataset requirement

`relative_chunks` requires a dataset processed with:

```yaml
action_pose_mode: absolute
```

The replay buffer denormalizes the absolute actions and current commanded pose before creating relative chunks. It rejects timestep-delta datasets to prevent mixing incompatible representations.
