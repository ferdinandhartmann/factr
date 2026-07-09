# Low-Dimensional Action Modes

Set the action representation once in `factr/cfg/train_bc_lowdim.yaml`:

```yaml
action_chunk_mode: relative  # absolute, delta, relative
```

The same value is used for training, evaluation, and plotting.

## Modes

- `absolute`: Every action is an absolute commanded 9D EE pose.
- `delta`: Each action is relative to the action immediately before it. An absolute trajectory is recovered with a cumulative sum.
- `relative`: Every action in a chunk is relative to the **current commanded pose at time `t`**. All `T` actions use this same anchor; they are not relative to each other.

For `relative`, position is encoded as:

```text
relative_position[k] = target_position[k] - command_position[t]
```

Orientation uses geometric relative rotation:

```text
relative_rotation[k] = command_rotation[t]^T @ target_rotation[k]
```

Evaluation and plotting reverse these operations to recover absolute commanded poses.

## Dataset requirement

`relative` requires a dataset processed with:

```yaml
action_pose_mode: relative
ac_chunk: 20
action_index_offset: 1
relative_chunk_normalized: true
```

Processing creates each chunk from raw absolute poses, fits action statistics on valid training targets only, and stores normalized `(T, 9)` chunks with padding masks. The replay buffer loads them directly and rejects mismatched chunk lengths, offsets, or action modes.
