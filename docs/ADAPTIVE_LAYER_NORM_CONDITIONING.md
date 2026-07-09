# Adaptive LayerNorm Conditioning

The low-dimensional CVAE policy now supports `use_adaptive_layer_norm`.

When this option is `false` (the default), enabled arrangement and goal conditions are embedded as separate transformer tokens, preserving the previous behavior and checkpoint shapes.

When it is `true`, enabled arrangement and/or goal vectors no longer add tokens. Instead, they are concatenated and passed through a small MLP that predicts a scale and shift for the final context LayerNorm:

```text
output = LayerNorm(context) * (1 + scale(condition)) + shift(condition)
```

The modulation output is initialized to zero, so training starts with ordinary LayerNorm behavior.

## Configuration

Set the options in `factr/cfg/train_bc_lowdim.yaml`:

```yaml
use_arrangement_conditioning: true
goal_label: true
use_adaptive_layer_norm: true
use_stiffness_goal_adaln_gate: true
goal_adaln_gate_min: 0.1
```

The existing condition flags still control which data is required:

- Arrangement conditioning expects vectors with shape `(B, 9)`.
- Goal conditioning expects vectors with shape `(B, 3)`.
- If both are enabled, arrangement is concatenated before goal.
- Enabling adaptive LayerNorm without either condition raises an error.

With `use_stiffness_goal_adaln_gate`, stiffness controls only the goal portion of
the AdaLN input. The lowest stiffness class scales the goal vector by
`goal_adaln_gate_min`; the highest class scales it by `1.0`, with intermediate
classes linearly interpolated. In the two-class mode override, mode 0/class 1 is
therefore weakly goal-conditioned and mode 1/class 2 is fully goal-conditioned.
This option requires stiffness conditioning, goal labels, and AdaLN.

The replay-buffer, trainer, evaluation, and rollout interfaces are unchanged.
