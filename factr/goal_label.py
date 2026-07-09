import numpy as np
import torch


def goal_label_to_group_one_hot(raw_goal):
    """Map scalar goal labels 1..9 into three groups of three goals."""
    values = np.asarray(raw_goal).reshape(-1)
    if values.size != 1:
        raise ValueError(f"Expected one goal label, got {values.size} values.")
    value = float(values[0])
    if not np.isfinite(value) or not value.is_integer():
        raise ValueError(f"Goal label must be a finite integer, got {values[0]!r}.")

    goal_label = int(value)
    if goal_label < 1 or goal_label > 9:
        raise ValueError(f"Goal label must be in [1, 9], got {goal_label}.")

    group_index = (goal_label - 1) // 3
    return np.eye(3, dtype=np.float32)[group_index]


def equal_goal_group_samples(sample_fn, goal_vectors, num_samples):
    """Sample every goal group equally while keeping the sample axis at dim 1."""
    if goal_vectors is None:
        return sample_fn(None, int(num_samples))
    if goal_vectors.ndim != 2 or goal_vectors.shape[1] != 3:
        raise ValueError(f"Expected goal_vectors shape (B, 3), got {tuple(goal_vectors.shape)}.")

    samples_per_goal = int(num_samples) // 3
    if samples_per_goal < 1:
        raise ValueError("Goal-conditioned evaluation requires num_samples >= 3.")

    batch_size = goal_vectors.shape[0]
    basis = torch.eye(3, device=goal_vectors.device, dtype=goal_vectors.dtype)
    grouped_samples = []
    for goal_index in range(3):
        condition = basis[goal_index].unsqueeze(0).expand(batch_size, -1)
        grouped_samples.append(sample_fn(condition, samples_per_goal))
    return torch.cat(grouped_samples, dim=1)


def format_real_goal_groups(goal_vectors):
    """Format the unique real grouped-goal ids represented by one plot."""
    if goal_vectors is None:
        return None
    if isinstance(goal_vectors, torch.Tensor):
        vectors = goal_vectors.detach().cpu().numpy()
    else:
        vectors = np.asarray(goal_vectors)
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError(f"Expected goal_vectors shape (B, 3), got {vectors.shape}.")
    groups = np.unique(np.argmax(vectors, axis=1) + 1)
    return "real_goal_group=" + ",".join(str(int(group)) for group in groups)
