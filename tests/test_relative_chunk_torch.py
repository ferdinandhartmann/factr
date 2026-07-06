import numpy as np
import torch

from factr.utils_plot import relative_chunk_to_absolute, relative_chunk_to_absolute_torch


def test_relative_chunk_to_absolute_torch_matches_numpy():
    rng = np.random.default_rng(7)
    actions = rng.normal(size=(3, 5, 4, 9)).astype(np.float32)
    anchors = rng.normal(size=(3, 9)).astype(np.float32)

    expected = relative_chunk_to_absolute(actions, anchors[:, None, :])
    actual = relative_chunk_to_absolute_torch(torch.from_numpy(actions), torch.from_numpy(anchors))

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_relative_chunk_to_absolute_torch_matches_rotation_fallbacks():
    actions = np.zeros((1, 2, 3, 9), dtype=np.float32)
    anchors = np.zeros((1, 9), dtype=np.float32)
    actions[0, 0, 0, 3:9] = np.nan
    actions[0, 1, 1, 3:9] = np.array([1, 0, 0, 2, 0, 0], dtype=np.float32)

    expected = relative_chunk_to_absolute(actions, anchors[:, None, :])
    actual = relative_chunk_to_absolute_torch(torch.from_numpy(actions), torch.from_numpy(anchors))

    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_relative_chunk_to_absolute_torch_preserves_device_and_dtype():
    actions = torch.randn(2, 3, 4, 9, dtype=torch.float64)
    anchors = torch.randn(2, 9, dtype=torch.float64)

    result = relative_chunk_to_absolute_torch(actions, anchors)

    assert result.shape == actions.shape
    assert result.device == actions.device
    assert result.dtype == actions.dtype
