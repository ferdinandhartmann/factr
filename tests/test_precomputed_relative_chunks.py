import pickle

import numpy as np
import yaml

from factr.replay_buffer import RobobufReplayBufferLowdim
from factr.utils import apply_grouped_transform
from factr.utils_plot import relative_chunk_to_absolute
from process_data.process_data import _build_relative_chunks, _normalize_relative_chunks
from process_data.utils_data_process import generate_robobuf


IDENTITY_ROT6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _pose(x, y=0.0, z=0.0):
    return np.concatenate([np.array([x, y, z], dtype=np.float32), IDENTITY_ROT6])


def test_precomputed_chunks_share_anchor_and_preserve_end_mask():
    actions = np.stack([_pose(float(i)) for i in range(4)])
    states = np.zeros((4, 12), dtype=np.float32)
    states[:, 3:12] = np.stack([_pose(10.0 + i) for i in range(4)])
    traj = {"states": states, "actions": actions.copy(), "num_steps": 4}

    _build_relative_chunks([traj], slice(3, 12), ac_chunk=3, action_index_offset=1)

    assert traj["actions"].shape == (4, 3, 9)
    np.testing.assert_array_equal(
        traj["action_mask"],
        np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32),
    )
    np.testing.assert_allclose(traj["actions"][0, :, 0], [-9.0, -8.0, -7.0], atol=1e-6)

    reconstructed = relative_chunk_to_absolute(traj["actions"][0:1], states[0:1, 3:12])[0]
    np.testing.assert_allclose(reconstructed, actions[1:4], atol=1e-6)


def test_relative_normalization_uses_only_valid_training_targets():
    train_actions = np.zeros((2, 2, 9), dtype=np.float32)
    train_actions[..., 3:] = IDENTITY_ROT6
    train_actions[0, 0, 0] = 1.0
    train_actions[0, 1, 0] = 3.0
    train_actions[1, :, 0] = 1000.0  # Invalid padding must not affect statistics.
    train = {"actions": train_actions, "action_mask": np.array([[1, 1], [0, 0]], dtype=np.float32)}

    test_actions = np.zeros((1, 2, 9), dtype=np.float32)
    test_actions[..., 3:] = IDENTITY_ROT6
    test_actions[..., 0] = 100.0  # Test values must not affect training statistics.
    test = {"actions": test_actions, "action_mask": np.ones((1, 2), dtype=np.float32)}

    stats = _normalize_relative_chunks([train], [train, test])

    assert stats["fit_on"] == "valid_train_chunks_only"
    assert stats["mean"][0] == 2.0
    assert stats["std"][0] == 1.0
    np.testing.assert_allclose(train["actions"][0, :, 0], [-1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        apply_grouped_transform(test["actions"], stats, inverse=True)[..., 0],
        100.0,
        atol=1e-5,
    )


def test_lowdim_replay_loads_precomputed_chunks_without_transform(tmp_path):
    chunks = np.arange(4 * 3 * 9, dtype=np.float32).reshape(4, 3, 9)
    masks = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
    trajectory = {
        "states": np.zeros((4, 12), dtype=np.float32),
        "actions": chunks.copy(),
        "action_mask": masks,
        "num_steps": 4,
    }
    buffer_path = tmp_path / "buf_train.pkl"
    with buffer_path.open("wb") as f:
        pickle.dump(generate_robobuf([trajectory]).to_traj_list(), f)
    rollout = {
        "norm_stats": {
            "action": {"mode": "gaussian", "mean": [0.0] * 9, "std": [1.0] * 9},
        },
        "processing_config": {
            "action_pose_mode": "relative",
            "ac_chunk": 3,
            "action_index_offset": 1,
            "relative_chunk_anchor": "current_command",
            "relative_chunk_normalized": True,
        },
    }
    with (tmp_path / "rollout_config.yaml").open("w") as f:
        yaml.safe_dump(rollout, f)

    replay = RobobufReplayBufferLowdim(
        str(buffer_path), ac_chunk=3, obs_window=2, obs_dim=12, action_index_offset=1,
        include_tracking_error=True, action_chunk_mode="relative", shuffle=False,
    )
    (_, obs), action, mask, _ = replay[0]
    assert tuple(obs.shape) == (2, 12)
    assert tuple(action.shape) == (3, 9)
    assert tuple(mask.shape) == (3, 9)
    np.testing.assert_array_equal(action.numpy(), chunks[0])
    np.testing.assert_array_equal(mask.numpy()[:, 0], masks[0])
