import importlib
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

from factr.arrangement import arrangement_id_to_one_hot
from factr.models.lowdim_action_transformer import LowdimStiffnessCVAEAgent


class ArrangementEncodingTest(unittest.TestCase):
    def test_representative_ids(self):
        expected = {
            0: [1, 0, 0, 1, 0, 0, 1, 0, 0],
            1: [1, 0, 0, 1, 0, 0, 0, 1, 0],
            5: [1, 0, 0, 0, 1, 0, 0, 0, 1],
            26: [0, 0, 1, 0, 0, 1, 0, 0, 1],
        }
        for arrangement_id, vector in expected.items():
            np.testing.assert_array_equal(arrangement_id_to_one_hot(arrangement_id), vector)

    def test_rejects_invalid_ids(self):
        for value in (-1, 27, 1.5, np.nan, [1, 2]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                arrangement_id_to_one_hot(value)


class ArrangementModelTest(unittest.TestCase):
    @staticmethod
    def _make_model(use_arrangement_conditioning):
        return LowdimStiffnessCVAEAgent(
            obs_dim=36,
            ac_dim=9,
            ac_chunk=3,
            obs_window=4,
            include_tracking_error=False,
            use_cls_token=False,
            stiffness_classes=1,
            use_stiffness_conditioning=False,
            use_arrangement_conditioning=use_arrangement_conditioning,
            d_z=4,
            latent_distribution="categorical",
            categorical_num_variables=2,
            categorical_num_categories=2,
            fixed_prior=True,
            token_dim=32,
            hidden_dim=64,
            encoder_layers=1,
            decoder_layers=1,
            posterior_layers=1,
            nhead=4,
            dropout=0.0,
        )

    def test_enabled_forward_and_inference_shapes(self):
        model = self._make_model(use_arrangement_conditioning=True)
        obs = torch.randn(2, 4, 30)
        actions = torch.randn(2, 3, 9)
        mask = torch.ones_like(actions)
        arrangements = torch.from_numpy(
            np.stack([arrangement_id_to_one_hot(0), arrangement_id_to_one_hot(26)])
        )

        output = model({}, obs, actions, mask, arrangement_vectors=arrangements)
        prior = model.get_actions_prior({}, obs, arrangement_vectors=arrangements, num_samples=2)
        posterior = model.get_actions_pos({}, obs, actions, arrangement_vectors=arrangements, num_samples=2)

        self.assertEqual(output["total_loss"].ndim, 0)
        self.assertEqual(prior.shape, (2, 2, 3, 9))
        self.assertEqual(posterior.shape, (2, 2, 3, 9))

    def test_disabled_context_shape_is_unchanged(self):
        model = self._make_model(use_arrangement_conditioning=False)
        context = model._build_context_tokens(torch.randn(2, 4, 30), class_labels=None)
        self.assertEqual(context.shape, (2, 4, 32))

    def test_enabled_requires_arrangement_vector(self):
        model = self._make_model(use_arrangement_conditioning=True)
        with self.assertRaisesRegex(ValueError, "requires arrangement_vectors"):
            model._build_context_tokens(torch.randn(2, 4, 30), class_labels=None)


class ArrangementReplayBufferTest(unittest.TestCase):
    @staticmethod
    def _replay_buffer_module():
        # The repository's system-test environment may not install robobuf.
        try:
            return importlib.import_module("factr.replay_buffer")
        except ModuleNotFoundError as exc:
            if exc.name != "robobuf":
                raise
            fake_robobuf = types.ModuleType("robobuf")
            fake_robobuf.ReplayBuffer = type("ReplayBuffer", (), {})
            sys.modules["robobuf"] = fake_robobuf
            return importlib.import_module("factr.replay_buffer")

    @staticmethod
    def _steps(include_arrangement=True):
        steps = []
        previous = None
        for step_idx in range(4):
            obs = {"state": np.zeros(36, dtype=np.float32)}
            if include_arrangement:
                obs["arrangement"] = 5
            step = types.SimpleNamespace(
                obs=obs,
                action=np.zeros(9, dtype=np.float32),
                prev=previous,
                first=step_idx == 0,
                is_first=False,
            )
            steps.append(step)
            previous = step
        return steps

    def test_enabled_and_disabled_batch_contracts(self):
        replay_buffer = self._replay_buffer_module()
        with mock.patch.object(replay_buffer, "_cached_load", return_value=self._steps()):
            enabled = replay_buffer.RobobufReplayBufferLowdim(
                "synthetic",
                ac_chunk=2,
                obs_window=2,
                obs_dim=36,
                include_tracking_error=False,
                use_arrangement_conditioning=True,
                shuffle=False,
            )
            disabled = replay_buffer.RobobufReplayBufferLowdim(
                "synthetic",
                ac_chunk=2,
                obs_window=2,
                obs_dim=36,
                include_tracking_error=False,
                use_arrangement_conditioning=False,
                shuffle=False,
            )

        self.assertEqual(len(enabled[0]), 5)
        self.assertEqual(len(disabled[0]), 4)
        np.testing.assert_array_equal(enabled[0][4].numpy(), arrangement_id_to_one_hot(5))

    def test_missing_arrangement_reports_episode_and_step(self):
        replay_buffer = self._replay_buffer_module()
        with mock.patch.object(replay_buffer, "_cached_load", return_value=self._steps(include_arrangement=False)):
            with self.assertRaisesRegex(ValueError, "episode 0, step 0"):
                replay_buffer.RobobufReplayBufferLowdim(
                    "synthetic-missing",
                    ac_chunk=2,
                    obs_window=2,
                    obs_dim=36,
                    include_tracking_error=False,
                    use_arrangement_conditioning=True,
                    shuffle=False,
                )


if __name__ == "__main__":
    unittest.main()
