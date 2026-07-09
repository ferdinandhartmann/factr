import importlib
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch

from factr.goal_label import equal_goal_group_samples, format_real_goal_groups, goal_label_to_group_one_hot
from factr.models.lowdim_action_transformer import LowdimStiffnessCVAEAgent


class GoalLabelEncodingTest(unittest.TestCase):
    def test_group_boundaries(self):
        expected = {
            1: [1, 0, 0],
            3: [1, 0, 0],
            4: [0, 1, 0],
            6: [0, 1, 0],
            7: [0, 0, 1],
            9: [0, 0, 1],
        }
        for label, vector in expected.items():
            np.testing.assert_array_equal(goal_label_to_group_one_hot(label), vector)

    def test_rejects_invalid_labels(self):
        for value in (0, 10, 1.5, np.nan, [1, 2]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                goal_label_to_group_one_hot(value)

    def test_equal_counterfactual_sampling_uses_floor_and_stable_order(self):
        real_goals = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        calls = []

        def sample_fn(condition, count):
            calls.append((condition.clone(), count))
            group = torch.argmax(condition, dim=1).float()
            return group[:, None, None, None].expand(-1, count, 2, 1)

        samples = equal_goal_group_samples(sample_fn, real_goals, 40)
        self.assertEqual(samples.shape, (2, 39, 2, 1))
        self.assertEqual([count for _, count in calls], [13, 13, 13])
        self.assertEqual(samples[0, :, 0, 0].tolist(), [0.0] * 13 + [1.0] * 13 + [2.0] * 13)

    def test_counterfactual_sampling_disabled_and_small_budget(self):
        marker = torch.zeros(2, 7, 1, 1)
        self.assertIs(equal_goal_group_samples(lambda condition, count: marker, None, 7), marker)
        with self.assertRaisesRegex(ValueError, "num_samples >= 3"):
            equal_goal_group_samples(lambda condition, count: marker, torch.eye(3), 2)

    def test_real_goal_group_title(self):
        title = format_real_goal_groups(torch.tensor([[1, 0, 0], [0, 0, 1]], dtype=torch.float32))
        self.assertEqual(title, "real_goal_group=1,3")


class GoalLabelModelTest(unittest.TestCase):
    @staticmethod
    def _make_model(goal_label=True, use_arrangement_conditioning=False):
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
            goal_label=goal_label,
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

    def test_forward_and_inference_shapes(self):
        model = self._make_model()
        obs = torch.randn(2, 4, 30)
        actions = torch.randn(2, 3, 9)
        mask = torch.ones_like(actions)
        goals = torch.tensor([[1, 0, 0], [0, 0, 1]], dtype=torch.float32)

        output = model({}, obs, actions, mask, goal_vectors=goals)
        prior = model.get_actions_prior({}, obs, goal_vectors=goals, num_samples=2)
        posterior = model.get_actions_pos({}, obs, actions, goal_vectors=goals, num_samples=2)

        self.assertEqual(output["total_loss"].ndim, 0)
        self.assertEqual(prior.shape, (2, 2, 3, 9))
        self.assertEqual(posterior.shape, (2, 2, 3, 9))

    def test_requires_three_wide_goal_vector(self):
        model = self._make_model()
        obs = torch.randn(2, 4, 30)
        with self.assertRaisesRegex(ValueError, "requires goal_vectors"):
            model._build_context_tokens(obs, class_labels=None)
        with self.assertRaisesRegex(ValueError, "Expected goal_vectors shape"):
            model._build_context_tokens(obs, class_labels=None, goal_vectors=torch.zeros(2, 4))

    def test_goal_and_arrangement_add_independent_tokens(self):
        model = self._make_model(use_arrangement_conditioning=True)
        context = model._build_context_tokens(
            torch.randn(2, 4, 30),
            class_labels=None,
            arrangement_vectors=torch.zeros(2, 9),
            goal_vectors=torch.zeros(2, 3),
        )
        self.assertEqual(context.shape, (2, 6, 32))


class GoalLabelReplayBufferTest(unittest.TestCase):
    @staticmethod
    def _replay_buffer_module():
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
    def _steps(include_goal=True):
        steps = []
        previous = None
        for step_idx in range(4):
            obs = {"state": np.zeros(36, dtype=np.float32), "arrangement": 5}
            if include_goal:
                obs["goals"] = 4
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

    def test_all_conditioning_batch_contracts(self):
        replay_buffer = self._replay_buffer_module()
        with mock.patch.object(replay_buffer, "_cached_load", return_value=self._steps()):
            disabled = replay_buffer.RobobufReplayBufferLowdim(
                "synthetic", ac_chunk=2, obs_window=2, obs_dim=36, include_tracking_error=False, shuffle=False
            )
            goal_only = replay_buffer.RobobufReplayBufferLowdim(
                "synthetic", ac_chunk=2, obs_window=2, obs_dim=36, include_tracking_error=False,
                goal_label=True, shuffle=False
            )
            arrangement_only = replay_buffer.RobobufReplayBufferLowdim(
                "synthetic", ac_chunk=2, obs_window=2, obs_dim=36, include_tracking_error=False,
                use_arrangement_conditioning=True, shuffle=False
            )
            both = replay_buffer.RobobufReplayBufferLowdim(
                "synthetic", ac_chunk=2, obs_window=2, obs_dim=36, include_tracking_error=False,
                use_arrangement_conditioning=True, goal_label=True, shuffle=False
            )

        self.assertEqual(len(disabled[0]), 4)
        self.assertEqual(goal_only[0][4].shape, (3,))
        self.assertEqual(arrangement_only[0][4].shape, (9,))
        self.assertEqual(len(both[0]), 6)
        np.testing.assert_array_equal(both[0][5].numpy(), [0, 1, 0])

    def test_missing_goal_reports_episode_and_step(self):
        replay_buffer = self._replay_buffer_module()
        with mock.patch.object(replay_buffer, "_cached_load", return_value=self._steps(include_goal=False)):
            with self.assertRaisesRegex(ValueError, "episode 0, step 0"):
                replay_buffer.RobobufReplayBufferLowdim(
                    "synthetic-missing", ac_chunk=2, obs_window=2, obs_dim=36,
                    include_tracking_error=False, goal_label=True, shuffle=False
                )


if __name__ == "__main__":
    unittest.main()
