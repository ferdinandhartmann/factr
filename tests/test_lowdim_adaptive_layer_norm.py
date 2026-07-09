import unittest

import torch

from factr.models.lowdim_action_transformer import LowdimStiffnessCVAEAgent


class AdaptiveLayerNormConditioningTest(unittest.TestCase):
    @staticmethod
    def _make_model(arrangement=False, goal=False, adaptive=True, stiffness_gate=False, gate_min=0.1):
        return LowdimStiffnessCVAEAgent(
            obs_dim=36,
            ac_dim=9,
            ac_chunk=3,
            obs_window=4,
            include_tracking_error=False,
            use_cls_token=False,
            stiffness_classes=2 if stiffness_gate else 1,
            use_stiffness_conditioning=stiffness_gate,
            use_arrangement_conditioning=arrangement,
            goal_label=goal,
            use_adaptive_layer_norm=adaptive,
            use_stiffness_goal_adaln_gate=stiffness_gate,
            goal_adaln_gate_min=gate_min,
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

    def test_requires_an_active_condition(self):
        with self.assertRaisesRegex(ValueError, "requires arrangement or goal conditioning"):
            self._make_model()

    def test_arrangement_goal_and_both_keep_base_token_count(self):
        obs = torch.randn(2, 4, 30)
        arrangements = torch.zeros(2, 9)
        goals = torch.zeros(2, 3)
        for arrangement, goal in ((True, False), (False, True), (True, True)):
            with self.subTest(arrangement=arrangement, goal=goal):
                model = self._make_model(arrangement=arrangement, goal=goal)
                context = model._build_context_tokens(
                    obs,
                    class_labels=None,
                    arrangement_vectors=arrangements if arrangement else None,
                    goal_vectors=goals if goal else None,
                )
                self.assertEqual(context.shape, (2, 4, 32))
                self.assertEqual(model.positional_tokens.shape, (1, 4, 32))
                self.assertIsNone(model.arrangement_encoder)
                self.assertIsNone(model.goal_encoder)

    def test_zero_initialization_matches_plain_layer_norm(self):
        model = self._make_model(arrangement=True)
        obs = torch.randn(2, 4, 30)
        first = model._build_context_tokens(obs, None, arrangement_vectors=torch.zeros(2, 9))
        second = model._build_context_tokens(obs, None, arrangement_vectors=torch.ones(2, 9))
        torch.testing.assert_close(first, second)

    def test_learned_modulation_changes_context_with_condition(self):
        model = self._make_model(goal=True)
        with torch.no_grad():
            model.adaln_modulation[-1].weight.fill_(0.05)
        obs = torch.randn(2, 4, 30)
        zeros = model._build_context_tokens(obs, None, goal_vectors=torch.zeros(2, 3))
        ones = model._build_context_tokens(obs, None, goal_vectors=torch.ones(2, 3))
        self.assertFalse(torch.allclose(zeros, ones))

    def test_forward_and_inference_shapes_with_both_conditions(self):
        model = self._make_model(arrangement=True, goal=True)
        obs = torch.randn(2, 4, 30)
        actions = torch.randn(2, 3, 9)
        arrangements = torch.zeros(2, 9)
        goals = torch.zeros(2, 3)
        output = model(
            {}, obs, actions, torch.ones_like(actions),
            arrangement_vectors=arrangements, goal_vectors=goals,
        )
        prior = model.get_actions_prior(
            {}, obs, arrangement_vectors=arrangements, goal_vectors=goals, num_samples=2
        )
        self.assertEqual(output["total_loss"].ndim, 0)
        self.assertEqual(prior.shape, (2, 2, 3, 9))

    def test_condition_shape_validation_is_preserved(self):
        model = self._make_model(arrangement=True, goal=True)
        obs = torch.randn(2, 4, 30)
        with self.assertRaisesRegex(ValueError, "requires arrangement_vectors"):
            model._build_context_tokens(obs, None, goal_vectors=torch.zeros(2, 3))
        with self.assertRaisesRegex(ValueError, "Expected goal_vectors shape"):
            model._build_context_tokens(
                obs, None, arrangement_vectors=torch.zeros(2, 9), goal_vectors=torch.zeros(2, 4)
            )

    def test_stiffness_gate_weakens_goal_in_mode_zero(self):
        model = self._make_model(goal=True, stiffness_gate=True, gate_min=0.1)
        captured = []
        handle = model.adaln_modulation[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].detach().clone())
        )
        try:
            model._build_context_tokens(
                torch.randn(2, 4, 30),
                class_labels=torch.tensor([1, 2]),
                goal_vectors=torch.ones(2, 3),
            )
        finally:
            handle.remove()
        torch.testing.assert_close(
            captured[0], torch.tensor([[0.1, 0.1, 0.1], [1.0, 1.0, 1.0]])
        )

    def test_stiffness_gate_requires_goal_adaln_and_stiffness(self):
        with self.assertRaisesRegex(ValueError, "requires adaptive LayerNorm"):
            LowdimStiffnessCVAEAgent(
                obs_dim=36,
                include_tracking_error=False,
                goal_label=True,
                use_adaptive_layer_norm=True,
                use_stiffness_conditioning=False,
                use_stiffness_goal_adaln_gate=True,
            )


if __name__ == "__main__":
    unittest.main()
