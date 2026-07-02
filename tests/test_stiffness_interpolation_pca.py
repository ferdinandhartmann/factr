import unittest

import numpy as np
import torch

from scripts.eval_stiffness_interpolation_pca import build_interpolation_vectors, fit_pca_2d, posterior_features, predict_and_reencode


class InterpolationVectorTest(unittest.TestCase):
    def test_default_grid_includes_endpoints(self):
        alpha, vectors = build_interpolation_vectors(0.1)
        self.assertEqual(alpha.shape, (11,))
        self.assertEqual(vectors.shape, (11, 2))
        np.testing.assert_allclose(vectors.sum(axis=1), 1.0)
        np.testing.assert_allclose(vectors[0], [1.0, 0.0])
        np.testing.assert_allclose(vectors[-1], [0.0, 1.0])

    def test_step_must_divide_unit_interval(self):
        with self.assertRaises(ValueError):
            build_interpolation_vectors(0.3)


class PCAHelperTest(unittest.TestCase):
    def test_repeated_fit_is_identical(self):
        values = np.random.default_rng(4).normal(size=(32, 7)).astype(np.float32)
        scores_a, components_a, ratio_a = fit_pca_2d(values)
        scores_b, components_b, ratio_b = fit_pca_2d(values)
        np.testing.assert_array_equal(scores_a, scores_b)
        np.testing.assert_array_equal(components_a, components_b)
        np.testing.assert_array_equal(ratio_a, ratio_b)
        self.assertEqual(scores_a.shape, (32, 2))


class _CategoricalPosterior:
    latent_distribution = "categorical"

    def posterior(self, context, actions):
        batch = actions.shape[0]
        logits = torch.arange(batch * 6, dtype=actions.dtype).reshape(batch, 2, 3)
        return {"logits": logits}


class _GaussianPosterior:
    latent_distribution = "gaussian"

    def posterior(self, context, actions):
        batch = actions.shape[0]
        return {"mu": torch.ones(batch, 4), "logvar": torch.zeros(batch, 4)}


class _DeterministicCategoricalModel(_CategoricalPosterior):
    def get_actions_pos(self, _, obs, target_action, class_labels, arrangement_vectors, sample, num_samples):
        assert sample is False
        assert num_samples == 1
        base = class_labels[:, 1].reshape(-1, 1, 1, 1)
        return target_action.unsqueeze(1) + base

    def _build_context_tokens(self, obs, class_labels, arrangement_vectors):
        return torch.zeros(obs.shape[0], 5, 8, device=obs.device)


class PosteriorFeatureTest(unittest.TestCase):
    def setUp(self):
        self.context = torch.randn(3, 5, 8)
        self.actions = torch.randn(3, 4, 9)

    def test_categorical_uses_flattened_probabilities(self):
        features = posterior_features(_CategoricalPosterior(), self.context, self.actions)
        self.assertEqual(tuple(features.shape), (3, 6))
        torch.testing.assert_close(features.reshape(3, 2, 3).sum(dim=-1), torch.ones(3, 2))

    def test_gaussian_uses_mean(self):
        features = posterior_features(_GaussianPosterior(), self.context, self.actions)
        self.assertEqual(tuple(features.shape), (3, 4))
        torch.testing.assert_close(features, torch.ones(3, 4))

    def test_prediction_and_reencoding_are_deterministic(self):
        model = _DeterministicCategoricalModel()
        condition = torch.tensor([[0.7, 0.3]]).expand(3, -1)
        prediction_a, features_a = predict_and_reencode(model, self.context, self.actions, condition, None)
        prediction_b, features_b = predict_and_reencode(model, self.context, self.actions, condition, None)
        self.assertEqual(tuple(prediction_a.shape), (3, 4, 9))
        self.assertEqual(tuple(features_a.shape), (3, 6))
        torch.testing.assert_close(prediction_a, prediction_b)
        torch.testing.assert_close(features_a, features_b)


if __name__ == "__main__":
    unittest.main()
