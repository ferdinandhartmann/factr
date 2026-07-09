import numpy as np
import pytest
import torch

from factr.models.lowdim_action_transformer import LowdimStiffnessCVAEAgent
from scripts.eval_single_episode_lowdim import (
    _collect_decoder_cross_attention,
    _decoder_memory_token_labels,
    _save_decoder_attention_video,
)


def _model(**overrides):
    config = dict(
        obs_dim=30,
        ac_dim=3,
        ac_chunk=4,
        obs_window=2,
        include_tracking_error=False,
        use_cls_token=False,
        stiffness_classes=2,
        use_stiffness_conditioning=True,
        d_z=2,
        latent_distribution="categorical",
        categorical_num_variables=1,
        categorical_num_categories=2,
        token_dim=8,
        hidden_dim=16,
        encoder_layers=1,
        decoder_layers=2,
        posterior_layers=1,
        nhead=2,
        dropout=0.0,
    )
    config.update(overrides)
    return LowdimStiffnessCVAEAgent(**config).eval()


@pytest.mark.parametrize("action_source", ["prior", "posterior"])
def test_collects_every_decoder_layer_and_head_without_changing_predictions(action_source):
    torch.manual_seed(4)
    model = _model()
    obs = torch.randn(3, 2, 30)
    actions = torch.randn(3, 4, 3)
    labels = torch.ones(3, dtype=torch.long)

    attention = _collect_decoder_cross_attention(model, obs, actions, labels, None, action_source)

    assert attention.shape == (3, 2, 2, 4, 6)
    assert np.allclose(attention.sum(axis=-1), 1.0, atol=1e-5)
    # The temporary hooks must be gone after collection.
    assert all(not layer.multihead_attn._forward_hooks for layer in model.decoder.layers)
    assert all(not layer.multihead_attn._forward_pre_hooks for layer in model.decoder.layers)
    prediction = model.get_actions_prior({}, obs, class_labels=labels, sample=False, num_samples=1)
    assert prediction.shape == (3, 1, 4, 3)


def test_memory_labels_follow_optional_context_tokens():
    model = _model(
        use_cls_token=True,
        include_tracking_error=True,
        obs_dim=36,
        use_arrangement_conditioning=True,
        goal_label=True,
    )
    assert _decoder_memory_token_labels(model) == [
        "z", "cls", "pose", "velocity", "wrench", "tracking",
        "stiffness", "arrangement", "goal", "command",
    ]


def test_adaln_conditions_are_not_reported_as_memory_tokens():
    model = _model(
        use_arrangement_conditioning=True,
        goal_label=True,
        use_adaptive_layer_norm=True,
    )
    assert _decoder_memory_token_labels(model) == [
        "z", "pose", "velocity", "wrench", "stiffness", "command",
    ]


def test_attention_video_smoke(tmp_path):
    if not animation_available():
        pytest.skip("ffmpeg is not available")
    attention = np.full((2, 1, 1, 3, 2), 0.5, dtype=np.float32)
    output_path = tmp_path / "attention.mp4"
    _save_decoder_attention_video(
        attention, ["z", "pose"], np.array([5, 6]), output_path,
        "ep_test", "prior", fps=2, dpi=40, frame_stride=1,
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def animation_available():
    from matplotlib import animation

    return animation.writers.is_available("ffmpeg")
