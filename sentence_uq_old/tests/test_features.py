"""Tests for src/features/extractor.py."""

import pytest
import torch

from src.features.extractor import (
    SentenceUQParams,
    extract_token_features,
    extract_sentence_token_features,
    extract_sentence_aggregate_feature,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T = 10
NUM_LAYERS = 4
HIDDEN_DIM = 16
PROJ_DIM = 8


@pytest.fixture()
def params() -> SentenceUQParams:
    return SentenceUQParams(
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        projection_dim=PROJ_DIM,
    )


@pytest.fixture()
def mock_inputs():
    torch.manual_seed(0)
    hidden_states = torch.randn(T, NUM_LAYERS, HIDDEN_DIM)
    entropy = torch.rand(T)
    top1_prob = torch.rand(T)
    return hidden_states, entropy, top1_prob


# ---------------------------------------------------------------------------
# SentenceUQParams
# ---------------------------------------------------------------------------


class TestSentenceUQParams:
    def test_feature_dim(self, params: SentenceUQParams):
        assert params.feature_dim == PROJ_DIM + 2

    def test_W_shape(self, params: SentenceUQParams):
        assert params.W.weight.shape == (PROJ_DIM, HIDDEN_DIM)

    def test_alpha_shape(self, params: SentenceUQParams):
        assert params.alpha.shape == (NUM_LAYERS,)

    def test_mu_0_shape(self, params: SentenceUQParams):
        assert params.mu_0.shape == (PROJ_DIM + 2,)

    def test_log_sigma_0_shape(self, params: SentenceUQParams):
        assert params.log_sigma_0.shape == (PROJ_DIM + 2,)

    def test_get_Sigma_0_inv_shape(self, params: SentenceUQParams):
        S_inv = params.get_Sigma_0_inv()
        k = PROJ_DIM + 2
        assert S_inv.shape == (k, k)

    def test_get_Sigma_0_shape(self, params: SentenceUQParams):
        S = params.get_Sigma_0()
        k = PROJ_DIM + 2
        assert S.shape == (k, k)

    def test_Sigma_0_inverse_consistency(self, params: SentenceUQParams):
        """Σ_0 @ Σ_0^{-1} should be identity (diagonal case)."""
        S = params.get_Sigma_0()
        S_inv = params.get_Sigma_0_inv()
        product = S @ S_inv
        eye = torch.eye(PROJ_DIM + 2)
        assert torch.allclose(product, eye, atol=1e-5)


# ---------------------------------------------------------------------------
# extract_token_features
# ---------------------------------------------------------------------------


class TestExtractTokenFeatures:
    def test_output_shape(self, mock_inputs, params):
        hidden_states, entropy, top1_prob = mock_inputs
        z = extract_token_features(hidden_states, entropy, top1_prob, params)
        k = PROJ_DIM + 2
        assert z.shape == (T, k), f"Expected ({T}, {k}), got {z.shape}"

    def test_requires_grad_W(self, mock_inputs, params):
        hidden_states, entropy, top1_prob = mock_inputs
        z = extract_token_features(hidden_states, entropy, top1_prob, params)
        loss = z.sum()
        loss.backward()
        assert params.W.weight.grad is not None

    def test_requires_grad_alpha(self, mock_inputs, params):
        hidden_states, entropy, top1_prob = mock_inputs
        z = extract_token_features(hidden_states, entropy, top1_prob, params)
        loss = z.sum()
        loss.backward()
        assert params.alpha.grad is not None

    def test_num_layers_1(self):
        """Edge case: single layer."""
        p = SentenceUQParams(hidden_dim=HIDDEN_DIM, num_layers=1, projection_dim=PROJ_DIM)
        hidden_states = torch.randn(T, 1, HIDDEN_DIM)
        entropy = torch.rand(T)
        top1_prob = torch.rand(T)
        z = extract_token_features(hidden_states, entropy, top1_prob, p)
        assert z.shape == (T, PROJ_DIM + 2)

    def test_num_layers_mismatch_raises(self, mock_inputs, params):
        hidden_states, entropy, top1_prob = mock_inputs
        bad_hs = torch.randn(T, NUM_LAYERS + 1, HIDDEN_DIM)
        with pytest.raises(AssertionError):
            extract_token_features(bad_hs, entropy, top1_prob, params)


# ---------------------------------------------------------------------------
# extract_sentence_token_features
# ---------------------------------------------------------------------------


class TestExtractSentenceTokenFeatures:
    def test_output_shape(self, mock_inputs, params):
        hidden_states, entropy, top1_prob = mock_inputs
        start, end = 2, 7
        z = extract_sentence_token_features(
            hidden_states, entropy, top1_prob, (start, end), params
        )
        assert z.shape == (end - start, PROJ_DIM + 2)

    def test_empty_range_raises(self, mock_inputs, params):
        hidden_states, entropy, top1_prob = mock_inputs
        with pytest.raises(ValueError):
            extract_sentence_token_features(
                hidden_states, entropy, top1_prob, (3, 3), params
            )

    def test_inverted_range_raises(self, mock_inputs, params):
        hidden_states, entropy, top1_prob = mock_inputs
        with pytest.raises(ValueError):
            extract_sentence_token_features(
                hidden_states, entropy, top1_prob, (5, 2), params
            )

    def test_single_token_sentence(self, mock_inputs, params):
        hidden_states, entropy, top1_prob = mock_inputs
        z = extract_sentence_token_features(
            hidden_states, entropy, top1_prob, (0, 1), params
        )
        assert z.shape == (1, PROJ_DIM + 2)


# ---------------------------------------------------------------------------
# extract_sentence_aggregate_feature
# ---------------------------------------------------------------------------


class TestExtractSentenceAggregateFeature:
    def test_output_dim(self):
        k = PROJ_DIM + 2
        z_tokens = torch.randn(5, k)
        agg = extract_sentence_aggregate_feature(z_tokens)
        assert agg.shape == (3 * k,)

    def test_single_token_std_zeros(self):
        k = PROJ_DIM + 2
        z_tokens = torch.randn(1, k)
        agg = extract_sentence_aggregate_feature(z_tokens)
        # std portion should be zeros
        std_part = agg[k : 2 * k]
        assert torch.all(std_part == 0.0)

    def test_single_token_mean_equals_last(self):
        k = PROJ_DIM + 2
        z_tokens = torch.randn(1, k)
        agg = extract_sentence_aggregate_feature(z_tokens)
        mean_part = agg[:k]
        last_part = agg[2 * k :]
        assert torch.allclose(mean_part, last_part)

    def test_multi_token_last_correct(self):
        k = PROJ_DIM + 2
        z_tokens = torch.randn(4, k)
        agg = extract_sentence_aggregate_feature(z_tokens)
        last_part = agg[2 * k :]
        assert torch.allclose(last_part, z_tokens[-1])
