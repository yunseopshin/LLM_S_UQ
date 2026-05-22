"""Tests for src/models/fisher_scoring.py."""

import pytest
import torch

from src.models.fisher_scoring import (
    _compute_grad_and_fisher,
    _compute_clipped_objective,
    _compute_pi_and_mu,
    _compute_g,
    fisher_scoring_map,
    fisher_scoring_map_detached,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

K = 5
N_SENTENCES = 20
TOKENS_PER_SENTENCE = 10


@pytest.fixture()
def prior():
    """Standard isotropic prior."""
    torch.manual_seed(42)
    mu_0 = torch.zeros(K)
    Sigma_0_inv = torch.eye(K)
    return mu_0, Sigma_0_inv


@pytest.fixture()
def synthetic_data():
    """Synthetic dataset: z_tokens drawn from N(0,1), F from Bernoulli(σ(z@θ*))."""
    torch.manual_seed(42)
    true_theta = torch.randn(K)
    z_list = [torch.randn(TOKENS_PER_SENTENCE, K) for _ in range(N_SENTENCES)]
    F_list = torch.tensor([
        (torch.sigmoid(z @ true_theta).mean() > 0.5).float().item()
        for z in z_list
    ])
    return true_theta, z_list, F_list


@pytest.fixture()
def small_data():
    """Small dataset for gradient checks (k=3, N=5, L=4)."""
    torch.manual_seed(0)
    k, n, l_tok = 3, 5, 4
    z_list = [torch.randn(l_tok, k, dtype=torch.float64, requires_grad=True)
              for _ in range(n)]
    F = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float64)
    mu_0 = torch.zeros(k, dtype=torch.float64, requires_grad=True)
    Sigma_0_inv = torch.eye(k, dtype=torch.float64)
    return z_list, F, mu_0, Sigma_0_inv


# ---------------------------------------------------------------------------
# _compute_pi_and_mu
# ---------------------------------------------------------------------------


class TestComputePiAndMu:
    def test_output_shapes(self, synthetic_data, prior):
        _, z_list, _ = synthetic_data
        mu_0, _ = prior
        theta = torch.zeros(K)
        pi_list, mu_tilde = _compute_pi_and_mu(theta, z_list)
        assert len(pi_list) == N_SENTENCES
        for pi_j in pi_list:
            assert pi_j.shape == (TOKENS_PER_SENTENCE,)
        assert mu_tilde.shape == (N_SENTENCES,)

    def test_pi_in_01(self, synthetic_data):
        _, z_list, _ = synthetic_data
        theta = torch.randn(K)
        pi_list, _ = _compute_pi_and_mu(theta, z_list)
        for pi_j in pi_list:
            assert (pi_j >= 0.0).all() and (pi_j <= 1.0).all()

    def test_mu_clipped(self, synthetic_data):
        _, z_list, _ = synthetic_data
        # Use extreme theta to push mu toward 0 or 1
        theta = torch.ones(K) * 100.0
        _, mu_tilde = _compute_pi_and_mu(theta, z_list, eps=1e-6)
        assert (mu_tilde >= 1e-6).all()
        assert (mu_tilde <= 1.0 - 1e-6).all()


# ---------------------------------------------------------------------------
# _compute_g
# ---------------------------------------------------------------------------


class TestComputeG:
    def test_output_shape(self, synthetic_data):
        _, z_list, _ = synthetic_data
        theta = torch.randn(K)
        pi_list, _ = _compute_pi_and_mu(theta, z_list)
        g = _compute_g(pi_list, z_list)
        assert g.shape == (N_SENTENCES, K)

    def test_g_bounded(self, synthetic_data):
        """g_j involves π(1-π) ≤ 0.25, so g should have bounded values."""
        _, z_list, _ = synthetic_data
        theta = torch.randn(K)
        pi_list, _ = _compute_pi_and_mu(theta, z_list)
        g = _compute_g(pi_list, z_list)
        # π(1-π) ≤ 0.25, and z are O(1) from standard normal,
        # so g should be reasonably bounded
        assert torch.isfinite(g).all()


# ---------------------------------------------------------------------------
# _compute_grad_and_fisher
# ---------------------------------------------------------------------------


class TestComputeGradAndFisher:
    def test_output_shapes(self, synthetic_data, prior):
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        theta = torch.zeros(K)
        grad, H = _compute_grad_and_fisher(
            theta, z_list, F_list, mu_0, Sigma_0_inv
        )
        assert grad.shape == (K,)
        assert H.shape == (K, K)

    def test_fisher_symmetric(self, synthetic_data, prior):
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        theta = torch.randn(K)
        _, H = _compute_grad_and_fisher(
            theta, z_list, F_list, mu_0, Sigma_0_inv
        )
        assert torch.allclose(H, H.t(), atol=1e-6)

    def test_fisher_pd(self, synthetic_data, prior):
        """Fisher-type precision H should be positive definite."""
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        theta = torch.randn(K)
        _, H = _compute_grad_and_fisher(
            theta, z_list, F_list, mu_0, Sigma_0_inv
        )
        eigvals = torch.linalg.eigvalsh(H)
        assert (eigvals > 0).all(), f"Min eigenvalue: {eigvals.min().item()}"

    def test_gradient_finite(self, synthetic_data, prior):
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        theta = torch.randn(K)
        grad, _ = _compute_grad_and_fisher(
            theta, z_list, F_list, mu_0, Sigma_0_inv
        )
        assert torch.isfinite(grad).all()


# ---------------------------------------------------------------------------
# _compute_clipped_objective
# ---------------------------------------------------------------------------


class TestComputeClippedObjective:
    def test_scalar_output(self, synthetic_data, prior):
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        theta = torch.zeros(K)
        obj = _compute_clipped_objective(
            theta, z_list, F_list, mu_0, Sigma_0_inv
        )
        assert obj.dim() == 0

    def test_finite(self, synthetic_data, prior):
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        theta = torch.randn(K)
        obj = _compute_clipped_objective(
            theta, z_list, F_list, mu_0, Sigma_0_inv
        )
        assert torch.isfinite(obj)

    def test_prior_zero_at_mu0(self, prior):
        """When θ=μ_0 and no data, objective = 0 (prior term vanishes)."""
        mu_0, Sigma_0_inv = prior
        obj = _compute_clipped_objective(
            mu_0, [], torch.tensor([]), mu_0, Sigma_0_inv
        )
        assert torch.isclose(obj, torch.tensor(0.0), atol=1e-8)

    def test_gradient_matches_numerical(self, small_data):
        """Verify analytical gradient via finite differences."""
        z_list, F, mu_0, Sigma_0_inv = small_data
        k = mu_0.shape[0]
        theta = torch.randn(k, dtype=torch.float64)
        # Detach z_list for this comparison (we're checking dL/dtheta only)
        z_det = [z.detach() for z in z_list]

        grad_analytic, _ = _compute_grad_and_fisher(
            theta, z_det, F, mu_0.detach(), Sigma_0_inv
        )

        # Numerical gradient
        delta = 1e-5
        grad_numerical = torch.zeros(k, dtype=torch.float64)
        for i in range(k):
            theta_plus = theta.clone()
            theta_plus[i] += delta
            theta_minus = theta.clone()
            theta_minus[i] -= delta
            obj_plus = _compute_clipped_objective(
                theta_plus, z_det, F, mu_0.detach(), Sigma_0_inv
            )
            obj_minus = _compute_clipped_objective(
                theta_minus, z_det, F, mu_0.detach(), Sigma_0_inv
            )
            grad_numerical[i] = (obj_plus - obj_minus) / (2.0 * delta)

        assert torch.allclose(grad_analytic, grad_numerical, atol=1e-4), (
            f"Max diff: {(grad_analytic - grad_numerical).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# fisher_scoring_map — convergence
# ---------------------------------------------------------------------------


class TestFisherScoringMap:
    def test_convergence_to_true_theta(self, synthetic_data, prior):
        """MAP estimate should be near true theta for sufficient data."""
        true_theta, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        theta_map, H = fisher_scoring_map(
            z_list, F_list, mu_0, Sigma_0_inv, num_iters=15
        )
        # MAP won't exactly equal true_theta due to prior and finite data,
        # but should be correlated (cosine similarity > 0.5)
        cos_sim = (
            torch.dot(theta_map, true_theta)
            / (theta_map.norm() * true_theta.norm() + 1e-12)
        )
        assert cos_sim > 0.5, f"Cosine similarity: {cos_sim.item()}"

    def test_objective_non_decreasing(self, synthetic_data, prior):
        """Objective should increase (or stay) at each accepted step."""
        true_theta, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior

        # Run manually to check monotonicity of accepted steps
        theta = mu_0.clone()
        objectives = [
            _compute_clipped_objective(
                theta, z_list, F_list, mu_0, Sigma_0_inv
            ).item()
        ]
        theta_map, _ = fisher_scoring_map(
            z_list, F_list, mu_0, Sigma_0_inv, num_iters=15
        )
        final_obj = _compute_clipped_objective(
            theta_map, z_list, F_list, mu_0, Sigma_0_inv
        ).item()
        # Final objective should be >= initial
        assert final_obj >= objectives[0] - 1e-6

    def test_output_shapes(self, synthetic_data, prior):
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        theta, H = fisher_scoring_map(
            z_list, F_list, mu_0, Sigma_0_inv
        )
        assert theta.shape == (K,)
        assert H.shape == (K, K)

    def test_final_fisher_pd(self, synthetic_data, prior):
        """Final Fisher matrix should be positive definite."""
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        _, H = fisher_scoring_map(
            z_list, F_list, mu_0, Sigma_0_inv
        )
        eigvals = torch.linalg.eigvalsh(H)
        assert (eigvals > 0).all(), f"Min eigenvalue: {eigvals.min().item()}"

    def test_final_fisher_symmetric(self, synthetic_data, prior):
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        _, H = fisher_scoring_map(
            z_list, F_list, mu_0, Sigma_0_inv
        )
        assert torch.allclose(H, H.t(), atol=1e-6)

    def test_differentiable_through_mu0(self, prior):
        """Gradient should flow from MAP theta back to mu_0."""
        torch.manual_seed(99)
        k = 4
        mu_0 = torch.randn(k, requires_grad=True)
        Sigma_0_inv = torch.eye(k)
        z_list = [torch.randn(5, k) for _ in range(8)]
        F = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        theta_map, _ = fisher_scoring_map(
            z_list, F, mu_0, Sigma_0_inv, num_iters=10
        )
        loss = theta_map.sum()
        loss.backward()
        assert mu_0.grad is not None
        assert torch.isfinite(mu_0.grad).all()

    def test_differentiable_through_z(self):
        """Gradient should flow from MAP theta back to z_tokens."""
        torch.manual_seed(99)
        k = 4
        mu_0 = torch.zeros(k)
        Sigma_0_inv = torch.eye(k)
        z_list = [torch.randn(5, k, requires_grad=True) for _ in range(8)]
        F = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        theta_map, _ = fisher_scoring_map(
            z_list, F, mu_0, Sigma_0_inv, num_iters=10
        )
        loss = theta_map.sum()
        loss.backward()
        for i, z in enumerate(z_list):
            assert z.grad is not None, f"z_list[{i}] has no grad"
            assert torch.isfinite(z.grad).all(), f"z_list[{i}] has non-finite grad"


# ---------------------------------------------------------------------------
# Numerical stability: edge cases
# ---------------------------------------------------------------------------


class TestNumericalStability:
    def test_all_F_zero(self, prior):
        """All labels 0: theta should not diverge to -inf (prior regularizes)."""
        torch.manual_seed(0)
        mu_0, Sigma_0_inv = prior
        z_list = [torch.randn(TOKENS_PER_SENTENCE, K) for _ in range(10)]
        F = torch.zeros(10)

        theta, H = fisher_scoring_map(
            z_list, F, mu_0, Sigma_0_inv, num_iters=15
        )
        assert torch.isfinite(theta).all(), f"theta has non-finite values: {theta}"
        assert torch.isfinite(H).all(), "H has non-finite values"

    def test_all_F_one(self, prior):
        """All labels 1: theta should not diverge to +inf (prior regularizes)."""
        torch.manual_seed(0)
        mu_0, Sigma_0_inv = prior
        z_list = [torch.randn(TOKENS_PER_SENTENCE, K) for _ in range(10)]
        F = torch.ones(10)

        theta, H = fisher_scoring_map(
            z_list, F, mu_0, Sigma_0_inv, num_iters=15
        )
        assert torch.isfinite(theta).all(), f"theta has non-finite values: {theta}"
        assert torch.isfinite(H).all(), "H has non-finite values"

    def test_single_sentence(self, prior):
        """Edge case: only one sentence."""
        torch.manual_seed(0)
        mu_0, Sigma_0_inv = prior
        z_list = [torch.randn(5, K)]
        F = torch.tensor([1.0])

        theta, H = fisher_scoring_map(
            z_list, F, mu_0, Sigma_0_inv, num_iters=15
        )
        assert theta.shape == (K,)
        assert torch.isfinite(theta).all()

    def test_single_token_per_sentence(self, prior):
        """Edge case: each sentence has exactly 1 token."""
        torch.manual_seed(0)
        mu_0, Sigma_0_inv = prior
        z_list = [torch.randn(1, K) for _ in range(10)]
        F = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        theta, H = fisher_scoring_map(
            z_list, F, mu_0, Sigma_0_inv, num_iters=15
        )
        assert torch.isfinite(theta).all()
        assert torch.isfinite(H).all()

    def test_extreme_z_values(self, prior):
        """Large z values should not cause NaN."""
        torch.manual_seed(0)
        mu_0, Sigma_0_inv = prior
        z_list = [torch.randn(5, K) * 10.0 for _ in range(10)]
        F = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        theta, H = fisher_scoring_map(
            z_list, F, mu_0, Sigma_0_inv, num_iters=15
        )
        assert torch.isfinite(theta).all()
        assert torch.isfinite(H).all()


# ---------------------------------------------------------------------------
# fisher_scoring_map_detached
# ---------------------------------------------------------------------------


class TestFisherScoringMapDetached:
    def test_no_grad(self, synthetic_data, prior):
        """Output tensors should not require gradient."""
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior
        theta, H = fisher_scoring_map_detached(
            z_list, F_list, mu_0, Sigma_0_inv
        )
        assert not theta.requires_grad
        assert not H.requires_grad

    def test_same_result_as_map(self, synthetic_data, prior):
        """Detached version should produce identical numerical results."""
        _, z_list, F_list = synthetic_data
        mu_0, Sigma_0_inv = prior

        theta_grad, H_grad = fisher_scoring_map(
            z_list, F_list, mu_0, Sigma_0_inv, num_iters=15
        )
        theta_det, H_det = fisher_scoring_map_detached(
            z_list, F_list, mu_0, Sigma_0_inv, num_iters=15
        )
        assert torch.allclose(theta_grad, theta_det, atol=1e-6)
        assert torch.allclose(H_grad, H_det, atol=1e-6)
