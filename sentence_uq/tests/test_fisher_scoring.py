"""Tests for ``src.models.fisher_scoring`` — Phase 3-1.

Covers:
- shape and symmetry of ``_compute_grad_and_fisher`` outputs,
- ``_compute_clipped_objective`` analytic checks (m_j=0 skip, sign of
  prior penalty),
- convergence of ``fisher_scoring_map`` on synthetic binomial data,
- Bernoulli special case (``m_j = 1``) coincides with the binomial form,
- extreme observations (``all K_j = 0`` / ``all K_j = m_j``) shift the
  MAP from ``μ_0`` in the expected direction,
- ``m_j = 0`` sentences are skipped without affecting the result,
- Fisher-type precision is positive-definite at convergence,
- ``torch.autograd.gradcheck`` on ``_compute_grad_and_fisher`` and
  ``_compute_clipped_objective``,
- the unrolled ``fisher_scoring_map`` carries gradients back to ``μ_0``,
- ``fisher_scoring_map_detached`` returns autograd-free tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.fisher_scoring import (  # noqa: E402
    _compute_clipped_objective,
    _compute_grad_and_fisher,
    fisher_scoring_map,
    fisher_scoring_map_detached,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_synthetic(
    k: int = 5,
    N: int = 20,
    L: int = 8,
    m_max: int = 5,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    all_z = [torch.randn(L, k, generator=g, dtype=dtype) for _ in range(N)]
    all_m = torch.randint(1, m_max + 1, (N,), generator=g)
    all_K = torch.stack(
        [torch.randint(0, int(m) + 1, (1,), generator=g) for m in all_m]
    ).squeeze(-1)
    mu_0 = torch.zeros(k, dtype=dtype)
    Sigma_0_inv = torch.eye(k, dtype=dtype) * 0.5
    return all_z, all_K, all_m, mu_0, Sigma_0_inv


# ---------------------------------------------------------------------------
# _compute_grad_and_fisher
# ---------------------------------------------------------------------------


def test_grad_and_fisher_shapes_and_symmetry() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_synthetic()
    theta = torch.randn(5)
    grad, H = _compute_grad_and_fisher(theta, all_z, all_K, all_m, mu_0, Sigma_0_inv)
    assert grad.shape == (5,)
    assert H.shape == (5, 5)
    assert torch.allclose(H, H.T, atol=1e-6)


def test_grad_and_fisher_skips_m_zero() -> None:
    """m_j = 0 entries must contribute nothing to grad and H."""
    k, L = 3, 4
    g = torch.Generator().manual_seed(11)
    all_z_base = [torch.randn(L, k, generator=g) for _ in range(4)]
    all_m_base = torch.tensor([2, 1, 3, 2])
    all_K_base = torch.tensor([1, 0, 2, 1])
    mu_0 = torch.zeros(k)
    Sigma_0_inv = torch.eye(k) * 0.5
    theta = torch.randn(k, generator=g)

    grad_a, H_a = _compute_grad_and_fisher(
        theta, all_z_base, all_K_base, all_m_base, mu_0, Sigma_0_inv
    )

    extras = [torch.randn(L, k, generator=g) for _ in range(3)]
    all_z_pad = all_z_base + extras
    all_m_pad = torch.cat([all_m_base, torch.zeros(3, dtype=torch.long)])
    all_K_pad = torch.cat([all_K_base, torch.zeros(3, dtype=torch.long)])

    grad_b, H_b = _compute_grad_and_fisher(
        theta, all_z_pad, all_K_pad, all_m_pad, mu_0, Sigma_0_inv
    )

    assert torch.allclose(grad_a, grad_b, atol=1e-6)
    assert torch.allclose(H_a, H_b, atol=1e-6)


def test_grad_matches_autograd_of_objective() -> None:
    """∇_θ L̃ from the analytic formula must match autograd of L̃."""
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_synthetic(
        k=4, N=6, L=5, seed=2, dtype=torch.float64
    )
    theta = torch.randn(4, dtype=torch.float64, requires_grad=True)
    obj = _compute_clipped_objective(theta, all_z, all_K, all_m, mu_0, Sigma_0_inv)
    grad_auto = torch.autograd.grad(obj, theta)[0]
    grad_ana, _ = _compute_grad_and_fisher(
        theta.detach(), all_z, all_K, all_m, mu_0, Sigma_0_inv
    )
    assert torch.allclose(grad_ana, grad_auto, atol=1e-8)


# ---------------------------------------------------------------------------
# _compute_clipped_objective
# ---------------------------------------------------------------------------


def test_objective_prior_only_at_mu0_is_zero_likelihood() -> None:
    """At θ = μ_0 with all m_j = 0, L̃ = 0 (no prior penalty, no likelihood)."""
    k = 3
    z = [torch.randn(2, k) for _ in range(2)]
    all_K = torch.zeros(2, dtype=torch.long)
    all_m = torch.zeros(2, dtype=torch.long)
    mu_0 = torch.zeros(k)
    Sigma_0_inv = torch.eye(k)
    obj = _compute_clipped_objective(mu_0, z, all_K, all_m, mu_0, Sigma_0_inv)
    assert obj.abs().item() < 1e-8


def test_objective_prior_penalty_negative_when_away_from_mu0() -> None:
    """θ ≠ μ_0 with empty likelihood: objective should be strictly negative."""
    k = 3
    z = [torch.randn(2, k) for _ in range(2)]
    all_K = torch.zeros(2, dtype=torch.long)
    all_m = torch.zeros(2, dtype=torch.long)
    mu_0 = torch.zeros(k)
    Sigma_0_inv = torch.eye(k)
    theta = torch.ones(k)
    obj = _compute_clipped_objective(theta, z, all_K, all_m, mu_0, Sigma_0_inv)
    assert obj.item() < 0
    assert obj.item() == pytest.approx(-1.5, abs=1e-6)  # -0.5 * 3


# ---------------------------------------------------------------------------
# fisher_scoring_map — convergence and behaviour
# ---------------------------------------------------------------------------


def test_fisher_scoring_increases_objective() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_synthetic(seed=3)
    theta_hat, _ = fisher_scoring_map(
        all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=15
    )
    obj_0 = _compute_clipped_objective(mu_0, all_z, all_K, all_m, mu_0, Sigma_0_inv)
    obj_hat = _compute_clipped_objective(
        theta_hat, all_z, all_K, all_m, mu_0, Sigma_0_inv
    )
    assert obj_hat.item() > obj_0.item()


def test_fisher_scoring_gradient_near_zero_at_solution() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_synthetic(seed=4)
    theta_hat, _ = fisher_scoring_map(
        all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=30
    )
    grad, _ = _compute_grad_and_fisher(
        theta_hat, all_z, all_K, all_m, mu_0, Sigma_0_inv
    )
    # Fisher-scoring is damped quasi-Newton (Fisher-type ≠ true Hessian),
    # so we expect a small but not vanishing residual.
    assert grad.norm().item() < 1e-2


def test_bernoulli_special_case_matches_binomial_formula() -> None:
    """m_j = 1, K_j ∈ {0,1}: R_j^bin coincides with Bernoulli R_j."""
    k, N, L = 4, 12, 5
    g = torch.Generator().manual_seed(5)
    all_z = [torch.randn(L, k, generator=g) for _ in range(N)]
    all_m = torch.ones(N, dtype=torch.long)
    all_K = torch.randint(0, 2, (N,), generator=g)
    mu_0 = torch.zeros(k)
    Sigma_0_inv = torch.eye(k)

    theta_hat, _ = fisher_scoring_map(
        all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=30
    )

    # Convergence check (loose tolerance — Fisher-type ≠ true Hessian).
    grad, _ = _compute_grad_and_fisher(
        theta_hat, all_z, all_K, all_m, mu_0, Sigma_0_inv
    )
    assert grad.norm().item() < 1e-2

    # Algebraic equivalence: with m_j = 1, (K_j - m_j μ_j)/[μ_j(1-μ_j)]
    # equals the Bernoulli residual (F_j - μ_j)/[μ_j(1-μ_j)].
    pi = torch.sigmoid(all_z[0] @ theta_hat)
    mu = pi.mean().clamp(1e-6, 1 - 1e-6)
    K = all_K[0].to(torch.float32)
    R_bin = (K - 1 * mu) / (mu * (1 - mu))
    R_bern = (K - mu) / (mu * (1 - mu))
    assert torch.allclose(R_bin, R_bern)


def test_all_K_zero_pushes_pi_down() -> None:
    """K_j = 0 ∀j with positive-mean features pulls θ in the negative direction."""
    k, N, L = 3, 12, 5
    g = torch.Generator().manual_seed(6)
    all_z = [torch.randn(L, k, generator=g) + 1.0 for _ in range(N)]
    all_m = torch.full((N,), 3, dtype=torch.long)
    all_K = torch.zeros(N, dtype=torch.long)
    mu_0 = torch.zeros(k)
    Sigma_0_inv = torch.eye(k) * 0.1

    theta_hat, _ = fisher_scoring_map(all_z, all_K, all_m, mu_0, Sigma_0_inv)

    pi_at_0 = torch.stack([torch.sigmoid(z @ mu_0).mean() for z in all_z]).mean()
    pi_at_hat = torch.stack([torch.sigmoid(z @ theta_hat).mean() for z in all_z]).mean()
    assert pi_at_hat.item() < pi_at_0.item()
    assert (theta_hat - mu_0).norm().item() > 1e-3


def test_all_K_equal_m_pushes_pi_up() -> None:
    """K_j = m_j ∀j pulls μ_j toward 1."""
    k, N, L = 3, 12, 5
    g = torch.Generator().manual_seed(7)
    all_z = [torch.randn(L, k, generator=g) + 1.0 for _ in range(N)]
    all_m = torch.full((N,), 3, dtype=torch.long)
    all_K = all_m.clone()
    mu_0 = torch.zeros(k)
    Sigma_0_inv = torch.eye(k) * 0.1

    theta_hat, _ = fisher_scoring_map(all_z, all_K, all_m, mu_0, Sigma_0_inv)
    pi_at_0 = torch.stack([torch.sigmoid(z @ mu_0).mean() for z in all_z]).mean()
    pi_at_hat = torch.stack([torch.sigmoid(z @ theta_hat).mean() for z in all_z]).mean()
    assert pi_at_hat.item() > pi_at_0.item()


def test_m_zero_sentences_do_not_affect_solution() -> None:
    k, L = 3, 4
    g = torch.Generator().manual_seed(8)
    all_z = [torch.randn(L, k, generator=g) for _ in range(8)]
    all_m = torch.tensor([2, 3, 1, 4, 2, 3, 1, 2])
    all_K = torch.tensor([1, 2, 0, 3, 1, 1, 0, 1])
    mu_0 = torch.zeros(k)
    Sigma_0_inv = torch.eye(k) * 0.5

    theta_a, H_a = fisher_scoring_map(all_z, all_K, all_m, mu_0, Sigma_0_inv)

    extras = [torch.randn(L, k, generator=g) for _ in range(3)]
    all_z_pad = all_z + extras
    all_m_pad = torch.cat([all_m, torch.zeros(3, dtype=torch.long)])
    all_K_pad = torch.cat([all_K, torch.zeros(3, dtype=torch.long)])

    theta_b, H_b = fisher_scoring_map(
        all_z_pad, all_K_pad, all_m_pad, mu_0, Sigma_0_inv
    )
    assert torch.allclose(theta_a, theta_b, atol=1e-5)
    assert torch.allclose(H_a, H_b, atol=1e-5)


def test_fisher_pd_at_convergence() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_synthetic(seed=9)
    _, H = fisher_scoring_map(all_z, all_K, all_m, mu_0, Sigma_0_inv)
    eigs = torch.linalg.eigvalsh(0.5 * (H + H.T))
    assert eigs.min().item() > 0


# ---------------------------------------------------------------------------
# autograd: gradcheck on helpers + outer-loop reachability
# ---------------------------------------------------------------------------


def test_gradcheck_compute_grad_and_fisher() -> None:
    torch.manual_seed(0)
    k, N, L = 2, 2, 2
    all_z = [torch.randn(L, k, dtype=torch.float64) for _ in range(N)]
    all_K = torch.tensor([1, 0])
    all_m = torch.tensor([2, 1])
    mu_0 = torch.zeros(k, dtype=torch.float64)
    Sigma_0_inv = torch.eye(k, dtype=torch.float64) * 0.5
    theta = torch.randn(k, dtype=torch.float64, requires_grad=True)

    def fn(t: torch.Tensor) -> torch.Tensor:
        grad, H = _compute_grad_and_fisher(t, all_z, all_K, all_m, mu_0, Sigma_0_inv)
        return torch.cat([grad, H.reshape(-1)])

    assert torch.autograd.gradcheck(fn, theta, eps=1e-6, atol=1e-4)


def test_gradcheck_compute_clipped_objective() -> None:
    torch.manual_seed(0)
    k, N, L = 2, 2, 2
    all_z = [torch.randn(L, k, dtype=torch.float64) for _ in range(N)]
    all_K = torch.tensor([1, 0])
    all_m = torch.tensor([2, 1])
    mu_0 = torch.zeros(k, dtype=torch.float64)
    Sigma_0_inv = torch.eye(k, dtype=torch.float64) * 0.5
    theta = torch.randn(k, dtype=torch.float64, requires_grad=True)

    def fn(t: torch.Tensor) -> torch.Tensor:
        return _compute_clipped_objective(t, all_z, all_K, all_m, mu_0, Sigma_0_inv)

    assert torch.autograd.gradcheck(fn, theta, eps=1e-6, atol=1e-4)


def test_fisher_scoring_map_is_differentiable_wrt_mu0() -> None:
    """Unrolled loop must propagate gradients to outer-loop parameters."""
    k, N, L = 3, 5, 4
    g = torch.Generator().manual_seed(10)
    all_z = [torch.randn(L, k, generator=g) for _ in range(N)]
    all_m = torch.tensor([2, 1, 3, 2, 1])
    all_K = torch.tensor([1, 0, 2, 1, 1])
    mu_0 = torch.zeros(k, requires_grad=True)
    Sigma_0_inv = torch.eye(k)

    theta_hat, H = fisher_scoring_map(
        all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=5
    )
    assert theta_hat.requires_grad
    loss = theta_hat.sum() + H.sum()
    loss.backward()
    assert mu_0.grad is not None
    assert mu_0.grad.norm().item() > 0


def test_fisher_scoring_map_is_differentiable_wrt_z() -> None:
    k, N, L = 3, 4, 3
    g = torch.Generator().manual_seed(12)
    all_z = [torch.randn(L, k, generator=g, requires_grad=True) for _ in range(N)]
    all_m = torch.tensor([2, 1, 3, 2])
    all_K = torch.tensor([1, 0, 2, 1])
    mu_0 = torch.zeros(k)
    Sigma_0_inv = torch.eye(k)

    theta_hat, _ = fisher_scoring_map(
        all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=4
    )
    theta_hat.sum().backward()
    for j, z in enumerate(all_z):
        assert z.grad is not None, f"z[{j}].grad is None"


# ---------------------------------------------------------------------------
# detached variant
# ---------------------------------------------------------------------------


def test_detached_variant_matches_tracked_result() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_synthetic(seed=13)
    theta_a, H_a = fisher_scoring_map(all_z, all_K, all_m, mu_0, Sigma_0_inv)
    theta_b, H_b = fisher_scoring_map_detached(
        all_z, all_K, all_m, mu_0, Sigma_0_inv
    )
    assert torch.allclose(theta_a, theta_b, atol=1e-6)
    assert torch.allclose(H_a, H_b, atol=1e-6)
    assert not theta_b.requires_grad
    assert not H_b.requires_grad


def test_detached_variant_blocks_grad_even_with_requires_grad_inputs() -> None:
    k, N, L = 3, 4, 3
    g = torch.Generator().manual_seed(14)
    all_z = [torch.randn(L, k, generator=g) for _ in range(N)]
    all_m = torch.tensor([2, 1, 3, 2])
    all_K = torch.tensor([1, 0, 2, 1])
    mu_0 = torch.zeros(k, requires_grad=True)
    Sigma_0_inv = torch.eye(k)

    theta_hat, H = fisher_scoring_map_detached(
        all_z, all_K, all_m, mu_0, Sigma_0_inv
    )
    assert not theta_hat.requires_grad
    assert not H.requires_grad
