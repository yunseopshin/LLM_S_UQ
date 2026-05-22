"""Tests for ``src.models.bayesian_aux`` — Phase 4-2.

Covers:
- :func:`safe_logit` boundary clipping and inverse-sigmoid identity,
- :class:`BayesianLogitRegression` constructor validation and default
  prior shape,
- closed-form posterior recovery on synthetic data with known ``θ``
  (low-noise + large ``N`` → ``θ_N ≈ θ_true``),
- sufficient statistics ``T_1 = Zᵀ Z``, ``T_2 = Zᵀ V`` reproduce the
  posterior exactly (re-derive ``Σ_N``, ``θ_N`` from ``T_1``, ``T_2``
  and the prior),
- predictive shapes, predictive variance decomposition
  ``logit_var = epistemic_logit + σ²``,
- residual-based ``estimate_noise_variance`` recovers ``σ²`` under
  large-N synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.bayesian_aux import (  # noqa: E402
    BayesianLogitRegression,
    safe_logit,
)


# ---------------------------------------------------------------------------
# safe_logit
# ---------------------------------------------------------------------------


def test_safe_logit_clips_boundary_values() -> None:
    u = torch.tensor([0.0, 1.0, 0.5])
    out = safe_logit(u, eps=1e-3)
    assert torch.isfinite(out).all()
    # 0.0 should hit -logit(1e-3), 1.0 should hit +logit(1 - 1e-3).
    expected_low = torch.log(torch.tensor(1e-3)) - torch.log(
        torch.tensor(1.0 - 1e-3)
    )
    expected_high = -expected_low
    assert torch.allclose(out[0], expected_low, atol=1e-6)
    assert torch.allclose(out[1], expected_high, atol=1e-6)
    assert torch.allclose(out[2], torch.tensor(0.0), atol=1e-6)


def test_safe_logit_is_inverse_of_sigmoid_in_interior() -> None:
    logits = torch.linspace(-3.0, 3.0, 11)
    u = torch.sigmoid(logits)
    recovered = safe_logit(u, eps=1e-6)
    assert torch.allclose(recovered, logits, atol=1e-4)


def test_safe_logit_validates_eps() -> None:
    u = torch.tensor([0.1, 0.9])
    with pytest.raises(ValueError):
        safe_logit(u, eps=0.0)
    with pytest.raises(ValueError):
        safe_logit(u, eps=0.5)


# ---------------------------------------------------------------------------
# constructor
# ---------------------------------------------------------------------------


def test_constructor_defaults() -> None:
    model = BayesianLogitRegression(feature_dim=4)
    assert model.feature_dim == 4
    assert model.prior_mu.shape == (4,)
    assert torch.allclose(model.prior_mu, torch.zeros(4, dtype=torch.float64))
    assert model.prior_Sigma.shape == (4, 4)
    # default prior_sigma=1.0 → identity covariance
    assert torch.allclose(
        model.prior_Sigma, torch.eye(4, dtype=torch.float64)
    )
    assert model.theta_N is None
    assert model.Sigma_N is None


def test_constructor_validates_inputs() -> None:
    with pytest.raises(ValueError):
        BayesianLogitRegression(feature_dim=0)
    with pytest.raises(ValueError):
        BayesianLogitRegression(feature_dim=3, noise_sigma=0.0)
    with pytest.raises(ValueError):
        BayesianLogitRegression(feature_dim=3, prior_sigma=-0.1)
    with pytest.raises(ValueError):
        BayesianLogitRegression(
            feature_dim=3, prior_mu=torch.zeros(2)  # wrong dim
        )


def test_constructor_accepts_diagonal_prior_sigma() -> None:
    ps = torch.tensor([0.5, 1.0, 2.0])
    model = BayesianLogitRegression(feature_dim=3, prior_sigma=ps)
    expected = torch.diag((ps * ps).to(torch.float64))
    assert torch.allclose(model.prior_Sigma, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _synthetic_dataset(
    N: int,
    k: int,
    theta_true: torch.Tensor,
    noise_sigma: float,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(N, k, generator=g, dtype=torch.float64)
    V = Z @ theta_true + noise_sigma * torch.randn(
        N, generator=g, dtype=torch.float64
    )
    U_star = torch.sigmoid(V)
    return Z, U_star


# ---------------------------------------------------------------------------
# fit — recovery
# ---------------------------------------------------------------------------


def test_fit_recovers_known_theta_under_low_noise() -> None:
    k = 5
    N = 4000
    torch.manual_seed(1)
    theta_true = torch.tensor([0.7, -1.2, 0.4, 0.1, -0.5], dtype=torch.float64)
    noise = 0.05
    Z, U_star = _synthetic_dataset(N, k, theta_true, noise, seed=1)

    model = BayesianLogitRegression(
        feature_dim=k, prior_sigma=10.0, noise_sigma=noise
    )
    model.fit(Z, U_star)

    assert model.theta_N is not None
    assert model.Sigma_N is not None
    assert model.theta_N.shape == (k,)
    assert model.Sigma_N.shape == (k, k)

    err = (model.theta_N - theta_true).abs().max().item()
    assert err < 0.05, f"recovered θ deviates by {err:.4f}"


def test_fit_sufficient_statistics_reproduce_posterior() -> None:
    """``Σ_N``, ``θ_N`` must be reconstructible from ``(T_1, T_2)`` + prior."""
    k = 4
    N = 200
    torch.manual_seed(2)
    theta_true = torch.tensor([0.3, -0.8, 1.1, 0.2], dtype=torch.float64)
    noise = 0.2
    Z, U_star = _synthetic_dataset(N, k, theta_true, noise, seed=2)

    model = BayesianLogitRegression(
        feature_dim=k, prior_sigma=2.0, noise_sigma=noise
    )
    model.fit(Z, U_star)

    # Recompute T_1, T_2 from raw inputs and verify they match what fit
    # stored. (Sufficient-statistics correctness.)
    Z64 = Z.to(torch.float64)
    V = safe_logit(U_star.to(torch.float64))
    T1_ref = Z64.T @ Z64
    T2_ref = Z64.T @ V
    assert torch.allclose(model.T1, T1_ref, atol=1e-9)
    assert torch.allclose(model.T2, T2_ref, atol=1e-9)

    # Now redo the conjugate update using *only* (T_1, T_2) + prior.
    sigma2 = noise * noise
    Sigma_N_inv = model.prior_Sigma_inv + T1_ref / sigma2
    Sigma_N_ref = torch.linalg.inv(0.5 * (Sigma_N_inv + Sigma_N_inv.T))
    rhs = model.prior_Sigma_inv @ model.prior_mu + T2_ref / sigma2
    theta_N_ref = Sigma_N_ref @ rhs

    assert torch.allclose(model.Sigma_N, Sigma_N_ref, atol=1e-9)
    assert torch.allclose(model.theta_N, theta_N_ref, atol=1e-9)


def test_fit_input_validation() -> None:
    model = BayesianLogitRegression(feature_dim=3)
    with pytest.raises(ValueError):
        model.fit(torch.zeros(5), torch.zeros(5))           # Z not 2-D
    with pytest.raises(ValueError):
        model.fit(torch.zeros(5, 4), torch.zeros(5))        # wrong feature_dim
    with pytest.raises(ValueError):
        model.fit(torch.zeros(5, 3), torch.zeros(5, 1))     # U_star not 1-D
    with pytest.raises(ValueError):
        model.fit(torch.zeros(5, 3), torch.zeros(4))        # length mismatch


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_requires_fit() -> None:
    model = BayesianLogitRegression(feature_dim=3)
    with pytest.raises(RuntimeError):
        model.predict(torch.zeros(3))


def test_predict_shapes_and_decomposition() -> None:
    k = 3
    N = 100
    torch.manual_seed(3)
    theta_true = torch.tensor([0.4, -0.6, 0.2], dtype=torch.float64)
    noise = 0.1
    Z, U_star = _synthetic_dataset(N, k, theta_true, noise, seed=3)
    model = BayesianLogitRegression(feature_dim=k, noise_sigma=noise).fit(
        Z, U_star
    )

    # batched prediction
    z_batch = torch.randn(7, k, dtype=torch.float64)
    out = model.predict(z_batch)
    for key in (
        "p_factual",
        "logit_mean",
        "logit_var",
        "epistemic_logit",
        "aleatoric_logit",
    ):
        assert key in out
        assert out[key].shape == (7,)

    # aleatoric == σ² (constant), variance = epistemic + aleatoric
    sigma2 = noise * noise
    assert torch.allclose(
        out["aleatoric_logit"],
        torch.full((7,), sigma2, dtype=torch.float64),
        atol=1e-12,
    )
    assert torch.allclose(
        out["logit_var"],
        out["epistemic_logit"] + out["aleatoric_logit"],
        atol=1e-12,
    )
    assert torch.allclose(out["p_factual"], torch.sigmoid(out["logit_mean"]))

    # single sample: scalar tensors
    out1 = model.predict(z_batch[0])
    for key in out:
        assert out1[key].dim() == 0


def test_predict_epistemic_nonnegative_and_matches_quadratic_form() -> None:
    k = 4
    N = 150
    torch.manual_seed(4)
    theta_true = torch.tensor([0.5, -0.5, 0.1, 0.3], dtype=torch.float64)
    Z, U_star = _synthetic_dataset(N, k, theta_true, 0.1, seed=4)
    model = BayesianLogitRegression(feature_dim=k, noise_sigma=0.1).fit(
        Z, U_star
    )

    z_new = torch.randn(5, k, dtype=torch.float64)
    out = model.predict(z_new)

    for i in range(5):
        expected = float(z_new[i] @ model.Sigma_N @ z_new[i])
        assert abs(float(out["epistemic_logit"][i]) - expected) < 1e-9
    assert (out["epistemic_logit"] >= 0.0).all()


# ---------------------------------------------------------------------------
# noise variance estimation
# ---------------------------------------------------------------------------


def test_estimate_noise_variance_recovers_true_sigma2() -> None:
    k = 4
    N = 3000
    torch.manual_seed(5)
    theta_true = torch.tensor([0.6, -0.3, 0.9, 0.1], dtype=torch.float64)
    true_noise = 0.25
    Z, U_star = _synthetic_dataset(N, k, theta_true, true_noise, seed=5)

    # Use a weak prior so θ_N is data-driven.
    model = BayesianLogitRegression(
        feature_dim=k, prior_sigma=20.0, noise_sigma=true_noise
    )
    model.fit(Z, U_star)
    sigma2_hat = model.estimate_noise_variance(Z, U_star)

    assert sigma2_hat > 0.0
    rel_err = abs(sigma2_hat - true_noise * true_noise) / (true_noise * true_noise)
    assert rel_err < 0.1, f"σ² estimate off by {rel_err:.3f}"


def test_estimate_noise_variance_requires_fit() -> None:
    model = BayesianLogitRegression(feature_dim=3)
    with pytest.raises(RuntimeError):
        model.estimate_noise_variance(torch.zeros(5, 3), torch.zeros(5))


def test_estimate_noise_variance_requires_sufficient_samples() -> None:
    k = 4
    model = BayesianLogitRegression(feature_dim=k, noise_sigma=0.1)
    Z = torch.randn(4, k, dtype=torch.float64)
    U = torch.rand(4, dtype=torch.float64)
    model.fit(Z, U)
    with pytest.raises(ValueError):
        model.estimate_noise_variance(Z, U)  # N == k


def test_set_noise_sigma_updates_aleatoric_in_predict() -> None:
    k = 3
    N = 50
    torch.manual_seed(6)
    theta_true = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float64)
    Z, U_star = _synthetic_dataset(N, k, theta_true, 0.1, seed=6)
    model = BayesianLogitRegression(feature_dim=k, noise_sigma=0.1).fit(
        Z, U_star
    )

    out_before = model.predict(Z[0])
    model.set_noise_sigma(0.3)
    out_after = model.predict(Z[0])

    assert float(out_after["aleatoric_logit"]) == pytest.approx(0.09, rel=1e-9)
    assert float(out_before["aleatoric_logit"]) == pytest.approx(0.01, rel=1e-9)

    with pytest.raises(ValueError):
        model.set_noise_sigma(-1.0)
