"""Tests for the Phase 10-1 Beta-Binomial observation model.

Covers the six requirements of ``prompts/phase_10_1_beta_binomial.md``
section 5:

1. **Binomial regression** -- injecting a ``BinomialLikelihood`` into the
   inner loop reproduces the no-``likelihood`` path *exactly*, and both
   match a hand-coded transcription of the historical Binomial arithmetic
   (the gradient / Fisher are bit-identical; the line-search objective is
   identical up to the documented <=1 ULP reassociation).
2. **phi -> infinity limit** -- with ``phi = 1e6`` the Beta-Binomial
   scalars collapse onto the Binomial limits (``score_mu``,
   ``neg_log_pmf``, ``log_prob_all_correct`` match ``BinomialLikelihood``;
   ``fisher_weight`` matches the observed-information closed form, and the
   Binomial *expected* information only at the optimum ``K = m mu``).
3. **Decomposition reduction** -- ``Predictor`` with ``rho = 0`` reproduces
   the current numbers; with ``rho > 0`` the ratio/count aleatoric scale by
   ``1 + (m - 1) rho`` and every term stays non-negative.
4. **Gradient flow** -- ``gradcheck`` on the Beta-Binomial scalars w.r.t.
   ``mu`` and ``phi``, plus an end-to-end check that one outer step yields a
   finite, non-zero gradient on ``log_phi`` through ``fisher_scoring_map``.
5. **Positivity** -- ``fisher_weight > 0`` for random valid
   ``(K, m, mu, phi)``.
6. **m_j = 0 skip** -- unaffected by the Beta-Binomial likelihood.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.extractor import SentenceUQParams  # noqa: E402
from src.inference.predict import Predictor  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.models.fisher_scoring import (  # noqa: E402
    _compute_clipped_objective,
    _compute_grad_and_fisher,
    fisher_scoring_map,
)
from src.models.likelihood import (  # noqa: E402
    BetaBinomialLikelihood,
    BinomialLikelihood,
    make_likelihood,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_synthetic(
    k: int = 5,
    N: int = 20,
    L: int = 8,
    m_max: int = 6,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
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


def _ref_binomial_grad_fisher(
    theta, all_z, all_K, all_m, mu_0, Sigma_0_inv, eps=1e-6
):
    """Hand-coded transcription of the *historical* Binomial arithmetic."""
    diff = theta - mu_0
    grad = -(Sigma_0_inv @ diff)
    H = Sigma_0_inv
    for j in range(len(all_K)):
        if int(all_m[j].item()) == 0:
            continue
        z_j = all_z[j]
        K_j = all_K[j].to(theta.dtype)
        m_j = all_m[j].to(theta.dtype)
        pi = torch.sigmoid(z_j @ theta)
        mu_c = torch.clamp(pi.mean(), eps, 1.0 - eps)
        w = pi * (1.0 - pi)
        g_j = (w.unsqueeze(1) * z_j).mean(dim=0)
        denom = mu_c * (1.0 - mu_c)
        R_j = (K_j - m_j * mu_c) / denom
        grad = grad + R_j * g_j
        H = H + (m_j / denom) * torch.outer(g_j, g_j)
    return grad, H


# ---------------------------------------------------------------------------
# 1. Binomial regression -- bit-identical to the historical path
# ---------------------------------------------------------------------------


def test_binomial_likelihood_matches_no_likelihood_path_exactly() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_synthetic(seed=1)
    theta = torch.randn(5, dtype=torch.float64, generator=torch.Generator().manual_seed(7))

    g_none, H_none = _compute_grad_and_fisher(
        theta, all_z, all_K, all_m, mu_0, Sigma_0_inv
    )
    g_bin, H_bin = _compute_grad_and_fisher(
        theta, all_z, all_K, all_m, mu_0, Sigma_0_inv,
        likelihood=BinomialLikelihood(),
    )
    # Spec test 1: BinomialLikelihood == no-likelihood path, exactly.
    assert torch.equal(g_none, g_bin)
    assert torch.equal(H_none, H_bin)

    # ... and both equal a hand-coded transcription of the OLD arithmetic.
    g_ref, H_ref = _ref_binomial_grad_fisher(
        theta, all_z, all_K, all_m, mu_0, Sigma_0_inv
    )
    assert torch.equal(g_bin, g_ref)
    assert torch.equal(H_bin, H_ref)


def test_binomial_objective_matches_no_likelihood_path() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_synthetic(seed=2)
    theta = torch.randn(5, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    obj_none = _compute_clipped_objective(
        theta, all_z, all_K, all_m, mu_0, Sigma_0_inv
    )
    obj_bin = _compute_clipped_objective(
        theta, all_z, all_K, all_m, mu_0, Sigma_0_inv,
        likelihood=BinomialLikelihood(),
    )
    # Same code path -> exact.
    assert torch.equal(obj_none, obj_bin)


def test_binomial_fisher_scoring_map_bit_identical() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_synthetic(seed=3)
    theta_a, H_a = fisher_scoring_map(all_z, all_K, all_m, mu_0, Sigma_0_inv)
    theta_b, H_b = fisher_scoring_map(
        all_z, all_K, all_m, mu_0, Sigma_0_inv,
        likelihood=BinomialLikelihood(),
    )
    assert torch.equal(theta_a, theta_b)
    assert torch.equal(H_a, H_b)


def _ref_original_clipped_objective(theta, all_z, all_K, all_m, mu_0, Sigma_0_inv, eps):
    """Pre-10-1 objective: term-wise ``(obj + A) + B`` accumulation."""
    diff = theta - mu_0
    obj = -0.5 * (diff @ (Sigma_0_inv @ diff))
    for j in range(len(all_K)):
        if int(all_m[j].item()) == 0:
            continue
        K_j = all_K[j].to(theta.dtype)
        m_j = all_m[j].to(theta.dtype)
        mu_c = torch.clamp(torch.sigmoid(all_z[j] @ theta).mean(), eps, 1.0 - eps)
        obj = obj + K_j * torch.log(mu_c) + (m_j - K_j) * torch.log(1.0 - mu_c)
    return obj


def _ref_original_core(all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters, eps, lam0):
    """Independent transcription of the PRE-10-1 Fisher-scoring core."""
    k = mu_0.shape[0]
    theta = mu_0.clone()
    lam = float(lam0)
    eye = torch.eye(k, dtype=mu_0.dtype)
    prev = _ref_original_clipped_objective(theta, all_z, all_K, all_m, mu_0, Sigma_0_inv, eps)
    for _ in range(num_iters):
        grad, H = _ref_binomial_grad_fisher(theta, all_z, all_K, all_m, mu_0, Sigma_0_inv, eps)
        try:
            delta = torch.linalg.solve(H + lam * eye, grad)
        except RuntimeError:
            lam *= 10.0
            if lam > 1e10:
                break
            continue
        theta_new = theta + delta
        new = _ref_original_clipped_objective(theta_new, all_z, all_K, all_m, mu_0, Sigma_0_inv, eps)
        if new.item() > prev.item():
            theta = theta_new
            prev = new
            lam = max(lam / 2.0, 1e-8)
        else:
            lam *= 10.0
            if lam > 1e10:
                break
    _, H_final = _ref_binomial_grad_fisher(theta, all_z, all_K, all_m, mu_0, Sigma_0_inv, eps)
    return theta, H_final


@pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
def test_fisher_map_float32_golden_matches_pre_10_1(seed: int) -> None:
    """Float32 golden-MAP pin: the refactored Binomial path must be bit-identical
    to an independent transcription of the pre-10-1 core (catches the line-search
    objective reassociation the loose-atol suite cannot)."""
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_synthetic(
        k=6, N=24, L=7, seed=seed, dtype=torch.float32
    )
    theta_new, H_new = fisher_scoring_map(
        all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=15, eps=1e-6, lambda_init=1e-4
    )
    theta_ref, H_ref = _ref_original_core(
        all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=15, eps=1e-6, lam0=1e-4
    )
    assert torch.equal(theta_new, theta_ref)
    assert torch.equal(H_new, H_ref)


# ---------------------------------------------------------------------------
# 2. phi -> infinity limit
# ---------------------------------------------------------------------------


def test_phi_to_infinity_recovers_binomial_scalars() -> None:
    torch.set_default_dtype(torch.float64)
    try:
        g = torch.Generator().manual_seed(11)
        bn = BinomialLikelihood()
        bb = BetaBinomialLikelihood(phi=torch.tensor(1e6, dtype=torch.float64))
        for _ in range(2000):
            m = float(torch.randint(1, 9, (1,), generator=g).item())
            K = float(torch.randint(0, int(m) + 1, (1,), generator=g).item())
            # Moderate mu keeps the analytic O(1/phi) score correction < 1e-3.
            mu = float(torch.empty(1).uniform_(0.2, 0.8, generator=g).item())
            Kt, mt, mut = (torch.tensor(x, dtype=torch.float64) for x in (K, m, mu))

            assert torch.allclose(
                bb.score_mu(Kt, mt, mut), bn.score_mu(Kt, mt, mut), atol=1e-3
            )
            assert torch.allclose(
                bb.neg_log_pmf(Kt, mt, mut), bn.neg_log_pmf(Kt, mt, mut), atol=1e-3
            )
            assert torch.allclose(
                bb.log_prob_all_correct(mt, mut),
                bn.log_prob_all_correct(mt, mut),
                atol=1e-3,
            )
            # fisher_weight -> observed information K/mu^2 + (m-K)/(1-mu)^2.
            obs = K / mu ** 2 + (m - K) / (1.0 - mu) ** 2
            assert torch.allclose(
                bb.fisher_weight(Kt, mt, mut),
                torch.tensor(obs, dtype=torch.float64),
                rtol=1e-3,
            )
    finally:
        torch.set_default_dtype(torch.float32)


def test_phi_to_infinity_fisher_weight_equals_expected_at_optimum() -> None:
    """At K = m*mu the observed info equals the Binomial expected info."""
    g = torch.Generator().manual_seed(12)
    bn = BinomialLikelihood()
    bb = BetaBinomialLikelihood(phi=torch.tensor(1e6, dtype=torch.float64))
    for _ in range(300):
        m = int(torch.randint(2, 9, (1,), generator=g).item())
        K = int(torch.randint(1, m, (1,), generator=g).item())
        mu = K / m
        Kt, mt, mut = (torch.tensor(float(x), dtype=torch.float64) for x in (K, m, mu))
        assert torch.allclose(
            bb.fisher_weight(Kt, mt, mut),
            bn.fisher_weight(Kt, mt, mut),
            rtol=1e-3,
        )


def test_phi_to_infinity_log_prob_all_correct_is_m_log_mu() -> None:
    bb = BetaBinomialLikelihood(phi=torch.tensor(1e7, dtype=torch.float64))
    m = torch.tensor(4.0, dtype=torch.float64)
    mu = torch.tensor(0.6, dtype=torch.float64)
    assert torch.allclose(
        bb.log_prob_all_correct(m, mu),
        m * torch.log(mu),
        atol=1e-3,
    )


# ---------------------------------------------------------------------------
# 3. Decomposition reduction
# ---------------------------------------------------------------------------


def _make_predictor(
    k: int = 6, sigma_scale: float = 0.1, seed: int = 0, likelihood=None
) -> Predictor:
    g = torch.Generator().manual_seed(seed)
    params = SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=k - 2)
    theta_hat = torch.randn(k, generator=g)
    A = torch.randn(k, k, generator=g)
    Sigma_hat = sigma_scale * (A @ A.T + 0.05 * torch.eye(k))
    return Predictor(
        theta_hat=theta_hat,
        Sigma_hat=Sigma_hat,
        feature_params=params,
        likelihood=likelihood,
    )


def test_decomposition_rho_zero_reproduces_current_numbers() -> None:
    k = 6
    g = torch.Generator().manual_seed(21)
    z = torch.randn(7, k, generator=g)
    pred_default = _make_predictor(k=k, seed=21)
    pred_binom = _make_predictor(
        k=k, seed=21, likelihood=make_likelihood("binomial")
    )
    for m in (1, 3, 8):
        a = pred_default.predict_sentence(z, m_j=m)
        b = pred_binom.predict_sentence(z, m_j=m)
        # Explicit Binomial likelihood == default (rho = 0).
        assert a["aleatoric_U"] == pytest.approx(b["aleatoric_U"], rel=1e-12)
        assert a["aleatoric_K"] == pytest.approx(b["aleatoric_K"], rel=1e-12)
        assert a["p_strict_factual"] == pytest.approx(b["p_strict_factual"], rel=1e-12)
        # ... and reproduces the historical closed forms.
        inner = max(0.0, a["mu_hat"] * (1.0 - a["mu_hat"]) - a["epi_mu"])
        assert a["aleatoric_U"] == pytest.approx(inner / m, rel=1e-9, abs=1e-12)
        assert a["aleatoric_K"] == pytest.approx(m * inner, rel=1e-9, abs=1e-12)
        assert a["p_strict_factual"] == pytest.approx(a["mu_hat"] ** m, rel=1e-9)


def test_decomposition_overdispersion_scales_aleatoric() -> None:
    k = 6
    g = torch.Generator().manual_seed(22)
    z = torch.randn(9, k, generator=g)
    phi = torch.tensor(4.0)  # rho = 0.2 -- clearly overdispersed
    pred_bin = _make_predictor(k=k, seed=22)
    pred_bb = _make_predictor(k=k, seed=22, likelihood=BetaBinomialLikelihood(phi=phi))
    # Use the predictor's own rho (float32 phi -> rho rounds in float32) so the
    # expected f matches the value predict actually uses.
    rho = pred_bb.rho

    for m in (1, 2, 5, 10):
        b = pred_bin.predict_sentence(z, m_j=m)
        bb = pred_bb.predict_sentence(z, m_j=m)
        f = 1.0 + (m - 1) * rho
        # Epistemic is likelihood-independent.
        assert bb["epi_mu"] == pytest.approx(b["epi_mu"], rel=1e-12)
        assert bb["epi_K"] == pytest.approx(b["epi_K"], rel=1e-12)
        # Ratio / count aleatoric scale by f.
        assert bb["aleatoric_U"] == pytest.approx(f * b["aleatoric_U"], rel=1e-9, abs=1e-12)
        assert bb["aleatoric_K"] == pytest.approx(f * b["aleatoric_K"], rel=1e-9, abs=1e-12)
        # m = 1 -> f = 1 -> unchanged.
        if m == 1:
            assert bb["aleatoric_U"] == pytest.approx(b["aleatoric_U"], rel=1e-9, abs=1e-12)
        # Non-negativity of every reported term.
        for key in ("aleatoric_U", "total_U", "epi_K", "aleatoric_K", "epi_mu"):
            assert bb[key] >= 0.0
        assert 0.0 <= bb["p_strict_factual"] <= 1.0


def test_decomposition_strict_uses_beta_binomial_form() -> None:
    """Beta-Binomial strict probability differs from the mu^m plug-in."""
    k = 6
    g = torch.Generator().manual_seed(23)
    z = torch.randn(5, k, generator=g)
    phi = torch.tensor(3.0)
    pred_bin = _make_predictor(k=k, seed=23)
    pred_bb = _make_predictor(k=k, seed=23, likelihood=BetaBinomialLikelihood(phi=phi))
    out_bin = pred_bin.predict_sentence(z, m_j=6)
    out_bb = pred_bb.predict_sentence(z, m_j=6)
    mu = out_bb["mu_hat"]
    # Reference: lgamma form for P(K = m), computed in float64 to match the
    # predictor's float64 strict path.
    phi64 = phi.to(torch.float64)
    m_t = torch.tensor(6.0, dtype=torch.float64)
    mu_t = torch.tensor(mu, dtype=torch.float64)
    a = phi64 * mu_t
    expected = float(
        torch.exp(
            torch.lgamma(m_t + a)
            - torch.lgamma(a)
            + torch.lgamma(phi64)
            - torch.lgamma(m_t + phi64)
        ).item()
    )
    assert out_bb["p_strict_factual"] == pytest.approx(expected, rel=1e-6)
    # Overdispersion makes "all m correct" more likely than the mu^m plug-in.
    assert out_bb["p_strict_factual"] > out_bin["p_strict_factual"]


# ---------------------------------------------------------------------------
# 4. Gradient flow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["score_mu", "fisher_weight", "neg_log_pmf"])
def test_gradcheck_beta_binomial_scalars_wrt_mu_and_phi(method: str) -> None:
    K = torch.tensor(2.0, dtype=torch.float64)
    m = torch.tensor(5.0, dtype=torch.float64)

    def fn(mu: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        L = BetaBinomialLikelihood(phi=phi)
        return getattr(L, method)(K, m, mu)

    mu = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(28.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (mu, phi), eps=1e-6, atol=1e-4)


def test_gradcheck_log_prob_all_correct_wrt_mu_and_phi() -> None:
    m = torch.tensor(5.0, dtype=torch.float64)

    def fn(mu: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return BetaBinomialLikelihood(phi=phi).log_prob_all_correct(m, mu)

    mu = torch.tensor(0.42, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (mu, phi), eps=1e-6, atol=1e-4)


def test_outer_step_produces_nonzero_log_phi_grad() -> None:
    """One compute_loss backward must hit log_phi through the unrolled MAP."""
    k = 6
    params = SentenceUQParams(
        hidden_dim=8, num_layers=3, projection_dim=k - 2,
        likelihood="beta_binomial", phi_init=20.0, learn_phi=True,
    )
    model = BayesianSentenceUQ(params, num_fisher_iters=4)
    g = torch.Generator().manual_seed(31)
    all_z = [torch.randn(4, k, generator=g) for _ in range(6)]
    all_m = torch.tensor([2, 3, 1, 4, 2, 5])
    all_K = torch.tensor([1, 0, 1, 4, 1, 2])

    loss = model.compute_loss(all_z, all_K, all_m)
    loss.backward()

    assert params.log_phi.grad is not None
    assert torch.isfinite(params.log_phi.grad).all()
    assert params.log_phi.grad.abs().item() > 0.0


def test_fisher_scoring_map_is_differentiable_wrt_log_phi() -> None:
    """log_phi receives gradient through the inner loop alone (no outer NLL)."""
    k = 4
    g = torch.Generator().manual_seed(32)
    all_z = [torch.randn(5, k, generator=g, dtype=torch.float64) for _ in range(5)]
    all_m = torch.tensor([2, 3, 4, 2, 5])
    all_K = torch.tensor([1, 2, 0, 1, 3])
    mu_0 = torch.zeros(k, dtype=torch.float64)
    Sigma_0_inv = torch.eye(k, dtype=torch.float64)

    log_phi = torch.tensor(math.log(30.0), dtype=torch.float64, requires_grad=True)
    like = BetaBinomialLikelihood(phi=torch.exp(log_phi))
    theta_hat, H = fisher_scoring_map(
        all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=5, likelihood=like
    )
    assert theta_hat.requires_grad
    (theta_hat.sum() + H.sum()).backward()
    assert log_phi.grad is not None
    assert torch.isfinite(log_phi.grad).all()
    assert log_phi.grad.abs().item() > 0.0


# ---------------------------------------------------------------------------
# 5. Positivity of the Fisher weight
# ---------------------------------------------------------------------------


def test_fisher_weight_strictly_positive() -> None:
    g = torch.Generator().manual_seed(41)
    for _ in range(3000):
        m = int(torch.randint(1, 12, (1,), generator=g).item())
        K = int(torch.randint(0, m + 1, (1,), generator=g).item())
        mu = float(torch.empty(1).uniform_(1e-3, 1 - 1e-3, generator=g).item())
        phi = float(torch.empty(1).uniform_(0.1, 500.0, generator=g).item())
        bb = BetaBinomialLikelihood(phi=torch.tensor(phi, dtype=torch.float64))
        w = bb.fisher_weight(
            torch.tensor(float(K), dtype=torch.float64),
            torch.tensor(float(m), dtype=torch.float64),
            torch.tensor(mu, dtype=torch.float64),
        )
        assert w.item() > 0.0, (K, m, mu, phi, w.item())


# ---------------------------------------------------------------------------
# 6. m_j = 0 skip (unaffected by the Beta-Binomial likelihood)
# ---------------------------------------------------------------------------


def test_m_zero_sentences_do_not_affect_beta_binomial_solution() -> None:
    k, L = 4, 5
    g = torch.Generator().manual_seed(51)
    all_z = [torch.randn(L, k, generator=g, dtype=torch.float64) for _ in range(8)]
    all_m = torch.tensor([2, 3, 1, 4, 2, 3, 1, 2])
    all_K = torch.tensor([1, 2, 0, 3, 1, 1, 0, 1])
    mu_0 = torch.zeros(k, dtype=torch.float64)
    Sigma_0_inv = torch.eye(k, dtype=torch.float64) * 0.5
    phi = torch.tensor(25.0, dtype=torch.float64)

    theta_a, H_a = fisher_scoring_map(
        all_z, all_K, all_m, mu_0, Sigma_0_inv,
        likelihood=BetaBinomialLikelihood(phi=phi),
    )

    extras = [torch.randn(L, k, generator=g, dtype=torch.float64) for _ in range(3)]
    all_z_pad = all_z + extras
    all_m_pad = torch.cat([all_m, torch.zeros(3, dtype=torch.long)])
    all_K_pad = torch.cat([all_K, torch.zeros(3, dtype=torch.long)])

    theta_b, H_b = fisher_scoring_map(
        all_z_pad, all_K_pad, all_m_pad, mu_0, Sigma_0_inv,
        likelihood=BetaBinomialLikelihood(phi=phi),
    )
    assert torch.allclose(theta_a, theta_b, atol=1e-9)
    assert torch.allclose(H_a, H_b, atol=1e-9)


# ---------------------------------------------------------------------------
# Extra: SentenceUQParams gating + persistence round-trip
# ---------------------------------------------------------------------------


def test_binomial_params_have_no_log_phi() -> None:
    p = SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=4)
    assert p.likelihood == "binomial"
    assert p.phi is None
    assert "log_phi" not in dict(p.named_parameters())


def test_beta_binomial_params_register_log_phi() -> None:
    p = SentenceUQParams(
        hidden_dim=8, num_layers=3, projection_dim=4,
        likelihood="beta_binomial", phi_init=50.0,
    )
    assert "log_phi" in dict(p.named_parameters())
    assert p.phi is not None
    assert float(p.phi.item()) == pytest.approx(50.0, rel=1e-6)


def test_learn_phi_false_freezes_log_phi() -> None:
    p = SentenceUQParams(
        hidden_dim=8, num_layers=3, projection_dim=4,
        likelihood="beta_binomial", phi_init=30.0, learn_phi=False,
    )
    assert p.log_phi.requires_grad is False


def test_invalid_likelihood_raises() -> None:
    with pytest.raises(ValueError):
        SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=4,
                         likelihood="poisson")


def test_save_load_round_trip_beta_binomial(tmp_path: Path) -> None:
    from src.inference.predict import load_trained_model, save_trained_model

    k = 6
    params = SentenceUQParams(
        hidden_dim=8, num_layers=3, projection_dim=k - 2,
        likelihood="beta_binomial", phi_init=12.0,
    )
    with torch.no_grad():
        params.log_phi.fill_(math.log(7.5))  # pretend training moved phi
    theta_hat = torch.randn(k)
    A = torch.randn(k, k)
    Sigma_hat = 0.1 * (A @ A.T + torch.eye(k))

    path = tmp_path / "bb_model.pt"
    save_trained_model(path, theta_hat, Sigma_hat, params)
    loaded = load_trained_model(path)

    fp = loaded["feature_params"]
    assert fp.likelihood == "beta_binomial"
    assert float(fp.phi.item()) == pytest.approx(7.5, rel=1e-5)
    like = loaded["likelihood"]
    assert isinstance(like, BetaBinomialLikelihood)
    assert like.rho == pytest.approx(1.0 / (7.5 + 1.0), rel=1e-5)
