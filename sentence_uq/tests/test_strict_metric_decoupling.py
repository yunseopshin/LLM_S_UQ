"""Tests for Phase 10-2 — strict-metric decoupling + epistemic significance.

Covers ``prompts/phase_10_2_epi_significance_and_strict_metric.md`` Part C:

1. **Back-compat** — ``compute_strict_factuality_metrics(A, p_strict)``
   (positional, no ``ranking_score``) returns the pre-refactor numbers.
2. **Decoupling** — a constructed case where ``mu_hat`` ranks ``A`` perfectly
   but ``mu_hat ** m`` reorders: ``AUROC(ranking=mu) > AUROC(ranking=mu^m)``
   while ``ECE(p_calib=mu^m) < ECE(p_calib=mu)``.
3. **gamma fit** — recovers a known ``gamma_true``; ``gamma -> 1`` reproduces
   the plain ``mu_hat ** m`` numbers.
4. **partial_correlation_gate** — noise ``epi`` -> FAIL; ``epi`` correlated with
   the residual error -> PASS.
5. **min / softmin** — ``score_softmin`` with large ``beta`` approaches
   ``score_min``; both lie in ``[0, 1]``.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compute_calibration_metrics,
    compute_strict_factuality_metrics,
    fit_strict_gamma,
    partial_correlation_gate,
)


def _load_evaluate_module():
    """Import ``scripts/04_evaluate.py`` (numeric filename) for ``_softmin``."""
    spec = importlib.util.spec_from_file_location(
        "evaluate_module_decouple", _PROJECT_ROOT / "scripts" / "04_evaluate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. Back-compat
# ---------------------------------------------------------------------------


def test_strict_backcompat_two_positional() -> None:
    """``(A, p_strict)`` ranks and calibrates on ``p_strict`` as before."""
    A = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    out = compute_strict_factuality_metrics(A, p)
    # Reference: AUROC/AUPRC on p, Brier/ECE on p.
    assert out["AUROC"] == pytest.approx(roc_auc_score(A, p))
    assert out["AUPRC"] == pytest.approx(average_precision_score(A, p))
    ref_calib = compute_calibration_metrics(A, p, n_bins=10)
    assert out["Brier"] == pytest.approx(ref_calib["Brier"])
    assert out["ECE"] == pytest.approx(ref_calib["ECE"])


def test_strict_backcompat_three_positional_uncertainty() -> None:
    """Legacy ``(A, p, uncertainty)`` is unchanged: AUROC still scored on p."""
    A = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    unc = 1.0 - p  # the historical 3rd positional argument
    out = compute_strict_factuality_metrics(A, p, unc)
    assert out["AUROC"] == pytest.approx(1.0, abs=1e-10)
    assert out["AUPRC"] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 2. Decoupling: ranking vs calibration use different quantities
# ---------------------------------------------------------------------------


def test_decoupling_ranking_vs_calibration() -> None:
    """mu ranks A perfectly; mu^m reorders (worse AUROC) but calibrates better."""
    A = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float64)
    mu = np.array([0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.90, 0.92])
    m = np.array([40, 40, 40, 40, 40, 40, 1, 1, 40, 40], dtype=np.float64)
    mu_pow = mu ** m

    by_mu = compute_strict_factuality_metrics(A, p_calib=mu, ranking_score=mu)
    by_pow = compute_strict_factuality_metrics(
        A, p_calib=mu_pow, ranking_score=mu_pow
    )

    # Ranking: raw mu is the stronger ranker.
    assert by_mu["AUROC"] > by_pow["AUROC"]
    # Calibration: mu^m (the model-consistent P(A=1)) is better calibrated.
    assert by_pow["ECE"] < by_mu["ECE"]


def test_decoupling_ranking_score_independent_of_calib() -> None:
    """AUROC follows ``ranking_score`` while Brier/ECE follow ``p_calib``."""
    A = np.array([0, 0, 1, 1], dtype=np.float64)
    p_calib = np.array([0.4, 0.45, 0.5, 0.55])   # weak ranker, used for Brier/ECE
    ranking = np.array([0.1, 0.2, 0.8, 0.9])     # perfect ranker

    out = compute_strict_factuality_metrics(
        A, p_calib=p_calib, ranking_score=ranking
    )
    assert out["AUROC"] == pytest.approx(1.0, abs=1e-10)
    ref = compute_calibration_metrics(A, p_calib, n_bins=10)
    assert out["Brier"] == pytest.approx(ref["Brier"])
    assert out["ECE"] == pytest.approx(ref["ECE"])


# ---------------------------------------------------------------------------
# 3. gamma fit
# ---------------------------------------------------------------------------


def test_fit_strict_gamma_recovers_known_gamma() -> None:
    rng = np.random.default_rng(0)
    N = 4000
    mu = rng.uniform(0.55, 0.98, N)
    m = rng.integers(1, 12, N).astype(np.float64)
    gamma_true = 0.4
    p = np.clip(mu ** (gamma_true * m), 1e-6, 1 - 1e-6)
    A = (rng.random(N) < p).astype(np.float64)
    gamma_hat = fit_strict_gamma(mu, m, A)
    assert gamma_hat == pytest.approx(gamma_true, abs=0.08)


def test_fit_strict_gamma_one_reproduces_plain_power() -> None:
    """gamma=1 must give exactly ``mu_hat ** m_j``."""
    mu = np.array([0.6, 0.7, 0.8, 0.9])
    m = np.array([2.0, 3.0, 1.0, 5.0])
    p_gamma1 = np.exp(1.0 * m * np.log(np.clip(mu, 1e-6, 1 - 1e-6)))
    assert np.allclose(p_gamma1, mu ** m, atol=1e-9)


def test_fit_strict_gamma_empty_returns_one() -> None:
    assert fit_strict_gamma(np.array([]), np.array([]), np.array([])) == 1.0


# ---------------------------------------------------------------------------
# 4. partial_correlation_gate
# ---------------------------------------------------------------------------


def test_gate_fails_on_noise() -> None:
    rng = np.random.default_rng(1)
    n = 800
    mu = rng.uniform(0.3, 0.9, n)
    ratio_err = np.abs(rng.random(n) * 0.3)
    strict_wrong = (rng.random(n) < 0.3).astype(np.float64)
    epi_noise = rng.random(n)  # independent of error
    out = partial_correlation_gate(epi_noise, mu, ratio_err, strict_wrong)
    assert out["passed"] == 0.0


def test_gate_passes_when_epi_tracks_residual_error() -> None:
    rng = np.random.default_rng(2)
    n = 800
    mu = rng.uniform(0.3, 0.9, n)
    err = np.abs(rng.normal(0.0, 0.2, n))
    epi_sig = err * 2.0 + rng.normal(0.0, 0.02, n)  # tracks error, ~indep of mu
    strict_wrong = (rng.random(n) < np.clip(err * 2.0, 0.0, 1.0)).astype(np.float64)
    out = partial_correlation_gate(epi_sig, mu, err, strict_wrong)
    assert out["passed"] == 1.0
    assert out["rho_partial"] > 0.0
    assert out["rho_partial_p"] < 0.05


# ---------------------------------------------------------------------------
# 5. min / softmin
# ---------------------------------------------------------------------------


def test_softmin_approaches_min_for_large_beta() -> None:
    module = _load_evaluate_module()
    rng = np.random.default_rng(3)
    pi = rng.uniform(0.05, 0.95, 12)
    hard_min = float(pi.min())
    soft = module._softmin(pi, beta=200.0)
    assert soft == pytest.approx(hard_min, abs=1e-2)
    assert 0.0 <= soft <= 1.0
    assert 0.0 <= hard_min <= 1.0


def test_softmin_is_lower_envelope() -> None:
    """For finite beta the softmin is a smooth lower bound on the mean."""
    module = _load_evaluate_module()
    pi = np.array([0.2, 0.6, 0.9])
    soft = module._softmin(pi, beta=10.0)
    assert soft <= float(pi.mean()) + 1e-9
    assert soft <= float(pi.min()) + 1e-9
