"""Tests for ``src.evaluation.metrics`` — Phase 6-1.

Covers the spec invariants from ``prompts/phase_6_1_metrics.md``:

* Ratio: perfect prediction ``U_true == mu_hat`` → MAE = 0, Pearson r = 1.
* Strict: perfect ranking → AUROC = 1.0.
* Perfect calibration ``p_pred = y_true`` → Brier = 0, ECE ≈ 0.
* Binomial NLL: closed-form numeric case.

Plus auxiliary checks for PRR, bootstrap CI, the reliability-diagram
plot helper, MC-vs-linear epistemic comparison (against the Phase 3-3
:class:`Predictor`), and ``full_evaluation`` returning a tidy DataFrame.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg")  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compare_mc_vs_linear_epistemic,
    compute_bootstrapped_ci,
    compute_calibration_metrics,
    compute_prr,
    compute_ratio_level_metrics,
    compute_strict_factuality_metrics,
    compute_bootstrapped_ci as _ci_alias,  # ensure import works
    full_evaluation,
    plot_reliability_diagram,
)
from src.features.extractor import SentenceUQParams  # noqa: E402
from src.inference.predict import Predictor  # noqa: E402


# ---------------------------------------------------------------------------
# Ratio-level metrics
# ---------------------------------------------------------------------------


def test_ratio_perfect_prediction() -> None:
    """U_true == mu_hat → MAE = 0, RMSE = 0, Pearson r = 1."""
    U = np.linspace(0.05, 0.95, 20)
    out = compute_ratio_level_metrics(U_true=U, mu_hat=U)
    assert out["MAE"] == pytest.approx(0.0, abs=1e-12)
    assert out["RMSE"] == pytest.approx(0.0, abs=1e-12)
    assert out["Pearson_r"] == pytest.approx(1.0, abs=1e-12)
    assert "binomial_NLL" not in out


def test_ratio_constant_pearson_is_nan() -> None:
    U = np.full(10, 0.5)
    mu = np.linspace(0.1, 0.9, 10)
    out = compute_ratio_level_metrics(U_true=U, mu_hat=mu)
    assert math.isnan(out["Pearson_r"])


def test_ratio_mae_rmse_closed_form() -> None:
    U = np.array([0.0, 0.5, 1.0])
    mu = np.array([0.1, 0.5, 0.9])
    out = compute_ratio_level_metrics(U_true=U, mu_hat=mu)
    # diffs = [0.1, 0.0, -0.1]
    assert out["MAE"] == pytest.approx((0.1 + 0.0 + 0.1) / 3.0)
    assert out["RMSE"] == pytest.approx(
        math.sqrt((0.01 + 0.0 + 0.01) / 3.0)
    )


def test_binomial_nll_closed_form() -> None:
    """U=0.5, m=2, K=1, μ̂=0.5 → NLL = -2·log(0.5) = 2 ln 2."""
    U = np.array([0.5])
    mu = np.array([0.5])
    m = np.array([2.0])
    out = compute_ratio_level_metrics(U_true=U, mu_hat=mu, m_j=m)
    assert "binomial_NLL" in out
    assert out["binomial_NLL"] == pytest.approx(2.0 * math.log(2.0), rel=1e-10)


def test_binomial_nll_multi_sample() -> None:
    """Average over N samples — extends the closed form to a vector."""
    U = np.array([0.5, 0.5])
    mu = np.array([0.5, 0.25])
    m = np.array([2.0, 4.0])
    K = U * m  # [1, 2]
    # nll_0 = -[1·log .5 + 1·log .5] = 2 ln 2
    # nll_1 = -[2·log .25 + 2·log .75] = 2(2 ln 2) + 2(ln(4/3))
    expected = (
        (-(K[0] * math.log(0.5) + (m[0] - K[0]) * math.log(0.5)))
        + (-(K[1] * math.log(0.25) + (m[1] - K[1]) * math.log(0.75)))
    ) / 2.0
    out = compute_ratio_level_metrics(U_true=U, mu_hat=mu, m_j=m)
    assert out["binomial_NLL"] == pytest.approx(expected, rel=1e-10)


def test_ratio_accepts_torch_tensors() -> None:
    U = torch.tensor([0.0, 0.25, 1.0])
    mu = torch.tensor([0.0, 0.25, 1.0])
    out = compute_ratio_level_metrics(U_true=U, mu_hat=mu)
    assert out["MAE"] == pytest.approx(0.0, abs=1e-12)


def test_ratio_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_ratio_level_metrics(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# Strict factuality metrics
# ---------------------------------------------------------------------------


def test_strict_perfect_ranking_auroc_is_one() -> None:
    """Perfect alignment between p_strict and A_true → AUROC = 1.0."""
    A = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    unc = 1.0 - p
    out = compute_strict_factuality_metrics(A, p, unc)
    assert out["AUROC"] == pytest.approx(1.0, abs=1e-10)
    assert out["AUPRC"] == pytest.approx(1.0, abs=1e-10)


def test_strict_random_predictions_auroc_near_half() -> None:
    rng = np.random.default_rng(0)
    A = rng.integers(0, 2, size=2000).astype(np.float64)
    p = rng.random(size=2000)
    out = compute_strict_factuality_metrics(A, p, p)
    assert 0.4 < out["AUROC"] < 0.6
    assert 0.0 <= out["Brier"] <= 1.0
    assert 0.0 <= out["ECE"] <= 1.0


def test_strict_degenerate_labels_returns_nan_auroc() -> None:
    A = np.zeros(10, dtype=np.float64)
    p = np.linspace(0, 1, 10)
    out = compute_strict_factuality_metrics(A, p, p)
    assert math.isnan(out["AUROC"])
    assert math.isnan(out["AUPRC"])


def test_strict_rejects_non_binary_targets() -> None:
    A = np.array([0.0, 0.5, 1.0])
    p = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError):
        compute_strict_factuality_metrics(A, p, p)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_perfect_zero_brier_zero_ece() -> None:
    """p_pred = y_true with y in {0, 1} → Brier = 0 and ECE = 0."""
    y = np.array([0, 0, 1, 1, 1], dtype=np.float64)
    out = compute_calibration_metrics(y_true=y, p_pred=y, n_bins=10)
    assert out["Brier"] == pytest.approx(0.0, abs=1e-12)
    assert out["ECE"] == pytest.approx(0.0, abs=1e-12)


def test_calibration_constant_prediction_brier_closed_form() -> None:
    """p ≡ 0.5 against y ∈ {0, 1} balanced → Brier = 0.25."""
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    p = np.full_like(y, 0.5)
    out = compute_calibration_metrics(y, p, n_bins=10)
    assert out["Brier"] == pytest.approx(0.25, abs=1e-12)
    # All mass in one bin → ECE = |0.5 - 0.5| = 0.
    assert out["ECE"] == pytest.approx(0.0, abs=1e-12)


def test_calibration_systematic_bias_ece_positive() -> None:
    """Predict 0.9 when truth is 0 → ECE = 0.9 (single bin)."""
    y = np.zeros(100, dtype=np.float64)
    p = np.full_like(y, 0.9)
    out = compute_calibration_metrics(y, p, n_bins=10)
    assert out["Brier"] == pytest.approx(0.81, abs=1e-12)
    assert out["ECE"] == pytest.approx(0.9, abs=1e-12)


# ---------------------------------------------------------------------------
# Prediction Rejection Ratio
# ---------------------------------------------------------------------------


def test_prr_perfect_uncertainty_pushes_quality_to_one() -> None:
    """When uncertainty correlates with errors, remaining_quality should
    monotonically increase toward the best samples."""
    # y_true = correctness in {0, 1}; uncertainty perfectly inverts y_true.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    unc = 1.0 - y  # high uncertainty exactly where wrong
    out = compute_prr(y_true=y, uncertainty=unc, num_thresholds=10)
    rq = out["remaining_quality"]
    # The first threshold keeps everyone (rejection_rate = 0).
    assert rq[0] == pytest.approx(y.mean())
    # As we reject the highest-uncertainty (wrong) samples first,
    # remaining quality must be non-decreasing.
    assert np.all(np.diff(rq) >= -1e-12)
    # After rejecting all 4 wrong samples, remaining is purely correct.
    assert rq[-1] == pytest.approx(1.0)
    assert out["prr_auc"] > rq[0]


def test_prr_random_uncertainty_roughly_flat() -> None:
    rng = np.random.default_rng(123)
    y = rng.integers(0, 2, size=500).astype(np.float64)
    unc = rng.random(size=500)  # independent of y
    out = compute_prr(y, unc, num_thresholds=50)
    # AUC is around base rate when uncertainty is uninformative.
    base = y.mean()
    assert abs(out["prr_auc"] - base) < 0.1


def test_prr_shapes_and_keys() -> None:
    y = np.array([0.0, 0.5, 1.0, 0.2])
    unc = np.array([0.4, 0.1, 0.3, 0.2])
    out = compute_prr(y, unc, num_thresholds=25)
    assert out["rejection_rates"].shape == (25,)
    assert out["remaining_quality"].shape == (25,)
    assert out["rejection_rates"][0] == 0.0
    assert isinstance(out["prr_auc"], float)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_mean_for_constant_data() -> None:
    """Constant inputs → metric is constant → CI is degenerate at the value."""
    y = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
    s = np.array([0.3, 0.3, 0.3, 0.3, 0.3])

    def mae(yt, sc) -> float:
        return float(np.mean(np.abs(yt - sc)))

    ci = compute_bootstrapped_ci(y, s, mae, n_bootstrap=200, seed=0)
    assert ci["mean"] == pytest.approx(0.0, abs=1e-12)
    assert ci["lower"] == pytest.approx(0.0, abs=1e-12)
    assert ci["upper"] == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_ci_brackets_true_mae() -> None:
    rng = np.random.default_rng(7)
    y = rng.random(200)
    s = y + 0.05 * rng.standard_normal(200)
    true_mae = float(np.mean(np.abs(y - s)))

    def mae(yt, sc) -> float:
        return float(np.mean(np.abs(yt - sc)))

    ci = compute_bootstrapped_ci(y, s, mae, n_bootstrap=500, seed=0)
    assert ci["lower"] <= true_mae <= ci["upper"]
    assert ci["lower"] <= ci["mean"] <= ci["upper"]


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------


def test_reliability_diagram_runs_and_saves(tmp_path) -> None:
    rng = np.random.default_rng(2)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(np.float64)
    save_path = tmp_path / "rel.png"
    fig = plot_reliability_diagram(
        y_true=y, p_pred=p, n_bins=10, save_path=save_path, title="test"
    )
    assert save_path.exists() and save_path.stat().st_size > 0
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# MC vs linear epistemic
# ---------------------------------------------------------------------------


def _toy_predictor(k: int = 6, seed: int = 0) -> Predictor:
    g = torch.Generator().manual_seed(seed)
    params = SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=k - 2)
    theta = torch.randn(k, generator=g)
    A = torch.randn(k, k, generator=g)
    # Small Σ → delta method should be tight against MC.
    Sigma = 0.02 * (A @ A.T + 0.05 * torch.eye(k))
    return Predictor(theta_hat=theta, Sigma_hat=Sigma, feature_params=params)


def test_mc_vs_linear_high_correlation_when_sigma_small() -> None:
    pred = _toy_predictor(k=6, seed=0)
    g = torch.Generator().manual_seed(42)
    sentences = [
        torch.randn(5 + i, 6, generator=g) for i in range(12)
    ]
    mc_gen = torch.Generator().manual_seed(123)
    out = compare_mc_vs_linear_epistemic(
        predictor=pred,
        test_sentences=sentences,
        num_mc_samples=300,
        generator=mc_gen,
    )
    assert out["linear_epi"].shape == (12,)
    assert out["mc_epi"].shape == (12,)
    # Small Σ — delta-method should track MC closely.
    assert out["Pearson_r"] > 0.9
    assert out["MAE"] < 0.05


# ---------------------------------------------------------------------------
# full_evaluation
# ---------------------------------------------------------------------------


def test_full_evaluation_returns_tidy_dataframe() -> None:
    rng = np.random.default_rng(3)
    N = 50
    m = rng.integers(1, 6, size=N).astype(np.float64)
    mu = rng.random(N) * 0.8 + 0.1
    # Sample K_j ~ Binomial(m, mu) so the data are consistent with the model.
    K = rng.binomial(m.astype(int), mu).astype(np.float64)
    p_strict = np.clip(mu, 1e-6, 1 - 1e-6) ** m
    unc = 1.0 - mu

    df = full_evaluation(
        predictions={"mu_hat": mu, "p_strict_factual": p_strict},
        K_true=K,
        m_true=m,
        uncertainties=unc,
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["metric", "tier", "value"]
    metrics_by_tier = set(zip(df["metric"], df["tier"]))
    # spot-check that the four spec metrics are present in each tier
    for name in ("MAE", "RMSE", "Pearson_r", "binomial_NLL", "ECE", "PRR_AUC"):
        assert (name, "ratio") in metrics_by_tier
    for name in ("AUROC", "AUPRC", "Brier", "ECE", "PRR_AUC"):
        assert (name, "strict") in metrics_by_tier
    assert (("n_sentences", "info")) in metrics_by_tier
    # n_sentences should equal N (no m_j = 0 in this fixture)
    n_row = df[(df["metric"] == "n_sentences") & (df["tier"] == "info")]
    assert float(n_row["value"].iloc[0]) == float(N)


def test_full_evaluation_skips_m0_rows() -> None:
    N = 8
    mu = np.linspace(0.1, 0.9, N)
    K = np.array([0, 1, 2, 0, 3, 1, 0, 4], dtype=np.float64)
    m = np.array([0, 2, 3, 0, 4, 2, 0, 5], dtype=np.float64)
    p_strict = np.clip(mu, 1e-6, 1 - 1e-6) ** np.where(m > 0, m, 1.0)
    unc = 1.0 - mu

    df = full_evaluation(
        predictions={"mu_hat": mu, "p_strict_factual": p_strict},
        K_true=K, m_true=m, uncertainties=unc,
    )
    n_kept = float(
        df[(df["metric"] == "n_sentences") & (df["tier"] == "info")]
        ["value"].iloc[0]
    )
    n_skip = float(
        df[(df["metric"] == "n_skipped_m0") & (df["tier"] == "info")]
        ["value"].iloc[0]
    )
    assert n_kept == 5.0
    assert n_skip == 3.0


def test_full_evaluation_requires_required_keys() -> None:
    with pytest.raises(KeyError):
        full_evaluation(
            predictions={"mu_hat": np.zeros(3)},
            K_true=np.zeros(3),
            m_true=np.ones(3),
            uncertainties=np.zeros(3),
        )
