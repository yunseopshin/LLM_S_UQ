"""Evaluation metrics for the Bayesian sentence-level factuality UQ.

Phase 6-1 — two-tiered evaluation:

* **Primary (ratio-level)**: ``U_j = K_j / m_j`` in ``[0, 1]`` — continuous.
  Metrics: MAE, RMSE, Pearson r, binomial NLL, ECE, PRR.
* **Secondary (strict factuality)**: ``A_j = 1{K_j = m_j}`` in ``{0, 1}`` —
  binary. Metrics: AUROC, AUPRC, Brier, ECE.

All numerics run in float64 NumPy (CLAUDE.md rule 10: compute in higher
precision, store lower). Sentences with ``m_j = 0`` must be filtered by
the caller before invoking the ratio / strict metrics (CLAUDE.md rule 8);
:func:`full_evaluation` does this filtering itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.validation import validate_binomial_counts


__all__ = [
    "compute_ratio_level_metrics",
    "compute_strict_factuality_metrics",
    "compute_strict_metrics",
    "compute_calibration_metrics",
    "compute_prr",
    "compute_bootstrapped_ci",
    "plot_reliability_diagram",
    "compare_mc_vs_linear_epistemic",
    "full_evaluation",
    "binomial_nll_full",
    "binomial_ce",
    "fit_strict_gamma",
    "partial_correlation_gate",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _to_numpy_1d(x: Any, name: str = "x") -> np.ndarray:
    """Coerce ``x`` to a 1-D float64 NumPy array (errors otherwise).

    Accepts NumPy arrays, Python sequences, and ``torch.Tensor`` (detached
    to CPU before conversion). Anything else raises ``TypeError``.

    Parameters
    ----------
    x : array-like or torch.Tensor
    name : str
        Argument name used in error messages.

    Returns
    -------
    np.ndarray of shape ``(N,)`` and dtype ``float64``.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(arr.shape)}")
    return arr


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson product-moment correlation in float64.

    Returns ``nan`` if either input is constant (zero variance).
    """
    if x.size < 2 or y.size < 2:
        return float("nan")
    dx = x - x.mean()
    dy = y - y.mean()
    denom = float(np.sqrt(np.dot(dx, dx)) * np.sqrt(np.dot(dy, dy)))
    if denom < _EPS:
        return float("nan")
    return float(np.dot(dx, dy) / denom)


def _equal_width_bins(
    y_true: np.ndarray, p_pred: np.ndarray, n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equal-width binning over ``[0, 1]`` for reliability/ECE.

    Each prediction is assigned to a bin by ``floor(p_pred * n_bins)``
    (with the right edge clipped into the final bin).

    Returns
    -------
    bin_counts : (n_bins,) int64
    bin_mean_pred : (n_bins,) float64 (NaN for empty bins)
    bin_mean_true : (n_bins,) float64 (NaN for empty bins)
    bin_centers : (n_bins,) float64
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    p = np.clip(p_pred, 0.0, 1.0)
    idx = np.minimum(np.floor(p * n_bins).astype(np.int64), n_bins - 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    sum_pred = np.zeros(n_bins, dtype=np.float64)
    sum_true = np.zeros(n_bins, dtype=np.float64)
    np.add.at(counts, idx, 1)
    np.add.at(sum_pred, idx, p)
    np.add.at(sum_true, idx, y_true)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_pred = np.where(counts > 0, sum_pred / np.maximum(counts, 1), np.nan)
        mean_true = np.where(counts > 0, sum_true / np.maximum(counts, 1), np.nan)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, mean_pred, mean_true, centers


# ---------------------------------------------------------------------------
# 1. Ratio-level metrics
# ---------------------------------------------------------------------------


def compute_ratio_level_metrics(
    U_true: Any,
    mu_hat: Any,
    m_j: Optional[Any] = None,
) -> Dict[str, float]:
    """Primary metrics for the ratio-level target ``U_j = K_j / m_j``.

    Implements §6 (primary panel) of ``research_document_v8.md``.

    Parameters
    ----------
    U_true : array-like of shape ``(N,)``
        Observed factuality ratio ``K_j / m_j`` in ``[0, 1]``.
    mu_hat : array-like of shape ``(N,)``
        Predicted ``μ̂_j`` in ``[0, 1]``.
    m_j : array-like of shape ``(N,)``, optional
        Per-sentence atomic-fact count. Required for ``binomial_NLL``;
        when omitted the NLL key is left out of the result.

    Returns
    -------
    dict with keys:
        ``MAE``         : float — mean absolute error
        ``RMSE``        : float — root mean squared error
        ``Pearson_r``   : float — Pearson correlation
        ``binomial_NLL``: float — mean ``-[K log μ̂ + (m-K) log(1-μ̂)]``
                          (only if ``m_j`` provided)
    """
    U = _to_numpy_1d(U_true, "U_true")
    mu = _to_numpy_1d(mu_hat, "mu_hat")
    if U.shape != mu.shape:
        raise ValueError(
            f"U_true and mu_hat shapes differ: {U.shape} vs {mu.shape}"
        )
    if U.size == 0:
        raise ValueError("Cannot compute metrics on empty inputs")

    diff = mu - U
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    pearson = _pearson_r(U, mu)

    out: Dict[str, float] = {
        "MAE": mae,
        "RMSE": rmse,
        "Pearson_r": pearson,
    }

    if m_j is not None:
        m = _to_numpy_1d(m_j, "m_j")
        if m.shape != U.shape:
            raise ValueError(
                f"m_j shape {m.shape} != U_true shape {U.shape}"
            )
        if np.any(m < 0):
            raise ValueError("m_j must be non-negative")
        K = U * m
        mu_safe = np.clip(mu, _EPS, 1.0 - _EPS)
        nll = -(K * np.log(mu_safe) + (m - K) * np.log(1.0 - mu_safe))
        out["binomial_NLL"] = float(np.mean(nll))

    return out


# ---------------------------------------------------------------------------
# 2. Strict factuality metrics
# ---------------------------------------------------------------------------


def compute_strict_factuality_metrics(
    A_true: Any,
    p_calib: Any,
    uncertainty: Any = None,
    ranking_score: Any = None,
) -> Dict[str, float]:
    """Secondary metrics for the binary target ``A_j = 1{K_j = m_j}``.

    Phase 10-2 (Part B) decouples *ranking* from *calibration* because they
    measure different things and need different inputs:

    * **Brier / ECE** (calibration) are computed on ``p_calib``, which must be
      an estimate of ``P(A_j = 1)`` — the model-consistent object is
      ``mu_hat ** m_j``, **not** the raw mean ``mu_hat`` (feeding raw ``mu_hat``
      is a type mismatch that inflates ECE).
    * **AUROC / AUPRC** (ranking, calibration-invariant) are computed on
      ``ranking_score`` (higher -> more likely ``A_j = 1``). When
      ``ranking_score is None`` it defaults to ``p_calib``.

    Backward compatibility
    ----------------------
    Legacy callers used the positional form
    ``compute_strict_factuality_metrics(A, p_strict, uncertainty)`` and got
    AUROC/AUPRC scored on ``p_strict``. With ``ranking_score`` defaulting to
    ``p_calib`` those calls are bit-for-bit unchanged. The Phase 10-2 spec
    lists ``ranking_score`` ahead of ``uncertainty`` in the signature; we keep
    ``uncertainty`` in its historical 3rd-positional slot instead so the
    pre-existing positional callers (and ``tests/test_metrics.py``) keep their
    exact behaviour (hard constraint 0.1). New call sites pass
    ``ranking_score=`` by keyword.

    Parameters
    ----------
    A_true : array-like of shape ``(N,)`` in ``{0, 1}``
    p_calib : array-like of shape ``(N,)`` in ``[0, 1]``
        Probability used for Brier/ECE; must estimate ``P(A_j = 1)``
        (e.g. ``mu_hat ** m_j``).
    uncertainty : array-like of shape ``(N,)``, optional
        Higher = more uncertain (kept for the rejection-curve API; not used by
        any returned metric). When ``None`` it is derived as
        ``1 - ranking_score``.
    ranking_score : array-like of shape ``(N,)``, optional
        Score used for AUROC/AUPRC (higher -> more likely ``A_j = 1``). When
        ``None``, defaults to ``p_calib`` (back-compat).

    Returns
    -------
    dict with keys ``{"AUROC", "AUPRC", "Brier", "ECE"}``.
    """
    A = _to_numpy_1d(A_true, "A_true")
    p = _to_numpy_1d(p_calib, "p_calib")
    s = p if ranking_score is None else _to_numpy_1d(ranking_score, "ranking_score")
    u = (1.0 - s) if uncertainty is None else _to_numpy_1d(uncertainty, "uncertainty")
    if not (A.shape == p.shape == s.shape == u.shape):
        raise ValueError(
            f"shape mismatch: A_true {A.shape}, p_calib {p.shape}, "
            f"ranking_score {s.shape}, uncertainty {u.shape}"
        )
    if A.size == 0:
        raise ValueError("Cannot compute metrics on empty inputs")
    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A_true must contain only {0, 1}")

    # AUROC / AUPRC need at least one of each class. Ranking uses ``ranking_score``.
    if A.sum() == 0 or A.sum() == A.size:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(A, s))
        auprc = float(average_precision_score(A, s))

    # Brier / ECE are calibration of ``p_calib`` (the estimate of P(A_j=1)).
    calib = compute_calibration_metrics(A, p, n_bins=10)
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Brier": calib["Brier"],
        "ECE": calib["ECE"],
    }


def compute_strict_metrics(
    K: Any,
    m: Any,
    mu_hat: Any,
) -> Dict[str, float]:
    """Strict factuality + error detection AUROC, both directions explicit.

    Phase 7-3 fix 6: the legacy ``compute_strict_factuality_metrics`` reports
    a single ``AUROC`` whose meaning depends on label / score direction. This
    helper returns **both** directions so the paper can pick the correct
    headline metric without ambiguity.

    Definitions
    -----------
    * ``A_strict[j] = 1{K_j == m_j}`` — sentence is fully factual.
    * ``p_strict[j] = μ̂_j ** m_j`` — model probability of "all atoms supported".
    * ``E_error[j]  = 1 - A_strict[j]`` — at least one atom unsupported.
    * ``p_error[j]  = 1 - p_strict[j]``.

    By the standard AUROC identity ``AUROC(1 - y, 1 - s) == AUROC(y, s)``,
    the two AUROCs are equal — exposing both names makes it impossible to
    accidentally swap the label direction when writing tables.

    Parameters
    ----------
    K : array-like of shape ``(N,)``
        Supported-atom counts.
    m : array-like of shape ``(N,)``
        Total atomic-fact counts. Must satisfy ``K_j ≤ m_j`` (validated).
    mu_hat : array-like of shape ``(N,)``
        Predicted per-sentence factuality.

    Returns
    -------
    dict with keys
        ``strict_factuality_auroc`` : float
        ``error_detection_auroc``   : float
    """
    K_arr = _to_numpy_1d(K, "K")
    m_arr = _to_numpy_1d(m, "m")
    mu_arr = _to_numpy_1d(mu_hat, "mu_hat")
    if not (K_arr.shape == m_arr.shape == mu_arr.shape):
        raise ValueError(
            f"shape mismatch: K {K_arr.shape}, m {m_arr.shape}, "
            f"mu_hat {mu_arr.shape}"
        )
    validate_binomial_counts(K_arr, m_arr, context="strict_metrics")

    A_strict = (K_arr == m_arr).astype(np.float64)
    p_strict = np.power(np.clip(mu_arr, 0.0, 1.0), m_arr)

    E_error = 1.0 - A_strict
    p_error = 1.0 - p_strict

    out: Dict[str, float] = {}
    if len(np.unique(A_strict)) > 1:
        out["strict_factuality_auroc"] = float(roc_auc_score(A_strict, p_strict))
        out["error_detection_auroc"] = float(roc_auc_score(E_error, p_error))
    else:
        out["strict_factuality_auroc"] = float("nan")
        out["error_detection_auroc"] = float("nan")
    return out


def binomial_nll_full(
    K: Any,
    m: Any,
    mu_hat: Any,
    eps: float = 1e-8,
) -> float:
    """Full binomial NLL including the combinatorial constant.

    ``NLL = -[ log C(m, K) + K log μ̂ + (m - K) log(1 - μ̂) ]``,
    averaged over sentences. The combinatorial term uses ``gammaln`` so it
    handles non-integer inputs gracefully.

    Parameters
    ----------
    K, m, mu_hat : array-like of shape ``(N,)``
    eps : float
        Clipping bound applied to ``mu_hat`` for log-stability.
    """
    from scipy.special import gammaln

    K_arr = _to_numpy_1d(K, "K")
    m_arr = _to_numpy_1d(m, "m")
    mu_arr = _to_numpy_1d(mu_hat, "mu_hat")
    if not (K_arr.shape == m_arr.shape == mu_arr.shape):
        raise ValueError(
            f"shape mismatch: K {K_arr.shape}, m {m_arr.shape}, "
            f"mu_hat {mu_arr.shape}"
        )
    mu_safe = np.clip(mu_arr, eps, 1.0 - eps)
    log_comb = gammaln(m_arr + 1.0) - gammaln(K_arr + 1.0) - gammaln(m_arr - K_arr + 1.0)
    nll = -(log_comb + K_arr * np.log(mu_safe) + (m_arr - K_arr) * np.log(1.0 - mu_safe))
    return float(nll.mean())


def binomial_ce(
    K: Any,
    m: Any,
    mu_hat: Any,
    eps: float = 1e-8,
) -> float:
    """Binomial cross-entropy (no combinatorial constant).

    ``CE = -[ K log μ̂ + (m - K) log(1 - μ̂) ]``, averaged over sentences.
    Matches the existing ``binomial_NLL`` field returned by
    :func:`compute_ratio_level_metrics`.
    """
    K_arr = _to_numpy_1d(K, "K")
    m_arr = _to_numpy_1d(m, "m")
    mu_arr = _to_numpy_1d(mu_hat, "mu_hat")
    if not (K_arr.shape == m_arr.shape == mu_arr.shape):
        raise ValueError(
            f"shape mismatch: K {K_arr.shape}, m {m_arr.shape}, "
            f"mu_hat {mu_arr.shape}"
        )
    mu_safe = np.clip(mu_arr, eps, 1.0 - eps)
    ce = -(K_arr * np.log(mu_safe) + (m_arr - K_arr) * np.log(1.0 - mu_safe))
    return float(ce.mean())


# ---------------------------------------------------------------------------
# 2b. Strict-metric decoupling helpers (Phase 10-2 Part B / A)
# ---------------------------------------------------------------------------


def fit_strict_gamma(
    mu_hat_val: Any,
    m_val: Any,
    A_val: Any,
    bounds: Tuple[float, float] = (1e-3, 5.0),
    eps: float = 1e-6,
) -> float:
    """1-parameter recalibration of the strict-event probability (Phase 10-2 B.3).

    Models ``P(A_j = 1; gamma) = mu_hat_j ** (gamma * m_j)`` and fits the
    single scalar ``gamma > 0`` on a **validation** split by minimising the
    strict negative log-likelihood

        ``-sum_j [ A_j log p_j + (1 - A_j) log(1 - p_j) ]``,

    with ``p_j`` clamped to ``[eps, 1 - eps]`` for log-stability. ``gamma = 1``
    recovers the Binomial plug-in ``mu_hat ** m_j``; ``gamma < 1`` softens the
    length penalty. This is fit on the strict event ``A_j`` directly, so it is
    distinct from the Beta-Binomial ``rho`` (fit on counts).

    Parameters
    ----------
    mu_hat_val : array-like of shape ``(N,)`` in ``[0, 1]``
        Validation per-sentence mean factuality ``mu_hat_j``.
    m_val : array-like of shape ``(N,)``
        Validation atomic-fact counts ``m_j``.
    A_val : array-like of shape ``(N,)`` in ``{0, 1}``
        Validation strict labels ``1{K_j = m_j}``.
    bounds : (float, float)
        Search interval for ``gamma`` (default ``(1e-3, 5.0)``).
    eps : float
        Probability clamp bound.

    Returns
    -------
    float — the fitted ``gamma_hat``. Returns ``1.0`` when the validation set
    is empty (no information to recalibrate).
    """
    from scipy.optimize import minimize_scalar

    mu = _to_numpy_1d(mu_hat_val, "mu_hat_val")
    m = _to_numpy_1d(m_val, "m_val")
    A = _to_numpy_1d(A_val, "A_val")
    if not (mu.shape == m.shape == A.shape):
        raise ValueError(
            f"shape mismatch: mu_hat_val {mu.shape}, m_val {m.shape}, "
            f"A_val {A.shape}"
        )
    if mu.size == 0:
        return 1.0

    mu_safe = np.clip(mu, eps, 1.0 - eps)
    log_mu = np.log(mu_safe)

    def _nll(gamma: float) -> float:
        # p = mu ** (gamma * m) computed via exp for stability.
        p = np.exp(gamma * m * log_mu)
        p = np.clip(p, eps, 1.0 - eps)
        return float(-np.sum(A * np.log(p) + (1.0 - A) * np.log(1.0 - p)))

    res = minimize_scalar(_nll, bounds=bounds, method="bounded")
    gamma_hat = float(res.x)
    # Keep strictly within bounds (bounded method can return an edge value).
    return float(min(max(gamma_hat, bounds[0]), bounds[1]))


def _residualize_quadratic(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residuals of ``y`` after an OLS regression on ``[1, x, x^2]``.

    Used to control for the point estimate ``mu_hat`` (and its curvature)
    before testing whether ``epi_mu`` adds predictive power.
    """
    X = np.column_stack([np.ones_like(x), x, x * x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _standardize(x: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-std standardisation (no-op scale when std ~ 0)."""
    mu = float(x.mean())
    sd = float(x.std())
    if sd < _EPS:
        return x - mu
    return (x - mu) / sd


def _logistic_wald(
    y: np.ndarray, X: np.ndarray, max_iter: int = 100, tol: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Newton-Raphson (IRLS) logistic fit with Wald standard errors.

    Fits ``P(y=1) = sigmoid(X beta)`` where ``X`` already contains an intercept
    column, and returns ``(beta, se, pvalues)`` with two-sided Wald p-values
    ``2 (1 - Phi(|beta / se|))``. A tiny ridge stabilises the Hessian for
    near-collinear / separable designs. Raises ``ValueError`` if ``y`` is
    single-class (no fit possible).

    Returns
    -------
    beta : (d,) float64
    se : (d,) float64
    pvalues : (d,) float64
    """
    from scipy.stats import norm

    y = y.astype(np.float64)
    X = X.astype(np.float64)
    if np.unique(y).size < 2:
        raise ValueError("logistic fit needs both classes present in y")
    n, d = X.shape
    beta = np.zeros(d, dtype=np.float64)
    ridge = 1e-8 * np.eye(d)
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1.0 - p), 1e-12, None)
        grad = X.T @ (y - p)
        hess = X.T @ (X * w[:, None]) + ridge
        delta = np.linalg.solve(hess, grad)
        beta = beta + delta
        if float(np.max(np.abs(delta))) < tol:
            break
    cov = np.linalg.inv(hess)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > _EPS, beta / se, 0.0)
    pvalues = 2.0 * (1.0 - norm.cdf(np.abs(z)))
    return beta, se, pvalues


def partial_correlation_gate(
    epi: Any,
    mu_hat: Any,
    ratio_err: Any,
    strict_wrong: Any,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Significance gate: does ``epi`` predict error *after* controlling ``mu_hat``?

    Phase 10-2 Part A primary diagnostic. Two independent checks, framed as
    significance (not magnitude):

    1. **Partial Spearman** between ``residualize(epi | mu_hat)`` and
       ``residualize(ratio_err | mu_hat)``, where ``residualize(y | x)``
       regresses ``y`` on ``[1, x, x^2]`` and returns residuals.
    2. **Logistic check** ``strict_wrong ~ 1 + z(mu_hat) + z(epi)`` (predictors
       standardised); we report the ``epi`` coefficient, its Wald p-value and
       sign.

    The gate **PASSES** iff ``epi`` has the correct sign (higher ``epi`` ->
    larger error) AND ``p < alpha`` in at least one of the two checks.

    Parameters
    ----------
    epi : array-like of shape ``(N,)``
        Epistemic signal (e.g. ``epi_mu``).
    mu_hat : array-like of shape ``(N,)``
        Point estimate to control for.
    ratio_err : array-like of shape ``(N,)``
        Continuous error ``|U_j - mu_hat|`` (target for the Spearman check).
    strict_wrong : array-like of shape ``(N,)`` in ``{0, 1}``
        Strict error ``1 - A_j`` (target for the logistic check).
    alpha : float
        Significance threshold (default 0.05).

    Returns
    -------
    dict with keys:
        ``rho_partial``, ``rho_partial_p``, ``spearman_pass``,
        ``logit_epi_coef``, ``logit_epi_p``, ``logit_epi_sign``,
        ``logistic_pass``, ``passed`` (1.0/0.0), ``n``.
    """
    from scipy.stats import spearmanr

    e = _to_numpy_1d(epi, "epi")
    mu = _to_numpy_1d(mu_hat, "mu_hat")
    rerr = _to_numpy_1d(ratio_err, "ratio_err")
    sw = _to_numpy_1d(strict_wrong, "strict_wrong")
    if not (e.shape == mu.shape == rerr.shape == sw.shape):
        raise ValueError(
            f"shape mismatch: epi {e.shape}, mu_hat {mu.shape}, "
            f"ratio_err {rerr.shape}, strict_wrong {sw.shape}"
        )
    if e.size == 0:
        raise ValueError("Cannot run the gate on empty inputs")

    # --- (1) partial Spearman -------------------------------------------------
    rho_partial = float("nan")
    rho_partial_p = float("nan")
    try:
        epi_res = _residualize_quadratic(e, mu)
        err_res = _residualize_quadratic(rerr, mu)
        if np.std(epi_res) > _EPS and np.std(err_res) > _EPS:
            rho, p_spear = spearmanr(epi_res, err_res)
            rho_partial = float(rho)
            rho_partial_p = float(p_spear)
    except (ValueError, FloatingPointError):
        pass
    spearman_pass = bool(
        np.isfinite(rho_partial)
        and np.isfinite(rho_partial_p)
        and rho_partial > 0.0
        and rho_partial_p < alpha
    )

    # --- (2) logistic strict_wrong ~ 1 + z(mu) + z(epi) -----------------------
    logit_epi_coef = float("nan")
    logit_epi_p = float("nan")
    logit_epi_sign = 0
    try:
        X = np.column_stack(
            [np.ones_like(mu), _standardize(mu), _standardize(e)]
        )
        beta, _se, pvals = _logistic_wald(sw, X)
        logit_epi_coef = float(beta[2])
        logit_epi_p = float(pvals[2])
        logit_epi_sign = int(np.sign(beta[2]))
    except (ValueError, np.linalg.LinAlgError):
        pass
    logistic_pass = bool(
        np.isfinite(logit_epi_coef)
        and np.isfinite(logit_epi_p)
        and logit_epi_coef > 0.0
        and logit_epi_p < alpha
    )

    passed = bool(spearman_pass or logistic_pass)
    return {
        "rho_partial": rho_partial,
        "rho_partial_p": rho_partial_p,
        "spearman_pass": float(spearman_pass),
        "logit_epi_coef": logit_epi_coef,
        "logit_epi_p": logit_epi_p,
        "logit_epi_sign": float(logit_epi_sign),
        "logistic_pass": float(logistic_pass),
        "passed": float(passed),
        "n": float(e.size),
    }


# ---------------------------------------------------------------------------
# 3. Calibration metrics
# ---------------------------------------------------------------------------


def compute_calibration_metrics(
    y_true: Any,
    p_pred: Any,
    n_bins: int = 10,
) -> Dict[str, float]:
    """General-purpose calibration: Brier score and equal-width ECE.

    Works for both the ratio target (``y_true ∈ [0, 1]``) and the strict
    target (``y_true ∈ {0, 1}``).

    Parameters
    ----------
    y_true : array-like of shape ``(N,)`` in ``[0, 1]``
    p_pred : array-like of shape ``(N,)`` in ``[0, 1]``
    n_bins : int
        Number of equal-width bins on ``[0, 1]``. Default ``10``.

    Returns
    -------
    dict with keys:
        ``Brier`` : float — ``mean((y - p)²)``
        ``ECE``   : float — ``Σ_b (|B_b| / N) · |mean(p|B_b) - mean(y|B_b)|``
    """
    y = _to_numpy_1d(y_true, "y_true")
    p = _to_numpy_1d(p_pred, "p_pred")
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape}, p_pred {p.shape}")
    if y.size == 0:
        raise ValueError("Cannot compute calibration on empty inputs")

    brier = float(np.mean((y - p) * (y - p)))

    counts, mean_pred, mean_true, _ = _equal_width_bins(y, p, n_bins)
    N = float(y.size)
    diffs = np.abs(mean_pred - mean_true)
    # Empty bins contribute 0 (their weight is 0).
    diffs = np.where(counts > 0, diffs, 0.0)
    ece = float(np.sum(counts / N * diffs))

    return {"Brier": brier, "ECE": ece}


# ---------------------------------------------------------------------------
# 4. Prediction Rejection Ratio (PRR)
# ---------------------------------------------------------------------------


def compute_prr(
    y_true: Any,
    uncertainty: Any,
    num_thresholds: int = 100,
) -> Dict[str, Any]:
    """Rejection-curve quality vs fraction of samples removed.

    Samples are sorted by ``uncertainty`` (descending) and the top fraction
    is removed; ``remaining_quality`` is the mean of ``y_true`` over the
    samples that remain. Higher ``y_true`` is interpreted as better
    (correctness / factuality ratio), so a useful uncertainty signal makes
    ``remaining_quality`` rise as the rejection rate grows. ``prr_auc`` is
    the trapezoidal area under the curve over ``rejection_rate ∈ [0, 1)``.

    Parameters
    ----------
    y_true : array-like of shape ``(N,)``
        Per-sample quality signal (``A_j`` for strict, ``U_j`` for ratio).
    uncertainty : array-like of shape ``(N,)``
        Higher = more uncertain.
    num_thresholds : int
        Number of rejection thresholds, evenly spaced in ``[0, 1)``.

    Returns
    -------
    dict with keys:
        ``rejection_rates``    : np.ndarray of shape ``(num_thresholds,)``
        ``remaining_quality``  : np.ndarray of shape ``(num_thresholds,)``
        ``prr_auc``            : float
    """
    y = _to_numpy_1d(y_true, "y_true")
    u = _to_numpy_1d(uncertainty, "uncertainty")
    if y.shape != u.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape}, uncertainty {u.shape}")
    if y.size == 0:
        raise ValueError("Cannot compute PRR on empty inputs")
    if num_thresholds <= 0:
        raise ValueError(f"num_thresholds must be positive, got {num_thresholds}")

    N = y.size
    # Ascending sort by uncertainty: low-uncertainty samples first.
    order = np.argsort(u, kind="mergesort")
    y_sorted = y[order]

    rejection_rates = np.linspace(0.0, 1.0, num_thresholds, endpoint=False)
    remaining_quality = np.empty(num_thresholds, dtype=np.float64)
    for i, r in enumerate(rejection_rates):
        keep = int(np.floor((1.0 - r) * N))
        keep = max(keep, 1)
        remaining_quality[i] = float(np.mean(y_sorted[:keep]))

    prr_auc = float(np.trapezoid(remaining_quality, rejection_rates))

    return {
        "rejection_rates": rejection_rates,
        "remaining_quality": remaining_quality,
        "prr_auc": prr_auc,
    }


# ---------------------------------------------------------------------------
# 5. Bootstrapped confidence intervals
# ---------------------------------------------------------------------------


def compute_bootstrapped_ci(
    y_true: Any,
    scores: Any,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Percentile bootstrap CI for an arbitrary scalar metric.

    Resamples paired ``(y_true, scores)`` with replacement ``n_bootstrap``
    times and applies ``metric_fn`` to each resample.

    Parameters
    ----------
    y_true : array-like of shape ``(N,)``
    scores : array-like of shape ``(N,)``
    metric_fn : callable
        ``metric_fn(y_true_np, scores_np) -> float``.
    n_bootstrap : int
        Number of resamples (default 1000).
    alpha : float
        Two-sided CI level (default 0.05 → 95 % CI).
    seed : int, optional
        Forwarded to ``np.random.default_rng`` for reproducibility.

    Returns
    -------
    dict with keys ``{"mean", "lower", "upper"}`` — all float.
    """
    y = _to_numpy_1d(y_true, "y_true")
    s = _to_numpy_1d(scores, "scores")
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape}, scores {s.shape}")
    if y.size == 0:
        raise ValueError("Cannot bootstrap empty inputs")
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    rng = np.random.default_rng(seed)
    N = y.size
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        samples[b] = float(metric_fn(y[idx], s[idx]))

    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "lower": float(np.quantile(finite, alpha / 2.0)),
        "upper": float(np.quantile(finite, 1.0 - alpha / 2.0)),
    }


# ---------------------------------------------------------------------------
# 6. Reliability diagram
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    y_true: Any,
    p_pred: Any,
    n_bins: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "",
) -> Any:
    """Reliability diagram with the ``y = x`` diagonal.

    Parameters
    ----------
    y_true : array-like of shape ``(N,)`` in ``[0, 1]``
    p_pred : array-like of shape ``(N,)`` in ``[0, 1]``
    n_bins : int
        Number of equal-width bins (default 10).
    save_path : str | Path, optional
        If given, the figure is written to this path (parent dirs created).
    title : str
        Optional axes title.

    Returns
    -------
    matplotlib.figure.Figure — the created figure (caller may further
    customise / close it).
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    y = _to_numpy_1d(y_true, "y_true")
    p = _to_numpy_1d(p_pred, "p_pred")
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape}, p_pred {p.shape}")

    counts, mean_pred, mean_true, centers = _equal_width_bins(y, p, n_bins)

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0,
            color="gray", label="Perfect calibration")

    mask = counts > 0
    if mask.any():
        ax.plot(mean_pred[mask], mean_true[mask], marker="o",
                linewidth=1.5, color="C0", label="Empirical")
        widths = 1.0 / n_bins * 0.8
        # Bar heights = mean accuracy in each bin (also visualises empty bins).
        ax.bar(centers, np.where(mask, mean_true, 0.0), width=widths,
               alpha=0.25, color="C0", edgecolor="C0", align="center")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)

    return fig


# ---------------------------------------------------------------------------
# 7. MC vs Linear epistemic comparison
# ---------------------------------------------------------------------------


def compare_mc_vs_linear_epistemic(
    predictor: Any,
    test_sentences: Sequence[Any],
    num_mc_samples: int = 100,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Pairwise comparison of delta-method vs Monte-Carlo epistemic.

    For each sentence, computes the linear (delta-method) ``epi_mu`` from
    :meth:`Predictor.predict_sentence` and the Monte-Carlo
    ``mc_epi_mu`` from :meth:`Predictor.predict_mc_epistemic` (the
    sample variance of ``μ̂(θ^(s))``). Returns the per-sentence arrays
    plus their Pearson correlation and MAE — the §6 sanity check for
    "MC vs linear epistemic correlation > 0.9".

    Parameters
    ----------
    predictor : Predictor
        Trained Phase 3-3 predictor.
    test_sentences : sequence
        Each element is either a ``(L_j, k)`` tensor of token features
        ``z_tokens`` or a ``(z_tokens, m_j)`` tuple. ``m_j`` is unused
        here (latent comparison only) but accepted for API symmetry.
    num_mc_samples : int
        ``S`` for the MC sampler (default 100).
    generator : torch.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    dict with keys:
        ``linear_epi`` : np.ndarray of shape ``(N,)``
        ``mc_epi``     : np.ndarray of shape ``(N,)``
        ``Pearson_r``  : float
        ``MAE``        : float
    """
    if len(test_sentences) == 0:
        raise ValueError("test_sentences must be non-empty")

    linear = np.empty(len(test_sentences), dtype=np.float64)
    mc = np.empty(len(test_sentences), dtype=np.float64)
    for i, item in enumerate(test_sentences):
        if isinstance(item, tuple):
            z = item[0]
        else:
            z = item
        lin_out = predictor.predict_sentence(z, m_j=None)
        mc_out = predictor.predict_mc_epistemic(
            z, num_samples=num_mc_samples, generator=generator
        )
        linear[i] = float(lin_out["epi_mu"])
        mc[i] = float(mc_out["mc_epi_mu"])

    pearson = _pearson_r(linear, mc)
    mae = float(np.mean(np.abs(linear - mc)))
    return {
        "linear_epi": linear,
        "mc_epi": mc,
        "Pearson_r": pearson,
        "MAE": mae,
    }


# ---------------------------------------------------------------------------
# 8. Full evaluation
# ---------------------------------------------------------------------------


def full_evaluation(
    predictions: Dict[str, Any],
    K_true: Any,
    m_true: Any,
    uncertainties: Any,
) -> pd.DataFrame:
    """All ratio + strict metrics at once, returned as a tidy DataFrame.

    Sentences with ``m_j = 0`` are dropped before any metric is computed
    (CLAUDE.md rule 8).

    Parameters
    ----------
    predictions : dict
        Must contain ``"mu_hat"`` (predicted ``μ̂_j``) and
        ``"p_strict_factual"`` (predicted ``μ̂_j^{m_j}``), each shape
        ``(N,)``. May also contain ``"epi_mu"`` and other diagnostics
        which are forwarded into the DataFrame ``info`` rows.
    K_true : array-like of shape ``(N,)``
        Observed supported-atom count per sentence.
    m_true : array-like of shape ``(N,)``
        Atomic-fact count per sentence (``m_j = 0`` rows are skipped).
    uncertainties : array-like of shape ``(N,)``
        Higher = more uncertain (used for PRR and the strict ranking).

    Returns
    -------
    pandas.DataFrame with columns ``["metric", "tier", "value"]`` —
    one row per scalar metric, sorted (ratio first, then strict, then
    info / counts).
    """
    if "mu_hat" not in predictions:
        raise KeyError("predictions must contain key 'mu_hat'")
    if "p_strict_factual" not in predictions:
        raise KeyError("predictions must contain key 'p_strict_factual'")

    mu = _to_numpy_1d(predictions["mu_hat"], "predictions['mu_hat']")
    p_str = _to_numpy_1d(predictions["p_strict_factual"], "predictions['p_strict_factual']")
    K = _to_numpy_1d(K_true, "K_true")
    m = _to_numpy_1d(m_true, "m_true")
    u = _to_numpy_1d(uncertainties, "uncertainties")
    if not (mu.shape == p_str.shape == K.shape == m.shape == u.shape):
        raise ValueError(
            "all of mu_hat, p_strict_factual, K_true, m_true, "
            f"uncertainties must share shape; got "
            f"{mu.shape}, {p_str.shape}, {K.shape}, {m.shape}, {u.shape}"
        )

    validate_binomial_counts(K, m, context="full_evaluation")

    keep = m > 0
    n_total = int(m.size)
    n_kept = int(keep.sum())
    n_skipped = n_total - n_kept
    if n_kept == 0:
        raise ValueError("All sentences have m_j = 0 — nothing to evaluate")

    mu = mu[keep]
    p_str = p_str[keep]
    K = K[keep]
    m = m[keep]
    u = u[keep]

    U = K / m
    # Strict factuality: label=1 means all atoms supported (K_j == m_j).
    A = (K == m).astype(np.float64)

    rows: List[Tuple[str, str, float]] = []

    # --- ratio-level ---
    ratio = compute_ratio_level_metrics(U, mu, m_j=m)
    for name in ("MAE", "RMSE", "Pearson_r", "binomial_NLL"):
        if name in ratio:
            rows.append((name, "ratio", float(ratio[name])))
    # binomial_NLL above is the cross-entropy (no combinatorial constant).
    # Phase 7-3 fix 7 surfaces both names explicitly + the full binomial NLL
    # (with the log C(m, K) term) so the paper can pick the right headline.
    rows.append(("binomial_ce", "ratio", float(binomial_ce(K, m, mu))))
    rows.append(("binomial_nll_full", "ratio", float(binomial_nll_full(K, m, mu))))
    ratio_calib = compute_calibration_metrics(U, mu, n_bins=10)
    rows.append(("Brier", "ratio", float(ratio_calib["Brier"])))
    rows.append(("ECE", "ratio", float(ratio_calib["ECE"])))
    ratio_prr = compute_prr(U, u, num_thresholds=100)
    rows.append(("PRR_AUC", "ratio", float(ratio_prr["prr_auc"])))

    # --- strict factuality ---
    strict = compute_strict_factuality_metrics(A, p_str, u)
    for name in ("AUROC", "AUPRC", "Brier", "ECE"):
        rows.append((name, "strict", float(strict[name])))
    strict_prr = compute_prr(A, u, num_thresholds=100)
    rows.append(("PRR_AUC", "strict", float(strict_prr["prr_auc"])))

    # --- error detection: label=1 means at least one atom unsupported ---
    strict_explicit = compute_strict_metrics(K, m, mu)
    rows.append((
        "strict_factuality_auroc",
        "strict",
        float(strict_explicit["strict_factuality_auroc"]),
    ))
    rows.append((
        "error_detection_auroc",
        "strict",
        float(strict_explicit["error_detection_auroc"]),
    ))

    # --- bookkeeping rows ---
    rows.append(("n_sentences", "info", float(n_kept)))
    rows.append(("n_skipped_m0", "info", float(n_skipped)))
    rows.append(("frac_strict_factual", "info", float(A.mean())))

    return pd.DataFrame(rows, columns=["metric", "tier", "value"])
