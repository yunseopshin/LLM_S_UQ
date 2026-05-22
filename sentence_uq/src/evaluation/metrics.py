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


__all__ = [
    "compute_ratio_level_metrics",
    "compute_strict_factuality_metrics",
    "compute_calibration_metrics",
    "compute_prr",
    "compute_bootstrapped_ci",
    "plot_reliability_diagram",
    "compare_mc_vs_linear_epistemic",
    "full_evaluation",
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
    p_strict: Any,
    uncertainty: Any,
) -> Dict[str, float]:
    """Secondary metrics for the binary target ``A_j = 1{K_j = m_j}``.

    AUROC / AUPRC are computed against ``p_strict`` (probability that
    ``A_j = 1``). When ``uncertainty`` ranks samples better than the
    probability does, AUROC may differ; both are reported for the
    probability-as-score convention. Brier and ECE are calibration
    measures of ``p_strict``.

    Parameters
    ----------
    A_true : array-like of shape ``(N,)`` in ``{0, 1}``
    p_strict : array-like of shape ``(N,)`` in ``[0, 1]``
        Predicted ``P(A_j = 1) = μ̂_j^{m_j}``.
    uncertainty : array-like of shape ``(N,)``
        Higher = more uncertain (kept for the rejection-curve API).

    Returns
    -------
    dict with keys ``{"AUROC", "AUPRC", "Brier", "ECE"}``.
    """
    A = _to_numpy_1d(A_true, "A_true")
    p = _to_numpy_1d(p_strict, "p_strict")
    u = _to_numpy_1d(uncertainty, "uncertainty")
    if not (A.shape == p.shape == u.shape):
        raise ValueError(
            f"shape mismatch: A_true {A.shape}, p_strict {p.shape}, "
            f"uncertainty {u.shape}"
        )
    if A.size == 0:
        raise ValueError("Cannot compute metrics on empty inputs")
    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A_true must contain only {0, 1}")

    # AUROC / AUPRC need at least one of each class.
    if A.sum() == 0 or A.sum() == A.size:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(A, p))
        auprc = float(average_precision_score(A, p))

    calib = compute_calibration_metrics(A, p, n_bins=10)
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Brier": calib["Brier"],
        "ECE": calib["ECE"],
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
    A = (K >= m).astype(np.float64)

    rows: List[Tuple[str, str, float]] = []

    # --- ratio-level ---
    ratio = compute_ratio_level_metrics(U, mu, m_j=m)
    for name in ("MAE", "RMSE", "Pearson_r", "binomial_NLL"):
        if name in ratio:
            rows.append((name, "ratio", float(ratio[name])))
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

    # --- bookkeeping rows ---
    rows.append(("n_sentences", "info", float(n_kept)))
    rows.append(("n_skipped_m0", "info", float(n_skipped)))
    rows.append(("frac_strict_factual", "info", float(A.mean())))

    return pd.DataFrame(rows, columns=["metric", "tier", "value"])
