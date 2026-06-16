"""Pre-flight overdispersion diagnostics for the Beta-Binomial model.

Phase 10-1 section 4. Before trusting any fitted ``phi``, we check two
things directly from the annotation counts ``(K_j, m_j)``:

1. The distribution of ``m_j`` -- in particular ``frac(m_j == 0)``,
   ``frac(m_j == 1)`` and ``frac(m_j >= 2)``. The concentration ``phi`` is
   identified *only* from ``m_j >= 2`` sentences (a single atom carries no
   within-sentence overdispersion signal: ``m_j (m_j - 1) = 0``). If that
   fraction is small, ``phi`` is weakly identified and the Beta-Binomial
   may barely move off its initial value.

2. A method-of-moments estimate of the intra-sentence correlation
   ``rho`` (hence ``phi = 1 / rho - 1``), as an independent sanity check
   on the learned ``phi_hat``.

Method-of-moments ``rho``
-------------------------
Restrict to sentences with ``m_j >= 2``. With the pooled mean
``mu_hat = sum K_j / sum m_j`` and the Beta-Binomial second moment
``E[(K_j - m_j mu)^2] = m_j mu (1 - mu) (1 + (m_j - 1) rho)``::

    S   = sum_j (K_j - m_j mu_hat)^2
    rho_hat = (S / (mu_hat (1 - mu_hat)) - sum_j m_j)
              / sum_j m_j (m_j - 1)

This is the standard Pearson-overdispersion intra-class-correlation
estimator. ``rho_hat`` is clamped to ``[0, 1)`` and mapped to
``phi_mom = 1 / rho_hat - 1`` (``inf`` when ``rho_hat == 0``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Union

import numpy as np
import torch


__all__ = ["dispersion_diagnostics", "format_dispersion_report"]


def _to_numpy(x: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def dispersion_diagnostics(
    all_K: Union[torch.Tensor, np.ndarray, list],
    all_m: Union[torch.Tensor, np.ndarray, list],
) -> Dict[str, Any]:
    """Report the ``m_j`` distribution and a moment estimate of ``rho``.

    Parameters
    ----------
    all_K : tensor / array / list of int
        Per-sentence supported-atom counts ``K_j``.
    all_m : tensor / array / list of int
        Per-sentence total atom counts ``m_j``.

    Returns
    -------
    dict with keys:
        ``n_total``        : int   -- number of sentences
        ``frac_m_eq_0``    : float
        ``frac_m_eq_1``    : float
        ``frac_m_ge_2``    : float
        ``n_m_ge_2``       : int   -- sentences that identify phi
        ``mu_pooled``      : float -- sum K / sum m over m_j >= 2 (NaN if none)
        ``rho_mom``        : float -- method-of-moments rho in [0, 1) (NaN if none)
        ``phi_mom``        : float -- 1/rho_mom - 1 (inf if rho_mom == 0)
        ``overdispersion`` : bool  -- rho_mom is finite and > 1e-3
    """
    K = _to_numpy(all_K).astype(np.float64).ravel()
    m = _to_numpy(all_m).astype(np.float64).ravel()
    if K.shape != m.shape:
        raise ValueError(
            f"all_K and all_m must have the same shape; got {K.shape} vs {m.shape}"
        )

    n_total = int(m.size)
    nan = float("nan")
    if n_total == 0:
        return {
            "n_total": 0,
            "frac_m_eq_0": nan,
            "frac_m_eq_1": nan,
            "frac_m_ge_2": nan,
            "n_m_ge_2": 0,
            "mu_pooled": nan,
            "rho_mom": nan,
            "phi_mom": nan,
            "overdispersion": False,
        }

    frac_m_eq_0 = float(np.mean(m == 0))
    frac_m_eq_1 = float(np.mean(m == 1))
    frac_m_ge_2 = float(np.mean(m >= 2))

    mask = m >= 2
    n_m_ge_2 = int(mask.sum())

    if n_m_ge_2 == 0:
        return {
            "n_total": n_total,
            "frac_m_eq_0": frac_m_eq_0,
            "frac_m_eq_1": frac_m_eq_1,
            "frac_m_ge_2": frac_m_ge_2,
            "n_m_ge_2": 0,
            "mu_pooled": nan,
            "rho_mom": nan,
            "phi_mom": nan,
            "overdispersion": False,
        }

    K2 = K[mask]
    m2 = m[mask]
    sum_m = float(m2.sum())
    mu_pooled = float(K2.sum() / sum_m) if sum_m > 0 else nan

    var_unit = mu_pooled * (1.0 - mu_pooled)
    sum_m_m1 = float((m2 * (m2 - 1.0)).sum())
    if not (var_unit > 0.0) or sum_m_m1 <= 0.0:
        rho_mom = nan
        phi_mom = nan
    else:
        S = float(np.sum((K2 - m2 * mu_pooled) ** 2))
        rho_raw = (S / var_unit - sum_m) / sum_m_m1
        # Clamp to a valid correlation; rho in [0, 1).
        rho_mom = float(min(max(rho_raw, 0.0), 1.0 - 1e-9))
        phi_mom = (1.0 / rho_mom - 1.0) if rho_mom > 0.0 else math.inf

    overdispersion = bool(np.isfinite(rho_mom) and rho_mom > 1e-3)

    return {
        "n_total": n_total,
        "frac_m_eq_0": frac_m_eq_0,
        "frac_m_eq_1": frac_m_eq_1,
        "frac_m_ge_2": frac_m_ge_2,
        "n_m_ge_2": n_m_ge_2,
        "mu_pooled": mu_pooled,
        "rho_mom": rho_mom,
        "phi_mom": phi_mom,
        "overdispersion": overdispersion,
    }


def format_dispersion_report(diag: Dict[str, Any]) -> str:
    """Render :func:`dispersion_diagnostics` output as a human-readable block."""
    lines = [
        "=== Pre-flight overdispersion diagnostics ===",
        f"  sentences:          {diag['n_total']}",
        f"  frac(m_j == 0):     {diag['frac_m_eq_0']:.3f}",
        f"  frac(m_j == 1):     {diag['frac_m_eq_1']:.3f}",
        f"  frac(m_j >= 2):     {diag['frac_m_ge_2']:.3f}"
        f"  (n={diag['n_m_ge_2']} identify phi)",
        f"  pooled mu (m>=2):   {diag['mu_pooled']:.4f}",
        f"  method-of-moments rho:  {diag['rho_mom']:.4f}",
        f"  method-of-moments phi:  {diag['phi_mom']:.2f}",
        f"  overdispersion present: {diag['overdispersion']}",
    ]
    if diag["frac_m_ge_2"] < 0.1:
        lines.append(
            "  WARNING: <10% of sentences have m_j >= 2 -- phi is weakly "
            "identified; the Beta-Binomial may not move off phi_init."
        )
    return "\n".join(lines)
