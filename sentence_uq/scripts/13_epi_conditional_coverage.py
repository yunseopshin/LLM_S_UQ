"""CLI for Phase 13 - where does epistemic actually add value?

Two settings, one narrative, on the existing annotated Setup-2 test split
(no retraining, no new generation/annotation):

* **Setting 1 (usefulness)** - conditional error-detection in ``mu_hat``-ambiguous
  bands. Confident regions (``mu_hat`` near 0 / 1) are already solved by the point
  estimate; we restrict to ambiguous bands and ask whether ``epi_mu`` adds
  discriminative power there, *after controlling for ``mu_hat``* (so we do not
  rediscover ``mu_hat`` through the ``w = mu(1 - mu)`` entanglement). The decisive
  group is ``narrow_symmetric`` ([0.45, 0.55], [0.4, 0.6]): there ``mu_hat``
  (hence ``w``) is ~constant, so any ``epi_mu`` variation is genuine.

* **Setting 2 (validity)** - binomial credible-interval coverage + sharpness.
  We build equal-tailed predictive intervals on ``U_j = K_j / m_j`` at nominal
  levels {0.50, 0.80, 0.90, 0.95} for two variants: (b) aleatoric-only
  ``K_j ~ Binomial(m_j, mu_hat)`` and (c) the full posterior-predictive (MC over
  ``theta``). We test whether (c) is calibrated and whether epistemic widens it
  usefully, with an adaptivity check stratified by ``epi_mu`` tercile.

This is read-only w.r.t. the trained model: it loads ``trained_model.pt`` and the
Setup-2 test split, and writes only under ``results/setup_{N}/conditional_epi/``.
It is additive (a new script plus additive helpers in ``metrics.py``); it does
not modify ``04_evaluate.py``, the inner loop, the posterior, ``predict.py``
return values, or any baseline.

Outputs (under ``results/setup_{N}/conditional_epi/``)::

    band_sweep.csv                 # one row per band (Setting 1)
    coverage.csv                   # level x variant x coverage x mean_width (Setting 2)
    interval_reliability.png       # nominal vs empirical coverage, (b) vs (c)
    sharpness_by_epi_tercile.csv   # coverage/width per epi_mu tercile @ headline level
    setting1_verdict.txt           # the Setting-1 reading-rule outcome
    setting2_verdict.txt           # the Setting-2 reading-rule outcome

Usage
-----
    python scripts/13_epi_conditional_coverage.py --setup 2 --device cpu
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    binomial_equal_tailed_interval,
    compute_bootstrapped_ci,
    equal_tailed_interval_from_samples,
    partial_correlation_gate,
    predictive_interval_coverage,
    sample_posterior_predictive_K,
)
from src.baselines.token_entropy import compute_token_entropy_baseline  # noqa: E402
from src.inference.predict import Predictor, load_trained_model  # noqa: E402
from src.train.trainer import SentenceUQTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# Config (config-driven defaults, per the spec)
# ---------------------------------------------------------------------------

#: Setting-1 band set. ``narrow_symmetric`` is the decisive group (within it
#: ``mu_hat`` and ``w = mu(1 - mu)`` are ~constant, so an ``epi_mu`` signal there
#: is genuine, not a ``mu_hat`` proxy). ``upper_overconfident`` is a secondary
#: probe for confident hallucinations.
DEFAULT_BANDS: Dict[str, List[Tuple[float, float]]] = {
    "asymmetric_lower_sweep": [
        (0.2, 0.7), (0.3, 0.7), (0.4, 0.7), (0.5, 0.7), (0.5, 0.8),
    ],
    "narrow_symmetric": [(0.4, 0.6), (0.45, 0.55)],
    "upper_overconfident": [(0.5, 0.9), (0.6, 0.95)],
}

#: Setting-2 nominal coverage levels; 0.95 is the headline.
LEVELS: Tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
HEADLINE_LEVEL: float = 0.95

#: Skip a band when there is too little data / no class balance to score AUROC.
MIN_BAND_N: int = 25
ERR_RATE_BOUNDS: Tuple[float, float] = (0.1, 0.9)


def _load_evaluate_module() -> Any:
    """Import the digit-prefixed ``04_evaluate.py`` to reuse its data loaders.

    Mirrors ``scripts/05_epi_significance.py``; we reuse ``_resolve_device``,
    ``_load_yaml``, ``_top_get``, ``_prepare_test_records``, ``_filter_positive``
    and ``_extract_z_tokens`` without modifying ``04_evaluate.py``.
    """
    path = _THIS_DIR / "04_evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate04", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# OOD data (weak-OOD fallback for Setting 2 -- spec section 2.3)
# ---------------------------------------------------------------------------


def prepare_ood_records(
    dataset: str,
    gen_dir: Path,
    cache_dir: Path,
    annotated_json: Path,
) -> List[Dict[str, Any]]:
    """Load an EXISTING labelled OOD set into the same record schema as the test
    split (no new generation/annotation -- spec hard constraint).

    Reuses the trainer's generation/cache loaders
    (:meth:`SentenceUQTrainer._build_generation_index`, ``_prompt_keys``,
    ``_load_prompt_tensors``) and the already-produced annotations
    (``K_j``, ``m_j`` per sentence). For ``dataset='longfact'`` this is the
    cross-domain set where ``epi_mu`` is larger than in-domain (prior phases).

    Returns one record per sentence (``m_j > 0``) carrying the prompt's
    ``hidden_states`` / ``entropy`` / ``top1`` (shared by reference across the
    prompt's sentences) plus its ``token_range``, ``m_j``, ``K_j`` -- exactly the
    keys :func:`collect_signals` / ``_extract_z_tokens`` consume.
    """
    if not annotated_json.exists():
        raise FileNotFoundError(f"OOD annotations not found: {annotated_json}")
    index = SentenceUQTrainer._build_generation_index(gen_dir)
    raw = json.load(open(annotated_json, encoding="utf-8"))
    prompt_recs = raw if isinstance(raw, list) else list(raw.values())

    records: List[Dict[str, Any]] = []
    n_prompts_used = 0
    for rec in prompt_recs:
        rel_path, source_id, _ann_key = SentenceUQTrainer._prompt_keys(rec, dataset)
        if rel_path is None or rel_path not in index:
            continue
        hs, ent, t1 = SentenceUQTrainer._load_prompt_tensors(
            gen_dir=gen_dir, cache_dir=cache_dir,
            rel_path=rel_path, cache_idx=index[rel_path],
        )
        n_prompts_used += 1
        for s in rec.get("sentences") or []:
            m_j = int(s.get("m_j", 0) or 0)
            if m_j <= 0:
                continue
            a, b = int(s["token_range"][0]), int(s["token_range"][1])
            if b <= a:
                continue
            records.append({
                "hidden_states": hs,
                "entropy": ent,
                "top1": t1,
                "token_range": (a, b),
                "m_j": m_j,
                "K_j": int(s.get("K_j", 0) or 0),
                "source_id": source_id,
            })
    print(f"OOD '{dataset}': {n_prompts_used} prompts matched, "
          f"{len(records)} sentences with m_j > 0.")
    return records


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _safe_auroc(y_true: Any, score: Any) -> float:
    """AUROC that returns ``nan`` when ``y_true`` is single-class.

    Higher ``score`` is interpreted as predicting ``y_true = 1`` (here the error
    label), so ``AUROC > 0.5`` means ``score`` ranks errors above non-errors.
    """
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def mu_band_mask(mu_hat: Any, lo: float, hi: float) -> np.ndarray:
    """Boolean mask for ``lo <= mu_hat <= hi`` (closed band).

    Parameters
    ----------
    mu_hat : array-like of shape ``(N,)``
        Per-sentence point estimate.
    lo, hi : float
        Inclusive band edges.

    Returns
    -------
    np.ndarray of shape ``(N,)`` dtype bool.
    """
    mu = np.asarray(mu_hat, dtype=np.float64)
    return (mu >= float(lo)) & (mu <= float(hi))


# ---------------------------------------------------------------------------
# Per-sentence signal collection
# ---------------------------------------------------------------------------


def collect_signals(
    predictor: Predictor,
    records: Sequence[Dict[str, Any]],
    feature_params: Any,
    device: torch.device,
    extract_z_tokens: Any,
    mc_samples: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Per-sentence quantities for both settings.

    For every sentence (already filtered to ``m_j > 0``) computes, via
    :meth:`Predictor.predict_sentence` and :meth:`Predictor.predict_mc_epistemic`:

    * ``mu_hat``        : point estimate
    * ``epi_mu``        : delta-method latent epistemic ``g_hat^T Sigma_hat g_hat``
    * ``epi_mc``        : MC latent variance of ``mu``
    * ``aleatoric_U``   : ratio-level aleatoric component
    * ``total_U``       : aleatoric + epistemic
    * ``mean_entropy``  : token-entropy contrast signal (cached entropy mean)

    plus the labels ``K``, ``m``, ``U = K/m``, ``A = 1{K=m}``, ``err = 1 - A``,
    ``abs_ratio_err = |U - mu_hat|`` and the per-sentence ``z_tokens`` (kept for
    Setting 2's posterior-predictive sampling). Arrays are float64 of length
    ``N`` and aligned with ``z_tokens_list``.
    """
    n = len(records)
    mu_hat = np.empty(n, dtype=np.float64)
    epi_mu = np.empty(n, dtype=np.float64)
    epi_mc = np.empty(n, dtype=np.float64)
    aleatoric_U = np.empty(n, dtype=np.float64)
    total_U = np.empty(n, dtype=np.float64)
    mean_entropy = np.empty(n, dtype=np.float64)
    K = np.empty(n, dtype=np.float64)
    m = np.empty(n, dtype=np.float64)
    z_tokens_list: List[torch.Tensor] = []

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    for i, rec in enumerate(records):
        z = extract_z_tokens(rec, feature_params, device)
        z_tokens_list.append(z)
        m_j = int(rec["m_j"])
        out = predictor.predict_sentence(z, m_j=m_j)
        mc = predictor.predict_mc_epistemic(
            z, num_samples=int(mc_samples), m_j=m_j, generator=gen
        )
        mu_hat[i] = float(out["mu_hat"])
        epi_mu[i] = float(out["epi_mu"])
        epi_mc[i] = float(mc["mc_epi_mu"])
        aleatoric_U[i] = float(out["aleatoric_U"] or 0.0)
        total_U[i] = float(out["total_U"] or 0.0)
        a, b = int(rec["token_range"][0]), int(rec["token_range"][1])
        # Reuse the Phase-5 token-entropy baseline (mean entropy over the span;
        # "higher -> more uncertain"), matching the spec's reuse list.
        mean_entropy[i] = compute_token_entropy_baseline(rec["entropy"], (a, b))
        K[i] = float(rec["K_j"])
        m[i] = float(m_j)

    U = K / np.maximum(m, 1.0)
    A = (K == m).astype(np.float64)
    err = 1.0 - A
    abs_ratio_err = np.abs(U - mu_hat)
    return {
        "mu_hat": mu_hat,
        "epi_mu": epi_mu,
        "epi_mc": epi_mc,
        "aleatoric_U": aleatoric_U,
        "total_U": total_U,
        "mean_entropy": mean_entropy,
        "K": K,
        "m": m,
        "U": U,
        "A": A,
        "err": err,
        "abs_ratio_err": abs_ratio_err,
        "z_tokens_list": z_tokens_list,
    }


# ---------------------------------------------------------------------------
# Setting 1 - conditional error-detection in mu_hat-ambiguous bands
# ---------------------------------------------------------------------------


def conditional_epi_band_report(
    group: str,
    band: Tuple[float, float],
    sig: Dict[str, Any],
    min_n: int = MIN_BAND_N,
    err_rate_bounds: Tuple[float, float] = ERR_RATE_BOUNDS,
    bootstrap_iters: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """One ``band_sweep.csv`` row: does ``epi_mu`` detect error within a band?

    For sentences with ``mu_hat in [lo, hi]`` reports ``N``, ``base_err_rate``,
    ``AUROC(err, epi_mu)`` and ``AUROC(err, epi_mc)`` with bootstrap 95% CIs, the
    partial-correlation gate (``epi_mu`` after residualising on
    ``[1, mu_hat, mu_hat^2]`` - reuses :func:`partial_correlation_gate`), and the
    ``AUROC(err, mean_entropy)`` / ``AUROC(err, -mu_hat)`` contrasts.

    The band is **skipped** (``skipped=True``, with a reason) when ``N < min_n``
    or ``base_err_rate`` is outside ``err_rate_bounds`` (AUROC unstable there).
    Skipped rows still carry ``group``, ``band_lo``, ``band_hi`` and ``N``.
    """
    lo, hi = float(band[0]), float(band[1])
    mask = mu_band_mask(sig["mu_hat"], lo, hi)
    n = int(mask.sum())
    row: Dict[str, Any] = {
        "group": group,
        "band_lo": lo,
        "band_hi": hi,
        "N": n,
        "base_err_rate": float("nan"),
        "skipped": True,
        "skip_reason": "",
        "auroc_epi_mu": float("nan"),
        "auroc_epi_mu_ci_lo": float("nan"),
        "auroc_epi_mu_ci_hi": float("nan"),
        "auroc_epi_mc": float("nan"),
        "auroc_epi_mc_ci_lo": float("nan"),
        "auroc_epi_mc_ci_hi": float("nan"),
        "partial_rho": float("nan"),
        "partial_rho_p": float("nan"),
        "partial_sign": "",
        "logit_epi_coef": float("nan"),
        "logit_epi_p": float("nan"),
        # partial_pass is the spec's "significant partial_corr" channel (partial
        # Spearman residualised on [1, mu, mu^2]); logistic_pass is the auxiliary
        # logistic check. gate_passed is their OR (reported, but NOT the Setting-1
        # decision gate -- see setting1_verdict_text).
        "partial_pass": 0.0,
        "logistic_pass": 0.0,
        "gate_passed": 0.0,
        "auroc_mean_entropy": float("nan"),
        "auroc_neg_mu_hat": float("nan"),
    }
    if n < int(min_n):
        row["skip_reason"] = f"N<{int(min_n)} (N={n})"
        return row

    err_b = sig["err"][mask]
    base = float(err_b.mean())
    row["base_err_rate"] = base
    if not (err_rate_bounds[0] <= base <= err_rate_bounds[1]):
        row["skip_reason"] = (
            f"base_err {base:.3f} outside [{err_rate_bounds[0]}, "
            f"{err_rate_bounds[1]}]"
        )
        return row

    epi_mu_b = sig["epi_mu"][mask]
    epi_mc_b = sig["epi_mc"][mask]
    mu_b = sig["mu_hat"][mask]
    ent_b = sig["mean_entropy"][mask]
    are_b = sig["abs_ratio_err"][mask]

    ci_epi = compute_bootstrapped_ci(
        err_b, epi_mu_b, _safe_auroc, n_bootstrap=int(bootstrap_iters), seed=seed
    )
    ci_mc = compute_bootstrapped_ci(
        err_b, epi_mc_b, _safe_auroc, n_bootstrap=int(bootstrap_iters), seed=seed
    )
    gate = partial_correlation_gate(
        epi=epi_mu_b, mu_hat=mu_b, ratio_err=are_b, strict_wrong=err_b
    )

    row.update({
        "skipped": False,
        "skip_reason": "",
        "auroc_epi_mu": _safe_auroc(err_b, epi_mu_b),
        "auroc_epi_mu_ci_lo": ci_epi["lower"],
        "auroc_epi_mu_ci_hi": ci_epi["upper"],
        "auroc_epi_mc": _safe_auroc(err_b, epi_mc_b),
        "auroc_epi_mc_ci_lo": ci_mc["lower"],
        "auroc_epi_mc_ci_hi": ci_mc["upper"],
        "partial_rho": gate["rho_partial"],
        "partial_rho_p": gate["rho_partial_p"],
        "partial_sign": (
            "na" if not np.isfinite(gate["rho_partial"])
            else "+" if gate["rho_partial"] > 0 else "-"
        ),
        "logit_epi_coef": gate["logit_epi_coef"],
        "logit_epi_p": gate["logit_epi_p"],
        "partial_pass": float(gate["spearman_pass"]),
        "logistic_pass": float(gate["logistic_pass"]),
        "gate_passed": gate["passed"],
        "auroc_mean_entropy": _safe_auroc(err_b, ent_b),
        "auroc_neg_mu_hat": _safe_auroc(err_b, -mu_b),
    })
    return row


def run_setting1(
    sig: Dict[str, Any],
    bands: Dict[str, List[Tuple[float, float]]],
    bootstrap_iters: int,
    seed: int,
    arena: str = "in-domain",
) -> Tuple[pd.DataFrame, str]:
    """Build the band-sweep table and the Setting-1 verdict text.

    Returns ``(band_sweep_df, verdict_text)``.
    """
    rows: List[Dict[str, Any]] = []
    for group, band_list in bands.items():
        for band in band_list:
            rows.append(
                conditional_epi_band_report(
                    group, band, sig,
                    bootstrap_iters=bootstrap_iters, seed=seed,
                )
            )
    df = pd.DataFrame(rows)
    return df, setting1_verdict_text(df, arena=arena)


def setting1_verdict_text(df: pd.DataFrame, arena: str = "in-domain") -> str:
    """Apply the Setting-1 reading rule (decisive on ``narrow_symmetric``).

    * Signal present in WIDE bands but **gone in narrow_symmetric** => it was
      ``mu_hat`` leaking through ``w`` => REJECT.
    * Signal **survives in narrow_symmetric** with ``AUROC(epi_mu)`` CI excluding
      0.5 (lower bound > 0.5) AND a significant ``partial_corr`` (correct sign)
      => genuine conditional epistemic value => ACCEPT.
    * ``upper_overconfident`` is reported as a separate, secondary finding.
    """
    rows = df.to_dict("records")
    narrow = [r for r in rows if r["group"] == "narrow_symmetric" and not r["skipped"]]
    narrow_skipped = [r for r in rows if r["group"] == "narrow_symmetric" and r["skipped"]]
    wide_groups = ("asymmetric_lower_sweep",)
    wide = [r for r in rows if r["group"] in wide_groups and not r["skipped"]]

    def _ci_excludes_half_positive(r: Dict[str, Any]) -> bool:
        lo = r["auroc_epi_mu_ci_lo"]
        return bool(np.isfinite(lo) and lo > 0.5)

    def _partial_corr_significant(r: Dict[str, Any]) -> bool:
        # The spec's "significant partial_corr (correct sign)" is the partial
        # SPEARMAN specifically (residualised on [1, mu, mu^2]) -- NOT the
        # OR-gate gate_passed (which also fires on the auxiliary logistic check,
        # whose linear-only mu control is weaker against the w = mu(1-mu) leak
        # this band is meant to rule out). Gate the verdict on partial_pass.
        rho = r["partial_rho"]
        p = r["partial_rho_p"]
        return bool(
            float(r["partial_pass"]) >= 0.5
            and np.isfinite(rho) and rho > 0.0
            and np.isfinite(p) and p < 0.05
        )

    narrow_pass = [
        r for r in narrow
        if _ci_excludes_half_positive(r) and _partial_corr_significant(r)
    ]
    wide_signal = [r for r in wide if _ci_excludes_half_positive(r)]

    lines: List[str] = []
    lines.append(
        f"=== Phase 13 Setting 1 verdict [{arena}] - conditional error-detection ==="
    )
    lines.append("")
    if narrow_pass:
        verdict = "ACCEPT"
        reason = (
            "epi_mu shows genuine conditional value: a narrow_symmetric band has "
            "AUROC(epi_mu) CI excluding 0.5 (lower > 0.5) AND a significant "
            "partial_corr (correct sign). This survives the w = mu(1-mu) control."
        )
    elif not narrow:
        verdict = "INCONCLUSIVE"
        reason = (
            "no narrow_symmetric band was scorable (all skipped: N<"
            f"{MIN_BAND_N} or base_err outside {list(ERR_RATE_BOUNDS)}). The "
            "decisive arena is empty in-domain; "
            + ("wide bands DO show a signal, but that is exactly the mu_hat-residue "
               "pattern we cannot disambiguate without the narrow band -> treat as "
               "NOT a pass." if wide_signal else
               "no wide-band signal either.")
        )
    elif wide_signal and not narrow_pass:
        verdict = "REJECT"
        reason = (
            "signal is present in WIDE bands but GONE in narrow_symmetric "
            "(no narrow band has both a positive CI and a significant "
            "partial_corr) => it was mu_hat leaking through w, not genuine "
            "epistemic value."
        )
    else:
        verdict = "REJECT"
        reason = (
            "no band (narrow or wide) shows a significant conditional epi_mu "
            "signal after controlling for mu_hat."
        )

    lines.append(f"VERDICT: {verdict}")
    lines.append(f"REASON:  {reason}")
    lines.append("")
    lines.append(f"narrow_symmetric scorable bands: {len(narrow)} "
                 f"(skipped: {len(narrow_skipped)})")
    for r in narrow:
        lines.append(
            f"  [{r['band_lo']},{r['band_hi']}] N={r['N']} "
            f"base_err={r['base_err_rate']:.3f} "
            f"AUROC(epi_mu)={r['auroc_epi_mu']:.3f} "
            f"[{r['auroc_epi_mu_ci_lo']:.3f},{r['auroc_epi_mu_ci_hi']:.3f}] "
            f"partial_rho={r['partial_rho']:.3f} (p={r['partial_rho_p']:.3g}, "
            f"sign={r['partial_sign']}) partial_pass={int(r['partial_pass'])} "
            f"[gate_or={int(r['gate_passed'])}]"
        )
    for r in narrow_skipped:
        lines.append(f"  [{r['band_lo']},{r['band_hi']}] SKIPPED: {r['skip_reason']}")
    lines.append("")
    lines.append("upper_overconfident (secondary - confident hallucinations):")
    for r in [x for x in rows if x["group"] == "upper_overconfident"]:
        if r["skipped"]:
            lines.append(f"  [{r['band_lo']},{r['band_hi']}] SKIPPED: {r['skip_reason']}")
        else:
            lines.append(
                f"  [{r['band_lo']},{r['band_hi']}] N={r['N']} "
                f"AUROC(epi_mu)={r['auroc_epi_mu']:.3f} "
                f"[{r['auroc_epi_mu_ci_lo']:.3f},{r['auroc_epi_mu_ci_hi']:.3f}] "
                f"partial_pass={int(r['partial_pass'])} "
                f"[gate_or={int(r['gate_passed'])}]"
            )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Setting 2 - binomial credible-interval coverage + sharpness
# ---------------------------------------------------------------------------


def _per_sentence_intervals(
    predictor: Predictor,
    sig: Dict[str, Any],
    levels: Sequence[float],
    num_samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Per-sentence count interval bounds for variants (b) and (c), all levels.

    Variant (b) uses the exact binomial PMF
    (:func:`binomial_equal_tailed_interval`); variant (c) draws the
    posterior-predictive ``{K^(s)}`` once per sentence
    (:func:`sample_posterior_predictive_K`, ``S = num_samples``) and reads off the
    equal-tailed quantiles at every level
    (:func:`equal_tailed_interval_from_samples`).

    Returns a dict keyed ``"<variant>_lo_<level>"`` / ``"<variant>_hi_<level>"``
    -> ``(N,)`` arrays, where ``variant in {b, c}``.
    """
    m = sig["m"]
    mu_hat = sig["mu_hat"]
    z_list = sig["z_tokens_list"]
    n = m.shape[0]
    out: Dict[str, np.ndarray] = {}
    for lvl in levels:
        for variant in ("b", "c"):
            out[f"{variant}_lo_{lvl}"] = np.empty(n, dtype=np.float64)
            out[f"{variant}_hi_{lvl}"] = np.empty(n, dtype=np.float64)

    gen = torch.Generator()
    gen.manual_seed(int(seed))
    for i in range(n):
        m_j = int(m[i])
        # variant (b): exact binomial
        for lvl in levels:
            lo_b, hi_b = binomial_equal_tailed_interval(m_j, float(mu_hat[i]), lvl)
            out[f"b_lo_{lvl}"][i] = lo_b
            out[f"b_hi_{lvl}"][i] = hi_b
        # variant (c): sample K once, read every level off the same samples
        K_samples = sample_posterior_predictive_K(
            predictor.theta_hat, predictor.Sigma_hat, z_list[i], m_j,
            num_samples=int(num_samples), generator=gen,
        )["K_samples"]
        for lvl in levels:
            lo_c, hi_c = equal_tailed_interval_from_samples(K_samples, lvl)
            out[f"c_lo_{lvl}"][i] = lo_c
            out[f"c_hi_{lvl}"][i] = hi_c
    return out


def run_setting2(
    predictor: Predictor,
    sig: Dict[str, Any],
    levels: Sequence[float],
    num_samples: int,
    seed: int,
    arena: str = "in-domain",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], str]:
    """Coverage / sharpness tables, tercile adaptivity, and the Setting-2 verdict.

    Returns ``(coverage_df, tercile_df, intervals, verdict_text)`` where
    ``intervals`` is the per-sentence bound dict (for plotting).
    """
    K = sig["K"]
    m = sig["m"]
    intervals = _per_sentence_intervals(predictor, sig, levels, num_samples, seed)

    # --- coverage.csv: level x variant ---
    cov_rows: List[Dict[str, Any]] = []
    for lvl in levels:
        for variant, label in (("b", "aleatoric_only"), ("c", "aleatoric+epistemic")):
            res = predictive_interval_coverage(
                intervals[f"{variant}_lo_{lvl}"],
                intervals[f"{variant}_hi_{lvl}"], K, m,
            )
            cov_rows.append({
                "level": lvl,
                "variant": label,
                "coverage": res["coverage"],
                "mean_width": res["mean_width"],
                "n": res["n"],
            })
    coverage_df = pd.DataFrame(cov_rows)

    # --- sharpness_by_epi_tercile.csv: headline level, per epi_mu tercile ---
    epi = sig["epi_mu"]
    tercile_df = _tercile_adaptivity(intervals, epi, K, m, HEADLINE_LEVEL)

    verdict = setting2_verdict_text(coverage_df, tercile_df, sig, arena=arena)
    return coverage_df, tercile_df, intervals, verdict


def _tercile_adaptivity(
    intervals: Dict[str, np.ndarray],
    epi: np.ndarray,
    K: np.ndarray,
    m: np.ndarray,
    level: float,
) -> pd.DataFrame:
    """Coverage + width per ``epi_mu`` tercile (low/mid/high) for (b) and (c).

    The adaptivity test: in the HIGH-epi stratum, does (c) widen relative to (b)
    and does that improve coverage toward nominal? Reports coverage / mean width
    / mean ``epi_mu`` per tercile for both variants at the given (headline) level.
    """
    n = epi.shape[0]
    if n == 0:
        return pd.DataFrame()
    # Tercile edges (33rd / 67th percentile of epi_mu).
    q1, q2 = np.quantile(epi, [1.0 / 3.0, 2.0 / 3.0])
    strata = {
        "low": epi <= q1,
        "mid": (epi > q1) & (epi <= q2),
        "high": epi > q2,
    }
    rows: List[Dict[str, Any]] = []
    for name, mask in strata.items():
        if int(mask.sum()) == 0:
            continue
        for variant, label in (("b", "aleatoric_only"), ("c", "aleatoric+epistemic")):
            res = predictive_interval_coverage(
                intervals[f"{variant}_lo_{level}"][mask],
                intervals[f"{variant}_hi_{level}"][mask],
                K[mask], m[mask],
            )
            rows.append({
                "tercile": name,
                "variant": label,
                "level": level,
                "n": int(mask.sum()),
                "mean_epi_mu": float(epi[mask].mean()),
                "coverage": res["coverage"],
                "mean_width": res["mean_width"],
            })
    return pd.DataFrame(rows)


def setting2_verdict_text(
    coverage_df: pd.DataFrame,
    tercile_df: pd.DataFrame,
    sig: Dict[str, Any],
    arena: str = "in-domain",
) -> str:
    """Apply the Setting-2 reading rule + the decomposition-honesty report.

    * (c) coverage closer to nominal than (b) - especially in the high-epi
      stratum => epistemic adds calibrated interval value => ACCEPT.
    * (b) ~= (c) everywhere (expected if the posterior is collapsed in-domain) =>
      epistemic adds negligible in-domain interval value => honest NEGATIVE.
    """
    def _cov(df: pd.DataFrame, variant: str) -> float:
        sub = df[(df["variant"] == variant) & (df["level"] == HEADLINE_LEVEL)]
        return float(sub["coverage"].iloc[0]) if len(sub) else float("nan")

    cov_b_all = _cov(coverage_df, "aleatoric_only")
    cov_c_all = _cov(coverage_df, "aleatoric+epistemic")

    def _tercile_cov(name: str, variant: str) -> float:
        if tercile_df.empty:
            return float("nan")
        sub = tercile_df[
            (tercile_df["tercile"] == name) & (tercile_df["variant"] == variant)
        ]
        return float(sub["coverage"].iloc[0]) if len(sub) else float("nan")

    cov_b_high = _tercile_cov("high", "aleatoric_only")
    cov_c_high = _tercile_cov("high", "aleatoric+epistemic")

    # Max coverage gap across levels (is (c) ever materially different from (b)?).
    pivot = coverage_df.pivot_table(
        index="level", columns="variant", values="coverage"
    )
    max_cov_gap = float(
        np.nanmax(np.abs(pivot["aleatoric+epistemic"] - pivot["aleatoric_only"]))
    ) if {"aleatoric+epistemic", "aleatoric_only"}.issubset(pivot.columns) else float("nan")

    d_b_all = abs(cov_b_all - HEADLINE_LEVEL)
    d_c_all = abs(cov_c_all - HEADLINE_LEVEL)
    d_b_high = abs(cov_b_high - HEADLINE_LEVEL)
    d_c_high = abs(cov_c_high - HEADLINE_LEVEL)

    # --- Decomposition honesty (so the epi contribution is not overstated) ---
    # Report mean aleatoric_U vs mean epi_mu overall AND per band (spec 2.2), so
    # the epi share -- which bounds how much epistemic can move coverage -- is
    # explicit everywhere, not just overall.
    def _decomp(mask: np.ndarray) -> Tuple[int, float, float, float]:
        nmask = int(mask.sum())
        if nmask == 0:
            return 0, float("nan"), float("nan"), float("nan")
        a = float(sig["aleatoric_U"][mask].mean())
        e = float(sig["epi_mu"][mask].mean())
        d = a + e
        return nmask, a, e, (e / d if d > 0 else float("nan"))

    _, mean_alea, mean_epi, epi_share_all = _decomp(
        np.ones_like(sig["epi_mu"], dtype=bool)
    )
    # Bands shown for the per-band decomposition: the mid-mu and overconfident
    # regions where the question "can epistemic move coverage?" is live.
    decomp_bands = [(0.4, 0.6), (0.45, 0.55), (0.5, 0.9)]
    decomp_rows = [
        (lo, hi) + _decomp(mu_band_mask(sig["mu_hat"], lo, hi))
        for (lo, hi) in decomp_bands
    ]

    # Reading rule (spec 2.3 + Go/No-Go): the outcome is ACCEPT vs honest-negative.
    # The honest-negative "(b) ~= (c)" case takes PRECEDENCE when the in-domain
    # posterior is collapsed (epistemic is a tiny share of total width): there the
    # (b)/(c) intervals differ by less than the epistemic share, so any sub-noise
    # coverage wiggle (here ~1 sentence) is NOT evidence of calibrated epistemic
    # value and must not be reported as a pass (this is what the decomposition-
    # honesty guard exists to prevent). ACCEPT therefore requires a *material*,
    # non-collapsed improvement in the decisive high-epi stratum.
    high_closer = (
        np.isfinite(d_c_high) and np.isfinite(d_b_high)
        and d_c_high < d_b_high - 1e-9
    )
    all_not_worse = (not np.isfinite(d_c_all)) or (not np.isfinite(d_b_all)) or (
        d_c_all <= d_b_all + 1e-9
    )
    # Direction of the (b) miscalibration at the headline level: under-coverage
    # (cov < nominal) means the model is OVERconfident; over-coverage means it is
    # conservative. This is the calibration story that matters on OOD.
    if np.isfinite(cov_b_all) and cov_b_all < HEADLINE_LEVEL - 0.02:
        cov_dir = (
            f"the (b) intervals UNDER-cover at {HEADLINE_LEVEL:.0%} "
            f"(cov={cov_b_all:.3f} < nominal -> the model is overconfident on "
            "this arena)"
        )
    elif np.isfinite(cov_b_all) and cov_b_all > HEADLINE_LEVEL + 0.02:
        cov_dir = (
            f"the (b) intervals OVER-cover at {HEADLINE_LEVEL:.0%} "
            f"(cov={cov_b_all:.3f} > nominal -> conservative)"
        )
    else:
        cov_dir = (
            f"the (b) intervals are well-calibrated at {HEADLINE_LEVEL:.0%} "
            f"(cov={cov_b_all:.3f})"
        )

    collapsed = (not np.isfinite(epi_share_all)) or epi_share_all < 0.05
    if collapsed:
        verdict = "NEGATIVE (honest)"
        reason = (
            f"on {arena}, the posterior is effectively collapsed (epistemic is "
            f"{epi_share_all:.1%} of mean total width; max coverage gap across "
            f"levels = {max_cov_gap:.4f}), so (b) ~= (c) and epistemic adds "
            f"negligible interval value: {cov_dir}, and (c) does NOT rescue it. "
            "Any sub-noise coverage difference is not evidence of calibrated "
            "epistemic value. The binomial interval's own calibration is the "
            "carry-forward result; the framing pivots cleanly to single-pass "
            "binomial calibration."
        )
    elif high_closer and all_not_worse:
        verdict = "ACCEPT"
        reason = (
            f"on {arena}, (c) is closer to nominal than (b) in the decisive "
            "high-epi stratum (|cov_c-0.95| < |cov_b-0.95| there) and is not worse "
            f"overall, with a non-collapsed epistemic share ({epi_share_all:.1%}): "
            "epistemic widens the interval where it should and improves coverage "
            "toward nominal."
        )
    else:
        verdict = "NEGATIVE (honest)"
        reason = (
            f"on {arena}, (c) is not closer to nominal than (b) in the high-epi "
            "stratum at the headline level, so epistemic does not add calibrated "
            f"interval value ({cov_dir}; epi_share={epi_share_all:.1%}, max "
            f"coverage gap across levels = {max_cov_gap:.4f})."
        )

    lines: List[str] = []
    lines.append(
        f"=== Phase 13 Setting 2 verdict [{arena}] - binomial interval coverage ==="
    )
    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    lines.append(f"REASON:  {reason}")
    lines.append("")
    lines.append(f"Headline level = {HEADLINE_LEVEL}")
    lines.append(
        f"  cov(all)  : (b)={cov_b_all:.3f}  (c)={cov_c_all:.3f}  "
        f"|.-nom|: (b)={d_b_all:.3f} (c)={d_c_all:.3f}"
    )
    lines.append(
        f"  cov(high-epi): (b)={cov_b_high:.3f}  (c)={cov_c_high:.3f}  "
        f"|.-nom|: (b)={d_b_high:.3f} (c)={d_c_high:.3f}"
    )
    lines.append(f"  max coverage gap (c vs b) across levels = {max_cov_gap:.4f}")
    lines.append("")
    lines.append("Decomposition honesty (mean aleatoric_U vs mean epi_mu):")
    lines.append(
        f"  overall  : aleatoric_U={mean_alea:.3e}  epi_mu={mean_epi:.3e}  "
        f"epi_share={epi_share_all:.3f}"
    )
    for lo, hi, nmask, a_b, e_b, share_b in decomp_rows:
        if nmask == 0:
            lines.append(f"  band [{lo},{hi}]: empty (no sentences)")
        else:
            lines.append(
                f"  band [{lo},{hi}] (N={nmask}): aleatoric_U={a_b:.3e}  "
                f"epi_mu={e_b:.3e}  epi_share={share_b:.3f}"
            )
    lines.append(
        "  (in mid-mu the interval width is aleatoric-dominated; the epi share "
        "above bounds how much epistemic can move coverage.)"
    )
    lines.append("")
    return "\n".join(lines)


def _plot_interval_reliability(
    coverage_df: pd.DataFrame,
    levels: Sequence[float],
    save_path: Path,
) -> None:
    """Nominal-vs-empirical interval reliability diagram, (b) and (c) overlaid.

    Note on the reuse list (hard constraint 4): ``metrics.plot_reliability_diagram``
    is a per-sample probability-calibration diagram (it bins ``(y_true, p_pred)``
    into a single empirical-frequency curve). The interval-reliability question
    here is different -- nominal coverage level vs empirical coverage, with the two
    variants (b) and (c) overlaid -- which that helper cannot draw, so this
    purpose-built plot is used instead. (``plot_reliability_diagram`` is still
    available to the rest of the pipeline; nothing about it changes.)
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0,
            color="gray", label="Perfect calibration")
    lvl = list(levels)
    for variant, label, color in (
        ("aleatoric_only", "(b) aleatoric-only", "C1"),
        ("aleatoric+epistemic", "(c) aleatoric+epistemic", "C0"),
    ):
        sub = coverage_df[coverage_df["variant"] == variant].sort_values("level")
        ax.plot(sub["level"], sub["coverage"], marker="o", linewidth=1.5,
                color=color, label=label)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nominal coverage level")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Binomial predictive-interval reliability")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """argparse parser for ``scripts/13_epi_conditional_coverage.py``."""
    p = argparse.ArgumentParser(
        description="Phase 13 - conditional epistemic value + interval coverage."
    )
    p.add_argument("--setup", type=int, default=2, help="Experimental setup.")
    p.add_argument("--config", type=str, default=None, help="YAML config.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for the feature extractor (default cpu).")
    p.add_argument("--trained-model", type=str, default=None,
                   help="Override path to trained_model.pt.")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Override output dir (default results/setup_{N}).")
    p.add_argument("--mc-samples", type=int, default=200,
                   help="theta samples for the MC latent epistemic (epi_mc).")
    p.add_argument("--pred-samples", type=int, default=500,
                   help="theta/K samples for the (c) posterior-predictive (S).")
    p.add_argument("--bootstrap-iters", type=int, default=1000,
                   help="Bootstrap resamples for the band-AUROC 95%% CIs.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on the number of test sentences (smoke).")
    p.add_argument("--ood-dataset", type=str, default=None,
                   help="Run on an existing labelled OOD set instead of the "
                        "in-domain test split (weak-OOD fallback for Setting 2, "
                        "spec 2.3). E.g. 'longfact'. Writes to a separate "
                        "conditional_epi_ood_{dataset}/ dir; the in-domain run is "
                        "untouched. No new generation/annotation is performed.")
    p.add_argument("--ood-gen-dir", type=str, default=None,
                   help="Override OOD generations dir (default data/generations/{ds}).")
    p.add_argument("--ood-cache-dir", type=str, default=None,
                   help="Override OOD cache dir (default data/cache/{ds}).")
    p.add_argument("--ood-annotated", type=str, default=None,
                   help="Override OOD annotations json "
                        "(default data/processed/{ds}/annotated.json).")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip the matplotlib figure (CSV outputs still produced).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point - see module docstring."""
    args = _build_parser().parse_args(argv)
    ev = _load_evaluate_module()
    setup = int(args.setup)
    device = torch.device(ev._resolve_device(args.device))

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = ev._load_yaml(args.config)

    results_dir = Path(
        args.results_dir or ev._top_get(cfg, "results_dir", f"results/setup_{setup}")
    )
    ood = args.ood_dataset
    out_dir = results_dir / (f"conditional_epi_ood_{ood}" if ood else "conditional_epi")
    out_dir.mkdir(parents=True, exist_ok=True)
    trained_path = Path(args.trained_model or results_dir / "trained_model.pt")

    arena = f"weak-OOD '{ood}'" if ood else f"in-domain (setup {setup})"
    print(f"=== Phase 13 - conditional epistemic + interval coverage [{arena}] ===")
    print(f"Trained model: {trained_path}")
    print(f"Results:       {out_dir}")
    if not trained_path.exists():
        print(f"error: trained model not found at {trained_path}", file=sys.stderr)
        return 2

    loaded = load_trained_model(trained_path, map_location="cpu")
    feature_params = loaded["feature_params"].to(device)
    feature_params.eval()
    # Raw MAP mu everywhere (no probit shrinkage): Setting 1 bands and Setting 2
    # variant (b) are defined on mu_hat, and variant (c) samples mu from the same
    # raw sigmoid. epi_mu / aleatoric_U / total_U are probit-independent anyway.
    predictor = Predictor(
        theta_hat=loaded["theta_hat"],
        Sigma_hat=loaded["Sigma_hat"],
        feature_params=feature_params,
        use_probit_shrinkage=False,
        likelihood=loaded.get("likelihood"),
    )

    if ood:
        gen_dir = Path(args.ood_gen_dir or f"data/generations/{ood}")
        cache_dir = Path(args.ood_cache_dir or f"data/cache/{ood}")
        annotated = Path(args.ood_annotated or f"data/processed/{ood}/annotated.json")
        try:
            records = prepare_ood_records(ood, gen_dir, cache_dir, annotated)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        records = ev._filter_positive(
            ev._prepare_test_records(cfg, setup, device, feature_params)
        )
    if not records:
        print("error: no positive-m_j sentences.", file=sys.stderr)
        return 2
    if args.limit is not None:
        records = records[: int(args.limit)]
    print(f"Sentences: {len(records)} with m_j > 0"
          + (f" (limited to {args.limit})" if args.limit is not None else "") + ".")

    sig = collect_signals(
        predictor, records, feature_params, device, ev._extract_z_tokens,
        mc_samples=int(args.mc_samples), seed=int(args.seed),
    )

    # --- Setting 1 ---------------------------------------------------------
    band_df, verdict1 = run_setting1(
        sig, DEFAULT_BANDS, bootstrap_iters=int(args.bootstrap_iters),
        seed=int(args.seed), arena=arena,
    )
    band_csv = out_dir / "band_sweep.csv"
    band_df.to_csv(band_csv, index=False)
    (out_dir / "setting1_verdict.txt").write_text(verdict1 + "\n", encoding="utf-8")

    # --- Setting 2 ---------------------------------------------------------
    coverage_df, tercile_df, intervals, verdict2 = run_setting2(
        predictor, sig, LEVELS, num_samples=int(args.pred_samples),
        seed=int(args.seed), arena=arena,
    )
    coverage_df.to_csv(out_dir / "coverage.csv", index=False)
    tercile_df.to_csv(out_dir / "sharpness_by_epi_tercile.csv", index=False)
    (out_dir / "setting2_verdict.txt").write_text(verdict2 + "\n", encoding="utf-8")
    if not args.no_plots:
        _plot_interval_reliability(
            coverage_df, LEVELS, out_dir / "interval_reliability.png"
        )

    # --- Console summary ---------------------------------------------------
    print("\n--- Setting 1: band sweep ---")
    show1 = band_df[[
        "group", "band_lo", "band_hi", "N", "base_err_rate", "auroc_epi_mu",
        "auroc_epi_mu_ci_lo", "auroc_epi_mu_ci_hi", "partial_rho_p",
        "gate_passed", "skipped",
    ]]
    print(show1.to_string(index=False))
    print("\n" + verdict1)

    print("\n--- Setting 2: coverage ---")
    print(coverage_df.to_string(index=False))
    print("\n--- Setting 2: sharpness by epi_mu tercile (headline level) ---")
    print(tercile_df.to_string(index=False))
    print("\n" + verdict2)

    print(f"\nSaved Phase-13 outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
