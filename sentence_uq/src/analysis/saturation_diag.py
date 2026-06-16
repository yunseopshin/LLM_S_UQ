"""Pooling diagnostic: token saturation by POS type (Phase 11-0, read-only).

Why this exists
---------------
The epistemic uncertainty of a sentence is

    Epi_mu(j) = g_j^T Sigma_hat g_j,   g_j = (1 / L_j) sum_l gate_l * z_l,

with the per-token *gate* ``gate_l = pi_l * (1 - pi_l)`` and
``pi_l = sigmoid(theta_hat^T z_l)``. On Setup-2 ``Epi_mu`` collapses toward a
near-zero floor (~8e-4) because most tokens are *saturated* (``pi_l`` near 0 or
1), so their gate ~ 0 and they dilute ``g_j``.

Content-mask pooling (averaging ``g_j`` over content tokens only) can rescue
``g_j`` **only if** content tokens (NOUN/PROPN/NUM/ADJ) are meaningfully less
saturated than function tokens. This module measures that directly on the
*existing* trained probe, with **no retraining** and **no model mutation**:

* token-level: saturation fraction and mean gate, split by content/function
  and by individual UPOS;
* sentence-level counterfactual: for each sentence, the exact ``Epi_mu`` under
  uniform pooling vs content-mask pooling using the *same* ``theta_hat`` /
  ``Sigma_hat``.

The counterfactual answers "if I had pooled over content tokens with this same
trained probe, how much larger would ``Epi_mu`` be?" It does NOT capture how
retraining under pooling would further move ``theta_hat`` (that is the actual,
gated-on-GO experiment).

Plain-ASCII math throughout (``pi_l``, ``gate_l``, ``g_j``, ``mu_j``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from src.data.pos_tags import DEFAULT_CONTENT_SET, is_content
from src.features.extractor import extract_token_features

__all__ = [
    "diagnose_saturation_by_pos",
    "format_verdict",
]

# Verdict thresholds (documented in :func:`diagnose_saturation_by_pos`).
DEFAULT_COLLAPSE_FLOOR = 8e-4   # reported Setup-2 Epi_mu collapse scale
DEFAULT_GATE_RATIO_GO = 2.0     # content mean-gate must exceed function by >= this
DEFAULT_EPI_RATIO_GO = 2.0      # median Epi_mask / Epi_unif must be >= this
DEFAULT_FLOOR_MULTIPLE = 2.0    # Epi_mask median must clear floor_multiple * floor


def _stats(values: Sequence[float]) -> Dict[str, float]:
    """Summary stats (count / median / mean / IQR) of a 1-D float sequence.

    NaN / +-inf are dropped first. Returns NaNs for empty input. ``iqr`` is the
    ``[q25, q75]`` pair (list of two floats).
    """
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        nan = float("nan")
        return {"n": 0, "median": nan, "mean": nan, "iqr": [nan, nan]}
    q25, q75 = (float(x) for x in np.percentile(arr, [25, 75]))
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "iqr": [q25, q75],
    }


def _quad_form(g: torch.Tensor, Sigma: torch.Tensor) -> float:
    """Return the scalar ``g^T Sigma g`` in fp32 (the per-sentence ``Epi_mu``)."""
    return float(torch.dot(g, Sigma @ g).item())


def diagnose_saturation_by_pos(
    eval_data: Sequence[Dict[str, Any]],
    feature_params: Any,
    theta_hat: torch.Tensor,
    Sigma_hat: torch.Tensor,
    pos_by_source: Dict[Any, Sequence[str]],
    *,
    sat_threshold: float = 0.05,
    content_set: Sequence[str] = DEFAULT_CONTENT_SET,
    collapse_floor: float = DEFAULT_COLLAPSE_FLOOR,
    gate_ratio_go: float = DEFAULT_GATE_RATIO_GO,
    epi_ratio_go: float = DEFAULT_EPI_RATIO_GO,
    floor_multiple: float = DEFAULT_FLOOR_MULTIPLE,
) -> Dict[str, Any]:
    """Read-only saturation-by-POS diagnostic. Returns a dict of statistics.

    No model state is modified: ``feature_params``, ``theta_hat``, and
    ``Sigma_hat`` are read but never updated, and feature extraction runs under
    ``torch.no_grad`` in fp32.

    Parameters
    ----------
    eval_data : sequence of per-sentence records
        Each record is shaped like ``SentenceUQTrainer.prepare_data`` output:
        ``{"dataset", "source_id", "token_range", "K_j", "m_j",
        "hidden_states", "entropy", "top1"}``. Records whose source is missing
        from ``pos_by_source`` (or whose POS length disagrees with ``T``) are
        skipped and counted.
    feature_params : SentenceUQParams
        Loaded feature extractor (any device). Used only to build ``z_l``.
    theta_hat : Tensor ``(k,)``
        Trained MAP estimate.
    Sigma_hat : Tensor ``(k, k)``
        Laplace posterior covariance.
    pos_by_source : dict
        ``(dataset, source_id) -> per-token UPOS list`` of length ``T``
        (the full generation token axis), as produced by
        :func:`src.data.pos_tags.compute_token_pos`.
    sat_threshold : float, optional
        A token is "saturated" iff ``pi_l < tau`` or ``pi_l > 1 - tau``.
        Default ``tau = 0.05``.
    content_set : sequence of str, optional
        UPOS tags counted as content. Default :data:`DEFAULT_CONTENT_SET`.
    collapse_floor, gate_ratio_go, epi_ratio_go, floor_multiple : float, optional
        Verdict knobs (see :func:`format_verdict`). GO requires
        ``mean_gate_ratio >= gate_ratio_go`` AND
        ``median(Epi_mask / Epi_unif) >= epi_ratio_go`` AND
        ``median(Epi_mask) >= floor_multiple * collapse_floor``.

    Returns
    -------
    dict
        Token-level fractions/gates (overall, by class, by UPOS), the
        sentence-level Epi counterfactual distributions, the mu shift, the
        degeneracy guard, the per-sentence rows, and the GO/NO-GO verdict.
    """
    content = set(content_set)
    param_device = next(feature_params.parameters()).device
    theta = theta_hat.detach().to("cpu", torch.float32)
    Sigma = Sigma_hat.detach().to("cpu", torch.float32)

    # Token-level accumulators (across all eval tokens).
    gate_content: List[float] = []
    gate_function: List[float] = []
    sat_content: List[bool] = []
    sat_function: List[bool] = []
    pos_count: Dict[str, int] = {}
    pos_gate_sum: Dict[str, float] = {}
    pos_sat_count: Dict[str, int] = {}

    # Sentence-level accumulators.
    per_sentence: List[Dict[str, Any]] = []
    epi_ratios: List[float] = []
    epi_unif_all: List[float] = []
    epi_mask_all: List[float] = []
    mu_shifts: List[float] = []

    n_total = len(eval_data)
    n_eval = 0
    n_skipped_no_pos = 0
    n_skipped_len_mismatch = 0
    n_skipped_bad_range = 0
    n_c0 = 0
    n_c_le1 = 0

    zfull_cache: Dict[Any, torch.Tensor] = {}

    for rec in eval_data:
        key = (rec.get("dataset"), rec.get("source_id"))
        pos_full = pos_by_source.get(key)
        if pos_full is None:
            n_skipped_no_pos += 1
            continue

        if key not in zfull_cache:
            with torch.no_grad():
                z_full = extract_token_features(
                    rec["hidden_states"].to(param_device),
                    rec["entropy"].to(param_device),
                    rec["top1"].to(param_device),
                    feature_params,
                )
            zfull_cache[key] = z_full.detach().to("cpu", torch.float32)
        z_full = zfull_cache[key]
        T = int(z_full.shape[0])

        if len(pos_full) != T:
            n_skipped_len_mismatch += 1
            continue

        start, end = int(rec["token_range"][0]), int(rec["token_range"][1])
        if end <= start or end > T:
            n_skipped_bad_range += 1
            continue

        z_span = z_full[start:end]                       # (L_j, k)
        pos_span = list(pos_full[start:end])
        L_j = int(z_span.shape[0])

        logit = z_span @ theta                           # (L_j,)
        pi = torch.sigmoid(logit)                        # (L_j,)
        gate = pi * (1.0 - pi)                           # (L_j,)
        sat = (pi < sat_threshold) | (pi > (1.0 - sat_threshold))

        content_flags = [is_content(p, content) for p in pos_span]
        content_mask = torch.tensor(content_flags, dtype=torch.bool)

        # --- token-level accumulation ---
        gate_list = gate.tolist()
        sat_list = [bool(s) for s in sat.tolist()]
        for p, g_val, s_val, is_c in zip(
            pos_span, gate_list, sat_list, content_flags
        ):
            pos_count[p] = pos_count.get(p, 0) + 1
            pos_gate_sum[p] = pos_gate_sum.get(p, 0.0) + float(g_val)
            pos_sat_count[p] = pos_sat_count.get(p, 0) + (1 if s_val else 0)
            if is_c:
                gate_content.append(float(g_val))
                sat_content.append(s_val)
            else:
                gate_function.append(float(g_val))
                sat_function.append(s_val)

        # --- sentence-level counterfactual ---
        g_unif = (gate.unsqueeze(1) * z_span).sum(dim=0) / float(L_j)
        epi_unif = _quad_form(g_unif, Sigma)
        mu_unif = float(pi.mean().item())
        epi_unif_all.append(epi_unif)

        n_content = int(content_mask.sum().item())
        if n_content == 0:
            n_c0 += 1
        if n_content <= 1:
            n_c_le1 += 1

        if n_content >= 1:
            z_c = z_span[content_mask]
            gate_c = gate[content_mask]
            pi_c = pi[content_mask]
            g_mask = (gate_c.unsqueeze(1) * z_c).sum(dim=0) / float(n_content)
            epi_mask = _quad_form(g_mask, Sigma)
            mu_mask = float(pi_c.mean().item())
            epi_mask_all.append(epi_mask)
            mu_shifts.append(mu_unif - mu_mask)
            ratio = (epi_mask / epi_unif) if epi_unif > 0.0 else float("nan")
            if np.isfinite(ratio):
                epi_ratios.append(ratio)
        else:
            # Degenerate: content-mask pooling would fall back to uniform in the
            # real implementation; exclude from the mask metrics here.
            epi_mask = None
            mu_mask = None
            ratio = None

        per_sentence.append(
            {
                "dataset": rec.get("dataset"),
                "source_id": rec.get("source_id"),
                "token_range": (start, end),
                "L_j": L_j,
                "n_content": n_content,
                "K_j": int(rec.get("K_j", 0) or 0),
                "m_j": int(rec.get("m_j", 0) or 0),
                "mu_unif": mu_unif,
                "mu_mask": mu_mask,
                "Epi_unif": epi_unif,
                "Epi_mask": epi_mask,
                "ratio": ratio,
            }
        )
        n_eval += 1

    # --- token-level aggregates ---
    all_sat = sat_content + sat_function
    n_tokens = len(all_sat)
    overall_sat = float(np.mean(all_sat)) if n_tokens else float("nan")

    def _frac(flags: List[bool]) -> float:
        return float(np.mean(flags)) if flags else float("nan")

    def _mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    mean_gate_content = _mean(gate_content)
    mean_gate_function = _mean(gate_function)
    gate_ratio = (
        mean_gate_content / mean_gate_function
        if (gate_function and mean_gate_function > 0.0)
        else float("nan")
    )

    by_pos: Dict[str, Dict[str, Any]] = {}
    for p in sorted(pos_count):
        cnt = pos_count[p]
        by_pos[p] = {
            "count": int(cnt),
            "sat_fraction": pos_sat_count[p] / cnt if cnt else float("nan"),
            "mean_gate": pos_gate_sum[p] / cnt if cnt else float("nan"),
            "is_content": is_content(p, content),
        }

    epi_ratio_stats = _stats(epi_ratios)
    epi_unif_stats = _stats(epi_unif_all)
    epi_mask_stats = _stats(epi_mask_all)
    mu_shift_stats = _stats(mu_shifts)

    # --- verdict ---
    cond_gate = np.isfinite(gate_ratio) and gate_ratio >= gate_ratio_go
    cond_ratio = (
        np.isfinite(epi_ratio_stats["median"])
        and epi_ratio_stats["median"] >= epi_ratio_go
    )
    cond_floor = (
        np.isfinite(epi_mask_stats["median"])
        and epi_mask_stats["median"] >= floor_multiple * collapse_floor
    )
    go = bool(cond_gate and cond_ratio and cond_floor)

    results: Dict[str, Any] = {
        "config": {
            "sat_threshold": float(sat_threshold),
            "content_set": sorted(content),
            "collapse_floor": float(collapse_floor),
            "gate_ratio_go": float(gate_ratio_go),
            "epi_ratio_go": float(epi_ratio_go),
            "floor_multiple": float(floor_multiple),
        },
        "counts": {
            "n_sentences_total": n_total,
            "n_sentences_evaluated": n_eval,
            "n_skipped_no_pos": n_skipped_no_pos,
            "n_skipped_len_mismatch": n_skipped_len_mismatch,
            "n_skipped_bad_range": n_skipped_bad_range,
            "n_tokens": n_tokens,
        },
        "overall_sat_fraction": overall_sat,
        "by_class": {
            "content": {
                "n_tokens": len(gate_content),
                "sat_fraction": _frac(sat_content),
                "mean_gate": mean_gate_content,
            },
            "function": {
                "n_tokens": len(gate_function),
                "sat_fraction": _frac(sat_function),
                "mean_gate": mean_gate_function,
            },
        },
        "mean_gate_ratio_content_over_function": gate_ratio,
        "by_pos": by_pos,
        "epi": {
            "ratio": epi_ratio_stats,
            "unif": epi_unif_stats,
            "mask": epi_mask_stats,
        },
        "mu_shift_unif_minus_mask": mu_shift_stats,
        "degeneracy": {
            "n_total": n_eval,
            "n_C0": n_c0,
            "n_C_le1": n_c_le1,
            "frac_C0": (n_c0 / n_eval) if n_eval else float("nan"),
            "frac_C_le1": (n_c_le1 / n_eval) if n_eval else float("nan"),
        },
        "per_sentence": per_sentence,
        "verdict": {
            "go": go,
            "cond_gate_ratio": bool(cond_gate),
            "cond_epi_ratio": bool(cond_ratio),
            "cond_floor": bool(cond_floor),
        },
    }
    return results


def format_verdict(
    results: Dict[str, Any],
    *,
    setup: Optional[int] = None,
    eval_split: Optional[str] = None,
) -> str:
    """Render the human-readable GO / NO-GO verdict block (Section 5).

    Parameters
    ----------
    results : dict
        Output of :func:`diagnose_saturation_by_pos`.
    setup, eval_split : optional
        Only used to label the header line.

    Returns
    -------
    str
        Multi-line ASCII block. The same string is printed to stdout and saved
        to ``verdict.md``.
    """
    cfg = results["config"]
    by_class = results["by_class"]
    epi = results["epi"]
    deg = results["degeneracy"]
    v = results["verdict"]

    content = by_class["content"]
    function = by_class["function"]
    gate_ratio = results["mean_gate_ratio_content_over_function"]
    tau = cfg["sat_threshold"]

    header_bits = []
    if setup is not None:
        header_bits.append(f"setup {setup}")
    if eval_split is not None:
        header_bits.append(f"{eval_split} split")
    header = (
        f"=== Pooling Diagnostic ({', '.join(header_bits)}) ==="
        if header_bits
        else "=== Pooling Diagnostic ==="
    )

    def f(x: float, fmt: str = ".3g") -> str:
        return "nan" if x is None or not np.isfinite(x) else format(x, fmt)

    ratio_iqr = epi["ratio"]["iqr"]

    lines = [
        header,
        f"Content set: {{{', '.join(cfg['content_set'])}}}",
        f"Overall saturation fraction (tau={tau:g}): "
        f"{f(results['overall_sat_fraction'], '.2f')}   [sanity vs known ~0.92]",
        f"Mean gate  content={f(content['mean_gate'], '.4f')}  "
        f"function={f(function['mean_gate'], '.4f')}   "
        f"ratio={f(gate_ratio, '.1f')}x",
        f"Saturation fraction  content={f(content['sat_fraction'], '.2f')}  "
        f"function={f(function['sat_fraction'], '.2f')}",
        f"Epi_mask / Epi_unif:  median={f(epi['ratio']['median'], '.2f')}x  "
        f"mean={f(epi['ratio']['mean'], '.2f')}x  "
        f"IQR=[{f(ratio_iqr[0], '.2f')}, {f(ratio_iqr[1], '.2f')}]  "
        f"(n={epi['ratio']['n']})",
        f"Absolute Epi_mu:  unif median={f(epi['unif']['median'])}   "
        f"mask median={f(epi['mask']['median'])}",
        f"mu shift (unif - mask):  "
        f"median={f(results['mu_shift_unif_minus_mask']['median'], '+.3f')}   "
        "(function tokens inflate mu toward 1 when positive)",
        f"Sentences with |C_j|==0: {f(deg['frac_C0'], '.1%')}   "
        f"|C_j|<=1: {f(deg['frac_C_le1'], '.1%')}",
        "",
    ]

    floor_target = cfg["floor_multiple"] * cfg["collapse_floor"]
    if v["go"]:
        lines.append(
            "VERDICT: GO. Content tokens are markedly less saturated "
            f"(gate {f(gate_ratio, '.1f')}x higher) and content-mask pooling "
            f"lifts Epi_mu ~{f(epi['ratio']['median'], '.1f')}x on this trained "
            "probe, clearing the collapse floor. Proceed to implement "
            "switchable pooling (model.pooling: content_mask | saliency)."
        )
    else:
        reasons = []
        if not v["cond_gate_ratio"]:
            reasons.append(
                f"gate ratio {f(gate_ratio, '.2f')}x < {cfg['gate_ratio_go']:g}x "
                "(content tokens are NOT clearly less saturated)"
            )
        if not v["cond_epi_ratio"]:
            reasons.append(
                f"median Epi lift {f(epi['ratio']['median'], '.2f')}x "
                f"< {cfg['epi_ratio_go']:g}x"
            )
        if not v["cond_floor"]:
            reasons.append(
                f"Epi_mask median {f(epi['mask']['median'])} still near the "
                f"collapse floor (< {f(floor_target)})"
            )
        lines.append(
            "VERDICT: NO-GO. " + "; ".join(reasons) + ". "
            "Pooling will not rescue g_j; redirect effort to the loss/prior "
            "side (binomial focal, prior scale, or feature/layer changes). "
            "A uniform-saturation result is itself reportable."
        )

    return "\n".join(lines)
