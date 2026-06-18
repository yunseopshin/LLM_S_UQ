"""CLI for Phase 10-2 (Part A) — epistemic significance diagnostic.

Usage
-----
    python scripts/05_epi_significance.py --setup 2 --device cpu

Question
~~~~~~~~
Is ``epi_mu`` *informative*, framed as **significance** rather than magnitude?
``epi_mu`` is structurally small on the in-domain Setup-2 model; the question
that matters for the paper is not "is it large" but "does it predict error,
after we control for the point estimate ``mu_hat``". This script is read-only
w.r.t. the trained model: it loads the Phase 4-1 ``trained_model.pt`` and the
Setup-2 test split and computes four diagnostics.

1. **Partial-correlation gate (PRIMARY)** — does ``epi_mu`` add predictive power
   beyond ``mu_hat``? Partial Spearman on the continuous error plus a logistic
   check on the strict error. PASS iff the correct sign and ``p < 0.05`` in at
   least one of the two checks (see :func:`partial_correlation_gate`).
2. **PRR-AUC** — rejection value of ``epi_mu`` vs a pure point-estimate
   confidence baseline ``-|mu_hat - 0.5|``.
3. **MC vs delta sanity check** — Pearson/Spearman + median ratio of the MC
   latent epistemic ``epi_mc`` against the delta-method ``epi_mu``. Tells us
   whether the smallness is structural (``MC ~= delta``) or an approximation
   artefact (``MC >> delta``).
4. **OOD ratio (conditional)** — ``median(epi_mu_ood) / median(epi_mu_indomain)``
   and a Mann-Whitney U p-value, when an OOD ``epi_mu`` array is available via
   ``--ood-results-dir``. Skipped with a printed note otherwise.

Outputs
~~~~~~~
* ``results/setup_2/epi_significance.csv`` — one row per diagnostic.
* ``results/setup_2/epi_significance_epi_mu.npy`` — in-domain ``epi_mu`` array
  (so a second run can serve as the OOD comparison set).
* Console: a compact summary table, the explicit ``PARTIAL_CORR_GATE: PASS|FAIL``
  line, and the MC-vs-delta interpretation line.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compute_prr,
    partial_correlation_gate,
)
from src.inference.predict import Predictor, load_trained_model  # noqa: E402


def _load_evaluate_module() -> Any:
    """Import the digit-prefixed ``04_evaluate.py`` to reuse its data loaders."""
    path = _THIS_DIR / "04_evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate04", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Per-sentence signals
# ---------------------------------------------------------------------------


def _collect_signals(
    predictor: Predictor,
    z_tokens_list: List[torch.Tensor],
    K_pos: np.ndarray,
    m_pos: np.ndarray,
    mc_samples: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Per-sentence ``mu_hat``, ``epi_mu`` (delta), ``epi_mc`` (MC) + errors.

    ``epi_mc`` is the latent-level MC variance of ``mu`` from
    :meth:`Predictor.predict_mc_epistemic` (``num_samples=mc_samples``).
    ``ratio_err = |U_j - mu_hat|`` with ``U_j = K_j / m_j``;
    ``strict_wrong = 1 - A_j`` with ``A_j = 1{K_j == m_j}``.
    """
    n = len(z_tokens_list)
    mu_hat = np.empty(n, dtype=np.float64)
    epi_mu = np.empty(n, dtype=np.float64)
    epi_mc = np.empty(n, dtype=np.float64)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    for i, (z, m) in enumerate(zip(z_tokens_list, m_pos)):
        out = predictor.predict_sentence(z, m_j=int(m))
        mc = predictor.predict_mc_epistemic(
            z, num_samples=int(mc_samples), m_j=int(m), generator=gen
        )
        mu_hat[i] = float(out["mu_hat"])
        epi_mu[i] = float(out["epi_mu"])
        epi_mc[i] = float(mc["mc_epi_mu"])
    U = K_pos / np.maximum(m_pos, 1.0)
    A = (K_pos == m_pos).astype(np.float64)
    ratio_err = np.abs(U - mu_hat)
    strict_wrong = 1.0 - A
    return {
        "mu_hat": mu_hat,
        "epi_mu": epi_mu,
        "epi_mc": epi_mc,
        "ratio_err": ratio_err,
        "strict_wrong": strict_wrong,
    }


# ---------------------------------------------------------------------------
# OOD (conditional)
# ---------------------------------------------------------------------------


def _ood_comparison(
    ood_results_dir: Optional[str], indomain_epi_mu: np.ndarray
) -> Optional[Dict[str, float]]:
    """median(OOD)/median(in-domain) + Mann-Whitney U, when an OOD set exists.

    Looks for ``<ood_results_dir>/epi_significance_epi_mu.npy`` (written by a
    prior run of this script). Returns ``None`` (skip) when unavailable.
    """
    if not ood_results_dir:
        return None
    ood_path = Path(ood_results_dir) / "epi_significance_epi_mu.npy"
    if not ood_path.exists():
        print(
            f"info: OOD epi_mu not found at {ood_path}; skipping OOD ratio "
            "(diagnostic 4). Run this script on the OOD results dir first.",
            file=sys.stderr,
        )
        return None
    from scipy.stats import mannwhitneyu

    ood = np.asarray(np.load(ood_path), dtype=np.float64)
    if ood.size == 0:
        return None
    med_in = float(np.median(indomain_epi_mu))
    med_ood = float(np.median(ood))
    try:
        _u, p = mannwhitneyu(ood, indomain_epi_mu, alternative="greater")
    except ValueError:
        p = float("nan")
    return {
        "median_indomain": med_in,
        "median_ood": med_ood,
        "ratio_ood_over_indomain": (med_ood / med_in if med_in > 0 else float("inf")),
        "mannwhitney_p_ood_greater": float(p),
        "n_ood": float(ood.size),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """argparse parser for ``scripts/05_epi_significance.py`` (mirrors 04)."""
    p = argparse.ArgumentParser(
        description="Phase 10-2 Part A — epistemic significance diagnostic."
    )
    p.add_argument("--setup", type=int, default=2, help="Experimental setup.")
    p.add_argument("--config", type=str, default=None, help="YAML config.")
    p.add_argument("--experiment", type=str, default=None,
                   help="Optional experiment label (recorded in the CSV).")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for the feature extractor (default cpu).")
    p.add_argument("--trained-model", type=str, default=None,
                   help="Override path to trained_model.pt.")
    p.add_argument("--results-dir", type=str, default=None,
                   help="Override output dir (default results/setup_{N}).")
    p.add_argument("--ood-results-dir", type=str, default=None,
                   help="Optional results dir holding an OOD epi_mu array "
                        "(epi_significance_epi_mu.npy) for diagnostic 4.")
    p.add_argument("--mc-samples", type=int, default=200,
                   help="theta samples for the MC latent epistemic.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for MC.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point — see module docstring."""
    args = _build_parser().parse_args(argv)
    ev = _load_evaluate_module()
    setup = int(args.setup)
    device = torch.device(ev._resolve_device(args.device))

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = ev._load_yaml(args.config)

    results_dir = Path(
        args.results_dir
        or ev._top_get(cfg, "results_dir", f"results/setup_{setup}")
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    trained_path = Path(args.trained_model or results_dir / "trained_model.pt")

    print(f"=== Phase 10-2 Part A — epistemic significance (setup {setup}) ===")
    print(f"Trained model: {trained_path}")
    print(f"Results:       {results_dir}")
    if not trained_path.exists():
        print(f"error: trained model not found at {trained_path}", file=sys.stderr)
        return 2

    loaded = load_trained_model(trained_path, map_location="cpu")
    feature_params = loaded["feature_params"].to(device)
    feature_params.eval()
    predictor = Predictor(
        theta_hat=loaded["theta_hat"],
        Sigma_hat=loaded["Sigma_hat"],
        feature_params=feature_params,
        use_probit_shrinkage=True,
        likelihood=loaded.get("likelihood"),
    )

    # --- Test split (positive m_j) -----------------------------------------
    records = ev._filter_positive(
        ev._prepare_test_records(cfg, setup, device, feature_params)
    )
    if not records:
        print("error: no positive-m_j test sentences.", file=sys.stderr)
        return 2
    print(f"Test sentences: {len(records)} with m_j > 0.")
    z_tokens_list = [ev._extract_z_tokens(r, feature_params, device) for r in records]
    K_pos = np.array([int(r["K_j"]) for r in records], dtype=np.float64)
    m_pos = np.array([int(r["m_j"]) for r in records], dtype=np.float64)

    sig = _collect_signals(
        predictor, z_tokens_list, K_pos, m_pos, args.mc_samples, args.seed
    )
    # Persist in-domain epi_mu so a second run can supply the OOD comparison.
    np.save(results_dir / "epi_significance_epi_mu.npy", sig["epi_mu"])

    # --- (1) Partial-correlation gate (PRIMARY) ----------------------------
    gate = partial_correlation_gate(
        epi=sig["epi_mu"], mu_hat=sig["mu_hat"],
        ratio_err=sig["ratio_err"], strict_wrong=sig["strict_wrong"],
        alpha=0.05,
    )
    gate_pass = bool(gate["passed"] >= 0.5)

    # --- (2) PRR-AUC: epi_mu vs point-estimate confidence ------------------
    # y_true = ratio_err (an error: LOWER remaining mean is better), reject the
    # highest-uncertainty sentences first.
    mu_conf_uncertainty = -np.abs(sig["mu_hat"] - 0.5)
    prr_epi = compute_prr(
        y_true=sig["ratio_err"], uncertainty=sig["epi_mu"], num_thresholds=100
    )["prr_auc"]
    prr_muconf = compute_prr(
        y_true=sig["ratio_err"], uncertainty=mu_conf_uncertainty, num_thresholds=100
    )["prr_auc"]

    # --- (3) MC vs delta ---------------------------------------------------
    from scipy.stats import pearsonr, spearmanr

    epi_mu = sig["epi_mu"]
    epi_mc = sig["epi_mc"]
    if np.std(epi_mu) > 1e-12 and np.std(epi_mc) > 1e-12:
        mc_pearson = float(pearsonr(epi_mc, epi_mu)[0])
        mc_spearman = float(spearmanr(epi_mc, epi_mu)[0])
    else:
        mc_pearson = float("nan")
        mc_spearman = float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(epi_mu > 0, epi_mc / epi_mu, np.nan)
    mc_delta_median_ratio = float(np.nanmedian(ratios))

    # --- (4) OOD ratio (conditional) ---------------------------------------
    ood = _ood_comparison(args.ood_results_dir, sig["epi_mu"])

    # --- Assemble CSV ------------------------------------------------------
    rows: List[Dict[str, Any]] = []

    def _row(diag: str, value: float, p: float, note: str) -> None:
        rows.append({
            "experiment": args.experiment or f"setup_{setup}",
            "diagnostic": diag, "value": value, "p_value": p, "note": note,
        })

    _row("n_sentences", float(len(records)), float("nan"), "positive m_j")
    _row("partial_spearman_ratio", gate["rho_partial"], gate["rho_partial_p"],
         f"sign={'+' if gate['rho_partial'] > 0 else '-'} (higher epi -> more error if +)")
    _row("logistic_epi_strict_coef", gate["logit_epi_coef"], gate["logit_epi_p"],
         f"sign={int(gate['logit_epi_sign'])} on standardized epi (strict_wrong)")
    _row("partial_corr_gate", gate["passed"], float("nan"),
         "PASS" if gate_pass else "FAIL")
    _row("prr_auc_epi_mu", float(prr_epi), float("nan"),
         "y=ratio_err -> LOWER is better")
    _row("prr_auc_mu_confidence", float(prr_muconf), float("nan"),
         "point-estimate baseline (-|mu-0.5|)")
    _row("mc_vs_delta_pearson", mc_pearson, float("nan"), "epi_mc vs epi_mu")
    _row("mc_vs_delta_spearman", mc_spearman, float("nan"), "epi_mc vs epi_mu")
    _row("mc_delta_median_ratio", mc_delta_median_ratio, float("nan"),
         "median epi_mc / epi_mu")
    if ood is not None:
        _row("ood_epi_ratio", ood["ratio_ood_over_indomain"],
             ood["mannwhitney_p_ood_greater"],
             f"median ood/indom; n_ood={int(ood['n_ood'])}")

    df = pd.DataFrame(rows)
    csv_path = results_dir / "epi_significance.csv"
    df.to_csv(csv_path, index=False)

    # --- Console summary ---------------------------------------------------
    print("\n=== Epistemic significance diagnostics ===")
    print(df.to_string(index=False))

    print(f"\nPARTIAL_CORR_GATE: {'PASS' if gate_pass else 'FAIL'}")
    print(
        f"  partial Spearman(ratio) rho={gate['rho_partial']:.4f} "
        f"(p={gate['rho_partial_p']:.4g}); "
        f"logistic epi coef={gate['logit_epi_coef']:.4f} "
        f"(p={gate['logit_epi_p']:.4g})"
    )
    print(
        f"  PRR-AUC(epi_mu)={prr_epi:.4f} vs PRR-AUC(mu-confidence)={prr_muconf:.4f} "
        f"(y=error; epi adds rejection value if {prr_epi:.4f} < {prr_muconf:.4f})"
    )

    # MC-vs-delta interpretation line.
    if np.isfinite(mc_delta_median_ratio) and mc_delta_median_ratio <= 2.0 and (
        np.isfinite(mc_pearson) and mc_pearson > 0.9
    ):
        interp = (
            "MC ~= delta (ratio ~1, high corr): smallness is structural, not an "
            "approximation artefact -> if the gate also FAILs, pivot the "
            "epistemic story to OOD / distance-aware."
        )
    elif np.isfinite(mc_delta_median_ratio) and mc_delta_median_ratio > 2.0:
        interp = (
            "MC >> delta (ratio >> 1): first-order approx underestimates -> "
            "report epi_mc as the rejection signal."
        )
    else:
        interp = (
            "MC vs delta inconclusive (check corr / ratio): "
            f"pearson={mc_pearson:.3f}, median ratio={mc_delta_median_ratio:.3f}."
        )
    print(f"MC_VS_DELTA: {interp}")
    if ood is not None:
        print(
            f"OOD epi ratio (ood/indom median) = "
            f"{ood['ratio_ood_over_indomain']:.3f} "
            f"(Mann-Whitney p(ood>indom)={ood['mannwhitney_p_ood_greater']:.4g})"
        )

    print(f"\nSaved diagnostics -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
