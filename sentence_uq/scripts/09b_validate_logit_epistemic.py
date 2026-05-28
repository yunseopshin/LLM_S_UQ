"""CLI for Phase 9.2-1: validate the logit-space epistemic signal.

Usage
-----
    python scripts/09b_validate_logit_epistemic.py --setup 2 --device cpu

Purpose
~~~~~~~
Phase 9.1 showed that the logit-space signal ``epi_logit = z̄ᵀ Σ̂ z̄`` lifts
ratio-level PRR-AUC from 0.139 (prob-space ``epi_μ``) to 0.317. But the signal
has *inverted* polarity vs. a textbook epistemic reading — higher ``epi_logit``
goes with **lower** error and **lower** ``U`` (Spearman(epi_logit, |μ̂−U|) ≈
−0.608). That raises the question: is ``epi_logit`` genuine parameter
uncertainty, or merely a confidence proxy that tracks ``μ̂``?

This script answers that with three checks on the Setup-2 test split:

1. **Partial Spearman** ``ρ(epi_logit, |μ̂−U| | μ̂)`` — rank-transform the
   three variables, partial out ``μ̂`` from both ``epi_logit`` and ``|μ̂−U|``
   by OLS on the ``μ̂`` rank, then correlate the residuals. **Primary gate**:
   ``|ρ| > 0.2`` means the signal carries error information beyond ``μ̂``.
2. **Stratified PRR-AUC** — within each ``μ̂`` tercile, rank by ``epi_logit``
   (passed *directly* to :func:`compute_prr`, highest rejected first) and
   compare PRR-AUC against the per-tercile no-skill baseline ``mean(U)``.
3. **VIF** ``= 1/(1−r²)`` between ``epi_logit`` and ``μ̂`` — *secondary*
   collinearity check only (VIF<5 ⟺ |r|≲0.89, too loose to settle the
   proxy question alone).

Decision rule (mirrors ``prompts/phase_9_2_logit_epistemic.md`` §Execution
Order): if the partial-Spearman gate FAILS (``|ρ| < 0.2``), STOP — the signal
is a μ̂/confidence proxy and must not be integrated.

Outputs
~~~~~~~
* ``results/setup_{N}/logit_epistemic_validation.json``
* ``results/setup_{N}/logit_epistemic_validation.png`` (2×2 figure)
* a summary table to stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import compute_prr  # noqa: E402
from src.inference.predict import load_trained_model  # noqa: E402


def _load_diag_module() -> Any:
    """Import the digit-prefixed Phase 9.1 module to reuse its data loaders."""
    path = _THIS_DIR / "09_diagnose_epistemic.py"
    spec = importlib.util.spec_from_file_location("diag09", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation in float64; NaN if either input is constant."""
    if x.size < 2:
        return float("nan")
    dx, dy = x - x.mean(), y - y.mean()
    denom = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(dx, dy) / denom)


def _ols_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residual of ``a`` after regressing it on ``[1, b]`` (OLS)."""
    design = np.vstack([np.ones_like(b), b]).T
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    return a - design @ coef


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """Spearman partial correlation of ``x`` and ``y`` controlling for ``z``.

    Rank-transforms all three inputs, partials ``z`` out of both ``x`` and
    ``y`` by OLS, and returns the Pearson correlation of the residuals (the
    standard rank-based partial correlation), plus the raw Spearman for
    reference.
    """
    from scipy.stats import rankdata

    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    raw = _pearson(rx, ry)  # == Spearman(x, y)
    ex = _ols_residual(rx.astype(np.float64), rz.astype(np.float64))
    ey = _ols_residual(ry.astype(np.float64), rz.astype(np.float64))
    return {"raw_spearman": raw, "partial_spearman": _pearson(ex, ey)}


def _vif(x: np.ndarray, y: np.ndarray) -> float:
    """Variance inflation factor ``1/(1−r²)`` for two variables."""
    r = _pearson(x, y)
    denom = 1.0 - r * r
    return float("inf") if denom <= 1e-12 else float(1.0 / denom)


def _terciles(values: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Split indices into low/mid/high terciles by ``values`` (ascending)."""
    order = np.argsort(values, kind="mergesort")
    thirds = np.array_split(order, 3)
    return [
        ("low", thirds[0]),
        ("mid", thirds[1]),
        ("high", thirds[2]),
    ]


# ---------------------------------------------------------------------------
# Per-sentence signal computation
# ---------------------------------------------------------------------------


def compute_signals(
    z_tokens_list: List[torch.Tensor],
    theta_hat: torch.Tensor,
    Sigma_hat: torch.Tensor,
    U_true: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-sentence epi_logit, epi_mu, μ̂, and |μ̂−U|."""
    theta = theta_hat.to(torch.float64)
    Sig = 0.5 * (Sigma_hat + Sigma_hat.T).to(torch.float64)

    n = len(z_tokens_list)
    epi_logit = np.empty(n, dtype=np.float64)
    epi_mu = np.empty(n, dtype=np.float64)
    mu_hat = np.empty(n, dtype=np.float64)
    for i, z in enumerate(z_tokens_list):
        zf = z.to(torch.float64)
        pi = torch.sigmoid(zf @ theta)
        w = pi * (1.0 - pi)
        g_hat = (w.unsqueeze(1) * zf).mean(dim=0)
        epi_mu[i] = float((g_hat @ (Sig @ g_hat)).clamp_min(0.0).item())
        z_bar = zf.mean(dim=0)
        epi_logit[i] = float((z_bar @ (Sig @ z_bar)).clamp_min(0.0).item())
        mu_hat[i] = float(pi.mean().item())

    return {
        "epi_logit": epi_logit,
        "epi_mu": epi_mu,
        "mu_hat": mu_hat,
        "abs_err": np.abs(mu_hat - U_true),
        "U_true": U_true,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_2x2(
    sig: Dict[str, np.ndarray],
    resid_x: np.ndarray,
    resid_y: np.ndarray,
    strat: Dict[str, Dict[str, float]],
    tercile_id: np.ndarray,
    save_path: Path,
) -> None:
    """2×2 validation figure (see module docstring §Outputs)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 9.0))

    # (0,0) residual scatter after μ̂ partialed out
    ax[0, 0].scatter(resid_x, resid_y, s=12, alpha=0.6, color="C3")
    ax[0, 0].set_title("epi_logit vs |err| residuals  (μ̂ partialed out)")
    ax[0, 0].set_xlabel("epi_logit rank residual")
    ax[0, 0].set_ylabel("|μ̂−U| rank residual")

    # (0,1) stratified PRR-AUC bars
    labels = ["low", "mid", "high", "overall"]
    prr_vals = [strat[k]["prr_auc"] for k in labels]
    noskill = [strat[k]["no_skill"] for k in labels]
    xpos = np.arange(len(labels))
    ax[0, 1].bar(xpos, prr_vals, width=0.6, color="C0", alpha=0.8, label="PRR-AUC")
    ax[0, 1].plot(xpos, noskill, "kD", ms=7, label="no-skill mean(U)")
    ax[0, 1].set_xticks(xpos)
    ax[0, 1].set_xticklabels(labels)
    ax[0, 1].set_title("Stratified PRR-AUC by μ̂ tercile")
    ax[0, 1].set_ylabel("PRR-AUC (rank by epi_logit)")
    ax[0, 1].legend(fontsize=8)

    # (1,0) epi_logit vs μ̂
    ax[1, 0].scatter(sig["mu_hat"], sig["epi_logit"], s=12, alpha=0.6, color="C2")
    ax[1, 0].set_title("epi_logit vs μ̂  (collinearity view)")
    ax[1, 0].set_xlabel("μ̂")
    ax[1, 0].set_ylabel("epi_logit")

    # (1,1) epi_logit vs |err| colored by μ̂ tercile
    colors = np.array(["C0", "C1", "C2"])[tercile_id]
    ax[1, 1].scatter(sig["epi_logit"], sig["abs_err"], s=12, alpha=0.6, c=colors)
    ax[1, 1].set_title("epi_logit vs |μ̂−U|  (color = μ̂ tercile)")
    ax[1, 1].set_xlabel("epi_logit")
    ax[1, 1].set_ylabel("|μ̂−U|")

    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("Phase 9.2-1 — is epi_logit a genuine epistemic signal?")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """argparse parser for ``scripts/09b_validate_logit_epistemic.py``."""
    p = argparse.ArgumentParser(
        description="Phase 9.2-1 — validate the logit-space epistemic signal."
    )
    p.add_argument("--setup", type=int, required=True, help="Experimental setup.")
    p.add_argument("--config", type=str, default=None, help="Optional YAML config.")
    p.add_argument("--device", type=str, default="cpu", help="Feature-extractor device.")
    p.add_argument("--trained-model", type=str, default=None, help="Override model path.")
    p.add_argument("--results-dir", type=str, default=None, help="Override output dir.")
    p.add_argument("--partial-threshold", type=float, default=0.2,
                   help="Pass threshold |ρ| for the partial-Spearman gate.")
    p.add_argument("--no-plots", action="store_true", help="Skip the figure.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point — see module docstring."""
    args = _build_parser().parse_args(argv)
    setup = int(args.setup)
    diag = _load_diag_module()
    device = torch.device(diag._resolve_device(args.device))
    cfg = diag._load_yaml(args.config) if args.config else {}

    results_dir = Path(args.results_dir or f"results/setup_{setup}")
    results_dir.mkdir(parents=True, exist_ok=True)
    trained_path = Path(args.trained_model or results_dir / "trained_model.pt")

    print(f"=== Phase 9.2-1 logit-epistemic validation — setup {setup} ===")
    print(f"Trained model: {trained_path}")
    if not trained_path.exists():
        print(f"error: trained model not found at {trained_path}", file=sys.stderr)
        return 2

    loaded = load_trained_model(trained_path, map_location="cpu")
    theta_hat = loaded["theta_hat"].to(torch.float32)
    Sigma_hat = loaded["Sigma_hat"].to(torch.float32)
    feature_params = loaded["feature_params"].to(device)
    feature_params.eval()

    test_records = [
        r for r in diag._prepare_test_records(cfg, setup, device, feature_params)
        if int(r.get("m_j", 0) or 0) > 0
    ]
    if not test_records:
        print("error: no test sentences with m_j > 0.", file=sys.stderr)
        return 2
    print(f"Test sentences (m_j > 0): {len(test_records)}")

    K = np.array([int(r["K_j"]) for r in test_records], dtype=np.float64)
    m = np.array([int(r["m_j"]) for r in test_records], dtype=np.float64)
    U_true = K / np.maximum(m, 1.0)
    z_tokens_list = [diag._extract_z_tokens(r, feature_params, device) for r in test_records]

    sig = compute_signals(z_tokens_list, theta_hat, Sigma_hat, U_true)

    # --- check 1: partial Spearman ------------------------------------------
    ps = partial_spearman(sig["epi_logit"], sig["abs_err"], sig["mu_hat"])
    partial_pass = abs(ps["partial_spearman"]) > args.partial_threshold

    # rank residuals for the plot
    from scipy.stats import rankdata
    rx = _ols_residual(rankdata(sig["epi_logit"]).astype(np.float64),
                        rankdata(sig["mu_hat"]).astype(np.float64))
    ry = _ols_residual(rankdata(sig["abs_err"]).astype(np.float64),
                        rankdata(sig["mu_hat"]).astype(np.float64))

    # --- check 2: stratified PRR-AUC ----------------------------------------
    tercile_id = np.empty(len(test_records), dtype=np.int64)
    strat: Dict[str, Dict[str, float]] = {}
    for tid, (label, idx) in enumerate(_terciles(sig["mu_hat"])):
        tercile_id[idx] = tid
        U_t = sig["U_true"][idx]
        prr = float(compute_prr(U_t, sig["epi_logit"][idx])["prr_auc"])
        ns = float(U_t.mean())
        strat[label] = {"prr_auc": prr, "no_skill": ns, "n": int(idx.size),
                        "pass": bool(prr > ns)}
    # overall
    overall_prr = float(compute_prr(sig["U_true"], sig["epi_logit"])["prr_auc"])
    strat["overall"] = {
        "prr_auc": overall_prr,
        "no_skill": float(sig["U_true"].mean()),
        "n": len(test_records),
        "pass": bool(overall_prr > float(sig["U_true"].mean())),
    }
    n_tercile_pass = sum(strat[k]["pass"] for k in ("low", "mid", "high"))
    strat_pass = n_tercile_pass >= 2

    # --- check 3: VIF -------------------------------------------------------
    vif = _vif(sig["epi_logit"], sig["mu_hat"])
    vif_pass = vif < 5.0

    # --- summary table ------------------------------------------------------
    print("\n--- Validation summary ---")
    print(f"{'Check':<46}{'Value':>12}{'Criterion':>16}{'Pass':>7}")

    def _row(name: str, val: str, crit: str, ok: bool) -> None:
        print(f"{name:<46}{val:>12}{crit:>16}{'YES' if ok else 'NO':>7}")

    _row("Partial Spearman(epi_logit,|err| | mu_hat)",
         f"{ps['partial_spearman']:+.3f}", "|rho|>0.2", partial_pass)
    print(f"   (raw Spearman, no control = {ps['raw_spearman']:+.3f})")
    for k in ("low", "mid", "high"):
        _row(f"Stratified PRR-AUC ({k} mu_hat tercile)",
             f"{strat[k]['prr_auc']:.4f}",
             f">{strat[k]['no_skill']:.3f}", strat[k]["pass"])
    _row("VIF(epi_logit, mu_hat)", f"{vif:.2f}", "<5", vif_pass)
    print(f"\nOverall PRR-AUC (epi_logit) = {overall_prr:.4f} "
          f"(no-skill = {strat['overall']['no_skill']:.4f})")

    overall_pass = partial_pass and strat_pass
    print(f"\n>>> GATE (partial |rho|>0.2 AND >=2/3 terciles lift): "
          f"{'PASS' if overall_pass else 'FAIL'}")
    if not overall_pass:
        print(">>> Recommendation: do NOT integrate epi_logit — it behaves like a "
              "mu_hat/confidence proxy. Reassess before Phase 9.2-2/9.2-3.")

    # --- persist ------------------------------------------------------------
    summary = {
        "setup": setup,
        "n_test_sentences": len(test_records),
        "partial_threshold": args.partial_threshold,
        "check_1_partial_spearman": {
            "raw_spearman": ps["raw_spearman"],
            "partial_spearman": ps["partial_spearman"],
            "pass": bool(partial_pass),
        },
        "check_2_stratified_prr": strat,
        "check_3_vif": {"vif": vif, "pass": bool(vif_pass)},
        "overall_prr_auc_epi_logit": overall_prr,
        "gate_pass": bool(overall_pass),
    }
    json_path = results_dir / "logit_epistemic_validation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary -> {json_path}")

    if not args.no_plots:
        plot_path = results_dir / "logit_epistemic_validation.png"
        _plot_2x2(sig, rx, ry, strat, tercile_id, plot_path)
        print(f"Saved figure  -> {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
