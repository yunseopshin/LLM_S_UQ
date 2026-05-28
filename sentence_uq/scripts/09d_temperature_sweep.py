"""CLI for Phase 9 (saturation attack, step 1): post-hoc temperature sweep.

Usage
-----
    python scripts/09d_temperature_sweep.py --setup 2 --device cpu

Question
~~~~~~~~
Phase 9.1 traced the epistemic collapse to **sigmoid saturation**: 92% of the
probe's per-token logits ``θ̂ᵀ z_ℓ`` are extreme, so ``π̂(1-π̂) ≈ 0`` and the
sensitivity ``ĝ = mean π̂(1-π̂) z`` vanishes → ``epi_μ = ĝᵀ Σ̂ ĝ ≈ 0``.

Before retraining the model with a logit regulariser, this script cheaply tests
**whether softening the per-token confidence even recovers epi_μ**, by tempering
the *fixed* trained logits at inference:

    π̃_ℓ(T) = σ(θ̂ᵀ z_ℓ / T)
    μ̃(T)   = (1/L) Σ_ℓ π̃_ℓ(T)

Mathematical caveat (why this is not a free lunch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Treating ``μ̃_T(θ) = (1/L) Σ σ(θᵀz/T)`` as the predictive and keeping the SAME
posterior ``Σ̂`` over θ:

    ∂μ̃_T/∂θ = (1/T) · ĝ_T,   ĝ_T = (1/L) Σ_ℓ π̃(1-π̃) z_ℓ
    epi_μ(T) = (1/T²) · ĝ_Tᵀ Σ̂ ĝ_T          ("honest" post-hoc value)

So raising T enlarges ``π̃(1-π̂)`` (less saturation) but the ``1/T²`` prefactor
fights it. We therefore report BOTH:

* ``epi_grad``   — the honest ``(1/T²) ĝ_Tᵀ Σ̂ ĝ_T``.
* ``epi_nofac``  — ``ĝ_Tᵀ Σ̂ ĝ_T`` without ``1/T²`` (the pure ĝ-channel "potential";
  a proxy for what *retraining* with smaller logits — which also enlarges Σ̂ —
  could unlock, since retraining would not carry the inference-time ``1/T``).

Decision rule
~~~~~~~~~~~~~
* If ``epi_grad`` has a sweet spot where it grows meaningfully while in-domain
  ECE/MAE stay acceptable → post-hoc tempering alone helps; cheap win.
* If ``epi_grad`` only falls but ``epi_nofac`` rises a lot → the ĝ-channel has
  headroom but the ``1/T²`` kills it post-hoc → **retraining** with a logit
  regulariser is the lever to pursue (it changes θ̂ and Σ̂, no ``1/T`` penalty).
* If neither moves → saturation is not the recoverable bottleneck; stop.

Outputs
~~~~~~~
* ``results/setup_{N}/document/temperature_sweep.json``
* ``results/setup_{N}/document/temperature_sweep.png``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import compute_calibration_metrics  # noqa: E402
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


def _sweep_one_T(
    logits_list: List[torch.Tensor],
    z_list: List[torch.Tensor],
    Sig: torch.Tensor,
    T: float,
    sat_thr: float,
) -> Dict[str, Any]:
    """Per-sentence epi readouts + μ̃ at temperature ``T``."""
    n = len(z_list)
    epi_grad = np.empty(n, dtype=np.float64)
    epi_nofac = np.empty(n, dtype=np.float64)
    mu = np.empty(n, dtype=np.float64)
    sat_frac = []
    for i, (logit, z) in enumerate(zip(logits_list, z_list)):
        pi = torch.sigmoid(logit / T)
        w = pi * (1.0 - pi)
        g = (w.unsqueeze(1) * z).mean(dim=0)
        q = float((g @ (Sig @ g)).clamp_min(0.0).item())
        epi_nofac[i] = q
        epi_grad[i] = q / (T * T)
        mu[i] = float(pi.mean().item())
        sat_frac.append(float(((pi < sat_thr) | (pi > 1.0 - sat_thr)).float().mean().item()))
    return {
        "epi_grad": epi_grad,
        "epi_nofac": epi_nofac,
        "mu": mu,
        "frac_saturated": float(np.mean(sat_frac)),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 9 saturation attack — post-hoc temperature sweep."
    )
    p.add_argument("--setup", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--trained-model", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--temperatures", type=float, nargs="+",
                   default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
    p.add_argument("--sat-threshold", type=float, default=0.05)
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point — see module docstring."""
    args = _build_parser().parse_args(argv)
    setup = int(args.setup)
    diag = _load_diag_module()
    device = torch.device(diag._resolve_device(args.device))

    results_dir = Path(args.results_dir or f"results/setup_{setup}")
    doc_dir = results_dir / "document"
    trained_path = Path(args.trained_model or results_dir / "trained_model.pt")

    print(f"=== Phase 9 temperature sweep — setup {setup} ===")
    if not trained_path.exists():
        print(f"error: trained model not found at {trained_path}", file=sys.stderr)
        return 2

    loaded = load_trained_model(trained_path, map_location="cpu")
    theta = loaded["theta_hat"].to(torch.float64)
    Sig = 0.5 * (loaded["Sigma_hat"] + loaded["Sigma_hat"].T).to(torch.float64)
    feature_params = loaded["feature_params"].to(device)
    feature_params.eval()

    records = [
        r for r in diag._prepare_test_records({}, setup, device, feature_params)
        if int(r.get("m_j", 0) or 0) > 0
    ]
    if not records:
        print("error: no test sentences with m_j > 0.", file=sys.stderr)
        return 2
    K = np.array([int(r["K_j"]) for r in records], dtype=np.float64)
    m = np.array([int(r["m_j"]) for r in records], dtype=np.float64)
    U = K / np.maximum(m, 1.0)

    z_list = [diag._extract_z_tokens(r, feature_params, device).to(torch.float64)
              for r in records]
    logits_list = [z @ theta for z in z_list]
    print(f"Test sentences: {len(records)}")

    # baseline logit-magnitude summary (saturation source)
    all_logit = torch.cat([l.abs() for l in logits_list]).numpy()
    print(f"|θ̂ᵀz| (T=1): median={np.median(all_logit):.2f}  "
          f"p90={np.percentile(all_logit,90):.2f}  max={all_logit.max():.2f}")

    rows: List[Dict[str, Any]] = []
    print(f"\n{'T':>5}{'epi_grad mean':>16}{'epi_nofac mean':>16}"
          f"{'ECE':>8}{'MAE':>8}{'sat%':>8}")
    for T in args.temperatures:
        s = _sweep_one_T(logits_list, z_list, Sig, float(T), args.sat_threshold)
        calib = compute_calibration_metrics(U, s["mu"], n_bins=10)
        mae = float(np.mean(np.abs(s["mu"] - U)))
        row = {
            "T": float(T),
            "epi_grad_mean": float(s["epi_grad"].mean()),
            "epi_grad_median": float(np.median(s["epi_grad"])),
            "epi_nofac_mean": float(s["epi_nofac"].mean()),
            "epi_nofac_median": float(np.median(s["epi_nofac"])),
            "ECE": float(calib["ECE"]),
            "MAE": mae,
            "frac_saturated": s["frac_saturated"],
        }
        rows.append(row)
        print(f"{T:>5.1f}{row['epi_grad_mean']:>16.3e}{row['epi_nofac_mean']:>16.3e}"
              f"{row['ECE']:>8.3f}{row['MAE']:>8.3f}{row['frac_saturated']:>8.3f}")

    summary = {
        "setup": setup,
        "n_test": len(records),
        "logit_abs_median": float(np.median(all_logit)),
        "logit_abs_p90": float(np.percentile(all_logit, 90)),
        "logit_abs_max": float(all_logit.max()),
        "sweep": rows,
        "note": ("epi_grad = (1/T^2) g^T Sig g (honest post-hoc); "
                 "epi_nofac = g^T Sig g (g-channel potential, proxy for retraining)."),
    }
    doc_dir.mkdir(parents=True, exist_ok=True)
    json_path = doc_dir / "temperature_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary -> {json_path}")

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        Ts = [r["T"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.5))
        ax[0].plot(Ts, [r["epi_grad_mean"] for r in rows], "o-", label="epi_grad (honest)")
        ax[0].plot(Ts, [r["epi_nofac_mean"] for r in rows], "s--", label="epi_nofac (ĝ-channel)")
        ax[0].axhline(8.07e-4, color="gray", ls=":", label="baseline epi_μ (T=1)")
        ax[0].set_yscale("log"); ax[0].set_xlabel("temperature T")
        ax[0].set_ylabel("mean epi (log)"); ax[0].set_title("epistemic vs T")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[1].plot(Ts, [r["ECE"] for r in rows], "o-", color="C3", label="ECE")
        ax[1].plot(Ts, [r["MAE"] for r in rows], "s-", color="C1", label="MAE")
        ax[1].plot(Ts, [r["frac_saturated"] for r in rows], "^-", color="C2", label="sat. frac")
        ax[1].set_xlabel("temperature T"); ax[1].set_title("calibration / saturation vs T")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        fig.suptitle("Post-hoc temperature sweep on probe logits (in-domain)")
        fig.tight_layout()
        plot_path = doc_dir / "temperature_sweep.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved figure  -> {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
