"""Compare logit-regularised models against the baseline (Phase 9.4 eval).

Usage
-----
    python scripts/09e_compare_logitreg.py --setup 2 \
        --models baseline=results/setup_2/trained_model.pt \
                 lam3e-3=results/setup_2_logitreg/lam3em3/trained_model.pt

For each model, on the in-domain Setup-2 test split, reports:
  epi_μ mean/median, frac saturated tokens, |logit| median, ratio-level ECE & MAE,
  Pearson r (μ̂ vs U), and error-detection AUROC.

The headline question (Phase 9.4): does the logit penalty raise `epi_μ` while
keeping ECE / MAE / AUROC close to baseline?
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

from src.evaluation.metrics import (  # noqa: E402
    compute_calibration_metrics,
    compute_strict_metrics,
)
from src.inference.predict import load_trained_model  # noqa: E402


def _load_diag_module() -> Any:
    path = _THIS_DIR / "09_diagnose_epistemic.py"
    spec = importlib.util.spec_from_file_location("diag09", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    dx, dy = x - x.mean(), y - y.mean()
    d = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    return float(np.dot(dx, dy) / d) if d > 1e-12 else float("nan")


def _eval_model(
    diag: Any, setup: int, device: torch.device, model_path: Path, sat_thr: float
) -> Dict[str, Any]:
    """Compute the comparison metrics for one trained model."""
    loaded = load_trained_model(model_path, map_location="cpu")
    theta = loaded["theta_hat"].to(torch.float64)
    Sig = 0.5 * (loaded["Sigma_hat"] + loaded["Sigma_hat"].T).to(torch.float64)
    fp = loaded["feature_params"].to(device)
    fp.eval()

    records = [
        r for r in diag._prepare_test_records({}, setup, device, fp)
        if int(r.get("m_j", 0) or 0) > 0
    ]
    K = np.array([int(r["K_j"]) for r in records], dtype=np.float64)
    m = np.array([int(r["m_j"]) for r in records], dtype=np.float64)
    U = K / np.maximum(m, 1.0)

    epi_mu, mu_hat, abs_logit, sat = [], [], [], []
    for r in records:
        z = diag._extract_z_tokens(r, fp, device).to(torch.float64)
        logit = z @ theta
        pi = torch.sigmoid(logit)
        w = pi * (1.0 - pi)
        g = (w.unsqueeze(1) * z).mean(dim=0)
        epi_mu.append(float((g @ (Sig @ g)).clamp_min(0.0).item()))
        mu_hat.append(float(pi.mean().item()))
        abs_logit.append(logit.abs())
        sat.append(float(((pi < sat_thr) | (pi > 1.0 - sat_thr)).float().mean().item()))

    epi_mu = np.asarray(epi_mu)
    mu_hat = np.asarray(mu_hat)
    all_abs_logit = torch.cat(abs_logit).numpy()
    calib = compute_calibration_metrics(U, mu_hat, n_bins=10)
    strict = compute_strict_metrics(K, m, mu_hat)
    return {
        "n": len(records),
        "epi_mu_mean": float(epi_mu.mean()),
        "epi_mu_median": float(np.median(epi_mu)),
        "frac_saturated": float(np.mean(sat)),
        "logit_abs_median": float(np.median(all_abs_logit)),
        "ECE": float(calib["ECE"]),
        "MAE": float(np.mean(np.abs(mu_hat - U))),
        "Pearson_r": _pearson(U, mu_hat),
        "error_detection_auroc": float(strict["error_detection_auroc"]),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 9.4 — compare logit-reg models.")
    p.add_argument("--setup", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--models", type=str, nargs="+", required=True,
                   help="name=path pairs, e.g. baseline=results/setup_2/trained_model.pt")
    p.add_argument("--sat-threshold", type=float, default=0.05)
    p.add_argument("--out", type=str,
                   default="results/setup_2/document/logitreg_compare.json")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    diag = _load_diag_module()
    device = torch.device(diag._resolve_device(args.device))

    results: Dict[str, Dict[str, Any]] = {}
    for spec in args.models:
        if "=" not in spec:
            print(f"error: --models entry '{spec}' must be name=path", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            print(f"  [skip] {name}: {p} not found")
            continue
        print(f"  evaluating {name} ...")
        results[name] = _eval_model(diag, int(args.setup), device, p, args.sat_threshold)

    if not results:
        print("error: no models evaluated.", file=sys.stderr)
        return 2

    cols = ["epi_mu_mean", "epi_mu_median", "frac_saturated", "logit_abs_median",
            "ECE", "MAE", "Pearson_r", "error_detection_auroc"]
    print(f"\n{'model':<14}" + "".join(f"{c[:13]:>15}" for c in cols))
    for name, r in results.items():
        print(f"{name:<14}" + "".join(
            f"{r[c]:>15.4g}" for c in cols))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
