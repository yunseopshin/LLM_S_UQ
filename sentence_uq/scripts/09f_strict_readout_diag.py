"""Phase 9.7 — Strict-factuality READOUT diagnostic (no retraining).

Question
--------
Our reported strict-factuality detector scores each sentence by the binomial
"all atoms supported" probability ``p_strict = μ̃_j ** m_j`` (``predict.py``).
On Setup-2 that gives strict AUROC ≈ 0.78, **below** Han's probe (≈0.81–0.86).
But the Binomial-vs-Bernoulli ablation already hinted that ranking by ``μ̂_j``
*directly* (the ``m=1`` case) lifts AUROC to ≈0.83.

This script quantifies, on the *same* promoted model, how the strict
**detection score** (the quantity AUROC/AUPRC rank) changes the discrimination
numbers — separating a *readout* artefact from a *representation* deficit.
AUROC/AUPRC are rank-only, so swapping the score is legitimate; the calibrated
probability ``μ̃^m`` is still what ECE/Brier should use.

Candidate scores compared (all derived from one model, zero retraining):
    μ̂          raw sentence mean factuality
    μ̃          probit-shrunk mean (Bayesian)
    μ̂ ** m     binomial, raw   (= Ours (Point) p_strict)
    μ̃ ** m     binomial, probit (= Ours (Bayesian) p_strict — CURRENT REPORT)
    μ̃ ** (β·m) length-tempered binomial, β grid (readout knob)

Usage
-----
    python scripts/09f_strict_readout_diag.py --setup 2 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import split_save_filename  # noqa: E402
from src.evaluation.metrics import compute_strict_factuality_metrics  # noqa: E402
from src.features.extractor import extract_sentence_token_features  # noqa: E402
from src.inference.predict import Predictor, load_trained_model  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.train.trainer import SentenceUQTrainer  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _load_yaml(path: str | None) -> Dict[str, Any]:
    """Parse a YAML config file; empty dict when missing/unset."""
    if not path or yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg: Dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    """Read ``cfg[section][key]`` with a default."""
    return ((cfg.get(section) or {}).get(key)) if cfg else default


def _auroc_auprc(A: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """AUROC + AUPRC of ``score`` against binary label ``A`` (rank-only)."""
    m = compute_strict_factuality_metrics(A, score, np.zeros_like(score))
    return float(m["AUROC"]), float(m["AUPRC"])


def main(argv: List[str] | None = None) -> int:
    """Compute strict AUROC/AUPRC under several readouts of one trained model."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup", type=int, default=2)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--trained-model", type=str,
        default=None, help="default: results/setup_{N}/trained_model.pt",
    )
    args = p.parse_args(argv)

    cfg = _load_yaml(args.config)
    device = torch.device(args.device)
    results_dir = Path(f"results/setup_{args.setup}")
    trained_path = Path(args.trained_model or results_dir / "trained_model.pt")

    loaded = load_trained_model(trained_path, map_location="cpu")
    theta_hat = loaded["theta_hat"]
    Sigma_hat = loaded["Sigma_hat"]
    feature_params = loaded["feature_params"].to(device)
    feature_params.eval()

    predictor = Predictor(
        theta_hat=theta_hat,
        Sigma_hat=Sigma_hat,
        feature_params=feature_params,
        use_probit_shrinkage=True,
    )

    # --- test records (same helper the trainer/eval uses) ------------------
    splits_dir = _cfg_get(cfg, "dataset", "splits_dir", "data/splits")
    split_file = _cfg_get(cfg, "dataset", "split_file") or str(
        Path(splits_dir) / split_save_filename(args.setup)
    )
    gdirs = {
        "factscore_bio": _cfg_get(cfg, "generation", "factscore_bio_dir",
                                  "data/generations/factscore_bio"),
        "longfact": _cfg_get(cfg, "generation", "longfact_dir",
                             "data/generations/longfact"),
    }
    cdirs = {
        "factscore_bio": _cfg_get(cfg, "cache", "factscore_bio_dir",
                                  "data/cache/factscore_bio"),
        "longfact": _cfg_get(cfg, "cache", "longfact_dir", "data/cache/longfact"),
    }
    pdirs = {
        "factscore_bio": _cfg_get(cfg, "processed", "factscore_bio_dir",
                                  "data/processed/factscore_bio"),
        "longfact": _cfg_get(cfg, "processed", "longfact_dir",
                             "data/processed/longfact"),
    }
    trainer = SentenceUQTrainer(
        model=BayesianSentenceUQ(feature_params=feature_params), device=device,
    )
    data = trainer.prepare_data(
        split_file=split_file, generations_dirs=gdirs,
        cache_dirs=cdirs, processed_dirs=pdirs,
    )
    records = [r for r in (data.get("test") or []) if int(r.get("m_j", 0) or 0) > 0]
    print(f"Model: {trained_path}\nTest sentences (m_j>0): {len(records)}")

    mu_hat, mu_tilde, m_arr, A_arr = [], [], [], []
    for rec in records:
        with torch.no_grad():
            z = extract_sentence_token_features(
                hidden_states=rec["hidden_states"].to(device),
                entropy=rec["entropy"].to(device),
                top1_prob=rec["top1"].to(device),
                token_range=(int(rec["token_range"][0]), int(rec["token_range"][1])),
                params=feature_params,
            ).detach().cpu().to(torch.float32)
        m_j = int(rec["m_j"])
        out = predictor.predict_sentence(z, m_j=m_j)
        mu_hat.append(float(out["mu_hat"]))
        mu_tilde.append(float(out["p_factual_probit"]))
        m_arr.append(m_j)
        A_arr.append(1.0 if int(rec["K_j"]) == m_j else 0.0)

    mu_hat = np.asarray(mu_hat)
    mu_tilde = np.asarray(mu_tilde)
    m_arr = np.asarray(m_arr, dtype=np.float64)
    A = np.asarray(A_arr)
    print(f"frac strict-factual (A=1): {A.mean():.4f}   median m_j: {np.median(m_arr):.0f}")

    # Note: μ̃^(β·m) = (μ̃^m)^β is a monotone reparam of μ̃^m → identical AUROC/
    # AUPRC. Uniform exponent scaling is a no-op for ranking; the only readout
    # that changes the ranking is *decoupling* the score from m (use μ̂ / μ̃).
    scores = {
        "mu_hat        (μ̂  — decoupled from m)": mu_hat,
        "mu_tilde      (μ̃ probit, decoupled)": mu_tilde,
        "mu_hat ** m   (Point p_strict)": np.power(mu_hat, m_arr),
        "mu_tilde ** m (Bayes p_strict, CURRENT)": np.power(mu_tilde, m_arr),
    }

    print("\n=== Strict-factuality detection by readout (same model) ===")
    print(f"{'readout':<46}{'AUROC':>9}{'AUPRC':>9}")
    out_json: Dict[str, Dict[str, float]] = {}
    for name, sc in scores.items():
        auroc, auprc = _auroc_auprc(A, sc)
        print(f"{name:<46}{auroc:>9.4f}{auprc:>9.4f}")
        out_json[name.split("(")[0].strip()] = {"AUROC": auroc, "AUPRC": auprc}

    # --- reference baselines from the existing eval table ------------------
    print("\n=== Reference (from final_metrics_strict.csv) ===")
    ref_csv = results_dir / "final_metrics_strict.csv"
    if ref_csv.exists():
        import csv
        with open(ref_csv, newline="") as f:
            for row in csv.DictReader(f):
                if "factuality_probe" in row["method"] or row["method"].startswith("Ours"):
                    print(f"{row['method']:<46}{float(row['AUROC']):>9.4f}{float(row['AUPRC']):>9.4f}")

    out_path = results_dir / "document" / "strict_readout_diag.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n": len(records), "frac_strict": float(A.mean()),
                   "median_m": float(np.median(m_arr)), "readouts": out_json}, f, indent=2)
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
