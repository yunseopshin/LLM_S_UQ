"""CLI for Phase 9.1: diagnose the epistemic-uncertainty collapse.

Usage
-----
    python scripts/09_diagnose_epistemic.py --setup 2

What it does
~~~~~~~~~~~~
Loads the Phase 4-1 ``trained_model.pt`` (Laplace posterior ``(θ̂, Σ̂)``
plus the feature extractor ``ψ``) and the test split, then runs the five
no-retraining diagnostics of ``prompts/phase_9_epistemic_collapse.md`` to
locate *why* ``Epi_μ = ĝᵀ Σ̂ ĝ`` has collapsed to ≈ 0:

1. **Σ̂ eigenspectrum** — is the posterior covariance uniformly tiny
   (Factor 1: posterior over-concentration)?
2. **ĝ norm + π̂ distribution** — are token predictions saturated so that
   ``π̂(1-π̂)`` damps ĝ to ≈ 0 (Factor 2: sigmoid saturation)?
3. **Upper-bound decomposition** — split the collapse into ‖ĝ‖² and
   λ_max(Σ̂) contributions, and an isotropy check.
4. **Learned σ₀** — did the learnable prior tighten during training?
5. **Fisher data term vs prior** — does the binomial Fisher term swamp the
   prior precision?

Bonus (informs the "swap to logit-space epistemic" remediation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For every test sentence we also compute a *logit-space* epistemic signal
``z̄ᵀ Σ̂ z̄`` (no ``π̂(1-π̂)`` damping, ``z̄`` = mean token feature) and
compare both signals against the per-sentence ratio error: Spearman
correlation with ``|μ̂ - U|`` and PRR-AUC when ranking by each. This tells
us whether a logit-space readout would rank error better than the current
probability-space ``Epi_μ`` *without* any data generation.

Outputs
~~~~~~~
* ``results/setup_{N}/epistemic_diagnostics.json`` — every scalar summary.
* ``results/setup_{N}/epistemic_diag_eigenspectrum.png``
* ``results/setup_{N}/epistemic_diag_distributions.png``
* ``results/setup_{N}/epistemic_diag_signal_vs_error.png``

All numerics run in fp32/fp64 (CLAUDE.md rule 10). No retraining; CPU is
fine and the whole thing takes a few minutes.
"""

from __future__ import annotations

import argparse
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

from src.data.dataset import SETUPS, split_save_filename  # noqa: E402
from src.evaluation.metrics import compute_prr  # noqa: E402
from src.features.extractor import extract_sentence_token_features  # noqa: E402
from src.inference.predict import load_trained_model  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.train.trainer import SentenceUQTrainer  # noqa: E402
from src.utils.validation import validate_binomial_counts  # noqa: E402


# ---------------------------------------------------------------------------
# Config / CLI plumbing (mirrors scripts/04_evaluate.py conventions)
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dict (empty dict when blank)."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg: Dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    """Read ``cfg[section][key]`` with a default fallback."""
    sec = cfg.get(section) or {}
    val = sec.get(key)
    return val if val is not None else default


def _resolve_device(name: str) -> str:
    """``cuda`` falls back to ``cpu`` when no GPU is visible."""
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for ``scripts/09_diagnose_epistemic.py``."""
    p = argparse.ArgumentParser(
        description="Phase 9.1 — diagnose the epistemic-uncertainty collapse."
    )
    p.add_argument(
        "--setup", type=int, choices=list(SETUPS), required=True,
        help="Experimental setup (1, 2, or 3).",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Optional YAML config (overrides default data paths).",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device for the feature extractor (default cpu).",
    )
    p.add_argument(
        "--trained-model", type=str, default=None,
        help="Override path to trained_model.pt "
             "(default: results/setup_{N}/trained_model.pt).",
    )
    p.add_argument(
        "--results-dir", type=str, default=None,
        help="Override output dir (default: results/setup_{N}).",
    )
    p.add_argument(
        "--sat-threshold", type=float, default=0.05,
        help="A token is 'saturated' when π̂ < thr or π̂ > 1-thr "
             "(default 0.05).",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib figures (JSON still produced).",
    )
    return p


# ---------------------------------------------------------------------------
# Data preparation (mirrors scripts/04_evaluate.py)
# ---------------------------------------------------------------------------


def _prepare_test_records(
    cfg: Dict[str, Any],
    setup: int,
    device: torch.device,
    feature_params: Any,
) -> List[Dict[str, Any]]:
    """Materialise the test-split sentence records via the Phase 4-1 trainer."""
    splits_dir = _cfg_get(cfg, "dataset", "splits_dir", "data/splits")
    split_file = _cfg_get(cfg, "dataset", "split_file") or str(
        Path(splits_dir) / split_save_filename(setup)
    )
    generations_dirs = {
        "factscore_bio": _cfg_get(
            cfg, "generation", "factscore_bio_dir",
            "data/generations/factscore_bio",
        ),
        "longfact": _cfg_get(
            cfg, "generation", "longfact_dir", "data/generations/longfact"
        ),
    }
    cache_dirs = {
        "factscore_bio": _cfg_get(
            cfg, "cache", "factscore_bio_dir", "data/cache/factscore_bio"
        ),
        "longfact": _cfg_get(
            cfg, "cache", "longfact_dir", "data/cache/longfact"
        ),
    }
    processed_dirs = {
        "factscore_bio": _cfg_get(
            cfg, "processed", "factscore_bio_dir",
            "data/processed/factscore_bio",
        ),
        "longfact": _cfg_get(
            cfg, "processed", "longfact_dir", "data/processed/longfact"
        ),
    }

    trainer = SentenceUQTrainer(
        model=BayesianSentenceUQ(feature_params=feature_params),
        device=device,
    )
    data = trainer.prepare_data(
        split_file=split_file,
        generations_dirs=generations_dirs,
        cache_dirs=cache_dirs,
        processed_dirs=processed_dirs,
    )
    return list(data.get("test") or [])


def _extract_z_tokens(
    record: Dict[str, Any],
    feature_params: Any,
    device: torch.device,
) -> torch.Tensor:
    """Run the Phase 2-1 feature extractor on a single sentence."""
    with torch.no_grad():
        z = extract_sentence_token_features(
            hidden_states=record["hidden_states"].to(device),
            entropy=record["entropy"].to(device),
            top1_prob=record["top1"].to(device),
            token_range=(
                int(record["token_range"][0]),
                int(record["token_range"][1]),
            ),
            params=feature_params,
        )
    return z.detach().cpu().to(torch.float32)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson r on ranks); NaN if degenerate."""
    if x.size < 2:
        return float("nan")

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(a.size, dtype=np.float64)
        return ranks

    rx, ry = _rank(x), _rank(y)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(dx, dy) / denom)


def diagnose_sigma_eigenspectrum(Sigma: torch.Tensor) -> Dict[str, Any]:
    """Diagnostic 1: eigenspectrum + isotropy summary of Σ̂."""
    sym = 0.5 * (Sigma + Sigma.T)
    eigvals = torch.linalg.eigvalsh(sym).to(torch.float64).numpy()
    lam_max = float(eigvals.max())
    lam_min = float(eigvals.min())
    lam_mean = float(eigvals.mean())
    return {
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "lambda_mean": lam_mean,
        "lambda_median": float(np.median(eigvals)),
        "trace": float(eigvals.sum()),
        "condition_number": lam_max / max(lam_min, 1e-15),
        # anisotropy: 1.0 == perfectly isotropic, large == one dominant axis
        "anisotropy_ratio": lam_max / max(lam_mean, 1e-15),
        "eigvals_sorted": eigvals[::-1].tolist(),
    }


def diagnose_g_and_pi(
    z_tokens_list: List[torch.Tensor],
    theta_hat: torch.Tensor,
    sat_threshold: float,
) -> Dict[str, Any]:
    """Diagnostic 2: ĝ norms and π̂ distribution across all test tokens."""
    theta = theta_hat.to(torch.float64)
    g_norms: List[float] = []
    mu_hats: List[float] = []
    all_pi: List[np.ndarray] = []
    for z in z_tokens_list:
        zf = z.to(torch.float64)
        logits = zf @ theta
        pi = torch.sigmoid(logits)
        w = pi * (1.0 - pi)
        g_tokens = w.unsqueeze(1) * zf
        g_hat = g_tokens.mean(dim=0)
        g_norms.append(float(torch.linalg.norm(g_hat).item()))
        mu_hats.append(float(pi.mean().item()))
        all_pi.append(pi.numpy())

    pi_cat = np.concatenate(all_pi) if all_pi else np.zeros(0)
    g_arr = np.asarray(g_norms, dtype=np.float64)
    sat = float(
        np.mean((pi_cat < sat_threshold) | (pi_cat > 1.0 - sat_threshold))
    ) if pi_cat.size else float("nan")
    return {
        "g_hat_norm": {
            "mean": float(g_arr.mean()),
            "median": float(np.median(g_arr)),
            "min": float(g_arr.min()),
            "max": float(g_arr.max()),
        },
        "pi": {
            "mean": float(pi_cat.mean()) if pi_cat.size else float("nan"),
            "median": float(np.median(pi_cat)) if pi_cat.size else float("nan"),
            "frac_saturated": sat,
            "frac_gt_0.9": float(np.mean(pi_cat > 0.9)) if pi_cat.size else float("nan"),
            "frac_lt_0.1": float(np.mean(pi_cat < 0.1)) if pi_cat.size else float("nan"),
            "num_tokens": int(pi_cat.size),
        },
        "mu_hat": {
            "mean": float(np.mean(mu_hats)),
            "median": float(np.median(mu_hats)),
            "frac_in_0.1_0.9": float(
                np.mean((np.asarray(mu_hats) > 0.1) & (np.asarray(mu_hats) < 0.9))
            ),
        },
        "_arrays": {  # kept for plotting, stripped before JSON dump
            "pi_cat": pi_cat,
            "g_norms": g_arr,
            "mu_hats": np.asarray(mu_hats, dtype=np.float64),
        },
    }


def diagnose_upper_bound(
    g_norm_mean: float, lam_max: float, lam_mean: float, epi_mu_mean: float
) -> Dict[str, Any]:
    """Diagnostic 3: ‖ĝ‖² vs λ contributions to the Epi_μ collapse."""
    epi_upper = (g_norm_mean ** 2) * lam_max
    epi_iso = (g_norm_mean ** 2) * lam_mean
    return {
        "g_norm_mean_sq": g_norm_mean ** 2,
        "epi_upper_bound": epi_upper,            # ‖ĝ‖²·λ_max
        "epi_isotropic_est": epi_iso,            # ‖ĝ‖²·λ_mean
        "epi_mu_mean_actual": epi_mu_mean,
        "actual_over_upper": epi_mu_mean / max(epi_upper, 1e-30),
        "actual_over_isotropic": epi_mu_mean / max(epi_iso, 1e-30),
    }


def diagnose_prior(feature_params: Any) -> Dict[str, Any]:
    """Diagnostic 4: learned σ₀ from the learnable log_sigma_0."""
    log_s = feature_params.log_sigma_0.detach().to(torch.float64)
    sigma_0 = torch.exp(log_s)
    return {
        "log_sigma_0": {
            "min": float(log_s.min().item()),
            "max": float(log_s.max().item()),
            "mean": float(log_s.mean().item()),
        },
        "sigma_0": {
            "min": float(sigma_0.min().item()),
            "max": float(sigma_0.max().item()),
            "mean": float(sigma_0.mean().item()),
        },
        "note": "init log_sigma_0=0 -> sigma_0=1; <1 means prior tightened.",
    }


def diagnose_fisher_vs_prior(
    Sigma: torch.Tensor, feature_params: Any
) -> Dict[str, Any]:
    """Diagnostic 5: Fisher data term (Σ̂⁻¹ − Σ₀⁻¹) vs prior precision."""
    sym = 0.5 * (Sigma + Sigma.T).to(torch.float64)
    Sigma_hat_inv = torch.linalg.inv(sym)
    Sigma_0_inv = feature_params.get_Sigma_0_inv().detach().to(torch.float64)
    fisher = Sigma_hat_inv - Sigma_0_inv
    fisher_sym = 0.5 * (fisher + fisher.T)
    fisher_eig = torch.linalg.eigvalsh(fisher_sym).numpy()
    prior_diag = torch.diag(Sigma_0_inv).numpy()
    return {
        "fisher_data_lambda_max": float(fisher_eig.max()),
        "fisher_data_lambda_min": float(fisher_eig.min()),
        "prior_inv_diag_max": float(prior_diag.max()),
        "prior_inv_diag_mean": float(prior_diag.mean()),
        "fisher_over_prior": float(fisher_eig.max()) / max(float(prior_diag.max()), 1e-15),
    }


def diagnose_logit_vs_prob_signal(
    z_tokens_list: List[torch.Tensor],
    theta_hat: torch.Tensor,
    Sigma: torch.Tensor,
    U_true: np.ndarray,
) -> Dict[str, Any]:
    """Bonus: prob-space Epi_μ vs logit-space z̄ᵀΣ̂z̄ as error rankers.

    For each sentence computes:
      * ``epi_mu  = ĝᵀ Σ̂ ĝ``           (probability space, π̂(1-π̂)-damped)
      * ``logit_epi = z̄ᵀ Σ̂ z̄``        (logit space, undamped; z̄ = mean z_ℓ)
      * ``logit_epi_tok = mean_ℓ z_ℓᵀ Σ̂ z_ℓ`` (per-token mean variant)
    then compares Spearman(signal, |μ̂-U|) and PRR-AUC(U, signal).
    """
    theta = theta_hat.to(torch.float64)
    Sig = 0.5 * (Sigma + Sigma.T).to(torch.float64)

    epi_mu = np.empty(len(z_tokens_list), dtype=np.float64)
    logit_epi = np.empty(len(z_tokens_list), dtype=np.float64)
    logit_epi_tok = np.empty(len(z_tokens_list), dtype=np.float64)
    mu_hat = np.empty(len(z_tokens_list), dtype=np.float64)
    for i, z in enumerate(z_tokens_list):
        zf = z.to(torch.float64)
        logits = zf @ theta
        pi = torch.sigmoid(logits)
        w = pi * (1.0 - pi)
        g_hat = (w.unsqueeze(1) * zf).mean(dim=0)
        epi_mu[i] = float((g_hat @ (Sig @ g_hat)).clamp_min(0.0).item())

        z_bar = zf.mean(dim=0)
        logit_epi[i] = float((z_bar @ (Sig @ z_bar)).clamp_min(0.0).item())

        Sz = zf @ Sig
        zSz = (Sz * zf).sum(dim=1).clamp_min(0.0)
        logit_epi_tok[i] = float(zSz.mean().item())
        mu_hat[i] = float(pi.mean().item())

    abs_err = np.abs(mu_hat - U_true)
    out: Dict[str, Any] = {}
    for name, sig in (
        ("epi_mu_prob", epi_mu),
        ("logit_epi_zbar", logit_epi),
        ("logit_epi_tokmean", logit_epi_tok),
    ):
        out[name] = {
            "mean": float(sig.mean()),
            "median": float(np.median(sig)),
            "spearman_vs_abs_err": _spearman(sig, abs_err),
            "prr_auc_ratio": float(compute_prr(U_true, sig)["prr_auc"]),
        }
    # baseline: PRR with a useless (constant) ranker == mean quality plateau
    out["_prr_no_skill_mean_U"] = float(U_true.mean())
    out["_arrays"] = {
        "epi_mu": epi_mu,
        "logit_epi_zbar": logit_epi,
        "abs_err": abs_err,
    }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_eigenspectrum(eigvals_desc: List[float], save_path: Path) -> None:
    """Log-scale plot of the sorted Σ̂ eigenvalues."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    vals = np.asarray(eigvals_desc, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(np.arange(1, vals.size + 1), np.maximum(vals, 1e-30), marker="o", ms=3)
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel("eigenvalue of Σ̂  (log)")
    ax.set_title("Diagnostic 1 — posterior covariance eigenspectrum")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_distributions(arrays: Dict[str, np.ndarray], save_path: Path) -> None:
    """π̂ histogram, ‖ĝ‖ histogram, μ̂ histogram side by side."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    axes[0].hist(arrays["pi_cat"], bins=40, color="C0", alpha=0.8)
    axes[0].set_title("per-token π̂  (Factor 2)")
    axes[0].set_xlabel("π̂")
    axes[1].hist(arrays["g_norms"], bins=30, color="C1", alpha=0.8)
    axes[1].set_title("per-sentence ‖ĝ‖₂")
    axes[1].set_xlabel("‖ĝ‖₂")
    axes[2].hist(arrays["mu_hats"], bins=30, color="C2", alpha=0.8)
    axes[2].set_title("per-sentence μ̂")
    axes[2].set_xlabel("μ̂")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Diagnostic 2 — gradient / prediction distributions")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_signal_vs_error(arrays: Dict[str, np.ndarray], save_path: Path) -> None:
    """Scatter of prob-space and logit-space epistemic vs |μ̂-U|."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    axes[0].scatter(arrays["epi_mu"], arrays["abs_err"], s=12, alpha=0.6, color="C3")
    axes[0].set_title("prob-space Epi_μ vs |μ̂-U|")
    axes[0].set_xlabel("Epi_μ = ĝᵀΣ̂ĝ")
    axes[0].set_ylabel("|μ̂ - U|")
    axes[1].scatter(arrays["logit_epi_zbar"], arrays["abs_err"], s=12, alpha=0.6, color="C0")
    axes[1].set_title("logit-space z̄ᵀΣ̂z̄ vs |μ̂-U|")
    axes[1].set_xlabel("z̄ᵀ Σ̂ z̄")
    axes[1].set_ylabel("|μ̂ - U|")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Bonus — does either epistemic signal track error?")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _strip_arrays(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively drop ``_arrays`` keys so the summary is JSON-serialisable."""
    return {k: (_strip_arrays(v) if isinstance(v, dict) else v)
            for k, v in d.items() if k != "_arrays"}


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point — see module docstring."""
    args = _build_parser().parse_args(argv)
    setup = int(args.setup)
    device = torch.device(_resolve_device(args.device))
    cfg = _load_yaml(args.config) if args.config else {}

    results_dir = Path(args.results_dir or f"results/setup_{setup}")
    results_dir.mkdir(parents=True, exist_ok=True)
    trained_path = Path(args.trained_model or results_dir / "trained_model.pt")

    print(f"=== Phase 9.1 epistemic diagnostics — setup {setup} ===")
    print(f"Trained model: {trained_path}")
    if not trained_path.exists():
        print(f"error: trained model not found at {trained_path}", file=sys.stderr)
        return 2

    loaded = load_trained_model(trained_path, map_location="cpu")
    theta_hat = loaded["theta_hat"].to(torch.float32)
    Sigma_hat = loaded["Sigma_hat"].to(torch.float32)
    feature_params = loaded["feature_params"].to(device)
    feature_params.eval()
    print(f"k = {theta_hat.shape[0]}  (feature dim)")

    test_records = [
        r for r in _prepare_test_records(cfg, setup, device, feature_params)
        if int(r.get("m_j", 0) or 0) > 0
    ]
    if not test_records:
        print("error: no test sentences with m_j > 0.", file=sys.stderr)
        return 2
    print(f"Test sentences (m_j > 0): {len(test_records)}")

    K = np.array([int(r["K_j"]) for r in test_records], dtype=np.float64)
    m = np.array([int(r["m_j"]) for r in test_records], dtype=np.float64)
    validate_binomial_counts(K, m, context="09_diagnose")
    U_true = K / np.maximum(m, 1.0)

    z_tokens_list = [_extract_z_tokens(r, feature_params, device) for r in test_records]

    # --- diagnostics ---------------------------------------------------------
    d1 = diagnose_sigma_eigenspectrum(Sigma_hat)
    d2 = diagnose_g_and_pi(z_tokens_list, theta_hat, args.sat_threshold)
    bonus = diagnose_logit_vs_prob_signal(z_tokens_list, theta_hat, Sigma_hat, U_true)
    epi_mu_mean = float(bonus["_arrays"]["epi_mu"].mean())
    d3 = diagnose_upper_bound(
        d2["g_hat_norm"]["mean"], d1["lambda_max"], d1["lambda_mean"], epi_mu_mean
    )
    d4 = diagnose_prior(feature_params)
    d5 = diagnose_fisher_vs_prior(Sigma_hat, feature_params)

    # --- console summary -----------------------------------------------------
    print("\n--- Diagnostic 1: Σ̂ eigenspectrum ---")
    print(f"  λ_max={d1['lambda_max']:.3e}  λ_min={d1['lambda_min']:.3e}  "
          f"λ_mean={d1['lambda_mean']:.3e}")
    print(f"  trace={d1['trace']:.3e}  cond={d1['condition_number']:.3e}  "
          f"anisotropy(λ_max/λ_mean)={d1['anisotropy_ratio']:.2f}")
    print("\n--- Diagnostic 2: ĝ norm & π̂ ---")
    print(f"  ‖ĝ‖ mean={d2['g_hat_norm']['mean']:.3e} median={d2['g_hat_norm']['median']:.3e}")
    print(f"  π̂ median={d2['pi']['median']:.3f}  frac_saturated(<{args.sat_threshold} or >"
          f"{1-args.sat_threshold:.2f})={d2['pi']['frac_saturated']:.3f}")
    print(f"  μ̂ in [0.1,0.9]: {d2['mu_hat']['frac_in_0.1_0.9']:.3f}")
    print("\n--- Diagnostic 3: upper-bound decomposition ---")
    print(f"  epi_mu actual mean = {epi_mu_mean:.3e}")
    print(f"  ‖ĝ‖²·λ_max (upper) = {d3['epi_upper_bound']:.3e}  "
          f"(actual/upper = {d3['actual_over_upper']:.3f})")
    print(f"  ‖ĝ‖²·λ_mean (iso)  = {d3['epi_isotropic_est']:.3e}  "
          f"(actual/iso = {d3['actual_over_isotropic']:.3f})")
    print("\n--- Diagnostic 4: learned σ₀ ---")
    print(f"  σ₀ min={d4['sigma_0']['min']:.4f} max={d4['sigma_0']['max']:.4f} "
          f"mean={d4['sigma_0']['mean']:.4f}")
    print("\n--- Diagnostic 5: Fisher data term vs prior ---")
    print(f"  Fisher λ_max={d5['fisher_data_lambda_max']:.3e}  "
          f"prior Σ₀⁻¹ diag max={d5['prior_inv_diag_max']:.3e}  "
          f"ratio={d5['fisher_over_prior']:.2f}")
    print("\n--- Bonus: prob-space vs logit-space epistemic as error rankers ---")
    print(f"  (no-skill PRR plateau = mean(U) = {bonus['_prr_no_skill_mean_U']:.4f})")
    for key in ("epi_mu_prob", "logit_epi_zbar", "logit_epi_tokmean"):
        b = bonus[key]
        print(f"  {key:>18}: spearman(|err|)={b['spearman_vs_abs_err']:+.3f}  "
              f"PRR_AUC={b['prr_auc_ratio']:.4f}  mean={b['mean']:.3e}")

    # --- persist -------------------------------------------------------------
    summary = {
        "setup": setup,
        "n_test_sentences": len(test_records),
        "k": int(theta_hat.shape[0]),
        "diagnostic_1_eigenspectrum": _strip_arrays(d1),
        "diagnostic_2_g_and_pi": _strip_arrays(d2),
        "diagnostic_3_upper_bound": d3,
        "diagnostic_4_prior": d4,
        "diagnostic_5_fisher_vs_prior": d5,
        "bonus_logit_vs_prob_signal": _strip_arrays(bonus),
    }
    json_path = results_dir / "epistemic_diagnostics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary -> {json_path}")

    if not args.no_plots:
        _plot_eigenspectrum(
            d1["eigvals_sorted"], results_dir / "epistemic_diag_eigenspectrum.png"
        )
        _plot_distributions(
            d2["_arrays"], results_dir / "epistemic_diag_distributions.png"
        )
        _plot_signal_vs_error(
            bonus["_arrays"], results_dir / "epistemic_diag_signal_vs_error.png"
        )
        print(f"Saved plots -> {results_dir}/epistemic_diag_*.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
