"""CLI for Phase 6-2: full two-tiered evaluation + ablations.

Usage
-----
    python scripts/04_evaluate.py --setup 2 --config configs/default.yaml

What it does
~~~~~~~~~~~~
1. Loads the Phase 4-1 trained model (``trained_model.pt``) and, when
   available, the Phase 4-2 auxiliary head (``aux_model.pt``) plus the
   Phase 5-1 baseline cache (``baselines.json``).
2. Runs predictive inference on the test split:

   - ``Ours (Bayesian)``      — :class:`Predictor` with probit shrinkage
   - ``Ours (Point)``         — same MAP, posterior covariance zeroed out
   - ``Ours (Aux)``           — :class:`BayesianLogitRegression` predictions
   - all baselines cached by Phase 5-1

3. Computes Phase 6-1 metrics:

   - **Ratio-level** (primary): MAE, RMSE, Pearson r, binomial NLL, ECE,
     PRR-AUC, Brier.
   - **Strict factuality** (secondary): AUROC, AUPRC, Brier, ECE,
     PRR-AUC, plus bootstrapped 95 % CIs on AUROC and ECE following
     Han et al.

4. Runs the spec ablations and saves them as separate CSV tables:

   - Bayesian vs Point estimate
   - Binomial vs Bernoulli scoring of our predictions
   - Linear (delta-method) vs Monte-Carlo epistemic
   - Han et al. (re-encoded) vs Han et al. (adapted, generation-time)

5. Persists every artefact under ``results/setup_{N}/``::

       final_metrics_ratio.csv        # method × ratio-level metric table
       final_metrics_strict.csv       # method × strict-factuality table
       ablation_bayesian_vs_point.csv
       ablation_binomial_vs_bernoulli.csv
       ablation_mc_vs_linear.csv
       alpha_distribution.csv         # learnt softmax(α) weights
       reliability_diagrams/{method}_{tier}.png
       prr_curves.png                 # all methods on one axis
       mc_vs_linear.png               # delta vs MC epistemic scatter
       token_heatmaps/{i}.png         # per-sentence attribution
       alpha_distribution.png         # learnt α as a bar chart

Notes
-----
* The metric routines from :mod:`src.evaluation.metrics` already handle
  the ``m_j = 0`` filter (CLAUDE.md rule 8). We pre-filter once and feed
  the kept rows to every downstream computation.
* The Bayesian variant always uses probit shrinkage so that the strict
  prediction reflects posterior uncertainty (research_document_v8 §5.2).
* All numerics run in fp32; only the cached hidden states stay in fp16.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import SETUPS, split_save_filename  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compare_mc_vs_linear_epistemic,
    compute_bootstrapped_ci,
    compute_calibration_metrics,
    compute_prr,
    compute_ratio_level_metrics,
    compute_strict_factuality_metrics,
    plot_reliability_diagram,
)
from src.features.extractor import (  # noqa: E402
    extract_sentence_aggregate_feature,
    extract_sentence_token_features,
)
from src.inference.predict import Predictor, load_trained_model  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.train.trainer import SentenceUQTrainer  # noqa: E402
from src.utils.validation import validate_binomial_counts  # noqa: E402


# ---------------------------------------------------------------------------
# Config / CLI plumbing (mirrors scripts/03_train.py conventions)
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a dict.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
        Empty dict when the YAML file is empty / contains nothing.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(
    cfg: Dict[str, Any], section: str, key: str, default: Any = None
) -> Any:
    """Read ``cfg[section][key]`` with a default fallback."""
    sec = cfg.get(section) or {}
    val = sec.get(key)
    return val if val is not None else default


def _top_get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Read a top-level YAML key with a default fallback."""
    val = cfg.get(key)
    return val if val is not None else default


def _resolve_setup(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
    """Pick the experimental setup from CLI args / YAML / fail loudly."""
    setup = args.setup
    if setup is None:
        setup = (cfg.get("dataset") or {}).get("setup")
    if setup is None:
        raise SystemExit(
            "error: --setup is required (or set dataset.setup in config)"
        )
    setup = int(setup)
    if setup not in SETUPS:
        raise SystemExit(f"error: unknown setup {setup}; valid: {list(SETUPS)}")
    return setup


def _resolve_device(name: str) -> str:
    """``cuda`` falls back to ``cpu`` when no GPU is visible."""
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for ``scripts/04_evaluate.py``."""
    p = argparse.ArgumentParser(
        description=(
            "Phase 6-2 — two-tiered evaluation (ratio + strict) plus the "
            "ablations listed in prompts/phase_6_2_evaluate.md."
        )
    )
    p.add_argument(
        "--setup", type=int, choices=list(SETUPS), required=False,
        help="Experimental setup (1, 2, or 3). Falls back to config value.",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="YAML config (e.g. configs/default.yaml).",
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device for the feature extractor (default cpu).",
    )
    p.add_argument(
        "--trained-model", type=str, default=None,
        help="Override path to the Phase 4-1 trained_model.pt.",
    )
    p.add_argument(
        "--aux-model", type=str, default=None,
        help="Optional Phase 4-2 aux_model.pt — when present the script "
             "reports 'Ours (Aux)' rows alongside the main predictions.",
    )
    p.add_argument(
        "--baselines-file", type=str, default=None,
        help="Optional Phase 5-1 baselines.json. When omitted the script "
             "looks for results_dir/baselines.json.",
    )
    p.add_argument(
        "--results-dir", type=str, default=None,
        help="Override the output directory (default: results/setup_{N}).",
    )
    p.add_argument(
        "--mc-samples", type=int, default=100,
        help="θ samples for the MC-vs-linear epistemic ablation.",
    )
    p.add_argument(
        "--bootstrap-iters", type=int, default=1000,
        help="Bootstrap resamples for the strict-factuality 95 %% CI.",
    )
    p.add_argument(
        "--num-heatmaps", type=int, default=5,
        help="How many test sentences to visualise as token heatmaps.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed used for bootstrap CIs and MC sampling.",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Skip every matplotlib figure (CSV outputs still produced).",
    )
    p.add_argument(
        "--compare-all", action="store_true",
        help="Evaluate setups 1, 2, 3 in turn and combine their metrics "
             "into results/cross_setup_comparison.csv.",
    )
    return p


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _prepare_test_records(
    cfg: Dict[str, Any],
    setup: int,
    device: torch.device,
    feature_params: Any,
) -> List[Dict[str, Any]]:
    """Materialise the test-split sentence records via the Phase 4-1 trainer helper.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config (may be empty).
    setup : int
        Experimental setup.
    device : torch.device
        Device for the throw-away :class:`BayesianSentenceUQ` instance that
        the trainer constructor demands.
    feature_params : SentenceUQParams
        Trained feature-extractor parameters from Phase 4-1.

    Returns
    -------
    list of per-sentence dicts (same schema as Phase 4-1 trainer).
    """
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


def _filter_positive(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only sentences with ``m_j > 0`` (CLAUDE.md rule 8)."""
    return [r for r in records if int(r.get("m_j", 0) or 0) > 0]


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
# Ours: Bayesian / Point predictions
# ---------------------------------------------------------------------------


def _ours_predictions(
    predictor: Predictor,
    z_tokens_list: Sequence[torch.Tensor],
    m_vec: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Per-sentence predictions for the ``Ours`` family.

    Calls :meth:`Predictor.predict_sentence` once per sentence and stacks
    the scalar outputs into arrays.

    Parameters
    ----------
    predictor : Predictor
        Single-sentence predictor (Bayesian or zero-Σ Point variant).
    z_tokens_list : sequence of ``(L_j, k)`` tensors.
    m_vec : sequence of int (length ``N``).

    Returns
    -------
    dict with NumPy arrays of length ``N``:
        ``mu_hat``         : μ̂_j
        ``mu_probit``      : probit-shrunk μ̃_j
        ``epi_mu``         : latent epistemic (delta method)
        ``p_strict``       : μ̂_j^{m_j} (Bayesian variant reuses probit-shrunk μ̃)
        ``total_U``        : ratio-level total uncertainty
        ``aleatoric_U``    : ratio-level aleatoric component
    """
    n = len(z_tokens_list)
    mu_hat = np.empty(n, dtype=np.float64)
    mu_probit = np.empty(n, dtype=np.float64)
    epi_mu = np.empty(n, dtype=np.float64)
    p_strict = np.empty(n, dtype=np.float64)
    total_U = np.empty(n, dtype=np.float64)
    aleatoric_U = np.empty(n, dtype=np.float64)
    for i, (z, m) in enumerate(zip(z_tokens_list, m_vec)):
        out = predictor.predict_sentence(z, m_j=int(m))
        mu_hat[i] = float(out["mu_hat"])
        mu_probit[i] = float(out["p_factual_probit"])
        epi_mu[i] = float(out["epi_mu"])
        p_strict[i] = float(out["p_strict_factual"] or 0.0)
        total_U[i] = float(out["total_U"] or 0.0)
        aleatoric_U[i] = float(out["aleatoric_U"] or 0.0)
    return {
        "mu_hat": mu_hat,
        "mu_probit": mu_probit,
        "epi_mu": epi_mu,
        "p_strict": p_strict,
        "total_U": total_U,
        "aleatoric_U": aleatoric_U,
    }


# ---------------------------------------------------------------------------
# Ours (Aux): Bayesian regression head
# ---------------------------------------------------------------------------


def _aux_predictions(
    aux_payload: Dict[str, Any],
    z_tokens_list: Sequence[torch.Tensor],
    m_vec: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Run the Phase 4-2 auxiliary regression on every test sentence.

    The aux model consumes the *sentence-level aggregate feature*
    ``ζ_j = concat([mean, std, last])`` of dimension ``3k``. ``μ̂_j`` is
    the closed-form ``σ(θ_Nᵀ ζ_j)``; epistemic uncertainty is the logit-
    space variance ``z_*ᵀ Σ_N z_*``. ``p_strict`` plugs ``μ̂_j`` into
    ``μ̂_j^{m_j}`` (same convention as the main predictor).
    """
    theta_N = aux_payload["theta_N"].to(torch.float64)
    Sigma_N = aux_payload["Sigma_N"].to(torch.float64)
    noise_sigma = float(aux_payload.get("noise_sigma", 0.1))

    n = len(z_tokens_list)
    mu_hat = np.empty(n, dtype=np.float64)
    epi_mu = np.empty(n, dtype=np.float64)
    p_strict = np.empty(n, dtype=np.float64)
    for i, (z, m) in enumerate(zip(z_tokens_list, m_vec)):
        zeta = extract_sentence_aggregate_feature(z).to(torch.float64)
        if zeta.shape[0] != theta_N.shape[0]:
            raise ValueError(
                f"aggregate feature dim {zeta.shape[0]} != aux feature_dim "
                f"{theta_N.shape[0]}; aux model and trained model are out "
                "of sync."
            )
        logit_mean = float((zeta @ theta_N).item())
        epi = float(((zeta @ Sigma_N) @ zeta).clamp_min(0.0).item())
        p = float(torch.sigmoid(torch.tensor(logit_mean)).item())
        mu_hat[i] = p
        epi_mu[i] = epi
        # logit-space aleatoric is σ²; here we only report the latent-mu
        # variance for the rejection curve.
        p_strict[i] = float(min(max(p, 0.0), 1.0) ** int(m))
    return {
        "mu_hat": mu_hat,
        "mu_probit": mu_hat.copy(),  # aux has no probit shrinkage
        "epi_mu": epi_mu,
        "p_strict": p_strict,
        "total_U": epi_mu + noise_sigma * noise_sigma,
        "aleatoric_U": np.full(n, noise_sigma * noise_sigma, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Metric tables
# ---------------------------------------------------------------------------


def _ratio_row(
    name: str,
    U: np.ndarray,
    mu: np.ndarray,
    m: np.ndarray,
    epi: Optional[np.ndarray],
    wall_ms: Optional[float],
    with_binom_nll: bool,
) -> Dict[str, Any]:
    """Build a single row of the ratio-level CSV.

    Uses :func:`compute_ratio_level_metrics`,
    :func:`compute_calibration_metrics`, and :func:`compute_prr`.
    Epistemic ``epi`` is used as the PRR rejection signal when present
    (higher uncertainty rejected first); otherwise we fall back to
    ``|μ - 0.5|⁻¹`` so baselines without an explicit signal still
    produce a PRR-AUC.
    """
    ratio = compute_ratio_level_metrics(
        U, mu, m_j=m if with_binom_nll else None
    )
    calib = compute_calibration_metrics(U, mu, n_bins=10)
    rejection = epi if epi is not None else -np.abs(mu - 0.5)
    prr = compute_prr(U, rejection, num_thresholds=100)
    row: Dict[str, Any] = {
        "method": name,
        "MAE": ratio["MAE"],
        "RMSE": ratio["RMSE"],
        "Pearson_r": ratio["Pearson_r"],
        "binomial_NLL": ratio.get("binomial_NLL", float("nan")),
        "Brier": calib["Brier"],
        "ECE": calib["ECE"],
        "PRR_AUC": prr["prr_auc"],
        "time_ms": wall_ms if wall_ms is not None else float("nan"),
        "n": int(U.size),
    }
    return row


def _strict_row(
    name: str,
    A: np.ndarray,
    p: np.ndarray,
    uncertainty: np.ndarray,
    wall_ms: Optional[float],
    bootstrap_iters: int,
    seed: int,
) -> Dict[str, Any]:
    """Build a single row of the strict-factuality CSV (with 95 % CIs)."""
    strict = compute_strict_factuality_metrics(A, p, uncertainty)
    prr = compute_prr(A, uncertainty, num_thresholds=100)
    auroc_ci = compute_bootstrapped_ci(
        A, p, _auroc_metric, n_bootstrap=bootstrap_iters, seed=seed
    )
    ece_ci = compute_bootstrapped_ci(
        A, p, _ece_metric, n_bootstrap=bootstrap_iters, seed=seed
    )
    return {
        "method": name,
        "AUROC": strict["AUROC"],
        "AUROC_CI_lo": auroc_ci["lower"],
        "AUROC_CI_hi": auroc_ci["upper"],
        "AUPRC": strict["AUPRC"],
        "Brier": strict["Brier"],
        "ECE": strict["ECE"],
        "ECE_CI_lo": ece_ci["lower"],
        "ECE_CI_hi": ece_ci["upper"],
        "PRR_AUC": prr["prr_auc"],
        "time_ms": wall_ms if wall_ms is not None else float("nan"),
        "n": int(A.size),
        "frac_strict": float(A.mean()),
    }


def _auroc_metric(y: np.ndarray, p: np.ndarray) -> float:
    """Bootstrap-safe AUROC: returns NaN when only one class is sampled."""
    if np.unique(y).size < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def _ece_metric(y: np.ndarray, p: np.ndarray) -> float:
    """ECE metric routine plumbed into :func:`compute_bootstrapped_ci`."""
    return float(compute_calibration_metrics(y, p, n_bins=10)["ECE"])


# ---------------------------------------------------------------------------
# Baselines (load Phase 5-1 cache)
# ---------------------------------------------------------------------------


def _load_baselines(path: Path) -> Dict[str, Any]:
    """Read ``baselines.json`` (Phase 5-1 output). Empty dict when absent."""
    if not path.exists():
        print(f"warning: baselines file not found at {path}", file=sys.stderr)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: failed to read baselines: {exc}", file=sys.stderr)
        return {}


def _baseline_rows(
    baselines: Dict[str, Any],
    K_pos: np.ndarray,
    m_pos: np.ndarray,
    bootstrap_iters: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert each cached baseline result into ratio + strict rows.

    Baselines whose ``mu_hat`` length does not match the positive-``m_j``
    test pool (e.g. semantic entropy that dropped prompts) are skipped
    with a warning so the table only reports comparable methods.
    """
    ratio_rows: List[Dict[str, Any]] = []
    strict_rows: List[Dict[str, Any]] = []
    U_pos = K_pos / np.maximum(m_pos, 1.0)
    # Phase 7-3 fix 4: strict factuality is K == m (all atoms supported).
    A_pos = (K_pos == m_pos).astype(np.float64)

    pool = baselines.get("baselines", {}) if "baselines" in baselines else {}
    for name, payload in pool.items():
        if not isinstance(payload, dict) or payload.get("skipped"):
            print(
                f"info: baseline '{name}' skipped — {payload.get('reason')}",
                file=sys.stderr,
            )
            continue
        mu_list = payload.get("mu_hat")
        if mu_list is None:
            print(
                f"info: baseline '{name}' has no mu_hat field; skipping",
                file=sys.stderr,
            )
            continue
        mu = np.asarray(mu_list, dtype=np.float64)
        if mu.shape[0] != U_pos.shape[0]:
            print(
                f"warning: baseline '{name}' length {mu.shape[0]} != "
                f"test pool {U_pos.shape[0]}; skipping (likely a "
                "differently-filtered sample pool)",
                file=sys.stderr,
            )
            continue
        wall_ms = (
            float(payload.get("wall_clock_seconds", float("nan"))) * 1000.0
        )
        uncertainty = -mu  # rank low-μ̂ as most uncertain
        ratio_rows.append(
            _ratio_row(name, U_pos, mu, m_pos, None, wall_ms, with_binom_nll=False)
        )
        strict_rows.append(
            _strict_row(name, A_pos, mu, uncertainty, wall_ms,
                        bootstrap_iters, seed)
        )
    return ratio_rows, strict_rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_prr_curves(
    pool: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
    target_label: str = "U",
) -> None:
    """Overlay every method's PRR curve on a single axis."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for name, payload in pool.items():
        ax.plot(
            payload["rejection_rates"],
            payload["remaining_quality"],
            marker="",
            linewidth=1.5,
            label=name,
        )
    ax.set_xlabel("Rejection rate")
    ax.set_ylabel(f"Mean {target_label} over kept samples")
    ax.set_title("Prediction–rejection curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_mc_vs_linear(
    linear: np.ndarray,
    mc: np.ndarray,
    save_path: Path,
) -> None:
    """Scatter the delta-method epistemic vs the MC sample variance."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(linear, mc, s=10, alpha=0.6, color="C0")
    lim = float(np.nanmax([linear.max(), mc.max()])) if linear.size else 1.0
    lim = max(lim, 1e-6)
    ax.plot([0.0, lim], [0.0, lim], "--", linewidth=1.0, color="gray")
    ax.set_xlabel("Linear (delta method) epistemic")
    ax.set_ylabel("Monte-Carlo epistemic")
    ax.set_title("Latent epistemic: linear vs MC")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_alpha_distribution(
    alpha_weights: np.ndarray,
    selected_layers: Sequence[int],
    save_path: Path,
    han_target_layer: int = 14,
) -> None:
    """Bar chart of the learnt softmax(α) layer-mixing weights."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    xs = np.arange(len(alpha_weights))
    ax.bar(xs, alpha_weights, color="C0", alpha=0.75, edgecolor="C0")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(l) for l in selected_layers])
    ax.set_xlabel("Absolute layer index")
    ax.set_ylabel("softmax(α)_l")
    ax.set_title("Learnt layer-mixing weights")
    if han_target_layer in selected_layers:
        i = int(list(selected_layers).index(han_target_layer))
        ax.axvline(i, color="C3", linestyle="--", linewidth=1.0,
                   label=f"Han et al. (layer {han_target_layer})")
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_token_heatmap(
    record: Dict[str, Any],
    token_pi: torch.Tensor,
    token_attr: torch.Tensor,
    token_local_epi: torch.Tensor,
    mu_hat: float,
    p_strict: float,
    save_path: Path,
) -> None:
    """Bar chart of token-level π̂_ℓ + attribution + LocalEpi_ℓ for one sentence."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    pi = token_pi.detach().cpu().numpy()
    attr = token_attr.detach().cpu().numpy()
    local_epi = token_local_epi.detach().cpu().numpy()
    L = pi.shape[0]
    xs = np.arange(L)

    fig, axes = plt.subplots(3, 1, figsize=(max(6.0, L * 0.3), 6.0), sharex=True)
    axes[0].bar(xs, pi, color="C0", alpha=0.75)
    axes[0].axhline(mu_hat, color="C1", linestyle="--", linewidth=1.0,
                    label=f"μ̂={mu_hat:.3f}")
    axes[0].set_ylabel("π̂_ℓ")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].bar(xs, attr, color="C2", alpha=0.75)
    axes[1].set_ylabel("Attribution Attr_ℓ")

    axes[2].bar(xs, local_epi, color="C3", alpha=0.75)
    axes[2].set_ylabel("LocalEpi_ℓ")
    axes[2].set_xlabel("Token offset within sentence")

    src = record.get("source_id", "")
    tr = record.get("token_range", ("?", "?"))
    fig.suptitle(
        f"{src}  tokens={tr}  p_strict={p_strict:.3f}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------


def _ablation_binom_vs_bernoulli(
    mu_hat: np.ndarray,
    p_strict: np.ndarray,
    K: np.ndarray,
    m: np.ndarray,
) -> pd.DataFrame:
    """Score our predictions under both observation models.

    * **Binomial** scoring uses the per-atom likelihood
      ``K log μ̂ + (m-K) log(1-μ̂)`` with the ratio target ``U=K/m``.
    * **Bernoulli** scoring collapses each sentence to ``A_j = 1{K_j=m_j}``
      and treats ``μ̂_j`` itself as the strict probability — i.e. the
      ``m_j=1`` ablation requested in the spec.

    Returns
    -------
    pandas.DataFrame
        Two rows (``Binomial``, ``Bernoulli``) with columns
        ``binomial_NLL``, ``ratio_MAE``, ``strict_ECE``, ``strict_AUROC``.
    """
    U = K / np.maximum(m, 1.0)
    # Phase 7-3 fix 4: strict factuality is K == m.
    A = (K == m).astype(np.float64)

    binom = compute_ratio_level_metrics(U, mu_hat, m_j=m)
    binom_strict = compute_calibration_metrics(A, p_strict, n_bins=10)
    binom_auroc = _auroc_metric(A, p_strict)

    # Bernoulli substitution: m_j=1 throughout → U becomes A, p_strict
    # collapses to μ̂ (μ̂^1).
    bern = compute_ratio_level_metrics(A, mu_hat, m_j=np.ones_like(m))
    bern_strict = compute_calibration_metrics(A, mu_hat, n_bins=10)
    bern_auroc = _auroc_metric(A, mu_hat)

    rows = [
        {
            "variant": "Binomial",
            "binomial_NLL": binom.get("binomial_NLL", float("nan")),
            "ratio_MAE": binom["MAE"],
            "strict_ECE": binom_strict["ECE"],
            "strict_AUROC": binom_auroc,
        },
        {
            "variant": "Bernoulli (m=1)",
            "binomial_NLL": bern.get("binomial_NLL", float("nan")),
            "ratio_MAE": bern["MAE"],
            "strict_ECE": bern_strict["ECE"],
            "strict_AUROC": bern_auroc,
        },
    ]
    return pd.DataFrame(rows)


def _ablation_bayesian_vs_point(
    ours_bayes: Dict[str, np.ndarray],
    ours_point: Dict[str, np.ndarray],
    U: np.ndarray,
    A: np.ndarray,
    m: np.ndarray,
) -> pd.DataFrame:
    """Side-by-side comparison of probit-shrunk vs raw MAP predictions.

    Reports AUROC / Brier / ECE on the strict target and the latent
    epistemic mean.
    """
    rows = []
    for name, pack in (
        ("Ours (Bayesian)", ours_bayes),
        ("Ours (Point)", ours_point),
    ):
        strict = compute_strict_factuality_metrics(A, pack["p_strict"], pack["epi_mu"])
        rows.append(
            {
                "variant": name,
                "AUROC": strict["AUROC"],
                "Brier": strict["Brier"],
                "ECE": strict["ECE"],
                "epi_mu_mean": float(pack["epi_mu"].mean()),
                "binomial_NLL": compute_ratio_level_metrics(
                    U, pack["mu_hat"], m_j=m
                )["binomial_NLL"],
            }
        )
    return pd.DataFrame(rows)


def _ablation_mc_vs_linear(
    predictor: Predictor,
    z_tokens_list: Sequence[torch.Tensor],
    num_samples: int,
    seed: int,
    save_path_plot: Optional[Path],
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compare delta-method and MC latent epistemic on every test sentence."""
    if not z_tokens_list:
        return pd.DataFrame(
            [{"linear_mean": float("nan"), "mc_mean": float("nan"),
              "Pearson_r": float("nan"), "MAE": float("nan")}]
        ), None, None
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    comp = compare_mc_vs_linear_epistemic(
        predictor,
        list(z_tokens_list),
        num_mc_samples=int(num_samples),
        generator=gen,
    )
    linear = np.asarray(comp["linear_epi"], dtype=np.float64)
    mc = np.asarray(comp["mc_epi"], dtype=np.float64)
    table = pd.DataFrame(
        [
            {
                "linear_mean": float(linear.mean()),
                "mc_mean": float(mc.mean()),
                "Pearson_r": float(comp["Pearson_r"]),
                "MAE": float(comp["MAE"]),
                "num_samples": int(num_samples),
            }
        ]
    )
    if save_path_plot is not None:
        _plot_mc_vs_linear(linear, mc, save_path_plot)
    return table, linear, mc


# ---------------------------------------------------------------------------
# Reliability diagrams
# ---------------------------------------------------------------------------


def _save_reliability_diagrams(
    pool: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    K_pos: np.ndarray,
    m_pos: np.ndarray,
) -> None:
    """Per-method reliability diagrams for both tiers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    U = K_pos / np.maximum(m_pos, 1.0)
    # Phase 7-3 fix 4: strict factuality is K == m.
    A = (K_pos == m_pos).astype(np.float64)
    for name, pack in pool.items():
        mu = pack["mu_hat"]
        p = pack["p_strict"]
        plot_reliability_diagram(
            U, mu, n_bins=10,
            save_path=out_dir / f"{_safe_filename(name)}_ratio.png",
            title=f"{name} — ratio",
        )
        plot_reliability_diagram(
            A, p, n_bins=10,
            save_path=out_dir / f"{_safe_filename(name)}_strict.png",
            title=f"{name} — strict",
        )


def _safe_filename(name: str) -> str:
    """Sanitise a method label so it can be used as a filename component."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "unnamed"


# ---------------------------------------------------------------------------
# Main per-setup evaluation
# ---------------------------------------------------------------------------


def _evaluate_single_setup(
    args: argparse.Namespace, cfg: Dict[str, Any]
) -> int:
    """Run the full Phase 6-2 pipeline for a single experimental setup."""
    setup = _resolve_setup(args, cfg)
    device = torch.device(_resolve_device(args.device))

    results_dir = Path(
        args.results_dir
        or _top_get(cfg, "results_dir", f"results/setup_{setup}")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    trained_path = Path(
        args.trained_model or results_dir / "trained_model.pt"
    )
    baselines_path = Path(
        args.baselines_file or results_dir / "baselines.json"
    )
    aux_path = (
        Path(args.aux_model) if args.aux_model else results_dir / "aux_model.pt"
    )

    print(f"=== Phase 6-2 evaluation — setup {setup} ===")
    print(f"Trained model: {trained_path}")
    print(f"Baselines:     {baselines_path}")
    print(f"Aux model:     {aux_path} (optional)")
    print(f"Results:       {results_dir}")

    if not trained_path.exists():
        print(
            f"error: trained model not found at {trained_path}. "
            "Run scripts/03_train.py first.",
            file=sys.stderr,
        )
        return 2

    loaded = load_trained_model(trained_path, map_location="cpu")
    theta_hat = loaded["theta_hat"]
    Sigma_hat = loaded["Sigma_hat"]
    feature_params = loaded["feature_params"].to(device)
    feature_params.eval()
    extra = loaded.get("extra", {}) or {}
    selected_layers = (
        (extra.get("model_dims") or {}).get("selected_layers")
        or extra.get("selected_layers")
        or _detect_selected_layers(cfg)
        or list(range(feature_params.num_layers))
    )

    bayesian_predictor = Predictor(
        theta_hat=theta_hat,
        Sigma_hat=Sigma_hat,
        feature_params=feature_params,
        use_probit_shrinkage=True,
    )
    point_predictor = Predictor(
        theta_hat=theta_hat,
        Sigma_hat=torch.zeros_like(Sigma_hat),
        feature_params=feature_params,
        use_probit_shrinkage=False,
    )

    # --- Test data ----------------------------------------------------------
    test_records_all = _prepare_test_records(cfg, setup, device, feature_params)
    test_records = _filter_positive(test_records_all)
    if not test_records:
        print(
            "error: no positive-m_j test sentences after the m_j>0 filter; "
            "check Phase 1-1/1-3/1-4 outputs.",
            file=sys.stderr,
        )
        return 2
    print(
        f"Test sentences: kept {len(test_records)}/{len(test_records_all)} "
        "with m_j > 0."
    )

    K_pos = np.array(
        [int(r["K_j"]) for r in test_records], dtype=np.float64
    )
    m_pos = np.array(
        [int(r["m_j"]) for r in test_records], dtype=np.float64
    )
    validate_binomial_counts(K_pos, m_pos, context="04_evaluate.main")
    U_pos = K_pos / np.maximum(m_pos, 1.0)
    # Phase 7-3 fix 4: strict factuality is K == m.
    A_pos = (K_pos == m_pos).astype(np.float64)

    # --- Feature extraction (shared across all "Ours" variants) ------------
    z_tokens_list: List[torch.Tensor] = [
        _extract_z_tokens(r, feature_params, device) for r in test_records
    ]

    # --- Ours predictions ---------------------------------------------------
    t0 = time.perf_counter()
    ours_bayes = _ours_predictions(bayesian_predictor, z_tokens_list, m_pos.astype(int))
    bayes_ms = (time.perf_counter() - t0) * 1000.0 / max(len(test_records), 1)

    t0 = time.perf_counter()
    ours_point = _ours_predictions(point_predictor, z_tokens_list, m_pos.astype(int))
    point_ms = (time.perf_counter() - t0) * 1000.0 / max(len(test_records), 1)

    # --- Optional Aux predictions ------------------------------------------
    aux_pack: Optional[Dict[str, np.ndarray]] = None
    aux_ms: float = float("nan")
    if aux_path.exists():
        try:
            aux_payload = torch.load(aux_path, map_location="cpu", weights_only=False)
            t0 = time.perf_counter()
            aux_pack = _aux_predictions(
                aux_payload, z_tokens_list, m_pos.astype(int)
            )
            aux_ms = (time.perf_counter() - t0) * 1000.0 / max(
                len(test_records), 1
            )
        except (RuntimeError, KeyError, ValueError) as exc:
            print(
                f"warning: failed to load aux model — {exc}", file=sys.stderr
            )
            aux_pack = None

    # --- Ratio + strict tables ---------------------------------------------
    ratio_rows: List[Dict[str, Any]] = []
    strict_rows: List[Dict[str, Any]] = []

    ratio_rows.append(
        _ratio_row("Ours (Bayesian)", U_pos, ours_bayes["mu_hat"], m_pos,
                   ours_bayes["epi_mu"], bayes_ms, with_binom_nll=True)
    )
    ratio_rows.append(
        _ratio_row("Ours (Point)", U_pos, ours_point["mu_hat"], m_pos,
                   ours_point["epi_mu"], point_ms, with_binom_nll=True)
    )
    strict_rows.append(
        _strict_row("Ours (Bayesian)", A_pos, ours_bayes["p_strict"],
                    ours_bayes["epi_mu"], bayes_ms,
                    args.bootstrap_iters, args.seed)
    )
    strict_rows.append(
        _strict_row("Ours (Point)", A_pos, ours_point["p_strict"],
                    -ours_point["mu_hat"], point_ms,
                    args.bootstrap_iters, args.seed)
    )

    if aux_pack is not None:
        ratio_rows.append(
            _ratio_row("Ours (Aux)", U_pos, aux_pack["mu_hat"], m_pos,
                       aux_pack["epi_mu"], aux_ms, with_binom_nll=True)
        )
        strict_rows.append(
            _strict_row("Ours (Aux)", A_pos, aux_pack["p_strict"],
                        aux_pack["epi_mu"], aux_ms,
                        args.bootstrap_iters, args.seed)
        )

    # --- Baselines ----------------------------------------------------------
    baselines = _load_baselines(baselines_path)
    base_ratio, base_strict = _baseline_rows(
        baselines, K_pos, m_pos, args.bootstrap_iters, args.seed
    )
    ratio_rows.extend(base_ratio)
    strict_rows.extend(base_strict)

    ratio_df = pd.DataFrame(ratio_rows)
    strict_df = pd.DataFrame(strict_rows)
    ratio_df.to_csv(results_dir / "final_metrics_ratio.csv", index=False)
    strict_df.to_csv(results_dir / "final_metrics_strict.csv", index=False)
    print("\n=== Ratio-Level Metrics (Primary) ===")
    print(ratio_df.to_string(index=False))
    print("\n=== Strict Factuality Metrics (Secondary, 95% CI) ===")
    print(strict_df.to_string(index=False))

    # --- Ablations ----------------------------------------------------------
    ab_bp = _ablation_bayesian_vs_point(
        ours_bayes, ours_point, U_pos, A_pos, m_pos
    )
    ab_bp.to_csv(results_dir / "ablation_bayesian_vs_point.csv", index=False)
    print("\n=== Bayesian vs Point Ablation ===")
    print(ab_bp.to_string(index=False))

    ab_bb = _ablation_binom_vs_bernoulli(
        ours_bayes["mu_hat"], ours_bayes["p_strict"], K_pos, m_pos
    )
    ab_bb.to_csv(results_dir / "ablation_binomial_vs_bernoulli.csv", index=False)
    print("\n=== Binomial vs Bernoulli Ablation ===")
    print(ab_bb.to_string(index=False))

    ab_mc, linear_epi, mc_epi = _ablation_mc_vs_linear(
        bayesian_predictor,
        z_tokens_list,
        args.mc_samples,
        args.seed,
        save_path_plot=None if args.no_plots else results_dir / "mc_vs_linear.png",
    )
    ab_mc.to_csv(results_dir / "ablation_mc_vs_linear.csv", index=False)
    print("\n=== MC vs Linear Epistemic ===")
    print(ab_mc.to_string(index=False))

    # --- α distribution -----------------------------------------------------
    with torch.no_grad():
        alpha_weights = (
            torch.softmax(feature_params.alpha.to(torch.float32), dim=0)
            .detach()
            .cpu()
            .numpy()
        )
    alpha_df = pd.DataFrame(
        {
            "selected_layer": [int(l) for l in selected_layers[: alpha_weights.shape[0]]],
            "softmax_alpha": alpha_weights,
        }
    )
    alpha_df.to_csv(results_dir / "alpha_distribution.csv", index=False)
    print("\n=== Layer Weight Analysis ===")
    print(alpha_df.to_string(index=False))
    if not args.no_plots:
        _plot_alpha_distribution(
            alpha_weights,
            selected_layers[: alpha_weights.shape[0]],
            save_path=results_dir / "alpha_distribution.png",
        )

    # --- Plots --------------------------------------------------------------
    if not args.no_plots:
        ratio_pool: Dict[str, Dict[str, np.ndarray]] = {
            "Ours (Bayesian)": {
                "mu_hat": ours_bayes["mu_hat"],
                "p_strict": ours_bayes["p_strict"],
            },
            "Ours (Point)": {
                "mu_hat": ours_point["mu_hat"],
                "p_strict": ours_point["p_strict"],
            },
        }
        if aux_pack is not None:
            ratio_pool["Ours (Aux)"] = {
                "mu_hat": aux_pack["mu_hat"],
                "p_strict": aux_pack["p_strict"],
            }
        pool = baselines.get("baselines", {}) if "baselines" in baselines else {}
        for name, payload in pool.items():
            if not isinstance(payload, dict) or payload.get("skipped"):
                continue
            mu_list = payload.get("mu_hat")
            if mu_list is None:
                continue
            mu_arr = np.asarray(mu_list, dtype=np.float64)
            if mu_arr.shape[0] != U_pos.shape[0]:
                continue
            ratio_pool[name] = {
                "mu_hat": mu_arr,
                "p_strict": mu_arr,
            }

        _save_reliability_diagrams(
            ratio_pool, results_dir / "reliability_diagrams", K_pos, m_pos
        )

        prr_pool: Dict[str, Dict[str, np.ndarray]] = {}
        for name, pack in ratio_pool.items():
            mu = pack["mu_hat"]
            rejection = (
                ours_bayes["epi_mu"]
                if name == "Ours (Bayesian)"
                else (-mu)
            )
            prr_pool[name] = compute_prr(U_pos, rejection, num_thresholds=100)
        _plot_prr_curves(prr_pool, results_dir / "prr_curves.png", target_label="U")

        # Token heatmaps for a handful of high-epistemic sentences.
        order = np.argsort(-ours_bayes["epi_mu"])
        heatmap_dir = results_dir / "token_heatmaps"
        for rank, i in enumerate(order[: int(args.num_heatmaps)]):
            rec = test_records[int(i)]
            out = bayesian_predictor.predict_sentence(
                z_tokens_list[int(i)], m_j=int(m_pos[int(i)])
            )
            _plot_token_heatmap(
                rec,
                out["token_pi"],
                out["token_attr"],
                out["token_local_epi"],
                mu_hat=float(out["mu_hat"]),
                p_strict=float(out["p_strict_factual"] or 0.0),
                save_path=heatmap_dir / f"{rank:02d}_{_safe_filename(str(rec.get('source_id', '')))}.png",
            )

    summary = {
        "setup": setup,
        "n_test_pool": int(len(test_records_all)),
        "n_test_positive": int(len(test_records)),
        "frac_strict_factual": float(A_pos.mean()),
        "selected_layers": [int(l) for l in selected_layers],
        "alpha_argmax_layer": (
            int(selected_layers[int(np.argmax(alpha_weights))])
            if alpha_weights.size and selected_layers
            else None
        ),
    }
    with open(results_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved evaluation artefacts to {results_dir}")
    return 0


def _detect_selected_layers(cfg: Dict[str, Any]) -> Optional[List[int]]:
    """Best-effort read of ``selected_layers`` from a Phase 1-1 ``.pt`` file."""
    generations_dirs = {
        "factscore_bio": _cfg_get(
            cfg, "generation", "factscore_bio_dir",
            "data/generations/factscore_bio",
        ),
        "longfact": _cfg_get(
            cfg, "generation", "longfact_dir", "data/generations/longfact"
        ),
    }
    for gen_dir in generations_dirs.values():
        p = Path(gen_dir)
        if not p.exists():
            continue
        files = sorted(
            (q for q in p.rglob("*.pt") if q.is_file()),
            key=lambda q: q.relative_to(p).as_posix(),
        )
        for f in files:
            try:
                payload = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:
                continue
            mc = payload.get("model_config") or {}
            sel = mc.get("selected_layers") or payload.get("selected_layers")
            if sel:
                return [int(x) for x in sel]
    return None


# ---------------------------------------------------------------------------
# Cross-setup comparison (--compare-all)
# ---------------------------------------------------------------------------


def _compare_all_setups(
    args: argparse.Namespace, cfg: Dict[str, Any]
) -> int:
    """Combine per-setup ratio + strict tables into a single comparison CSV."""
    base_results = Path(
        args.results_dir or _top_get(cfg, "results_dir", "results")
    ).parent if args.results_dir else Path("results")
    rows: List[Dict[str, Any]] = []
    for setup in sorted(SETUPS):
        per_setup = base_results / f"setup_{setup}"
        ratio_csv = per_setup / "final_metrics_ratio.csv"
        strict_csv = per_setup / "final_metrics_strict.csv"
        if not (ratio_csv.exists() and strict_csv.exists()):
            print(
                f"info: skipping setup {setup} — run scripts/04_evaluate.py "
                f"--setup {setup} first.",
                file=sys.stderr,
            )
            continue
        ratio_df = pd.read_csv(ratio_csv).assign(tier="ratio", setup=setup)
        strict_df = pd.read_csv(strict_csv).assign(tier="strict", setup=setup)
        rows.append(ratio_df)
        rows.append(strict_df)

    if not rows:
        print("error: nothing to compare; produce per-setup CSVs first.",
              file=sys.stderr)
        return 2

    combined = pd.concat(rows, ignore_index=True, sort=False)
    out_path = base_results / "cross_setup_comparison.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved cross-setup comparison to {out_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for ``python scripts/04_evaluate.py``."""
    args = _build_parser().parse_args(argv)
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)
    if args.compare_all:
        return _compare_all_setups(args, cfg)
    return _evaluate_single_setup(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
