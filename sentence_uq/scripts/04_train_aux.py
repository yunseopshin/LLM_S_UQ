"""CLI for Phase 4-2: train the auxiliary Bayesian regression head.

Usage
-----
    python scripts/04_train_aux.py \\
        --setup 2 \\
        --config configs/default.yaml \\
        --trained-model results/setup_2/trained_model.pt \\
        --u-star data/processed/u_star_setup_2.json

The auxiliary model is a closed-form logit-transformed Bayesian Gaussian
regression (Part VIII / Phase 4-2). It learns to predict the
sentence-level target uncertainty ``U_j^*`` produced by an *expensive*
offline method (e.g. semantic entropy or LUQ) from the cheap
sentence-level aggregate feature ``ζ_j`` already available at
generation time.

Inputs
~~~~~~
* The Phase 4-1 trained-model ``.pt`` (for the feature-extractor ψ).
* A U_star file aligning each sentence with its target uncertainty.
  Supported formats (auto-detected by extension):

      .json
          ``[{"dataset": str,
              "source_id": str,
              "token_range": [int, int],
              "U_star": float}, ...]``

      .pt
          dict with the same fields, or a list of such dicts.

  Sentences whose ``(dataset, source_id, token_range)`` are absent
  from the U_star file are skipped.

Outputs
~~~~~~~
``results/setup_{N}/aux_model.pt`` containing::

    {
        "theta_N":          (k,)  fp32 CPU Tensor,
        "Sigma_N":          (k,k) fp32 CPU Tensor,
        "prior_mu":         (k,)  fp32 CPU Tensor,
        "prior_Sigma":      (k,k) fp32 CPU Tensor,
        "noise_sigma":      float,
        "feature_dim":      int,
        "aggregate_feature_dim": int (= 3 * k_token),
        "extra":            { setup, trained_model, u_star_file, n_train, n_test },
    }

``results/setup_{N}/aux_train_summary.json`` containing per-split
sentence counts and (optionally) the estimated noise variance.

The aggregate feature ``ζ_j`` is the Part VIII §8.4 concat of the
per-coordinate mean / std / last-token feature of ``{z_ℓ}_{ℓ ∈ s_j}``
— i.e. :func:`src.features.extractor.extract_sentence_aggregate_feature`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import SETUPS, split_save_filename  # noqa: E402
from src.features.extractor import (  # noqa: E402
    extract_sentence_aggregate_feature,
    extract_sentence_token_features,
)
from src.inference.predict import load_trained_model  # noqa: E402
from src.models.bayesian_aux import BayesianLogitRegression  # noqa: E402
from src.train.trainer import SentenceUQTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# Config / CLI plumbing (mirrors scripts/03_train.py conventions)
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(
    cfg: dict[str, Any], section: str, key: str, default: Any = None
) -> Any:
    sec = cfg.get(section) or {}
    val = sec.get(key)
    return val if val is not None else default


def _top_get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    val = cfg.get(key)
    return val if val is not None else default


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train the auxiliary Bayesian regression head (Phase 4-2): "
            "logit-transformed Gaussian regression from sentence-level "
            "aggregate features to an offline target uncertainty U^*."
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
        "--trained-model", type=str, required=True,
        help="Phase 4-1 trained_model.pt — supplies the feature extractor ψ.",
    )
    p.add_argument(
        "--u-star", type=str, required=True,
        help=(
            "Path to the offline-precomputed U^* file (.json or .pt). "
            "Entries are matched on (dataset, source_id, token_range)."
        ),
    )
    p.add_argument(
        "--prior-sigma", type=float, default=1.0,
        help="Isotropic prior std for θ. Default 1.0.",
    )
    p.add_argument(
        "--noise-sigma", type=float, default=0.1,
        help="Initial likelihood noise std σ. Default 0.1.",
    )
    p.add_argument(
        "--estimate-noise", action="store_true",
        help=(
            "Run residual-based σ² estimation after the first fit and "
            "refit with the updated noise (residual DOF: N - k)."
        ),
    )
    p.add_argument(
        "--device", type=str, default="cpu",
        help="Device for the feature extractor (default cpu — aux fit is closed-form).",
    )
    p.add_argument(
        "--results-dir", type=str, default=None,
        help="Override the output directory (default: results/setup_{N}).",
    )
    return p


def _resolve_device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def _resolve_setup(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
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


# ---------------------------------------------------------------------------
# U^* alignment
# ---------------------------------------------------------------------------


def _load_u_star_records(path: Path) -> List[Dict[str, Any]]:
    """Load the U^* file (.json or .pt) into a flat list of dicts."""
    if not path.exists():
        raise SystemExit(f"error: --u-star file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise SystemExit(
            f"error: --u-star must be .json or .pt (got {suffix})"
        )

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise SystemExit(
            "error: --u-star payload must be a list of records "
            "(or a single record dict)."
        )
    return data


def _build_u_star_index(
    records: Iterable[Dict[str, Any]]
) -> Dict[Tuple[str, str, int, int], float]:
    """Index U^* records by ``(dataset, source_id, start, end)``."""
    out: Dict[Tuple[str, str, int, int], float] = {}
    for rec in records:
        ds = rec.get("dataset")
        sid = rec.get("source_id")
        tr = rec.get("token_range")
        u = rec.get("U_star")
        if ds is None or sid is None or tr is None or u is None:
            continue
        if len(tr) != 2:
            continue
        try:
            start = int(tr[0])
            end = int(tr[1])
            u_val = float(u)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= u_val <= 1.0):
            # Skip out-of-range targets but warn — user should clip first.
            print(
                f"warning: U_star={u_val} outside [0, 1] for "
                f"({ds}, {sid}, {start}, {end}); skipping",
                file=sys.stderr,
            )
            continue
        out[(str(ds), str(sid), start, end)] = u_val
    return out


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _build_aggregate_features(
    sentence_records: Sequence[Dict[str, Any]],
    u_star_index: Dict[Tuple[str, str, int, int], float],
    feature_params,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Compute ``(Z, U_star)`` for every sentence with a matched target.

    Returns ``Z`` of shape ``(N, 3k_token)``, ``U_star`` of shape ``(N,)``,
    and the number of records dropped because they had no U^* match.
    """
    Z_rows: List[torch.Tensor] = []
    U_list: List[float] = []
    n_skipped = 0

    with torch.no_grad():
        for sent in sentence_records:
            key = (
                str(sent.get("dataset")),
                str(sent.get("source_id")),
                int(sent["token_range"][0]),
                int(sent["token_range"][1]),
            )
            u_star = u_star_index.get(key)
            if u_star is None:
                n_skipped += 1
                continue

            hidden_states = sent["hidden_states"].to(device)
            entropy = sent["entropy"].to(device)
            top1 = sent["top1"].to(device)

            z_tokens = extract_sentence_token_features(
                hidden_states=hidden_states,
                entropy=entropy,
                top1_prob=top1,
                token_range=(int(sent["token_range"][0]), int(sent["token_range"][1])),
                params=feature_params,
            )
            zeta = extract_sentence_aggregate_feature(z_tokens)
            Z_rows.append(zeta.detach().cpu().to(torch.float64))
            U_list.append(float(u_star))

    if not Z_rows:
        return (
            torch.zeros((0, 0), dtype=torch.float64),
            torch.zeros((0,), dtype=torch.float64),
            n_skipped,
        )

    Z = torch.stack(Z_rows, dim=0)
    U = torch.tensor(U_list, dtype=torch.float64)
    return Z, U, n_skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _evaluate(
    model: BayesianLogitRegression,
    Z: torch.Tensor,
    U: torch.Tensor,
) -> Dict[str, float]:
    if Z.shape[0] == 0:
        return {"n": 0, "MAE": float("nan"), "RMSE": float("nan")}
    out = model.predict(Z)
    p = out["p_factual"]
    mae = float((p - U).abs().mean().item())
    rmse = float(((p - U).pow(2).mean()).sqrt().item())
    return {"n": int(Z.shape[0]), "MAE": mae, "RMSE": rmse}


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)

    setup = _resolve_setup(args, cfg)
    device = torch.device(_resolve_device(args.device))

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

    results_dir = Path(
        args.results_dir
        or _top_get(cfg, "results_dir", f"results/setup_{setup}")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 4-2 auxiliary training — setup {setup} ===")
    print(f"Trained model: {args.trained_model}")
    print(f"U^* file:      {args.u_star}")
    print(f"Split file:    {split_file}")
    print(f"Results:       {results_dir}")
    print(
        f"Prior σ={args.prior_sigma}, noise σ_init={args.noise_sigma}, "
        f"estimate_noise={args.estimate_noise}"
    )

    # ---- Load feature extractor ψ from Phase 4-1 trained model ----
    loaded = load_trained_model(args.trained_model, map_location="cpu")
    feature_params = loaded["feature_params"].to(device)
    feature_params.eval()

    # ---- Load + index U^* ----
    u_star_records = _load_u_star_records(Path(args.u_star))
    u_star_index = _build_u_star_index(u_star_records)
    if not u_star_index:
        print(
            "error: no usable U^* records (need dataset/source_id/token_range/U_star).",
            file=sys.stderr,
        )
        return 2

    # ---- Materialise sentence records via the Phase 4-1 trainer helper ----
    from src.models.bayesian_main import BayesianSentenceUQ  # local import

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

    Z_train, U_train, skipped_train = _build_aggregate_features(
        data.get("train", []), u_star_index, feature_params, device
    )
    Z_val, U_val, skipped_val = _build_aggregate_features(
        data.get("val", []), u_star_index, feature_params, device
    )
    Z_test, U_test, skipped_test = _build_aggregate_features(
        data.get("test", []), u_star_index, feature_params, device
    )
    print(
        f"Sentences: train={Z_train.shape[0]} (skipped {skipped_train}), "
        f"val={Z_val.shape[0]} (skipped {skipped_val}), "
        f"test={Z_test.shape[0]} (skipped {skipped_test})"
    )

    if Z_train.shape[0] == 0:
        print(
            "error: no training sentences matched U^* — check that "
            "(dataset, source_id, token_range) keys agree with "
            "Phase 1-1 / 1-4 outputs.",
            file=sys.stderr,
        )
        return 2

    feature_dim = int(Z_train.shape[1])

    # ---- Fit ----
    aux = BayesianLogitRegression(
        feature_dim=feature_dim,
        prior_sigma=float(args.prior_sigma),
        noise_sigma=float(args.noise_sigma),
    )
    aux.fit(Z_train, U_train)

    sigma2_estimate: Optional[float] = None
    if args.estimate_noise:
        try:
            sigma2_estimate = aux.estimate_noise_variance(Z_train, U_train)
            new_sigma = float(sigma2_estimate) ** 0.5
            print(
                f"Estimated σ² = {sigma2_estimate:.6f} → refitting with σ = "
                f"{new_sigma:.6f}"
            )
            aux.set_noise_sigma(new_sigma)
            aux.fit(Z_train, U_train)
        except ValueError as exc:
            print(f"warning: σ² estimation skipped — {exc}", file=sys.stderr)

    train_metrics = _evaluate(aux, Z_train, U_train)
    val_metrics = _evaluate(aux, Z_val, U_val) if Z_val.shape[0] > 0 else None
    test_metrics = (
        _evaluate(aux, Z_test, U_test) if Z_test.shape[0] > 0 else None
    )

    print(f"Train: {train_metrics}")
    if val_metrics is not None:
        print(f"Val:   {val_metrics}")
    if test_metrics is not None:
        print(f"Test:  {test_metrics}")

    # ---- Persist ----
    aux_path = results_dir / "aux_model.pt"
    payload: Dict[str, Any] = {
        "theta_N": aux.theta_N.to(torch.float32).contiguous(),
        "Sigma_N": aux.Sigma_N.to(torch.float32).contiguous(),
        "prior_mu": aux.prior_mu.to(torch.float32).contiguous(),
        "prior_Sigma": aux.prior_Sigma.to(torch.float32).contiguous(),
        "noise_sigma": float(aux.noise_sigma),
        "feature_dim": feature_dim,
        "aggregate_feature_dim": feature_dim,
        "extra": {
            "setup": setup,
            "trained_model": str(args.trained_model),
            "u_star_file": str(args.u_star),
            "n_train": int(Z_train.shape[0]),
            "n_val": int(Z_val.shape[0]),
            "n_test": int(Z_test.shape[0]),
            "skipped": {
                "train": skipped_train,
                "val": skipped_val,
                "test": skipped_test,
            },
            "sigma2_estimate": sigma2_estimate,
        },
    }
    torch.save(payload, aux_path)

    summary = {
        "setup": setup,
        "feature_dim": feature_dim,
        "n_train": int(Z_train.shape[0]),
        "n_val": int(Z_val.shape[0]),
        "n_test": int(Z_test.shape[0]),
        "skipped": {
            "train": skipped_train,
            "val": skipped_val,
            "test": skipped_test,
        },
        "prior_sigma": float(args.prior_sigma),
        "noise_sigma": float(aux.noise_sigma),
        "sigma2_estimate": sigma2_estimate,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(results_dir / "aux_train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved auxiliary model to {aux_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
