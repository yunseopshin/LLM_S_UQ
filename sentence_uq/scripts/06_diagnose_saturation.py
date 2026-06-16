"""Phase 11-0 - Pooling diagnostic: saturation by token type (read-only).

Measures whether content tokens (NOUN/PROPN/NUM/ADJ) are less saturated than
function tokens on the *already-trained* probe, and how much content-mask
pooling would lift the per-sentence epistemic ``Epi_mu = g_j^T Sigma_hat g_j``
WITHOUT retraining. The verdict gates the later switchable-pooling phase.

Usage
-----
    python scripts/06_diagnose_saturation.py --setup 2 \
        --eval-split test \
        --sat-threshold 0.05 \
        --content-set PROPN NUM NOUN ADJ

Outputs (under ``results/{setup}/diagnostics/``):
    saturation_by_pos.csv     per-UPOS: count, sat_fraction, mean_gate
    epi_counterfactual.csv     per-sentence: L_j, n_content, mu_*, Epi_*, ratio
    saturation_diag.png        mean-gate-by-UPOS bars + Epi-ratio histogram
    saturation_diag.json       full statistics dict
    verdict.md                 the printed GO / NO-GO block

This script never modifies any model code path; it only reads the trained
``theta_hat`` / ``Sigma_hat`` / ``feature_params`` and the cached per-token
tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.saturation_diag import (  # noqa: E402
    diagnose_saturation_by_pos,
    format_verdict,
)
from src.data.dataset import split_save_filename  # noqa: E402
from src.data.generation import _safe_filename  # noqa: E402
from src.data.pos_tags import compute_token_pos  # noqa: E402
from src.data.sentence_split import load_spacy_model  # noqa: E402
from src.inference.predict import load_trained_model  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.train.trainer import SentenceUQTrainer  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Parse a YAML config; empty dict when missing/unset."""
    if not path or yaml is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg: Dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    """Read ``cfg[section][key]`` with a default."""
    return ((cfg.get(section) or {}).get(key)) if cfg else default


def _gen_rel_path(dataset: str, source_id: str) -> Optional[str]:
    """Reconstruct the generation ``.pt`` relative path for a source.

    Mirrors :func:`src.data.generation._output_path_for` /
    ``SentenceUQTrainer._prompt_keys``:

    - ``factscore_bio``: ``{safe(entity)}.pt`` (source_id == entity).
    - ``longfact``: ``{safe(topic)}/{idx:03d}.pt`` (source_id == "topic/idx").
    """
    if dataset == "factscore_bio":
        return f"{_safe_filename(source_id)}.pt"
    if dataset == "longfact":
        if "/" not in source_id:
            return None
        topic, idx_str = source_id.rsplit("/", 1)
        try:
            idx = int(idx_str)
        except (TypeError, ValueError):
            return None
        return f"{_safe_filename(topic)}/{idx:03d}.pt"
    return None


def _load_tokenizer(model_name: str) -> Any:
    """Load the generation tokenizer (fast) for ``model_name``."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def _build_pos_by_source(
    records: List[Dict[str, Any]],
    gen_dirs: Dict[str, str],
    nlp: Any,
    log: Any,
) -> Dict[Tuple[str, str], List[str]]:
    """Load each distinct source once, run ``compute_token_pos``, cache the result."""
    distinct: List[Tuple[str, str]] = []
    seen = set()
    for r in records:
        key = (r["dataset"], r["source_id"])
        if key not in seen:
            seen.add(key)
            distinct.append(key)

    tokenizer: Optional[Any] = None
    pos_by_source: Dict[Tuple[str, str], List[str]] = {}
    n_ok = 0
    n_fail = 0
    for ds, sid in distinct:
        rel = _gen_rel_path(ds, sid)
        if rel is None:
            log(f"  [skip] cannot map source ({ds}, {sid}) to a .pt path")
            n_fail += 1
            continue
        gen_path = Path(gen_dirs[ds]) / rel
        if not gen_path.exists():
            log(f"  [skip] missing generation file {gen_path}")
            n_fail += 1
            continue
        payload = torch.load(gen_path, map_location="cpu", weights_only=False)
        text = payload.get("text", "") or ""
        token_ids = payload.get("token_ids")
        if token_ids is None:
            log(f"  [skip] no token_ids in {gen_path}")
            n_fail += 1
            continue
        if tokenizer is None:
            model_name = (payload.get("model_config") or {}).get("name")
            if not model_name:
                raise SystemExit(
                    f"error: generation record {gen_path} has no "
                    "model_config.name; cannot load the tokenizer."
                )
            log(f"Loading tokenizer: {model_name}")
            tokenizer = _load_tokenizer(model_name)
        pos = compute_token_pos(text, token_ids, tokenizer, nlp)
        pos_by_source[(ds, sid)] = pos
        n_ok += 1

    log(f"POS computed for {n_ok}/{len(distinct)} sources ({n_fail} skipped)")
    return pos_by_source


def _write_saturation_by_pos_csv(path: Path, by_pos: Dict[str, Dict[str, Any]]) -> None:
    """Write per-UPOS count / sat_fraction / mean_gate, sorted by mean_gate desc."""
    rows = sorted(
        by_pos.items(),
        key=lambda kv: (
            kv[1]["mean_gate"] if np.isfinite(kv[1]["mean_gate"]) else -1.0
        ),
        reverse=True,
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pos", "count", "sat_fraction", "mean_gate", "is_content"])
        for pos, d in rows:
            w.writerow(
                [
                    pos,
                    d["count"],
                    f"{d['sat_fraction']:.6f}",
                    f"{d['mean_gate']:.6f}",
                    int(bool(d["is_content"])),
                ]
            )


def _fmt(x: Any) -> str:
    """Format a possibly-None float for CSV (empty cell for None)."""
    if x is None:
        return ""
    if isinstance(x, float) and not np.isfinite(x):
        return "nan"
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def _write_epi_csv(path: Path, per_sentence: List[Dict[str, Any]]) -> None:
    """Write the per-sentence Epi counterfactual table."""
    cols = [
        "dataset", "source_id", "L_j", "n_content",
        "mu_unif", "mu_mask", "Epi_unif", "Epi_mask", "ratio",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in per_sentence:
            w.writerow([_fmt(s.get(c)) for c in cols])


def _make_plot(path: Path, results: Dict[str, Any]) -> None:
    """Two-panel diagnostic: mean gate by UPOS + Epi-ratio histogram."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_pos = results["by_pos"]
    items = sorted(
        by_pos.items(),
        key=lambda kv: (
            kv[1]["mean_gate"] if np.isfinite(kv[1]["mean_gate"]) else -1.0
        ),
        reverse=True,
    )
    tags = [p for p, _ in items]
    gates = [d["mean_gate"] for _, d in items]
    colors = ["tab:green" if d["is_content"] else "tab:gray" for _, d in items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(len(tags)), gates, color=colors)
    ax1.set_xticks(range(len(tags)))
    ax1.set_xticklabels(tags, rotation=45, ha="right")
    ax1.set_ylabel("mean gate  pi*(1-pi)")
    ax1.set_title("Mean gate by UPOS (green=content, gray=function)")
    cf = results["by_class"]["function"]["mean_gate"]
    cc = results["by_class"]["content"]["mean_gate"]
    if np.isfinite(cf):
        ax1.axhline(cf, color="tab:gray", ls="--", lw=1, label=f"function {cf:.3g}")
    if np.isfinite(cc):
        ax1.axhline(cc, color="tab:green", ls="--", lw=1, label=f"content {cc:.3g}")
    ax1.legend(fontsize=8)

    ratios = np.asarray(
        [s["ratio"] for s in results["per_sentence"] if s["ratio"] is not None],
        dtype=np.float64,
    )
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size:
        lo, hi = ratios.min(), ratios.max()
        bins = np.logspace(np.log10(lo), np.log10(max(hi, lo * 1.01)), 30)
        ax2.hist(ratios, bins=bins, color="tab:blue", alpha=0.8)
        ax2.set_xscale("log")
        med = float(np.median(ratios))
        ax2.axvline(med, color="red", ls="--", lw=2, label=f"median {med:.2f}x")
        ax2.axvline(1.0, color="black", ls=":", lw=1, label="no change (1x)")
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "no non-degenerate sentences", ha="center")
    ax2.set_xlabel("Epi_mask / Epi_unif  (log scale)")
    ax2.set_ylabel("# sentences")
    ax2.set_title("Epistemic lift from content-mask pooling")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup", type=int, default=2)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--eval-split", type=str, default="test", choices=["test", "train", "val"],
        help="Which split to diagnose (default: test).",
    )
    p.add_argument("--sat-threshold", type=float, default=0.05)
    p.add_argument(
        "--content-set", type=str, nargs="+",
        default=["PROPN", "NUM", "NOUN", "ADJ"],
        help="UPOS tags counted as content.",
    )
    p.add_argument("--collapse-floor", type=float, default=8e-4)
    p.add_argument("--gate-ratio-go", type=float, default=2.0)
    p.add_argument("--epi-ratio-go", type=float, default=2.0)
    p.add_argument("--floor-multiple", type=float, default=2.0)
    p.add_argument(
        "--trained-model", type=str, default=None,
        help="default: results/setup_{N}/trained_model.pt",
    )
    p.add_argument(
        "--results-dir", type=str, default=None,
        help="default: results/setup_{N}",
    )
    args = p.parse_args(argv)

    cfg = _load_yaml(args.config)
    device = torch.device(args.device)
    results_dir = Path(args.results_dir or f"results/setup_{args.setup}")
    diag_dir = results_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    trained_path = Path(args.trained_model or results_dir / "trained_model.pt")

    print(f"=== Phase 11-0 saturation diagnostic (setup {args.setup}) ===")
    print(f"Trained model: {trained_path}")
    print(f"Eval split:    {args.eval_split}")
    print(f"Content set:   {args.content_set}")
    print(f"Output dir:    {diag_dir}")

    # --- load trained artifacts (read-only) --------------------------------
    loaded = load_trained_model(trained_path, map_location="cpu")
    theta_hat = loaded["theta_hat"].to(torch.float32)
    Sigma_hat = loaded["Sigma_hat"].to(torch.float32)
    feature_params = loaded["feature_params"].to(device)
    feature_params.eval()

    # --- rebuild the eval-split records (same path as 03_train.py) ----------
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
    records = list(data.get(args.eval_split) or [])
    print(f"Eval sentences ({args.eval_split}): {len(records)}")
    if not records:
        print("error: no eval sentences after joining splits.", file=sys.stderr)
        return 2

    # --- POS per source (load each generation once) ------------------------
    print("Computing POS tags on the BPE axis...")
    nlp = load_spacy_model("en")
    pos_by_source = _build_pos_by_source(records, gdirs, nlp, log=print)

    # --- diagnostic --------------------------------------------------------
    results = diagnose_saturation_by_pos(
        records,
        feature_params,
        theta_hat,
        Sigma_hat,
        pos_by_source,
        sat_threshold=args.sat_threshold,
        content_set=tuple(args.content_set),
        collapse_floor=args.collapse_floor,
        gate_ratio_go=args.gate_ratio_go,
        epi_ratio_go=args.epi_ratio_go,
        floor_multiple=args.floor_multiple,
    )

    # --- artifacts ---------------------------------------------------------
    _write_saturation_by_pos_csv(diag_dir / "saturation_by_pos.csv", results["by_pos"])
    _write_epi_csv(diag_dir / "epi_counterfactual.csv", results["per_sentence"])
    _make_plot(diag_dir / "saturation_diag.png", results)
    with open(diag_dir / "saturation_diag.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)

    verdict = format_verdict(results, setup=args.setup, eval_split=args.eval_split)
    with open(diag_dir / "verdict.md", "w", encoding="utf-8") as f:
        f.write(verdict + "\n")

    print()
    print(verdict)
    print()
    print(f"Saved artifacts -> {diag_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
