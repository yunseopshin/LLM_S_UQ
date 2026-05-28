"""CLI for Phase 5-1: run every baseline and persist results.

Usage
-----
    python scripts/05_baselines.py --setup 2 --config configs/default.yaml

What it does
~~~~~~~~~~~~
1. Loads the Phase 1-0 split and joins it with the Phase 1-1 / 1-3 /
   1-4 outputs via :meth:`SentenceUQTrainer.prepare_data` (so the
   sentence universe matches the Phase 4-1 trainer).
2. Runs each baseline on the *test* split:

   - ``token_entropy`` (cheap, ``μ̂_j`` derived from cached entropy)
   - ``logistic_regression`` (sklearn, sentence features cached)
   - ``factuality_probe_adapted`` (Han et al., generation-time hidden
     states; trained on the train split, evaluated on test)
   - ``factuality_probe_original`` (only when ``--run-original`` is
     passed — it needs the LLM weights and is expensive)
   - ``semantic_entropy`` / ``luq`` (only when ``--samples-dir`` is
     supplied — these require the precomputed sample cache)

3. For every baseline records wall-clock time, the per-sentence
   ``μ̂_j`` (or uncertainty score), and the ratio-level + strict
   metrics that ``src.train.trainer.SentenceUQTrainer._ratio_metrics``
   already exposes. Strict-factuality AUROC / AUPRC are computed
   inline (sklearn).
4. Saves everything to ``results/setup_{N}/baselines.json``.

Caching
~~~~~~~
Semantic-entropy / LUQ samples should be precomputed offline and
passed in via ``--samples-dir``. The expected layout is::

    samples_dir/
        {dataset}/{source_id}.json     # JSON list of {sample_text} strings
                                       # (one file per prompt)

The runner falls back gracefully when a prompt has no cached samples
— that prompt is dropped from the baseline's evaluation pool.

The Phase 5-1 spec also requires a *singleton* NLI scorer; we
instantiate :class:`NLIScorer` once and share it across the SE / LUQ
calls (or skip both if ``transformers`` cannot be imported).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.baselines.factuality_probe import (  # noqa: E402
    DEFAULT_TARGET_LAYER,
    FactualityProbeBaseline,
)
from src.baselines.logistic_regression import (  # noqa: E402
    LogisticRegressionBaseline,
    collate_sentence_features,
)
from src.baselines.token_entropy import (  # noqa: E402
    compute_token_entropy_baseline,
)
from src.data.dataset import SETUPS, split_save_filename  # noqa: E402
from src.train.trainer import SentenceUQTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# Config plumbing (mirrors scripts/03_train.py)
# ---------------------------------------------------------------------------


_UNSAFE_FNAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(s: str, max_len: int = 200) -> str:
    cleaned = _UNSAFE_FNAME_RE.sub("_", (s or "").strip()).strip("._")
    if not cleaned:
        cleaned = "unnamed"
    return cleaned[:max_len]


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(
    cfg: Dict[str, Any], section: str, key: str, default: Any = None
) -> Any:
    sec = cfg.get(section) or {}
    val = sec.get(key)
    return val if val is not None else default


def _top_get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    val = cfg.get(key)
    return val if val is not None else default


def _resolve_setup(args: argparse.Namespace, cfg: Dict[str, Any]) -> int:
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
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Phase 5-1 — run every baseline on the test split and dump "
            "results to results/setup_{N}/baselines.json."
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
        help="cuda | cpu for the optional NLI / LM models.",
    )
    p.add_argument(
        "--samples-dir", type=str, default=None,
        help=(
            "Directory of precomputed semantic-entropy / LUQ samples. "
            "Layout: {samples_dir}/{dataset}/{source_id}.json with a "
            "JSON list of strings."
        ),
    )
    p.add_argument(
        "--run-original", action="store_true",
        help=(
            "Also run the original Han et al. variant (re-encodes claims "
            "through the LLM). Requires --model-name."
        ),
    )
    p.add_argument(
        "--model-name", type=str, default=None,
        help=(
            "HuggingFace LM id for the original factuality-probe variant. "
            "Defaults to model.name in the config."
        ),
    )
    p.add_argument(
        "--target-layer", type=int, default=DEFAULT_TARGET_LAYER,
        help=f"Target absolute layer for the factuality probes (default {DEFAULT_TARGET_LAYER}).",
    )
    p.add_argument(
        "--C-lr", type=float, default=1.0,
        help="Inverse regularisation strength for the LogisticRegression baseline.",
    )
    p.add_argument(
        "--C-l1", type=float, default=1.0,
        help="Inverse L1 regularisation strength for the Han et al. probe.",
    )
    p.add_argument(
        "--skip", nargs="*", default=[],
        help=(
            "Baseline names to skip (e.g. --skip semantic_entropy luq). "
            "Useful when a precomputed sample cache is not yet available."
        ),
    )
    p.add_argument(
        "--results-dir", type=str, default=None,
        help="Override the output directory (default: results/setup_{N}).",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ratio_metrics(
    mu_hats: torch.Tensor,
    K: torch.Tensor,
    m: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Re-use Phase 4-1 metric definitions on a baseline's μ̂ vector."""
    return SentenceUQTrainer._ratio_metrics(mu_hats, K, m, eps)


def _strict_metrics(probs: torch.Tensor, A: torch.Tensor) -> Dict[str, float]:
    """AUROC / AUPRC / Brier on the binary ``A_j = 1{K_j = m_j}`` target.

    Returns ``nan`` metrics when only one class is present (sklearn raises
    otherwise).
    """
    if probs.numel() == 0:
        return {
            "n": 0,
            "AUROC": float("nan"),
            "AUPRC": float("nan"),
            "Brier": float("nan"),
        }
    y = A.detach().cpu().to(torch.int64).numpy()
    p = probs.detach().cpu().to(torch.float32).numpy()

    out: Dict[str, float] = {"n": int(y.shape[0])}
    try:
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            roc_auc_score,
        )
    except ImportError:
        out.update({"AUROC": float("nan"), "AUPRC": float("nan"), "Brier": float("nan")})
        return out

    if len(set(y.tolist())) < 2:
        out.update({"AUROC": float("nan"), "AUPRC": float("nan")})
    else:
        out["AUROC"] = float(roc_auc_score(y, p))
        out["AUPRC"] = float(average_precision_score(y, p))
    out["Brier"] = float(brier_score_loss(y, p))
    return out


def _strict_labels(K: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``A_j = 1{K_j = m_j}`` restricted to ``m_j > 0`` rows."""
    mask = m > 0
    return (K[mask] == m[mask]).to(torch.long)


def _filter_positive(
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep only sentences with ``m_j > 0`` (CLAUDE.md rule 8)."""
    return [r for r in records if int(r.get("m_j", 0) or 0) > 0]


def _normalise_entropy_to_unit(scores: torch.Tensor) -> torch.Tensor:
    """Map an unbounded uncertainty score to a μ̂ ∈ [0, 1] estimate.

    Token-entropy / semantic-entropy / LUQ return *uncertainty* scores,
    but the ratio-level metric expects a *factuality probability*
    ``μ̂_j``. We invert and min-max normalise within the evaluation
    pool so the score becomes a monotonic factuality estimate. This is
    intentionally crude — Phase 5-1 cares about comparing the *ranking
    + calibration* of baselines, not their absolute scale.
    """
    if scores.numel() == 0:
        return scores.clone()
    s = scores.detach().to(torch.float32)
    lo = float(torch.nan_to_num(s, nan=0.0).min().item())
    hi = float(torch.nan_to_num(s, nan=0.0).max().item())
    if hi <= lo:
        return torch.full_like(s, 0.5)
    norm = (s - lo) / (hi - lo)
    return (1.0 - norm).clamp(0.0, 1.0)


def _entity_or_source(rec: Dict[str, Any]) -> Tuple[str, str]:
    """Return ``(dataset, source_id)`` for a per-sentence record."""
    return str(rec.get("dataset")), str(rec.get("source_id"))


def _samples_path(samples_dir: Path, dataset: str, source_id: str) -> Path:
    return samples_dir / dataset / f"{_safe_filename(source_id)}.json"


def _load_prompt_samples(
    samples_dir: Path, dataset: str, source_id: str
) -> Optional[List[str]]:
    path = _samples_path(samples_dir, dataset, source_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    return [str(x) for x in data if isinstance(x, str)]


def _group_records_by_prompt(
    records: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[int]]:
    """Map ``(dataset, source_id)`` → list of indices into ``records``."""
    out: Dict[Tuple[str, str], List[int]] = {}
    for i, r in enumerate(records):
        key = _entity_or_source(r)
        out.setdefault(key, []).append(i)
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _run_token_entropy(
    test_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Per-sentence mean entropy → inverted/normalised to μ̂."""
    t0 = time.perf_counter()
    test_pos = _filter_positive(test_records)
    K = torch.tensor([int(r["K_j"]) for r in test_pos], dtype=torch.long)
    m = torch.tensor([int(r["m_j"]) for r in test_pos], dtype=torch.long)
    scores = torch.tensor(
        [
            compute_token_entropy_baseline(
                r["entropy"], (int(r["token_range"][0]), int(r["token_range"][1]))
            )
            for r in test_pos
        ],
        dtype=torch.float32,
    )
    mu_hat = _normalise_entropy_to_unit(scores)
    elapsed = time.perf_counter() - t0
    return {
        "name": "token_entropy",
        "n_test": int(scores.shape[0]),
        "wall_clock_seconds": float(elapsed),
        "ratio_metrics": _ratio_metrics(mu_hat, K, m),
        "strict_metrics": _strict_metrics(mu_hat, _strict_labels(K, m)),
        "scores_raw": scores.tolist(),
        "mu_hat": mu_hat.tolist(),
    }


def _run_logistic_regression(
    train_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    C: float,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    train_pos = _filter_positive(train_records)
    test_pos = _filter_positive(test_records)
    train_pack = collate_sentence_features(train_pos)
    test_pack = collate_sentence_features(test_pos)
    if train_pack["Z"].shape[0] == 0 or test_pack["Z"].shape[0] == 0:
        return {
            "name": "logistic_regression",
            "skipped": True,
            "reason": "no usable rows after m_j > 0 filter",
        }

    clf = LogisticRegressionBaseline(target="strict", C=C)
    clf.fit(train_pack["Z"], train_pack["K"], train_pack["m"])
    mu_hat = clf.predict_proba(test_pack["Z"])
    elapsed = time.perf_counter() - t0
    return {
        "name": "logistic_regression",
        "n_train": int(train_pack["Z"].shape[0]),
        "n_test": int(test_pack["Z"].shape[0]),
        "wall_clock_seconds": float(elapsed),
        "ratio_metrics": _ratio_metrics(mu_hat, test_pack["K"], test_pack["m"]),
        "strict_metrics": _strict_metrics(
            mu_hat, _strict_labels(test_pack["K"], test_pack["m"])
        ),
        "mu_hat": mu_hat.tolist(),
        "config": {"C": float(C)},
    }


def _detect_selected_layers(
    generations_dirs: Dict[str, str | Path],
) -> Optional[List[int]]:
    """Read ``selected_layers`` from the first generation ``.pt`` found."""
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


def _run_factuality_probe_adapted(
    train_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    selected_layers: Sequence[int],
    target_layer: int,
    C: float,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    probe = FactualityProbeBaseline(
        variant="adapted", target_layer=target_layer, C=C
    )
    train_pack = probe.build_adapted_dataset(train_records, selected_layers)
    test_pack = probe.build_adapted_dataset(test_records, selected_layers)
    if train_pack["H"].shape[0] == 0 or test_pack["H"].shape[0] == 0:
        return {
            "name": "factuality_probe_adapted",
            "skipped": True,
            "reason": "no usable rows after m_j > 0 filter",
        }

    probe.fit(train_pack["H"], train_pack["A"])
    p_hat = probe.predict_proba(test_pack["H"])

    # Same probability serves both ratio and strict scoring (Phase 5-1
    # spec — "For ratio-level comparison, their predicted probability
    # maps to μ̂_j. For strict factuality, their probability is
    # compared against A_j via standard binary metrics.").
    # Reconstruct K, m for the test pack:
    K_list: List[int] = []
    m_list: List[int] = []
    for rec in test_records:
        m_j = int(rec.get("m_j", 0) or 0)
        if m_j == 0:
            continue
        K_list.append(int(rec.get("K_j", 0) or 0))
        m_list.append(m_j)
    K = torch.tensor(K_list, dtype=torch.long)
    m = torch.tensor(m_list, dtype=torch.long)

    elapsed = time.perf_counter() - t0
    return {
        "name": "factuality_probe_adapted",
        "n_train": int(train_pack["H"].shape[0]),
        "n_test": int(test_pack["H"].shape[0]),
        "wall_clock_seconds": float(elapsed),
        "ratio_metrics": _ratio_metrics(p_hat, K, m),
        "strict_metrics": _strict_metrics(p_hat, test_pack["A"]),
        "mu_hat": p_hat.tolist(),
        "config": {
            "target_layer": int(target_layer),
            "C": float(C),
            "selected_layers": list(map(int, selected_layers)),
        },
    }


def _run_factuality_probe_original(
    train_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    model_name: str,
    target_layer: int,
    C: float,
    device: str,
) -> Dict[str, Any]:
    from src.data.generation import load_model

    t0 = time.perf_counter()
    model, tokenizer, _ = load_model(model_name, device=device)
    probe = FactualityProbeBaseline(
        variant="original", target_layer=target_layer, C=C
    )
    train_pack = probe.build_original_dataset(train_records, model, tokenizer)
    test_pack = probe.build_original_dataset(test_records, model, tokenizer)
    if train_pack["H"].shape[0] == 0 or test_pack["H"].shape[0] == 0:
        return {
            "name": "factuality_probe_original",
            "skipped": True,
            "reason": "no claims to re-encode (Phase 1-4 outputs missing?)",
        }

    probe.fit(train_pack["H"], train_pack["y"])
    claim_probs = probe.predict_proba(test_pack["H"])
    sent_probs = probe.aggregate_sentence_scores(
        claim_probs, test_pack["sentence_to_claims"], agg="mean"
    )

    K_list: List[int] = []
    m_list: List[int] = []
    A_list: List[int] = []
    for rec in test_pack["sentence_records"]:
        m_j = int(rec.get("m_j", 0) or 0)
        K_j = int(rec.get("K_j", 0) or 0)
        K_list.append(K_j)
        m_list.append(m_j)
        A_list.append(1 if K_j == m_j and m_j > 0 else 0)
    K = torch.tensor(K_list, dtype=torch.long)
    m = torch.tensor(m_list, dtype=torch.long)
    A = torch.tensor(A_list, dtype=torch.long)

    elapsed = time.perf_counter() - t0
    return {
        "name": "factuality_probe_original",
        "n_train_claims": int(train_pack["H"].shape[0]),
        "n_test_claims": int(test_pack["H"].shape[0]),
        "n_test_sentences": int(sent_probs.shape[0]),
        "wall_clock_seconds": float(elapsed),
        "ratio_metrics": _ratio_metrics(sent_probs, K, m),
        "strict_metrics": _strict_metrics(sent_probs, A),
        "mu_hat": sent_probs.tolist(),
        "claim_probs": claim_probs.tolist(),
        "config": {
            "model_name": model_name,
            "target_layer": int(target_layer),
            "C": float(C),
            "aggregation": "mean",
        },
    }


def _run_semantic_entropy(
    test_records: Sequence[Dict[str, Any]],
    samples_dir: Path,
    nli_scorer: Any,
) -> Dict[str, Any]:
    from src.baselines.semantic_entropy import compute_semantic_entropy_from_samples

    t0 = time.perf_counter()
    test_pos = _filter_positive(test_records)
    groups = _group_records_by_prompt(test_pos)
    prompt_scores: Dict[Tuple[str, str], float] = {}
    n_missing = 0
    for key in groups:
        samples = _load_prompt_samples(samples_dir, key[0], key[1])
        if not samples:
            n_missing += 1
            continue
        prompt_scores[key] = compute_semantic_entropy_from_samples(
            samples, nli_scorer
        )
    if not prompt_scores:
        return {
            "name": "semantic_entropy",
            "skipped": True,
            "reason": "no prompts had cached samples",
        }

    # Broadcast the per-prompt score to every sentence in the prompt.
    keep_idx: List[int] = []
    scores_list: List[float] = []
    K_list: List[int] = []
    m_list: List[int] = []
    for key, indices in groups.items():
        if key not in prompt_scores:
            continue
        s = float(prompt_scores[key])
        for i in indices:
            r = test_pos[i]
            keep_idx.append(i)
            scores_list.append(s)
            K_list.append(int(r["K_j"]))
            m_list.append(int(r["m_j"]))
    scores = torch.tensor(scores_list, dtype=torch.float32)
    K = torch.tensor(K_list, dtype=torch.long)
    m = torch.tensor(m_list, dtype=torch.long)
    mu_hat = _normalise_entropy_to_unit(scores)
    elapsed = time.perf_counter() - t0
    return {
        "name": "semantic_entropy",
        "n_test_sentences": int(scores.shape[0]),
        "n_prompts_scored": len(prompt_scores),
        "n_prompts_missing_samples": int(n_missing),
        "wall_clock_seconds": float(elapsed),
        "ratio_metrics": _ratio_metrics(mu_hat, K, m),
        "strict_metrics": _strict_metrics(mu_hat, _strict_labels(K, m)),
        "scores_raw": scores.tolist(),
        "mu_hat": mu_hat.tolist(),
    }


def _run_luq(
    test_records: Sequence[Dict[str, Any]],
    samples_dir: Path,
    nli_scorer: Any,
) -> Dict[str, Any]:
    from src.baselines.luq import compute_luq_for_sentences

    t0 = time.perf_counter()
    test_pos = _filter_positive(test_records)
    groups = _group_records_by_prompt(test_pos)
    keep_idx: List[int] = []
    scores_list: List[float] = []
    K_list: List[int] = []
    m_list: List[int] = []
    n_missing = 0
    for key, indices in groups.items():
        samples = _load_prompt_samples(samples_dir, key[0], key[1])
        if not samples:
            n_missing += 1
            continue
        sentences = [
            str((test_pos[i].get("text") or "")).strip() for i in indices
        ]
        # The trainer's prepare_data doesn't populate "text"; pass an
        # empty placeholder when it's missing — LUQ then returns nan
        # which the metrics drop.
        per_sentence = compute_luq_for_sentences(sentences, samples, nli_scorer)
        for i, s in zip(indices, per_sentence):
            if s != s:  # NaN
                continue
            r = test_pos[i]
            keep_idx.append(i)
            scores_list.append(float(s))
            K_list.append(int(r["K_j"]))
            m_list.append(int(r["m_j"]))
    if not scores_list:
        return {
            "name": "luq",
            "skipped": True,
            "reason": (
                "no sentences scored — check that samples-dir is populated "
                "and that test_records carry a 'text' field per sentence"
            ),
            "n_prompts_missing_samples": int(n_missing),
        }
    scores = torch.tensor(scores_list, dtype=torch.float32)
    K = torch.tensor(K_list, dtype=torch.long)
    m = torch.tensor(m_list, dtype=torch.long)
    # LUQ already returns U ∈ [0, 1]; μ̂ = 1 - U.
    mu_hat = (1.0 - scores).clamp(0.0, 1.0)
    elapsed = time.perf_counter() - t0
    return {
        "name": "luq",
        "n_test_sentences": int(scores.shape[0]),
        "n_prompts_missing_samples": int(n_missing),
        "wall_clock_seconds": float(elapsed),
        "ratio_metrics": _ratio_metrics(mu_hat, K, m),
        "strict_metrics": _strict_metrics(mu_hat, _strict_labels(K, m)),
        "scores_raw": scores.tolist(),
        "mu_hat": mu_hat.tolist(),
    }


# ---------------------------------------------------------------------------
# Sentence text join (Phase 1-4 annotation supplies it)
# ---------------------------------------------------------------------------


def _attach_sentence_texts(
    records: Sequence[Dict[str, Any]],
    processed_dirs: Dict[str, str | Path],
) -> None:
    """Mutate ``records`` in place, adding ``text`` from the Phase 1-4 dump.

    The Phase 4-1 trainer does not populate ``text`` because the
    Bayesian model never needs it; LUQ does. We re-walk the
    annotation files and match on ``(dataset, source_id, token_range)``.
    Missing matches leave ``text`` unset (LUQ then drops the row).
    """
    by_dataset: Dict[str, Dict[Tuple[str, int, int], str]] = {}
    for dataset, processed_dir in processed_dirs.items():
        annotations = SentenceUQTrainer._load_annotations(  # type: ignore[attr-defined]
            dataset, Path(processed_dir)
        )
        idx: Dict[Tuple[str, int, int], str] = {}
        for source_id, record in annotations.items():
            for sent in record.get("sentences", []) or []:
                tr = sent.get("token_range")
                if not tr or len(tr) != 2:
                    continue
                idx[(source_id, int(tr[0]), int(tr[1]))] = str(
                    sent.get("text", "") or ""
                )
        by_dataset[dataset] = idx

    for rec in records:
        ds = str(rec.get("dataset"))
        key = (
            str(rec.get("source_id")),
            int(rec["token_range"][0]),
            int(rec["token_range"][1]),
        )
        rec["text"] = by_dataset.get(ds, {}).get(key, "")


def _attach_sentence_claims(
    records: Sequence[Dict[str, Any]],
    processed_dirs: Dict[str, str | Path],
) -> None:
    """Mutate ``records`` in place, adding the per-claim list from Phase 1-4.

    ``SentenceUQTrainer.prepare_data`` carries only ``K_j`` / ``m_j`` (all
    the Bayesian model needs), but the *original* Han et al. probe
    re-encodes every atomic claim, so it requires ``claims`` — each entry
    exposing ``text`` and ``label`` (see
    :meth:`FactualityProbeBaseline.build_original_dataset`). We re-walk the
    annotation files and match on ``(source_id, token_range)`` exactly as
    :func:`_attach_sentence_texts`. Missing matches leave ``claims`` empty
    (the probe then drops the row).
    """
    by_dataset: Dict[str, Dict[Tuple[str, int, int], List[Dict[str, Any]]]] = {}
    for dataset, processed_dir in processed_dirs.items():
        annotations = SentenceUQTrainer._load_annotations(  # type: ignore[attr-defined]
            dataset, Path(processed_dir)
        )
        idx: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
        for source_id, record in annotations.items():
            for sent in record.get("sentences", []) or []:
                tr = sent.get("token_range")
                if not tr or len(tr) != 2:
                    continue
                idx[(source_id, int(tr[0]), int(tr[1]))] = list(
                    sent.get("claims", []) or []
                )
        by_dataset[dataset] = idx

    for rec in records:
        ds = str(rec.get("dataset"))
        key = (
            str(rec.get("source_id")),
            int(rec["token_range"][0]),
            int(rec["token_range"][1]),
        )
        rec["claims"] = by_dataset.get(ds, {}).get(key, [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)

    setup = _resolve_setup(args, cfg)
    device = _resolve_device(args.device)

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

    print(f"=== Phase 5-1 baselines — setup {setup} ===")
    print(f"Split file: {split_file}")
    print(f"Results:    {results_dir}")

    # ---- Materialise sentence records via the Phase 4-1 trainer helper ----
    from src.features.extractor import SentenceUQParams
    from src.models.bayesian_main import BayesianSentenceUQ

    # Trainer needs a model + feature params just so prepare_data works.
    # The Bayesian model itself is never trained here.
    dummy_params = SentenceUQParams(hidden_dim=8, num_layers=2, projection_dim=4)
    trainer = SentenceUQTrainer(
        model=BayesianSentenceUQ(feature_params=dummy_params),
        device=torch.device("cpu"),
    )
    data = trainer.prepare_data(
        split_file=split_file,
        generations_dirs=generations_dirs,
        cache_dirs=cache_dirs,
        processed_dirs=processed_dirs,
    )
    train_records: List[Dict[str, Any]] = list(data.get("train") or [])
    test_records: List[Dict[str, Any]] = list(data.get("test") or [])

    print(
        f"Sentences: train={len(train_records)}, test={len(test_records)} "
        f"(includes m_j == 0 rows, dropped per-baseline)"
    )
    if not test_records:
        print(
            "error: no test sentences — check that Phases 1-1/1-3/1-4 have "
            "been run for this setup.",
            file=sys.stderr,
        )
        return 2

    # Add ``text`` for LUQ — fast no-op when annotations are missing.
    _attach_sentence_texts(test_records, processed_dirs)

    selected_layers = _detect_selected_layers(generations_dirs) or []
    if not selected_layers:
        print(
            "warning: could not detect selected_layers from generation .pt — "
            "factuality_probe_adapted will fall back to layer index 0.",
            file=sys.stderr,
        )
        selected_layers = [0]

    skip = set(args.skip or [])
    results: Dict[str, Any] = {
        "setup": setup,
        "selected_layers": selected_layers,
        "split_file": str(split_file),
        "baselines": {},
    }

    # ---- Cheap baselines first ----
    if "token_entropy" not in skip:
        print("\n[token_entropy] running…")
        out = _run_token_entropy(test_records)
        results["baselines"]["token_entropy"] = out
        print(
            f"  n={out['n_test']} | t={out['wall_clock_seconds']:.2f}s | "
            f"ratio={out['ratio_metrics']} | strict={out['strict_metrics']}"
        )

    if "logistic_regression" not in skip:
        print("\n[logistic_regression] running…")
        out = _run_logistic_regression(
            train_records, test_records, C=float(args.C_lr)
        )
        results["baselines"]["logistic_regression"] = out
        if out.get("skipped"):
            print(f"  skipped: {out.get('reason')}")
        else:
            print(
                f"  n_train={out['n_train']} n_test={out['n_test']} | "
                f"t={out['wall_clock_seconds']:.2f}s | "
                f"ratio={out['ratio_metrics']} | strict={out['strict_metrics']}"
            )

    if "factuality_probe_adapted" not in skip:
        print("\n[factuality_probe_adapted] running…")
        out = _run_factuality_probe_adapted(
            train_records,
            test_records,
            selected_layers,
            target_layer=int(args.target_layer),
            C=float(args.C_l1),
        )
        results["baselines"]["factuality_probe_adapted"] = out
        if out.get("skipped"):
            print(f"  skipped: {out.get('reason')}")
        else:
            print(
                f"  n_train={out['n_train']} n_test={out['n_test']} | "
                f"t={out['wall_clock_seconds']:.2f}s | "
                f"ratio={out['ratio_metrics']} | strict={out['strict_metrics']}"
            )

    # ---- Optional / expensive baselines ----
    nli_scorer = None
    if (
        ("semantic_entropy" not in skip or "luq" not in skip)
        and args.samples_dir
    ):
        try:
            from src.baselines.semantic_entropy import NLIScorer

            print("\nLoading NLI scorer (microsoft/deberta-large-mnli)…")
            nli_scorer = NLIScorer(device=device)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"warning: failed to load NLI scorer — {exc}", file=sys.stderr)
            nli_scorer = None

    if "semantic_entropy" not in skip:
        if args.samples_dir is None:
            results["baselines"]["semantic_entropy"] = {
                "skipped": True,
                "reason": "--samples-dir not provided",
            }
        elif nli_scorer is None:
            results["baselines"]["semantic_entropy"] = {
                "skipped": True,
                "reason": "NLI scorer unavailable (transformers not installed?)",
            }
        else:
            print("\n[semantic_entropy] running…")
            out = _run_semantic_entropy(
                test_records, Path(args.samples_dir), nli_scorer
            )
            results["baselines"]["semantic_entropy"] = out
            if out.get("skipped"):
                print(f"  skipped: {out.get('reason')}")
            else:
                print(
                    f"  n={out['n_test_sentences']} "
                    f"(prompts: {out['n_prompts_scored']} scored, "
                    f"{out['n_prompts_missing_samples']} missing) | "
                    f"t={out['wall_clock_seconds']:.2f}s | "
                    f"ratio={out['ratio_metrics']} | strict={out['strict_metrics']}"
                )

    if "luq" not in skip:
        if args.samples_dir is None:
            results["baselines"]["luq"] = {
                "skipped": True,
                "reason": "--samples-dir not provided",
            }
        elif nli_scorer is None:
            results["baselines"]["luq"] = {
                "skipped": True,
                "reason": "NLI scorer unavailable (transformers not installed?)",
            }
        else:
            print("\n[luq] running…")
            out = _run_luq(test_records, Path(args.samples_dir), nli_scorer)
            results["baselines"]["luq"] = out
            if out.get("skipped"):
                print(f"  skipped: {out.get('reason')}")
            else:
                print(
                    f"  n={out['n_test_sentences']} | "
                    f"t={out['wall_clock_seconds']:.2f}s | "
                    f"ratio={out['ratio_metrics']} | strict={out['strict_metrics']}"
                )

    if args.run_original and "factuality_probe_original" not in skip:
        model_name = args.model_name or _cfg_get(cfg, "model", "name")
        if not model_name:
            results["baselines"]["factuality_probe_original"] = {
                "skipped": True,
                "reason": "no --model-name and no model.name in config",
            }
        else:
            print(f"\n[factuality_probe_original] running… (model={model_name})")
            # prepare_data drops the per-claim list; the original probe needs it.
            _attach_sentence_claims(train_records, processed_dirs)
            _attach_sentence_claims(test_records, processed_dirs)
            out = _run_factuality_probe_original(
                train_records,
                test_records,
                model_name=model_name,
                target_layer=int(args.target_layer),
                C=float(args.C_l1),
                device=device,
            )
            results["baselines"]["factuality_probe_original"] = out
            if out.get("skipped"):
                print(f"  skipped: {out.get('reason')}")
            else:
                print(
                    f"  n_train_claims={out['n_train_claims']} "
                    f"n_test_claims={out['n_test_claims']} "
                    f"n_test_sentences={out['n_test_sentences']} | "
                    f"t={out['wall_clock_seconds']:.2f}s | "
                    f"ratio={out['ratio_metrics']} | strict={out['strict_metrics']}"
                )

    out_path = results_dir / "baselines.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baselines to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
