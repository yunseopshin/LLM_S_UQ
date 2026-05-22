"""CLI for Phase 1-4: factuality annotation → binomial counts (K_j, m_j).

Loads generation ``.pt`` files (Phase 1-1 outputs), runs sentence splitting
(Phase 1-2) on each, calls the auxiliary LM (GPT-4o-mini by default) to
decompose / revise / subjectivity-filter the claims, retrieves the knowledge
context, and writes ``(m_j, K_j)`` per sentence.

Usage
-----
    # Annotate FActScore-Bio (Setup 2 default).
    python scripts/02_annotate_factuality.py --setup 2 --config configs/default.yaml

    # Annotate both datasets (Setup 1).
    python scripts/02_annotate_factuality.py --setup 1

    # Annotate LongFact only (Setup 3).
    python scripts/02_annotate_factuality.py --setup 3

Setup → dataset map
~~~~~~~~~~~~~~~~~~~
- Setup 1: FActScore (test) + LongFact (train) — both datasets.
- Setup 2: FActScore only.
- Setup 3: LongFact only.

Output layout
~~~~~~~~~~~~~
::

    data/processed/factscore_bio/{entity}.json     # per-prompt resume files
    data/processed/factscore_bio/annotated.json    # combined
    data/processed/factscore_bio/knowledge/...     # Wikipedia cache
    data/processed/longfact/{topic}/{prompt_idx:03d}.json
    data/processed/longfact/annotated.json

Each per-sentence dict carries ``m_j``, ``K_j``, and ``claims``; downstream
phases (3+) consume the combined ``annotated.json`` files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.annotation import (  # noqa: E402
    DEFAULT_AUX_MODEL,
    DEFAULT_TEMPERATURE,
    OpenAIChatClient,
    RateLimiter,
    annotate_batch,
)
from src.data.sentence_split import (  # noqa: E402
    load_spacy_model,
    process_generation,
)

_DATASETS = ("factscore_bio", "longfact")

#: Setup → datasets to annotate (matches the phase_1_4 spec).
_SETUP_DATASETS: dict[int, tuple[str, ...]] = {
    1: ("factscore_bio", "longfact"),
    2: ("factscore_bio",),
    3: ("longfact",),
}


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    sec = cfg.get(section) or {}
    val = sec.get(key)
    return val if val is not None else default


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the factuality annotation pipeline on generation outputs and "
            "produce per-sentence (m_j, K_j) binomial counts."
        )
    )
    p.add_argument(
        "--setup",
        type=int,
        choices=list(_SETUP_DATASETS),
        required=False,
        help="Experimental setup (1, 2, or 3). Falls back to config value.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config (e.g. configs/default.yaml).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(_DATASETS),
        help="Restrict to a single dataset (overrides --setup mapping).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only annotate the first N generation files per dataset.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Auxiliary LM (chat) model id. Default: {DEFAULT_AUX_MODEL}.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="LM temperature for the annotation pipeline (default: 0).",
    )
    p.add_argument(
        "--rps",
        type=float,
        default=0.0,
        help=(
            "Max requests per second to the auxiliary LM (0 = no rate limit). "
            "Use to respect provider rate limits."
        ),
    )
    p.add_argument(
        "--tokenizer-name",
        type=str,
        default=None,
        help=(
            "Tokenizer to load for sentence ↔ token alignment. Defaults to "
            "model.name from the config (same model used for generation)."
        ),
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    p.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip writing the combined annotated.json (only per-record files).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List records that would be annotated; skip API calls.",
    )
    return p


def _resolve_setup(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    setup = args.setup
    if setup is None:
        setup = (cfg.get("dataset") or {}).get("setup")
    if setup is None:
        raise SystemExit(
            "error: --setup is required (or set dataset.setup in config)"
        )
    setup = int(setup)
    if setup not in _SETUP_DATASETS:
        raise SystemExit(f"error: unknown setup {setup}; valid: {list(_SETUP_DATASETS)}")
    return setup


def _resolve_datasets(args: argparse.Namespace, setup: int) -> tuple[str, ...]:
    if args.dataset is not None:
        return (args.dataset,)
    return _SETUP_DATASETS[setup]


def _dataset_dirs(cfg: dict[str, Any], dataset: str) -> tuple[Path, Path]:
    """Return ``(generations_dir, processed_dir)`` for ``dataset``."""
    gen = _cfg_get(
        cfg, "generation", f"{dataset}_dir", f"data/generations/{dataset}"
    )
    proc = _cfg_get(
        cfg, "processed", f"{dataset}_dir", f"data/processed/{dataset}"
    )
    return Path(gen), Path(proc)


def _iter_generation_files(gen_dir: Path, dataset: str) -> Iterable[Path]:
    """Yield ``.pt`` generation files in deterministic order."""
    if dataset == "factscore_bio":
        yield from sorted(gen_dir.glob("*.pt"))
    elif dataset == "longfact":
        for topic_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
            yield from sorted(topic_dir.glob("*.pt"))


def _load_processed_records(
    gen_dir: Path, dataset: str, tokenizer: Any, nlp: Any, limit: int | None
) -> list[dict[str, Any]]:
    """Load every generation .pt file under ``gen_dir`` and attach sentences.

    The output dicts carry everything :func:`annotate_record` needs:
    ``text``, ``sentences`` (with ``token_range``), and the dataset-specific
    context fields (``entity`` / ``topic`` / ``prompt`` / ``prompt_idx``).
    """
    records: list[dict[str, Any]] = []
    files = list(_iter_generation_files(gen_dir, dataset))
    if limit is not None and limit >= 0:
        files = files[:limit]

    for pt_path in files:
        try:
            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - log & continue
            print(f"  WARN load {pt_path}: {exc}", file=sys.stderr)
            continue

        split = process_generation(payload, tokenizer, nlp)
        record: dict[str, Any] = {
            "dataset": dataset,
            "text": payload.get("text", ""),
            "sentences": split.get("sentences", []),
            "meta": dict(payload.get("meta") or {}),
        }
        if dataset == "factscore_bio":
            record["entity"] = (
                payload.get("meta", {}).get("entity") if payload.get("meta") else None
            ) or pt_path.stem
        elif dataset == "longfact":
            record["topic"] = (
                payload.get("meta", {}).get("topic") if payload.get("meta") else None
            ) or pt_path.parent.name
            record["prompt_idx"] = (
                payload.get("meta", {}).get("prompt_idx") if payload.get("meta") else None
            )
            if record["prompt_idx"] is None:
                try:
                    record["prompt_idx"] = int(pt_path.stem)
                except ValueError:
                    record["prompt_idx"] = 0
            record["prompt"] = payload.get("prompt", "")
        records.append(record)
    return records


def _load_tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, use_fast=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)

    setup = _resolve_setup(args, cfg)
    datasets = _resolve_datasets(args, setup)

    tokenizer_name = args.tokenizer_name or _cfg_get(cfg, "model", "name")
    if not tokenizer_name:
        print(
            "error: tokenizer model name not set (use --tokenizer-name or model.name in config)",
            file=sys.stderr,
        )
        return 2

    aux_model = args.model or DEFAULT_AUX_MODEL

    print(f"=== Phase 1-4 annotation — setup {setup} ===")
    print(f"Datasets:        {', '.join(datasets)}")
    print(f"Tokenizer:       {tokenizer_name}")
    print(f"Aux LM:          {aux_model}")
    print(f"Temperature:     {args.temperature}")
    print(f"Rate limit:      {args.rps} req/s" if args.rps > 0 else "Rate limit:      (none)")

    # Heavy imports happen only after argv validation.
    print("Loading spaCy + tokenizer ...")
    nlp = load_spacy_model("en")
    tokenizer = _load_tokenizer(tokenizer_name)

    if args.dry_run:
        client: Any = None
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "warning: OPENAI_API_KEY not set; OpenAIChatClient.generate will "
                "fail. Use --dry-run for a planning pass.",
                file=sys.stderr,
            )
        client = OpenAIChatClient(model=aux_model)
    rate_limiter = RateLimiter(rps=args.rps) if args.rps > 0 else None

    totals = {"annotated": 0, "skipped": 0, "errors": 0}
    for dataset in datasets:
        gen_dir, processed_dir = _dataset_dirs(cfg, dataset)
        if not gen_dir.exists():
            print(
                f"[{dataset}] skip: generations dir not found at {gen_dir}",
                file=sys.stderr,
            )
            continue

        print(f"\n-- {dataset} --")
        print(f"  generations:  {gen_dir}")
        print(f"  processed:    {processed_dir}")

        records = _load_processed_records(
            gen_dir, dataset, tokenizer, nlp, args.limit
        )
        n_sents = sum(len(r.get("sentences") or []) for r in records)
        print(f"  records:      {len(records)}  (sentences: {n_sents})")

        if args.dry_run:
            for r in records[:5]:
                key = r.get("entity") or f"{r.get('topic')}/{r.get('prompt_idx')}"
                print(f"    [{key}]  sentences={len(r.get('sentences') or [])}")
            continue

        result = annotate_batch(
            records,
            dataset_type=dataset,
            api_client=client,
            out_dir=processed_dir,
            knowledge_cache_dir=processed_dir / "knowledge",
            rate_limiter=rate_limiter,
            temperature=args.temperature,
            progress=not args.no_progress,
            write_combined=not args.no_combined,
        )

        print(
            f"  annotated={result['annotated']}, "
            f"skipped={result['skipped']}, "
            f"errors={len(result['errors'])}"
        )
        if result.get("combined_path"):
            print(f"  combined -> {result['combined_path']}")
        for rec, msg in result["errors"][:10]:
            key = rec.get("entity") or f"{rec.get('topic')}/{rec.get('prompt_idx')}"
            print(f"    ERROR  {key}: {msg}", file=sys.stderr)

        totals["annotated"] += result["annotated"]
        totals["skipped"] += result["skipped"]
        totals["errors"] += len(result["errors"])

    print(
        f"\nDone: annotated={totals['annotated']}, "
        f"skipped={totals['skipped']}, errors={totals['errors']}"
    )
    return 0 if totals["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
