"""CLI for Phase 1-3: cache per-token entropy and top-1 probability.

Reads ``generations_dir`` and ``cache_dir`` for each dataset (FActScore-Bio
and LongFact) from the YAML config and runs
:func:`src.features.cached_scalars.cache_scalars_for_directory` with a
``tqdm`` progress bar.

Examples
--------
    python scripts/01b_cache_scalars.py --config configs/default.yaml
    python scripts/01b_cache_scalars.py --dataset factscore_bio
    python scripts/01b_cache_scalars.py \
        --generations-dir data/generations/factscore_bio \
        --cache-dir data/cache/factscore_bio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.cached_scalars import cache_scalars_for_directory  # noqa: E402


_DATASETS = ("factscore_bio", "longfact")


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
            "Compute per-token entropy and top-1 probability for every "
            "generation .pt file and write them to the offline cache."
        )
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
        help="Restrict to a single dataset (default: process both).",
    )
    p.add_argument(
        "--generations-dir",
        type=str,
        default=None,
        help="Override generations dir (requires --cache-dir).",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache dir (requires --generations-dir).",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    return p


def _resolve_dataset_paths(
    cfg: dict[str, Any], dataset: str
) -> tuple[Path, Path]:
    """Return ``(generations_dir, cache_dir)`` paths for ``dataset``."""
    gen = _cfg_get(
        cfg, "generation", f"{dataset}_dir", f"data/generations/{dataset}"
    )
    cache = _cfg_get(cfg, "cache", f"{dataset}_dir", f"data/cache/{dataset}")
    return Path(gen), Path(cache)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)

    # Explicit path overrides take precedence and run a single pass.
    if (args.generations_dir is not None) != (args.cache_dir is not None):
        print(
            "error: --generations-dir and --cache-dir must be provided together",
            file=sys.stderr,
        )
        return 2

    pairs: list[tuple[str, Path, Path]] = []
    if args.generations_dir is not None:
        pairs.append(
            ("explicit", Path(args.generations_dir), Path(args.cache_dir))
        )
    else:
        datasets = (args.dataset,) if args.dataset else _DATASETS
        for ds in datasets:
            gen, cache = _resolve_dataset_paths(cfg, ds)
            pairs.append((ds, gen, cache))

    print("=== Phase 1-3 entropy/top-1 cache ===")

    total_cached = 0
    total_errors = 0
    progress = not args.no_progress

    for label, gen_dir, cache_dir in pairs:
        if not gen_dir.exists():
            print(
                f"[{label}] skip: generations dir not found at {gen_dir}",
                file=sys.stderr,
            )
            continue

        print(f"\n[{label}] {gen_dir} -> {cache_dir}")
        result = cache_scalars_for_directory(
            gen_dir, cache_dir, progress=progress
        )
        total_cached += result["cached"]
        total_errors += len(result["errors"])
        print(
            f"  cached={result['cached']}, errors={len(result['errors'])}"
        )
        for src, msg in result["errors"][:10]:
            print(f"    ERROR  {src}: {msg}", file=sys.stderr)

    print(f"\nDone: cached={total_cached}, errors={total_errors}")
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
