"""CLI for Phase 1-0: prepare raw datasets and build a setup split.

Examples
--------
    python scripts/00_prepare_dataset.py --setup 2 --seed 42
    python scripts/00_prepare_dataset.py --setup 1 --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Make `src` importable when this script is executed directly.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    SETUPS,
    prepare_all_and_split,
    split_save_filename,
    summarise_split,
)


def _load_yaml_config(path: str) -> dict[str, Any]:
    import yaml  # local import so the script still runs without pyyaml installed

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prepare FActScore-Bio and LongFact-Objects raw data and build the "
            "train/val/test split for the chosen experimental setup."
        )
    )
    p.add_argument(
        "--setup",
        type=int,
        choices=list(SETUPS),
        required=False,
        help="Experimental setup (1, 2, or 3). Falls back to config value.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for shuffling (default: from config or 42).",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config (e.g. configs/default.yaml).",
    )
    p.add_argument(
        "--factscore-dir",
        type=str,
        default=None,
        help="Override factscore_bio_dir.",
    )
    p.add_argument(
        "--longfact-dir",
        type=str,
        default=None,
        help="Override longfact_dir.",
    )
    p.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Override splits_dir.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-prepare raw datasets even if outputs already exist.",
    )
    return p


def _resolve(cfg: dict[str, Any], key: str, override: Any, default: Any) -> Any:
    if override is not None:
        return override
    section = cfg.get("dataset") or {}
    val = section.get(key)
    return val if val is not None else default


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml_config(args.config)

    setup = args.setup
    if setup is None:
        setup = (cfg.get("dataset") or {}).get("setup")
    if setup is None:
        print("error: --setup is required (or set dataset.setup in config)", file=sys.stderr)
        return 2

    seed = _resolve(cfg, "seed", args.seed, 42)
    factscore_dir = _resolve(cfg, "factscore_bio_dir", args.factscore_dir, "data/raw/factscore_bio")
    longfact_dir = _resolve(cfg, "longfact_dir", args.longfact_dir, "data/raw/longfact")
    splits_dir = _resolve(cfg, "splits_dir", args.splits_dir, "data/splits")

    split = prepare_all_and_split(
        setup=int(setup),
        factscore_dir=factscore_dir,
        longfact_dir=longfact_dir,
        splits_dir=splits_dir,
        seed=int(seed),
        force_redownload=args.force,
    )

    out_path = os.path.join(splits_dir, split_save_filename(int(setup)))
    print(summarise_split(split))
    print(f"Split saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
