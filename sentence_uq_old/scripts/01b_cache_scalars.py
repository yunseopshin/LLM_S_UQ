"""
Script 01b: Compute and cache per-token entropy / top-1 probability.

Reads generation .pt files produced by 01_generate_data.py and writes
scalar cache files to data/cache/.

Usage:
    python scripts/01b_cache_scalars.py --config configs/default.yaml
    python scripts/01b_cache_scalars.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.features.cached_scalars import cache_scalars_for_directory


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1-3: Cache per-token entropy and top-1 prob.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    generations_dir: str = cfg["data"]["generations_dir"]
    cache_dir: str = cfg["data"]["cache_dir"]

    print(f"[INFO] Reading generations from: {generations_dir}")
    print(f"[INFO] Writing scalar cache to:  {cache_dir}")

    cache_scalars_for_directory(generations_dir, cache_dir)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
