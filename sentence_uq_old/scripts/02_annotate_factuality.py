"""
Script 02: Annotate factuality of generated sentences using LLM-as-judge.

Reads processed sentence data from data/processed/sentences.json and writes
annotated results to data/processed/annotated.json.

Usage:
    ANTHROPIC_API_KEY=sk-... python scripts/02_annotate_factuality.py --config configs/default.yaml

Environment variables:
    ANTHROPIC_API_KEY : Anthropic API key (required)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data.annotation import annotate_batch


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1-4: Annotate sentence factuality.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    processed_path = Path(cfg["data"]["processed_path"])
    # Input sentences live one level up: data/processed/sentences.json
    sentences_path = processed_path.parent / "sentences.json"
    save_path = processed_path  # data/processed/annotated.json

    # ------------------------------------------------------------------
    # API client
    # ------------------------------------------------------------------
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[ERROR] ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Load sentences
    # ------------------------------------------------------------------
    if not sentences_path.exists():
        print(f"[ERROR] Sentence file not found: {sentences_path}")
        print("        Run 01_generate_data.py and the sentence-splitting step first.")
        sys.exit(1)

    with open(sentences_path) as f:
        sentences = json.load(f)
    print(f"[INFO] Loaded {len(sentences)} sentences from {sentences_path}")

    # ------------------------------------------------------------------
    # Annotate
    # ------------------------------------------------------------------
    print(f"[INFO] Annotating with LLM-as-judge (resume={save_path.exists()}) ...")
    results = annotate_batch(
        processed_sentences=sentences,
        api_client=client,
        use_wiki=True,
        resume=True,
        save_path=save_path,
        rate_limit_sleep=0.5,
        save_interval=100,
    )

    n_supported = sum(1 for r in results if r.get("label") == 1)
    n_not_supported = sum(1 for r in results if r.get("label") == 0)
    n_failed = sum(1 for r in results if r.get("label") is None)

    print(f"[INFO] Done. SUPPORTED={n_supported}, NOT_SUPPORTED={n_not_supported}, failed={n_failed}")
    print(f"[INFO] Results saved to: {save_path}")


if __name__ == "__main__":
    main()
