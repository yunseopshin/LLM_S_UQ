"""
Script 01: Generate LLM responses + hidden states for each entity.

Usage:
    python scripts/01_generate_data.py --config configs/default.yaml
    python scripts/01_generate_data.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the sentence_uq/ root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data.generation import batch_generate, load_model


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1-1: Generate LLM responses with hidden states.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Config fields
    # ------------------------------------------------------------------
    model_name: str = cfg["model"]["name"]
    device: str = cfg["model"].get("device", "cuda")
    dtype: str = cfg["model"].get("dtype", "float16")
    selected_layers: list[int] = cfg["model"].get("selected_layers", None)
    max_new_tokens: int = cfg["model"].get("max_new_tokens", 512)

    entity_list_path: str = cfg["data"]["entity_list_path"]
    generations_dir: str = cfg["data"]["generations_dir"]

    # ------------------------------------------------------------------
    # Load entity list
    # ------------------------------------------------------------------
    entity_file = Path(entity_list_path)
    if not entity_file.exists():
        # Fallback: small hardcoded list for quick testing
        entities = [
            "Albert Einstein",
            "Marie Curie",
            "Isaac Newton",
            "Charles Darwin",
            "Galileo Galilei",
        ]
        print(f"[WARN] Entity file not found at {entity_file}. Using 5 hardcoded entities.")
    else:
        with open(entity_file) as f:
            entities = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(entities)} entities from {entity_file}")

    prompts = [f"Tell me a bio of {entity}." for entity in entities]

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"[INFO] Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device, dtype=dtype)
    print("[INFO] Model loaded.")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        save_dir=generations_dir,
        selected_layers=selected_layers,
        max_new_tokens=max_new_tokens,
        entities=entities,
    )
    print(f"[INFO] Done. Results saved to: {generations_dir}")


if __name__ == "__main__":
    main()
