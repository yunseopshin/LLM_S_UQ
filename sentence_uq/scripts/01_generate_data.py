"""CLI for Phase 1-1: LLM generation + hidden-state extraction.

Usage
-----
    python scripts/01_generate_data.py --setup 2 --config configs/default.yaml

Setup behaviour
~~~~~~~~~~~~~~~
- **Setup 1** — generate train+val (LongFact) *and* test (FActScore-Bio).
- **Setup 2** — generate train+val+test (all FActScore-Bio).
- **Setup 3** — generate train+val+test (all LongFact).

Output layout
~~~~~~~~~~~~~
::

    data/generations/factscore_bio/{entity_name}.pt
    data/generations/longfact/{topic}/{prompt_idx:03d}.pt
    data/generations/{dataset}/metadata.json

Resume support: any ``.pt`` that already exists is skipped, so Setup 1 and
Setup 2 share the FActScore generations without recomputing them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import SETUPS, split_save_filename  # noqa: E402
from src.data.generation import (  # noqa: E402
    batch_generate,
    load_model,
    resolve_selected_layers,
    write_dataset_metadata,
)


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_dtype(s: str | None) -> torch.dtype:
    if not s:
        return torch.float16
    key = s.lower()
    if key not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype {s!r}; choose from {sorted(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[key]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate LLM responses + per-token hidden states for the chosen "
            "experimental setup."
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
        "--config",
        type=str,
        default=None,
        help="YAML config (e.g. configs/default.yaml).",
    )
    p.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Directory containing setup_{N}.json (default: data/splits).",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override model.name from the config.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Model dtype: float16 | bfloat16 | float32.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda | cpu (default: cuda; falls back to cpu if unavailable).",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument(
        "--no-sample",
        action="store_true",
        help="Greedy decoding (overrides temperature).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only generate the first N items per split section.",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Regenerate even when an output .pt already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan but do not load the model or generate.",
    )
    return p


def _cfg_get(cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    sec = cfg.get(section) or {}
    val = sec.get(key)
    return val if val is not None else default


def _select_items_for_setup(
    split: dict[str, Any], setup: int, limit: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (factscore_items, longfact_items) to generate for ``setup``.

    Setup 1: train+val are LongFact, test is FActScore-Bio.
    Setup 2: train+val+test are FActScore-Bio.
    Setup 3: train+val+test are LongFact.
    """
    all_items: list[dict[str, Any]] = []
    for part in ("train", "val", "test"):
        all_items.extend(split.get(part) or [])

    fs = [it for it in all_items if it.get("dataset") == "factscore_bio"]
    lf = [it for it in all_items if it.get("dataset") == "longfact"]

    if limit is not None and limit >= 0:
        fs = fs[:limit]
        lf = lf[:limit]
    return fs, lf


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)

    setup = args.setup
    if setup is None:
        setup = (cfg.get("dataset") or {}).get("setup")
    if setup is None:
        print(
            "error: --setup is required (or set dataset.setup in config)",
            file=sys.stderr,
        )
        return 2
    setup = int(setup)

    splits_dir = args.splits_dir or _cfg_get(cfg, "dataset", "splits_dir", "data/splits")
    split_path = Path(splits_dir) / split_save_filename(setup)
    if not split_path.exists():
        print(
            f"error: split file not found at {split_path}. "
            f"Run scripts/00_prepare_dataset.py --setup {setup} first.",
            file=sys.stderr,
        )
        return 2
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    model_name = args.model_name or _cfg_get(cfg, "model", "name")
    if not model_name:
        print(
            "error: model name not set (use --model-name or model.name in config)",
            file=sys.stderr,
        )
        return 2
    dtype = _parse_dtype(args.dtype or _cfg_get(cfg, "model", "dtype"))
    selected_cfg = _cfg_get(cfg, "model", "selected_layers", None)

    factscore_dir = _cfg_get(
        cfg, "generation", "factscore_bio_dir", "data/generations/factscore_bio"
    )
    longfact_dir = _cfg_get(
        cfg, "generation", "longfact_dir", "data/generations/longfact"
    )
    max_new_tokens = int(
        args.max_new_tokens
        if args.max_new_tokens is not None
        else _cfg_get(cfg, "generation", "max_new_tokens", 512)
    )
    temperature = float(
        args.temperature
        if args.temperature is not None
        else _cfg_get(cfg, "generation", "temperature", 0.7)
    )
    top_p = float(
        args.top_p
        if args.top_p is not None
        else _cfg_get(cfg, "generation", "top_p", 1.0)
    )
    do_sample = not args.no_sample

    fs_items, lf_items = _select_items_for_setup(split, setup, args.limit)

    print(f"=== Phase 1-1 generation — setup {setup} ===")
    print(f"Model:               {model_name}  (dtype={dtype})")
    print(f"Split file:          {split_path}")
    print(f"FActScore-Bio items: {len(fs_items):4d} -> {factscore_dir}")
    print(f"LongFact items:      {len(lf_items):4d} -> {longfact_dir}")
    print(
        f"Generation:          max_new_tokens={max_new_tokens}, "
        f"temperature={temperature}, top_p={top_p}, do_sample={do_sample}"
    )

    if args.dry_run:
        print("(dry-run) skipping model load + generation")
        return 0

    if not fs_items and not lf_items:
        print("Nothing to generate for this setup. Done.")
        return 0

    model, tokenizer, model_info = load_model(
        model_name=model_name, device=args.device, dtype=dtype
    )
    selected_layers = resolve_selected_layers(
        model_info["num_hidden_layers"], selected_cfg
    )
    print(f"hidden_dim={model_info['hidden_dim']}, "
          f"num_hidden_layers={model_info['num_hidden_layers']}, "
          f"selected_layers={selected_layers}")

    gen_cfg = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
    }
    skip_existing = not args.no_skip_existing

    totals = {"generated": 0, "skipped": 0, "errors": 0}

    if fs_items:
        print(f"\n-- FActScore-Bio ({len(fs_items)} items) --")
        result = batch_generate(
            fs_items,
            model=model,
            tokenizer=tokenizer,
            model_info=model_info,
            selected_layers=selected_layers,
            factscore_dir=factscore_dir,
            longfact_dir=longfact_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            skip_existing=skip_existing,
        )
        write_dataset_metadata(
            factscore_dir,
            model_info=model_info,
            selected_layers=selected_layers,
            generation_config=gen_cfg,
            dataset_tag="factscore_bio",
            items=fs_items,
        )
        totals["generated"] += result["generated"]
        totals["skipped"] += result["skipped"]
        totals["errors"] += len(result["errors"])
        for it, msg in result["errors"][:10]:
            print(f"  ERROR  {it.get('entity', it.get('topic'))}: {msg}", file=sys.stderr)

    if lf_items:
        print(f"\n-- LongFact ({len(lf_items)} items) --")
        result = batch_generate(
            lf_items,
            model=model,
            tokenizer=tokenizer,
            model_info=model_info,
            selected_layers=selected_layers,
            factscore_dir=factscore_dir,
            longfact_dir=longfact_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            skip_existing=skip_existing,
        )
        write_dataset_metadata(
            longfact_dir,
            model_info=model_info,
            selected_layers=selected_layers,
            generation_config=gen_cfg,
            dataset_tag="longfact",
            items=lf_items,
        )
        totals["generated"] += result["generated"]
        totals["skipped"] += result["skipped"]
        totals["errors"] += len(result["errors"])
        for it, msg in result["errors"][:10]:
            print(
                f"  ERROR  {it.get('topic')}/{it.get('prompt_idx')}: {msg}",
                file=sys.stderr,
            )

    print(
        f"\nDone: generated={totals['generated']}, "
        f"skipped={totals['skipped']}, errors={totals['errors']}"
    )
    return 0 if totals["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
