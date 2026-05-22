"""
Phase 1-3: Per-token entropy and top-1 probability offline cache.

These scalars depend only on the model's logits (not on the learnable
parameters ψ), so they are computed once and stored to disk.

Math (CLAUDE.md notation):
    entropy_ℓ = -∑_v p_v log p_v   (Shannon entropy in nats)
    top1_ℓ    = max_v p_v
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_token_entropy_and_top1(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token predictive entropy and top-1 probability from logits.

    Args:
        logits: (T, V) Tensor — raw logits, may be fp16.

    Returns:
        (entropy, top1_prob):
            entropy   : (T,) fp32 — Shannon entropy in nats
            top1_prob : (T,) fp32 — maximum softmax probability
    """
    logits_f = logits.float()  # (T, V) fp32 — required for numerical stability

    log_probs = F.log_softmax(logits_f, dim=-1)   # (T, V)
    probs = log_probs.exp()                        # (T, V)

    # nansum handles the 0 * log(0) = 0 convention automatically
    entropy = -(probs * log_probs).nansum(dim=-1)  # (T,)
    top1_prob = probs.max(dim=-1).values           # (T,)

    return entropy, top1_prob


def cache_scalars_for_directory(
    generations_dir: str | Path,
    cache_dir: str | Path,
) -> None:
    """Compute and cache entropy/top1 scalars for all .pt files in generations_dir.

    Skips files that already exist in cache_dir (resume-safe).

    Args:
        generations_dir: Directory containing generation .pt files
                         (output of batch_generate).  Each file must have
                         keys "logits" (T, V) and "token_ids" (T,).
        cache_dir: Destination directory.  Each output file is named
                   {idx:05d}.pt and contains:
                       "entropy"   : (T,) fp32
                       "top1_prob" : (T,) fp32
                       "token_ids" : (T,) LongTensor
    """
    generations_dir = Path(generations_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(generations_dir.glob("*.pt"))
    if not pt_files:
        return

    for pt_file in tqdm(pt_files, desc="Caching scalars"):
        idx = int(pt_file.stem)  # e.g. "00003" → 3
        out_path = cache_dir / f"{idx:05d}.pt"

        if out_path.exists():
            continue  # Resume: skip already-cached entries

        data = torch.load(pt_file, weights_only=False)
        logits = data["logits"]      # (T, V) fp16
        token_ids = data["token_ids"]  # (T,) long

        entropy, top1_prob = compute_token_entropy_and_top1(logits)

        torch.save(
            {"entropy": entropy, "top1_prob": top1_prob, "token_ids": token_ids},
            out_path,
        )


def load_scalars(idx: int, cache_dir: str | Path) -> Dict[str, torch.Tensor]:
    """Load a cached scalar file by index.

    Args:
        idx: Integer index (e.g. 3 → loads 00003.pt).
        cache_dir: Directory written by cache_scalars_for_directory.

    Returns:
        Dict with keys "entropy", "top1_prob", "token_ids".
    """
    path = Path(cache_dir) / f"{idx:05d}.pt"
    return torch.load(path, weights_only=False)
