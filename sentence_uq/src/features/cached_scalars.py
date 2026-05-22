"""Offline cache of ψ-independent per-token scalars.

Phase 1-3. Computes predictive **entropy** ``H_ℓ`` and **top-1 probability**
``p^(1)_ℓ`` from the per-token logits stored by Phase 1-1 generation. Both
quantities depend only on the LLM output distribution (not on the learnable
parameters ψ = {W, α, μ_0, σ_0}), so they are cached once per generation and
re-used for every outer-loop iteration.

Math reference
--------------
Given logits ``ℓ_ℓ ∈ ℝ^V`` for generated token ``ℓ``,

    p_ℓ = softmax(ℓ_ℓ)                ∈ Δ^{V-1}
    H_ℓ = -Σ_v p_{ℓ,v} log p_{ℓ,v}    (predictive entropy, nats)
    p^(1)_ℓ = max_v p_{ℓ,v}           (top-1 probability)

These are the last two coordinates of the Phase 2-1 feature vector
``z_ℓ = [W · Σ_l α_l h_ℓ^(l), H_ℓ, p^(1)_ℓ] ∈ ℝ^{D'+2}`` (see CLAUDE.md
Core Math).

Numerics
--------
- Logits are stored in fp16 (CLAUDE.md "Store hidden states in fp16; always
  compute numerics in fp32"). We cast to fp32 before any softmax / log.
- Stable computation uses ``log_softmax`` (which internally subtracts
  ``logits.max``) rather than ``log(softmax(x))``.
- ``0 · log 0 = 0`` is enforced via ``torch.where`` to avoid ``NaN`` when a
  vocabulary entry has zero probability (e.g. when logits contain ``-inf``).

File layout
-----------
For a generations directory laid out as in Phase 1-1::

    data/generations/factscore_bio/{entity}.pt
    data/generations/longfact/{topic}/{prompt_idx:03d}.pt

the cache mirrors the per-dataset root::

    data/cache/{dataset}/{idx:05d}.pt

where ``idx`` is the position of the source file in the **sorted recursive
listing** of ``generations_dir``. The cache file contains::

    {
        "entropy":     (T,) fp32 Tensor,
        "top1_prob":   (T,) fp32 Tensor,
        "token_ids":   (T,) LongTensor,         # sanity check vs. source
        "source_path": str,                     # relative to generations_dir
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Core numerics
# ---------------------------------------------------------------------------


def compute_token_entropy_and_top1(logits: Tensor) -> tuple[Tensor, Tensor]:
    """Compute per-token predictive entropy and top-1 probability.

    Math
    ----
    For each token row ``ℓ_t ∈ ℝ^V``::

        p_t       = softmax(ℓ_t)
        H_t       = -Σ_v p_{t,v} log p_{t,v}      (nats)
        p^(1)_t   = max_v p_{t,v}

    Numerical stability
    -------------------
    - ``log_softmax`` is used directly; it subtracts ``logits.max`` internally,
      so no overflow even for large logit magnitudes.
    - ``0 · log 0`` is forced to ``0`` via :func:`torch.where`, avoiding the
      ``0 * -inf = NaN`` trap when some logit equals ``-inf``.
    - Logits are cast to fp32 before any reduction (CLAUDE.md rule:
      "compute numerics in fp32").

    Parameters
    ----------
    logits : Tensor
        Shape ``(T, vocab_size)``. Any floating dtype; cast to fp32 internally.

    Returns
    -------
    entropy : Tensor
        Shape ``(T,)``, fp32, nats.
    top1_prob : Tensor
        Shape ``(T,)``, fp32, in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``logits`` is not 2-D or has a non-floating dtype.
    """
    if logits.dim() != 2:
        raise ValueError(
            f"logits must be 2-D (T, vocab_size); got shape {tuple(logits.shape)}"
        )
    if not torch.is_floating_point(logits):
        raise ValueError(
            f"logits must be a floating-point tensor; got dtype {logits.dtype}"
        )

    logits_f32 = logits.to(torch.float32)
    log_probs = F.log_softmax(logits_f32, dim=-1)
    probs = log_probs.exp()

    # 0 * log(0) = 0; mask the (possibly NaN) product where p == 0.
    plogp = torch.where(
        probs > 0.0, probs * log_probs, torch.zeros_like(probs)
    )
    entropy = -plogp.sum(dim=-1)
    top1_prob = probs.max(dim=-1).values
    return entropy, top1_prob


# ---------------------------------------------------------------------------
# Directory-level cache
# ---------------------------------------------------------------------------


def _iter_generation_files(generations_dir: Path) -> list[Path]:
    """Return all ``.pt`` files under ``generations_dir`` in sorted order.

    Sorting is by POSIX-style relative path so the assigned ``idx`` is stable
    across operating systems and re-runs.
    """
    files = [p for p in generations_dir.rglob("*.pt") if p.is_file()]
    files.sort(key=lambda p: p.relative_to(generations_dir).as_posix())
    return files


def cache_scalars_for_directory(
    generations_dir: str | os.PathLike,
    cache_dir: str | os.PathLike,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    """Compute and persist entropy + top-1 probability for every generation.

    For each ``.pt`` file found recursively under ``generations_dir`` (sorted
    by relative path for determinism), loads the stored ``logits``, computes
    :func:`compute_token_entropy_and_top1`, and writes the result to
    ``{cache_dir}/{idx:05d}.pt`` with the schema described in the module
    docstring. The original generation file is **not** modified or deleted.

    Parameters
    ----------
    generations_dir : str | PathLike
        Root containing per-prompt ``.pt`` files (flat or nested).
    cache_dir : str | PathLike
        Output root for cached scalars. Created if missing.
    progress : bool, default True
        Show a ``tqdm`` progress bar when available.

    Returns
    -------
    dict
        ``{"cached": int, "errors": [(source_path, message), ...]}``.

    Notes
    -----
    Empty generations (``T == 0``) are still cached so that downstream code
    that relies on a one-to-one (generation, cache) correspondence works
    without special-casing.
    """
    gen_dir = Path(generations_dir)
    if not gen_dir.exists():
        raise FileNotFoundError(f"generations_dir does not exist: {gen_dir}")

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_generation_files(gen_dir)

    iterator: Iterable[Path] = files
    if progress and files:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(files, desc=f"cache[{gen_dir.name}]", unit="file")
        except ImportError:
            pass

    cached = 0
    errors: list[tuple[str, str]] = []

    for idx, gen_path in enumerate(iterator):
        rel = gen_path.relative_to(gen_dir).as_posix()
        try:
            payload = torch.load(gen_path, map_location="cpu", weights_only=False)
            logits: Tensor = payload["logits"]
            token_ids: Tensor = payload["token_ids"]

            if logits.shape[0] != token_ids.shape[0]:
                raise ValueError(
                    f"logits/token_ids length mismatch in {rel}: "
                    f"{tuple(logits.shape)} vs {tuple(token_ids.shape)}"
                )

            if logits.shape[0] == 0:
                # No generated tokens; emit empty tensors with the right shape.
                entropy = torch.empty((0,), dtype=torch.float32)
                top1_prob = torch.empty((0,), dtype=torch.float32)
            else:
                entropy, top1_prob = compute_token_entropy_and_top1(logits)

            cache_payload = {
                "entropy": entropy.to(torch.float32).contiguous(),
                "top1_prob": top1_prob.to(torch.float32).contiguous(),
                "token_ids": token_ids.to(torch.long).contiguous(),
                "source_path": rel,
            }
            out_path = out_dir / f"{idx:05d}.pt"
            torch.save(cache_payload, out_path)
            cached += 1
        except Exception as exc:  # noqa: BLE001 - record & continue
            errors.append((rel, repr(exc)))

    return {"cached": cached, "errors": errors}


def load_scalars(idx: int, cache_dir: str | os.PathLike) -> dict[str, Any]:
    """Load cached entropy / top-1 probability / token ids for one prompt.

    Parameters
    ----------
    idx : int
        Position index assigned by :func:`cache_scalars_for_directory`.
    cache_dir : str | PathLike
        Root containing ``{idx:05d}.pt`` cache files.

    Returns
    -------
    dict
        ``{"entropy": (T,) fp32, "top1_prob": (T,) fp32,
            "token_ids": (T,) Long, "source_path": str}``.

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    ValueError
        If ``idx`` is negative.
    """
    if idx < 0:
        raise ValueError(f"idx must be >= 0, got {idx}")
    path = Path(cache_dir) / f"{idx:05d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"cache file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


__all__ = [
    "compute_token_entropy_and_top1",
    "cache_scalars_for_directory",
    "load_scalars",
]
