"""Token-entropy baseline for sentence-level factuality UQ.

Phase 5-1 baseline #1. The simplest possible reference: for each
sentence, average the per-token predictive entropy ``H_ℓ`` cached in
Phase 1-3. Higher mean entropy is treated as *more uncertain*.

This baseline produces an unbounded uncertainty score, not a
probability. The Phase 5-1 runner (``scripts/05_baselines.py``)
normalises / inverts the score before comparing with ``U_j = K_j/m_j``
or the binary strict-factuality target ``A_j = 1{K_j = m_j}``.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
from torch import Tensor


__all__ = ["compute_token_entropy_baseline"]


_Range = Union[Tuple[int, int], Sequence[int]]


def compute_token_entropy_baseline(
    entropy: Tensor,
    token_range: _Range,
) -> float:
    """Return the mean per-token entropy over ``token_range`` (half-open).

    Math::

        score_j = (1 / L_j) Σ_{ℓ ∈ s_j} H_ℓ,
        L_j     = end - start,
        s_j     = tokens with indices in ``[start, end)``.

    Parameters
    ----------
    entropy : Tensor of shape ``(T,)``.
        Cached per-token predictive entropy from Phase 1-3.
    token_range : tuple ``(start, end)``
        Half-open sentence span ``[start, end)``. ``end`` must satisfy
        ``end <= T``; ``start == end`` raises (an empty sentence has no
        well-defined mean entropy).

    Returns
    -------
    float
        Mean entropy in nats (matches the unit produced by
        :mod:`src.features.cached_scalars`). Higher → more uncertain.
    """
    if not torch.is_tensor(entropy):
        raise TypeError(
            f"entropy must be a torch.Tensor; got {type(entropy).__name__}"
        )
    if entropy.dim() != 1:
        raise ValueError(
            f"entropy must be 1-D (T,); got shape {tuple(entropy.shape)}"
        )

    start, end = int(token_range[0]), int(token_range[1])
    T = int(entropy.shape[0])
    if start < 0 or end < start:
        raise ValueError(
            f"token_range must satisfy 0 <= start <= end; got ({start}, {end})"
        )
    if end > T:
        raise ValueError(
            f"token_range end={end} exceeds sequence length T={T}"
        )
    if end == start:
        raise ValueError(
            "token_range covers zero tokens; cannot compute mean entropy"
        )

    slice_ = entropy[start:end].to(torch.float32)
    return float(slice_.mean().item())
