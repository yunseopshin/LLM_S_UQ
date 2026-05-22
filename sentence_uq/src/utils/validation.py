"""Data validation utilities.

Phase 7-3 (code-review fix 4). Centralises the binomial-count
sanity check used at every entry point that consumes ``(K_j, m_j)``
pairs (the bilevel trainer, the evaluation script, and the Fisher
scoring inner loop). Invalid counts (``K_j > m_j``, ``K_j < 0``, or
``m_j < 0``) are otherwise silently accepted and can hide annotation
bugs that corrupt experimental results.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch


__all__ = ["validate_binomial_counts"]


def validate_binomial_counts(
    K: Union[torch.Tensor, np.ndarray],
    m: Union[torch.Tensor, np.ndarray],
    context: str = "",
) -> None:
    """Raise ``ValueError`` if binomial counts are invalid.

    Checks ``0 ≤ K_j ≤ m_j`` for every sentence and ``m_j ≥ 0``.

    Parameters
    ----------
    K : torch.Tensor or numpy.ndarray
        Per-sentence supported-atom count.
    m : torch.Tensor or numpy.ndarray
        Per-sentence total atomic-fact count.
    context : str, optional
        Caller identifier prepended to the error message.

    Raises
    ------
    ValueError
        If any entry violates ``K_j >= 0``, ``m_j >= 0``, or ``K_j <= m_j``.
    """
    prefix = f"[{context}] " if context else ""
    if isinstance(K, torch.Tensor):
        if torch.any(K < 0):
            raise ValueError(f"{prefix}Found K_j < 0")
        if torch.any(m < 0):
            raise ValueError(f"{prefix}Found m_j < 0")
        if torch.any(K > m):
            raise ValueError(f"{prefix}Found K_j > m_j")
    else:
        if np.any(K < 0):
            raise ValueError(f"{prefix}Found K_j < 0")
        if np.any(m < 0):
            raise ValueError(f"{prefix}Found m_j < 0")
        if np.any(K > m):
            raise ValueError(f"{prefix}Found K_j > m_j")
