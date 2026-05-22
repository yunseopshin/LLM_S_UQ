"""Per-token feature extractor for the Bayesian sentence-level UQ model.

Phase 2-1. Implements the learnable parameters
``ψ = (W, α, μ_0, log σ_0)`` and the deterministic feature map

    z_ℓ = [W · h_ℓ^agg,  ent_ℓ,  top1_ℓ] ∈ R^k,
    h_ℓ^agg = Σ_l α_l h_ℓ^(l),
    α_l = softmax(α)_l,
    k = p + 2.

This is the building block consumed by the inner Fisher-scoring loop
(Phase 3-1) and the auxiliary Bayesian regression head (Phase 4-2).
See ``research_document_v8.md`` Part VI for the math and CLAUDE.md
"Core Math" for the project-level summary.

Design notes
------------
* **Model-agnostic** — ``hidden_dim`` and ``num_layers`` are *required*
  ``__init__`` arguments; nothing here may assume Llama-3 dimensions.
* **Numerical safety** — all computation runs in fp32 even if the
  cached hidden states arrive as fp16 (see CLAUDE.md rule 10).
* **No in-place ops** — every tensor is constructed via functional
  ops so autograd through ψ stays clean.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "SentenceUQParams",
    "extract_token_features",
    "extract_sentence_token_features",
    "extract_sentence_aggregate_feature",
]


class SentenceUQParams(nn.Module):
    """Learnable feature-extractor parameters ψ = (W, α, μ_0, log σ_0).

    Parameters
    ----------
    hidden_dim : int
        LLM hidden-state dimension ``d`` (e.g. 4096 for Llama-3-8B,
        3584 for Gemma-2-9B). Read from ``model.config.hidden_size``;
        no default is provided on purpose.
    num_layers : int
        Number of layers retained for aggregation ``L_layers``. This is
        the size of ``selected_layers`` from Phase 1-1, *not* the full
        depth of the transformer.
    projection_dim : int, optional
        Output dimension ``p`` of the linear projection ``W``. Defaults
        to 64, matching Part VI §6.1.

    Attributes
    ----------
    W : nn.Linear
        Projection ``W ∈ R^{p × d}`` (``bias=False``).
    alpha : nn.Parameter
        Pre-softmax layer-mixing logits ``α ∈ R^{L_layers}``.
    mu_0 : nn.Parameter
        Prior mean ``μ_0 ∈ R^k`` with ``k = p + 2``.
    log_sigma_0 : nn.Parameter
        Log of the diagonal prior std ``log σ_0 ∈ R^k``.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if projection_dim <= 0:
            raise ValueError(
                f"projection_dim must be positive, got {projection_dim}"
            )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.projection_dim = projection_dim

        self.W = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.alpha = nn.Parameter(torch.zeros(num_layers))
        self.mu_0 = nn.Parameter(torch.zeros(projection_dim + 2))
        self.log_sigma_0 = nn.Parameter(torch.zeros(projection_dim + 2))

    @property
    def feature_dim(self) -> int:
        """Dimension ``k = p + 2`` of the per-token feature vector ``z_ℓ``."""
        return self.projection_dim + 2

    def get_Sigma_0(self) -> torch.Tensor:
        """Diagonal prior covariance ``Σ_0 = diag(exp(2 · log σ_0)) ∈ R^{k×k}``.

        Returns
        -------
        Tensor of shape ``(k, k)``.
        """
        return torch.diag(torch.exp(2.0 * self.log_sigma_0))

    def get_Sigma_0_inv(self) -> torch.Tensor:
        """Diagonal prior precision ``Σ_0^{-1} = diag(exp(-2 · log σ_0)) ∈ R^{k×k}``.

        Returns
        -------
        Tensor of shape ``(k, k)``.
        """
        return torch.diag(torch.exp(-2.0 * self.log_sigma_0))


def _validate_token_inputs(
    hidden_states: torch.Tensor,
    entropy: torch.Tensor,
    top1_prob: torch.Tensor,
    params: SentenceUQParams,
) -> None:
    if hidden_states.dim() != 3:
        raise ValueError(
            "hidden_states must be (T, num_layers, hidden_dim); "
            f"got shape {tuple(hidden_states.shape)}"
        )
    T, L, D = hidden_states.shape
    if L != params.num_layers:
        raise ValueError(
            f"hidden_states has num_layers={L} but params expects "
            f"{params.num_layers}"
        )
    if D != params.hidden_dim:
        raise ValueError(
            f"hidden_states has hidden_dim={D} but params expects "
            f"{params.hidden_dim}"
        )
    if entropy.shape != (T,):
        raise ValueError(
            f"entropy must have shape (T={T},); got {tuple(entropy.shape)}"
        )
    if top1_prob.shape != (T,):
        raise ValueError(
            f"top1_prob must have shape (T={T},); got {tuple(top1_prob.shape)}"
        )


def extract_token_features(
    hidden_states: torch.Tensor,
    entropy: torch.Tensor,
    top1_prob: torch.Tensor,
    params: SentenceUQParams,
) -> torch.Tensor:
    """Compute per-token feature vectors ``z_ℓ`` for an entire generation.

    Math (Part VI §6.1)::

        w = softmax(params.alpha)              ∈ R^{L_layers}
        h_ℓ^agg = Σ_l w_l · h_ℓ^(l)            ∈ R^d
        h_ℓ^proj = W · h_ℓ^agg                 ∈ R^p
        z_ℓ = concat([h_ℓ^proj, ent_ℓ, top1_ℓ]) ∈ R^k,  k = p + 2

    Parameters
    ----------
    hidden_states : Tensor of shape ``(T, num_layers, hidden_dim)``.
        Layer-wise hidden states for ``T`` generated tokens. May arrive
        as fp16; computation is promoted to fp32 internally.
    entropy : Tensor of shape ``(T,)``.
        Cached predictive entropy ``H_ℓ`` (Phase 1-3).
    top1_prob : Tensor of shape ``(T,)``.
        Cached top-1 probability ``p^(1)_ℓ`` (Phase 1-3).
    params : SentenceUQParams
        Learnable parameters ψ.

    Returns
    -------
    Tensor of shape ``(T, k)`` where ``k = params.feature_dim``.
    """
    _validate_token_inputs(hidden_states, entropy, top1_prob, params)

    # Promote to fp32 to keep numerics stable (CLAUDE.md rule 10).
    h = hidden_states.to(torch.float32)
    ent = entropy.to(torch.float32)
    top1 = top1_prob.to(torch.float32)

    w = F.softmax(params.alpha.to(torch.float32), dim=0)         # (L_layers,)
    h_agg = torch.einsum("l,tld->td", w, h)                       # (T, d)
    h_proj = params.W(h_agg)                                      # (T, p)

    z = torch.cat([h_proj, ent.unsqueeze(1), top1.unsqueeze(1)], dim=1)
    return z


def extract_sentence_token_features(
    hidden_states: torch.Tensor,
    entropy: torch.Tensor,
    top1_prob: torch.Tensor,
    token_range: Tuple[int, int],
    params: SentenceUQParams,
) -> torch.Tensor:
    """Extract ``{z_ℓ}_{ℓ ∈ s_j}`` for a single sentence ``s_j``.

    Parameters
    ----------
    hidden_states, entropy, top1_prob, params
        As in :func:`extract_token_features`. Inputs span the entire
        generation; this function slices to ``token_range``.
    token_range : tuple ``(start, end)``
        Half-open interval ``[start, end)`` of token indices belonging
        to the sentence, as returned by Phase 1-2 sentence splitting.

    Returns
    -------
    Tensor of shape ``(L_j, k)`` where ``L_j = end - start``.
    """
    start, end = token_range
    if not (isinstance(start, int) and isinstance(end, int)):
        raise TypeError(
            f"token_range must be (int, int); got ({type(start).__name__}, "
            f"{type(end).__name__})"
        )
    if start < 0 or end < start:
        raise ValueError(
            f"token_range must satisfy 0 <= start <= end; got ({start}, {end})"
        )
    T = hidden_states.shape[0]
    if end > T:
        raise ValueError(
            f"token_range end={end} exceeds sequence length T={T}"
        )

    h_slice = hidden_states[start:end]
    ent_slice = entropy[start:end]
    top1_slice = top1_prob[start:end]
    return extract_token_features(h_slice, ent_slice, top1_slice, params)


def extract_sentence_aggregate_feature(z_tokens: torch.Tensor) -> torch.Tensor:
    """Aggregate per-token features into a sentence-level vector.

    Used by the auxiliary Bayesian regression model (Part VIII): the
    aggregate concatenates the per-coordinate mean, std (population,
    not Bessel-corrected), and the last token's feature::

        ζ_j = concat([ mean_ℓ z_ℓ,  std_ℓ z_ℓ,  z_{ℓ = last} ]) ∈ R^{3k}

    Edge case ``L_j == 1``: std is set to zero rather than NaN.

    Parameters
    ----------
    z_tokens : Tensor of shape ``(L_j, k)``.
        Per-token features for a single sentence (Phase 2-1 output).

    Returns
    -------
    Tensor of shape ``(3 * k,)``.
    """
    if z_tokens.dim() != 2:
        raise ValueError(
            f"z_tokens must be 2-D (L_j, k); got shape "
            f"{tuple(z_tokens.shape)}"
        )
    L_j, k = z_tokens.shape
    if L_j == 0:
        raise ValueError("z_tokens must contain at least one token (L_j >= 1)")

    z = z_tokens.to(torch.float32)
    mean = z.mean(dim=0)                                          # (k,)
    if L_j == 1:
        std = torch.zeros(k, dtype=z.dtype, device=z.device)
    else:
        std = z.std(dim=0, unbiased=False)                        # (k,)
    last = z[-1]                                                  # (k,)
    return torch.cat([mean, std, last], dim=0)
