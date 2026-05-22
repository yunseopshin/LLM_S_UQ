"""Feature extractor for sentence-level Bayesian UQ.

Computes per-token features:
    z_ℓ = [W · h_ℓ^agg, entropy_ℓ, top1_ℓ] ∈ R^k
where
    h_ℓ^agg = Σ_l α_l · h_ℓ^(l),  α_l = softmax(α)_l
    k = projection_dim + 2
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SentenceUQParams(nn.Module):
    """Learnable parameters shared across the Bayesian UQ model.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension d of each LLM layer (default 4096 for Llama-3-8B).
    num_layers : int
        Number of LLM layers used for aggregation L_layers.
    projection_dim : int
        Projection dimension p.  Feature dim k = p + 2.

    Attributes
    ----------
    W : nn.Linear
        Weight matrix W ∈ R^{p × d} (no bias).
    alpha : nn.Parameter
        Layer mixing weights before softmax, α ∈ R^{L_layers}.
    mu_0 : nn.Parameter
        Prior mean μ_0 ∈ R^k.
    log_sigma_0 : nn.Parameter
        Log prior std log σ_0 ∈ R^k.
    feature_dim : int
        k = projection_dim + 2.
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_layers: int = 8,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.projection_dim = projection_dim
        self.feature_dim = projection_dim + 2  # k

        self.W = nn.Linear(hidden_dim, projection_dim, bias=False)
        self.alpha = nn.Parameter(torch.zeros(num_layers))  # before softmax
        self.mu_0 = nn.Parameter(torch.zeros(self.feature_dim))
        self.log_sigma_0 = nn.Parameter(torch.zeros(self.feature_dim))

    def get_Sigma_0_inv(self) -> torch.Tensor:
        """Return prior precision matrix Σ_0^{-1} = diag(exp(-2 log σ_0)).

        Returns
        -------
        torch.Tensor
            Shape (k, k), diagonal precision matrix.
        """
        return torch.diag(torch.exp(-2.0 * self.log_sigma_0))

    def get_Sigma_0(self) -> torch.Tensor:
        """Return prior covariance Σ_0 = diag(exp(2 log σ_0)).

        Returns
        -------
        torch.Tensor
            Shape (k, k), diagonal covariance matrix.
        """
        return torch.diag(torch.exp(2.0 * self.log_sigma_0))


def extract_token_features(
    hidden_states: torch.Tensor,
    entropy: torch.Tensor,
    top1_prob: torch.Tensor,
    params: SentenceUQParams,
) -> torch.Tensor:
    """Compute per-token feature vectors z_ℓ for all T tokens.

    z_ℓ = [W · h_ℓ^agg, entropy_ℓ, top1_ℓ] ∈ R^k

    where h_ℓ^agg = Σ_l α_l · h_ℓ^(l) with α_l = softmax(α)_l.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape (T, num_layers, hidden_dim), fp32.
    entropy : torch.Tensor
        Shape (T,), token-level entropy values, fp32.
    top1_prob : torch.Tensor
        Shape (T,), top-1 probability per token, fp32.
    params : SentenceUQParams
        Model parameters containing W, alpha.

    Returns
    -------
    torch.Tensor
        Shape (T, k) where k = projection_dim + 2.
    """
    T, num_layers, hidden_dim = hidden_states.shape
    assert num_layers == params.num_layers, (
        f"num_layers mismatch: hidden_states has {num_layers}, "
        f"params has {params.num_layers}"
    )

    w = torch.softmax(params.alpha, dim=0)  # (num_layers,)
    h_agg = torch.einsum("l,tlh->th", w, hidden_states)  # (T, hidden_dim)
    h_proj = params.W(h_agg)  # (T, projection_dim)
    z = torch.cat(
        [h_proj, entropy.unsqueeze(1), top1_prob.unsqueeze(1)],
        dim=1,
    )  # (T, k)
    return z


def extract_sentence_token_features(
    hidden_states: torch.Tensor,
    entropy: torch.Tensor,
    top1_prob: torch.Tensor,
    token_range: tuple[int, int],
    params: SentenceUQParams,
) -> torch.Tensor:
    """Compute per-token feature vectors for a single sentence slice.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape (T, num_layers, hidden_dim), fp32.
    entropy : torch.Tensor
        Shape (T,), fp32.
    top1_prob : torch.Tensor
        Shape (T,), fp32.
    token_range : tuple[int, int]
        (start, end) indices (end exclusive) identifying the sentence tokens.
    params : SentenceUQParams
        Model parameters.

    Returns
    -------
    torch.Tensor
        Shape (L_j, k) where L_j = end - start.

    Raises
    ------
    ValueError
        If token_range is empty (start >= end).
    """
    start, end = token_range
    if start >= end:
        raise ValueError(
            f"token_range must be non-empty, got ({start}, {end})"
        )

    hs_slice = hidden_states[start:end]      # (L_j, num_layers, hidden_dim)
    ent_slice = entropy[start:end]           # (L_j,)
    top1_slice = top1_prob[start:end]        # (L_j,)

    return extract_token_features(hs_slice, ent_slice, top1_slice, params)


def extract_sentence_aggregate_feature(z_tokens: torch.Tensor) -> torch.Tensor:
    """Compute sentence-level aggregate feature for the auxiliary model.

    Concatenates [mean, std, last] of token features.
    When L_j == 1, std is set to zeros.

    Parameters
    ----------
    z_tokens : torch.Tensor
        Shape (L_j, k), per-token features for one sentence.

    Returns
    -------
    torch.Tensor
        Shape (3k,).
    """
    L_j, k = z_tokens.shape
    mean = z_tokens.mean(dim=0)           # (k,)
    if L_j == 1:
        std = torch.zeros(k, dtype=z_tokens.dtype, device=z_tokens.device)
    else:
        std = z_tokens.std(dim=0, unbiased=True)  # (k,)
    last = z_tokens[-1]                   # (k,)
    return torch.cat([mean, std, last], dim=0)  # (3k,)
