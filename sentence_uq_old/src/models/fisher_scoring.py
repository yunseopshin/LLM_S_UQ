"""Damped Fisher scoring inner loop for sentence-level Bayesian UQ.

Solves MAP estimation:
    θ* = argmax L̃(θ)
    L̃(θ) = Σ_j [F_j log μ̃_j + (1-F_j) log(1-μ̃_j)]
            - (1/2)(θ-μ_0)^T Σ_0^{-1} (θ-μ_0)

where μ̃_j = clamp(μ_j, ε, 1-ε) and μ_j = (1/L_j) Σ_{ℓ∈s_j} σ(θ^T z_ℓ).

All functions are fully differentiable (no detach) so that outer-loop
gradients can flow through the unrolled iterations to learnable parameters ψ.
"""

from __future__ import annotations

import warnings

import torch


def _compute_pi_and_mu(
    theta: torch.Tensor,
    all_z_tokens: list[torch.Tensor],
    eps: float = 1e-6,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Compute per-token π_ℓ and per-sentence μ̃_j.

    π_ℓ(θ) = σ(θ^T z_ℓ)
    μ_j(θ) = (1/L_j) Σ_{ℓ∈s_j} π_ℓ(θ)
    μ̃_j   = clamp(μ_j, ε, 1-ε)

    Parameters
    ----------
    theta : torch.Tensor
        Shape (k,).
    all_z_tokens : list[torch.Tensor]
        Length N list; each element has shape (L_j, k).
    eps : float
        Clipping bound for μ̃_j.

    Returns
    -------
    pi_list : list[torch.Tensor]
        Length N; each (L_j,) with per-token π_ℓ values.
    mu_tilde : torch.Tensor
        Shape (N,), clipped sentence-level means.
    """
    pi_list: list[torch.Tensor] = []
    mu_vals: list[torch.Tensor] = []
    for z_j in all_z_tokens:
        # z_j: (L_j, k), theta: (k,)
        logits = z_j @ theta  # (L_j,)
        pi_j = torch.sigmoid(logits)  # (L_j,)
        pi_list.append(pi_j)
        mu_j = pi_j.mean()  # scalar
        mu_vals.append(mu_j)
    if len(mu_vals) == 0:
        mu_tilde = torch.zeros(0, device=theta.device, dtype=theta.dtype)
    else:
        mu = torch.stack(mu_vals)  # (N,)
        mu_tilde = torch.clamp(mu, eps, 1.0 - eps)  # (N,)
    return pi_list, mu_tilde


def _compute_g(
    pi_list: list[torch.Tensor],
    all_z_tokens: list[torch.Tensor],
) -> torch.Tensor:
    """Compute per-sentence gradient factors g_j.

    g_j = (1/L_j) Σ_{ℓ∈s_j} π_ℓ(1-π_ℓ) z_ℓ

    Parameters
    ----------
    pi_list : list[torch.Tensor]
        Length N; each (L_j,).
    all_z_tokens : list[torch.Tensor]
        Length N; each (L_j, k).

    Returns
    -------
    torch.Tensor
        Shape (N, k), the g_j vectors stacked.
    """
    g_list: list[torch.Tensor] = []
    for pi_j, z_j in zip(pi_list, all_z_tokens):
        # pi_j: (L_j,), z_j: (L_j, k)
        weight = pi_j * (1.0 - pi_j)  # (L_j,)
        # weighted mean: (1/L_j) Σ weight_ℓ * z_ℓ
        g_j = (weight.unsqueeze(1) * z_j).mean(dim=0)  # (k,)
        g_list.append(g_j)
    return torch.stack(g_list)  # (N, k)


def _compute_grad_and_fisher(
    theta: torch.Tensor,
    all_z_tokens: list[torch.Tensor],
    all_F: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient ∇L̃ and Fisher-type Hessian H.

    Gradient:
        ∇L̃ = -Σ_0^{-1}(θ-μ_0) + Σ_j R_j g_j
        R_j = (F_j - μ̃_j) / (μ̃_j(1-μ̃_j))

    Fisher-type precision (expected information):
        H = Σ_0^{-1} + Σ_j (1/(μ̃_j(1-μ̃_j))) g_j g_j^T

    Parameters
    ----------
    theta : torch.Tensor
        Shape (k,), current parameter estimate.
    all_z_tokens : list[torch.Tensor]
        Length N; each (L_j, k).
    all_F : torch.Tensor
        Shape (N,), binary factuality labels.
    mu_0 : torch.Tensor
        Shape (k,), prior mean.
    Sigma_0_inv : torch.Tensor
        Shape (k, k), prior precision.
    eps : float
        Clipping bound for numerical safety.

    Returns
    -------
    grad : torch.Tensor
        Shape (k,), gradient of clipped log-posterior.
    H_fisher : torch.Tensor
        Shape (k, k), Fisher-type precision matrix.
    """
    pi_list, mu_tilde = _compute_pi_and_mu(theta, all_z_tokens, eps)
    g = _compute_g(pi_list, all_z_tokens)  # (N, k)

    # Residuals R_j = (F_j - μ̃_j) / (μ̃_j(1-μ̃_j))
    var_mu = mu_tilde * (1.0 - mu_tilde)  # (N,)
    R = (all_F - mu_tilde) / var_mu  # (N,)

    # Prior contribution to gradient: -Σ_0^{-1}(θ - μ_0)
    prior_grad = -Sigma_0_inv @ (theta - mu_0)  # (k,)

    # Data contribution: Σ_j R_j g_j
    data_grad = (R.unsqueeze(1) * g).sum(dim=0)  # (k,)

    grad = prior_grad + data_grad  # (k,)

    # Fisher-type precision: Σ_0^{-1} + Σ_j (1/var_μ_j) g_j g_j^T
    inv_var = 1.0 / var_mu  # (N,)
    # Weighted outer product sum: Σ_j w_j g_j g_j^T = (g * sqrt(w))^T (g * sqrt(w))
    g_weighted = g * torch.sqrt(inv_var).unsqueeze(1)  # (N, k)
    H_fisher = Sigma_0_inv + g_weighted.t() @ g_weighted  # (k, k)

    return grad, H_fisher


def _compute_clipped_objective(
    theta: torch.Tensor,
    all_z_tokens: list[torch.Tensor],
    all_F: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute clipped log-posterior L̃(θ).

    L̃(θ) = Σ_j [F_j log μ̃_j + (1-F_j) log(1-μ̃_j)]
            - (1/2)(θ-μ_0)^T Σ_0^{-1} (θ-μ_0)

    Parameters
    ----------
    theta : torch.Tensor
        Shape (k,).
    all_z_tokens : list[torch.Tensor]
        Length N; each (L_j, k).
    all_F : torch.Tensor
        Shape (N,), binary factuality labels.
    mu_0 : torch.Tensor
        Shape (k,), prior mean.
    Sigma_0_inv : torch.Tensor
        Shape (k, k), prior precision.
    eps : float
        Clipping bound.

    Returns
    -------
    torch.Tensor
        Scalar, the clipped log-posterior value.
    """
    _, mu_tilde = _compute_pi_and_mu(theta, all_z_tokens, eps)

    # Log-likelihood: Σ_j [F_j log μ̃_j + (1-F_j) log(1-μ̃_j)]
    log_lik = (all_F * torch.log(mu_tilde)
               + (1.0 - all_F) * torch.log(1.0 - mu_tilde)).sum()

    # Prior: -(1/2)(θ-μ_0)^T Σ_0^{-1} (θ-μ_0)
    diff = theta - mu_0  # (k,)
    log_prior = -0.5 * diff @ Sigma_0_inv @ diff

    return log_lik + log_prior


def fisher_scoring_map(
    all_z_tokens: list[torch.Tensor],
    all_F: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    num_iters: int = 15,
    eps: float = 1e-6,
    lambda_init: float = 1e-4,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Damped Fisher scoring MAP estimation — fully differentiable.

    All tensor operations preserve the computation graph so that
    outer-loop gradients (bilevel optimization on ψ) can flow through
    the unrolled iterations.

    Algorithm:
        θ ← μ_0
        for each iteration:
            grad, H = compute_grad_and_fisher(θ, ...)
            δ = solve(H + λI, grad)
            θ_new = θ + δ
            if L̃(θ_new) > L̃(θ):  accept, reduce λ
            else:                    reject, increase λ

    Parameters
    ----------
    all_z_tokens : list[torch.Tensor]
        Length N; each (L_j, k).
    all_F : torch.Tensor
        Shape (N,), binary factuality labels.
    mu_0 : torch.Tensor
        Shape (k,), prior mean.
    Sigma_0_inv : torch.Tensor
        Shape (k, k), prior precision.
    num_iters : int
        Maximum number of iterations.
    eps : float
        Clipping bound.
    lambda_init : float
        Initial damping factor.
    verbose : bool
        If True, print iteration info.

    Returns
    -------
    theta : torch.Tensor
        Shape (k,), MAP estimate.
    H_final : torch.Tensor
        Shape (k, k), Fisher-type precision at the MAP estimate.
    """
    k = mu_0.shape[0]
    device = mu_0.device
    dtype = mu_0.dtype

    theta = mu_0.clone()  # (k,) — keeps gradient connection to mu_0
    lam = lambda_init
    prev_obj = _compute_clipped_objective(
        theta, all_z_tokens, all_F, mu_0, Sigma_0_inv, eps
    )

    for it in range(num_iters):
        grad, H = _compute_grad_and_fisher(
            theta, all_z_tokens, all_F, mu_0, Sigma_0_inv, eps
        )

        # Damped solve: δ = (H + λI)^{-1} grad
        eye_k = torch.eye(k, device=device, dtype=dtype)
        try:
            delta = torch.linalg.solve(H + lam * eye_k, grad)
        except torch._C._LinAlgError:
            lam *= 10.0
            if verbose:
                warnings.warn(
                    f"Fisher scoring iter {it}: linalg solve failed, "
                    f"increasing lambda to {lam:.2e}"
                )
            continue

        theta_new = theta + delta
        new_obj = _compute_clipped_objective(
            theta_new, all_z_tokens, all_F, mu_0, Sigma_0_inv, eps
        )

        if new_obj > prev_obj + 1e-8:
            theta = theta_new
            prev_obj = new_obj
            lam = max(lam / 2.0, 1e-8)
            if verbose:
                print(
                    f"  iter {it}: obj={new_obj.item():.6f}, "
                    f"lambda={lam:.2e} [accepted]"
                )
        else:
            lam *= 10.0
            if verbose:
                print(
                    f"  iter {it}: obj={new_obj.item():.6f} <= "
                    f"{prev_obj.item():.6f}, lambda={lam:.2e} [rejected]"
                )
            if lam > 1e10:
                if verbose:
                    warnings.warn(
                        "Fisher scoring: lambda exceeded 1e10, stopping early."
                    )
                break

    # Recompute Fisher at final theta
    _, H_final = _compute_grad_and_fisher(
        theta, all_z_tokens, all_F, mu_0, Sigma_0_inv, eps
    )
    return theta, H_final


def fisher_scoring_map_detached(
    all_z_tokens: list[torch.Tensor],
    all_F: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    num_iters: int = 15,
    eps: float = 1e-6,
    lambda_init: float = 1e-4,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference-only Fisher scoring MAP estimation (no gradient tracking).

    Identical to :func:`fisher_scoring_map` but runs inside
    ``torch.no_grad()`` for faster inference.

    Parameters
    ----------
    all_z_tokens : list[torch.Tensor]
        Length N; each (L_j, k).
    all_F : torch.Tensor
        Shape (N,), binary factuality labels.
    mu_0 : torch.Tensor
        Shape (k,), prior mean.
    Sigma_0_inv : torch.Tensor
        Shape (k, k), prior precision.
    num_iters : int
        Maximum number of iterations.
    eps : float
        Clipping bound.
    lambda_init : float
        Initial damping factor.
    verbose : bool
        If True, print iteration info.

    Returns
    -------
    theta : torch.Tensor
        Shape (k,), MAP estimate (no grad).
    H_final : torch.Tensor
        Shape (k, k), Fisher-type precision at MAP (no grad).
    """
    with torch.no_grad():
        return fisher_scoring_map(
            all_z_tokens,
            all_F,
            mu_0,
            Sigma_0_inv,
            num_iters=num_iters,
            eps=eps,
            lambda_init=lambda_init,
            verbose=verbose,
        )
