"""Damped Fisher-scoring inner loop for the Bayesian sentence-level UQ model.

Phase 3-1. Implements the binomial-likelihood MAP estimator that the
outer trainer (Phase 4-1) backpropagates through. See
``research_document_v8.md`` Parts III and VII for the math and
``prompts/phase_3_1_fisher_scoring.md`` for the spec.

Mathematical definition
-----------------------
Per-token latent factuality::

    π_ℓ(θ) = σ(θᵀ z_ℓ)

Sentence factuality (token average)::

    μ_j(θ) = (1 / L_j) Σ_{ℓ∈s_j} π_ℓ(θ)

Clipped binomial log-posterior (skip sentences with ``m_j = 0``)::

    L̃(θ) = Σ_j [ K_j log μ̃_j + (m_j - K_j) log(1 - μ̃_j) ]
            - 0.5 (θ - μ_0)ᵀ Σ_0⁻¹ (θ - μ_0),
    μ̃_j  = clip(μ_j, ε, 1 - ε)

Gradient::

    ∇L̃ = -Σ_0⁻¹ (θ - μ_0) + Σ_j R_j^bin · g_j,
    R_j^bin = (K_j - m_j μ̃_j) / (μ̃_j (1 - μ̃_j)),
    g_j     = (1 / L_j) Σ_ℓ π_ℓ (1 - π_ℓ) z_ℓ

Fisher-type precision (m_j-weighted)::

    H_fisher = Σ_0⁻¹ + Σ_j (m_j / (μ̃_j (1 - μ̃_j))) · g_j g_jᵀ

Damped Fisher-scoring update with adaptive damping λ::

    θ ← θ + (H_fisher + λ I)⁻¹ ∇L̃

Bernoulli is recovered when ``m_j = 1`` and ``K_j ∈ {0, 1}``.

Design notes
------------
* **Differentiable** — ``fisher_scoring_map`` and the two helpers must
  be backprop-friendly (no ``.detach()`` on the optimization path),
  because the outer trainer (Phase 4-1) backprops through the unrolled
  Fisher loop to update ψ = (W, α, μ_0, log σ_0).
* **No in-place ops** on autograd-tracked tensors.
* **Skip ``m_j = 0`` sentences** consistently in grad / Hessian /
  objective (CLAUDE.md rule 8).
* The damping schedule follows the spec: success → λ ← max(λ/2, 1e-8);
  failure or linear-solve failure → λ ← 10λ; give up if λ > 1e10.
* ``fisher_scoring_map_detached`` wraps the same algorithm in
  ``torch.no_grad()`` for inference-only callers.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


__all__ = [
    "_compute_clipped_objective",
    "_compute_grad_and_fisher",
    "fisher_scoring_map",
    "fisher_scoring_map_detached",
]


def _compute_grad_and_fisher(
    theta: torch.Tensor,
    all_z_tokens: List[torch.Tensor],
    all_K: torch.Tensor,
    all_m: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gradient of L̃ and binomial Fisher-type precision at ``theta``.

    Parameters
    ----------
    theta : Tensor of shape ``(k,)``.
        Current latent direction.
    all_z_tokens : list of N tensors, each of shape ``(L_j, k)``.
        Per-sentence per-token features from Phase 2-1.
    all_K : Tensor of shape ``(N,)``, integer dtype.
        Supported-atom counts per sentence.
    all_m : Tensor of shape ``(N,)``, integer dtype.
        Total atomic-fact counts per sentence. ``m_j = 0`` sentences are
        skipped entirely.
    mu_0 : Tensor of shape ``(k,)``.
        Prior mean.
    Sigma_0_inv : Tensor of shape ``(k, k)``.
        Prior precision.
    eps : float
        Clipping bound for ``μ_j`` to avoid log singularities.

    Returns
    -------
    grad : Tensor of shape ``(k,)``.
        ``∇_θ L̃(θ)``.
    H_fisher : Tensor of shape ``(k, k)``.
        Binomial Fisher-type precision.
    """
    if len(all_K) != len(all_m) or len(all_K) != len(all_z_tokens):
        raise ValueError(
            "all_z_tokens, all_K, all_m must have the same length; "
            f"got {len(all_z_tokens)}, {len(all_K)}, {len(all_m)}"
        )

    diff = theta - mu_0
    grad = -(Sigma_0_inv @ diff)
    H = Sigma_0_inv

    for j in range(len(all_K)):
        m_j_int = int(all_m[j].item()) if torch.is_tensor(all_m[j]) else int(all_m[j])
        if m_j_int == 0:
            continue

        z_j = all_z_tokens[j]
        if z_j.dim() != 2 or z_j.shape[1] != theta.shape[0]:
            raise ValueError(
                f"all_z_tokens[{j}] must be (L_j, k={theta.shape[0]}); "
                f"got shape {tuple(z_j.shape)}"
            )

        K_j = all_K[j].to(theta.dtype)
        m_j = all_m[j].to(theta.dtype)

        logits = z_j @ theta                                  # (L_j,)
        pi_j = torch.sigmoid(logits)                          # (L_j,)
        mu_raw = pi_j.mean()                                  # ()
        mu_clamped = torch.clamp(mu_raw, eps, 1.0 - eps)      # ()

        weights = pi_j * (1.0 - pi_j)                         # (L_j,)
        g_j = (weights.unsqueeze(1) * z_j).mean(dim=0)        # (k,)

        denom = mu_clamped * (1.0 - mu_clamped)
        R_j = (K_j - m_j * mu_clamped) / denom

        grad = grad + R_j * g_j
        H = H + (m_j / denom) * torch.outer(g_j, g_j)

    return grad, H


def _compute_clipped_objective(
    theta: torch.Tensor,
    all_z_tokens: List[torch.Tensor],
    all_K: torch.Tensor,
    all_m: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scalar clipped binomial log-posterior ``L̃(θ)``.

    Sentences with ``m_j = 0`` are skipped (contribute zero).

    Parameters
    ----------
    theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
        See :func:`_compute_grad_and_fisher`.

    Returns
    -------
    Tensor of shape ``()``.
    """
    if len(all_K) != len(all_m) or len(all_K) != len(all_z_tokens):
        raise ValueError(
            "all_z_tokens, all_K, all_m must have the same length; "
            f"got {len(all_z_tokens)}, {len(all_K)}, {len(all_m)}"
        )

    diff = theta - mu_0
    obj = -0.5 * (diff @ (Sigma_0_inv @ diff))

    for j in range(len(all_K)):
        m_j_int = int(all_m[j].item()) if torch.is_tensor(all_m[j]) else int(all_m[j])
        if m_j_int == 0:
            continue

        z_j = all_z_tokens[j]
        K_j = all_K[j].to(theta.dtype)
        m_j = all_m[j].to(theta.dtype)

        pi_j = torch.sigmoid(z_j @ theta)
        mu_clamped = torch.clamp(pi_j.mean(), eps, 1.0 - eps)

        obj = obj + K_j * torch.log(mu_clamped) + (m_j - K_j) * torch.log(1.0 - mu_clamped)

    return obj


def _fisher_scoring_core(
    all_z_tokens: List[torch.Tensor],
    all_K: torch.Tensor,
    all_m: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    num_iters: int,
    eps: float,
    lambda_init: float,
    verbose: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared loop body for the autograd-tracked and detached variants."""
    k = mu_0.shape[0]
    if Sigma_0_inv.shape != (k, k):
        raise ValueError(
            f"Sigma_0_inv must be ({k}, {k}); got {tuple(Sigma_0_inv.shape)}"
        )

    theta = mu_0.clone()
    lam = float(lambda_init)
    eye = torch.eye(k, device=mu_0.device, dtype=mu_0.dtype)

    prev_obj = _compute_clipped_objective(
        theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
    )

    for it in range(num_iters):
        grad, H = _compute_grad_and_fisher(
            theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
        )

        try:
            delta = torch.linalg.solve(H + lam * eye, grad)
        except RuntimeError:
            lam *= 10.0
            if lam > 1e10:
                break
            continue

        theta_new = theta + delta
        new_obj = _compute_clipped_objective(
            theta_new, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
        )

        if new_obj.item() > prev_obj.item():
            theta = theta_new
            prev_obj = new_obj
            lam = max(lam / 2.0, 1e-8)
            if verbose:
                print(f"[fisher_scoring] iter {it}: obj={new_obj.item():.6f} lam={lam:.2e}")
        else:
            lam *= 10.0
            if verbose:
                print(f"[fisher_scoring] iter {it}: rejected, lam->{lam:.2e}")
            if lam > 1e10:
                break

    _, H_final = _compute_grad_and_fisher(
        theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
    )
    return theta, H_final


def fisher_scoring_map(
    all_z_tokens: List[torch.Tensor],
    all_K: torch.Tensor,
    all_m: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    num_iters: int = 15,
    eps: float = 1e-6,
    lambda_init: float = 1e-4,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Damped Fisher-scoring MAP for the binomial latent model.

    Unrolled, autograd-tracked variant — used inside the outer bilevel
    loop so that ``∂θ̂ / ∂ψ`` is available via backprop through the
    iterations. For inference use :func:`fisher_scoring_map_detached`.

    Update rule per iteration::

        θ ← θ + (H_fisher(θ) + λ I)⁻¹ ∇L̃(θ)

    with adaptive damping:

    * ``new_obj > prev_obj`` → accept, ``λ ← max(λ/2, 1e-8)``;
    * otherwise reject and ``λ ← 10 λ``;
    * if ``torch.linalg.solve`` raises, ``λ ← 10 λ`` and retry;
    * stop when ``λ > 1e10``.

    Parameters
    ----------
    all_z_tokens : list of N tensors of shape ``(L_j, k)``.
    all_K : Tensor of shape ``(N,)``, integer dtype.
    all_m : Tensor of shape ``(N,)``, integer dtype. ``m_j = 0`` rows skipped.
    mu_0 : Tensor of shape ``(k,)``. Prior mean (initial θ).
    Sigma_0_inv : Tensor of shape ``(k, k)``. Prior precision.
    num_iters : int
        Max outer iterations. Keep moderate (10–15) to bound the memory
        cost of the unrolled backward pass (CLAUDE.md rule 9).
    eps : float
        Clipping bound for ``μ_j``.
    lambda_init : float
        Initial damping.
    verbose : bool
        Print per-iteration status.

    Returns
    -------
    theta_hat : Tensor of shape ``(k,)``.
    H_fisher_final : Tensor of shape ``(k, k)``.
        Fisher-type precision recomputed at ``theta_hat``.
    """
    return _fisher_scoring_core(
        all_z_tokens=all_z_tokens,
        all_K=all_K,
        all_m=all_m,
        mu_0=mu_0,
        Sigma_0_inv=Sigma_0_inv,
        num_iters=num_iters,
        eps=eps,
        lambda_init=lambda_init,
        verbose=verbose,
    )


def fisher_scoring_map_detached(
    all_z_tokens: List[torch.Tensor],
    all_K: torch.Tensor,
    all_m: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    num_iters: int = 15,
    eps: float = 1e-6,
    lambda_init: float = 1e-4,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inference-only Fisher-scoring MAP — same algorithm under ``no_grad``.

    See :func:`fisher_scoring_map` for the algorithm and parameters.
    Returned tensors are detached from the autograd graph.
    """
    with torch.no_grad():
        theta, H = _fisher_scoring_core(
            all_z_tokens=all_z_tokens,
            all_K=all_K,
            all_m=all_m,
            mu_0=mu_0,
            Sigma_0_inv=Sigma_0_inv,
            num_iters=num_iters,
            eps=eps,
            lambda_init=lambda_init,
            verbose=verbose,
        )
    return theta, H
