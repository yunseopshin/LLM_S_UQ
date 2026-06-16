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

Epsilon-stabilized gradient (§7.2.2 of research_document_v8)::

    ∇L̃ ≈ -Σ_0⁻¹ (θ - μ_0) + Σ_j R_j^ε · g_j,
    R_j^ε   = (K_j - m_j μ_j) / max(μ_j (1 - μ_j), ε),
    g_j     = (1 / L_j) Σ_ℓ π_ℓ (1 - π_ℓ) z_ℓ

Note: this is NOT the true gradient of the clipped objective L̃. At
clipping boundaries the true clipped gradient would be zero for the
affected sentence; the implementation instead stabilises only the
denominator, keeping every sentence contributing to the gradient. The
clipped objective L̃ is used only for line-search accept/reject.

Fisher-type precision (m_j-weighted, denominator stabilised the same way)::

    H_fisher = Σ_0⁻¹ + Σ_j (m_j / max(μ_j (1 - μ_j), ε)) · g_j g_jᵀ

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

from typing import List, Optional, Tuple

import torch

from src.models.likelihood import BinomialLikelihood, Likelihood
from src.utils.validation import validate_binomial_counts


__all__ = [
    "_compute_clipped_objective",
    "_compute_grad_and_fisher",
    "fisher_scoring_map",
    "fisher_scoring_map_detached",
    "_last_diagnostics",
]


#: Diagnostics from the most recent ``_compute_grad_and_fisher`` call.
#:
#: Populated as a side effect so that callers (the trainer's epoch loop,
#: notebooks) can read post-hoc statistics without changing the public
#: return signature. Keys:
#:
#: * ``boundary_fraction`` — fraction of ``m_j > 0`` sentences whose raw
#:   ``μ_j`` falls outside ``(eps, 1 - eps)`` (i.e. where the denominator
#:   stabiliser is active).
#: * ``boundary_count``    — numerator of the above ratio.
#: * ``total_sentences``   — denominator of the above ratio (sentences with
#:   ``m_j > 0`` that contributed to the gradient).
_last_diagnostics: dict = {
    "boundary_fraction": 0.0,
    "boundary_count": 0,
    "total_sentences": 0,
    "phi": None,
    "rho": 0.0,
}


def _compute_grad_and_fisher(
    theta: torch.Tensor,
    all_z_tokens: List[torch.Tensor],
    all_K: torch.Tensor,
    all_m: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    eps: float = 1e-6,
    likelihood: Optional[Likelihood] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Epsilon-stabilised gradient and binomial Fisher-type precision at ``theta``.

    Note: the gradient returned is epsilon-stabilised (denominator-only
    clipping), not the true gradient of the clipped objective L̃. See
    §7.2.2 of ``research_document_v8.md``. At the boundary the true
    clipped gradient would be zero for the affected sentence, but this
    implementation keeps every sentence contributing by stabilising the
    ``μ_j (1 - μ_j)`` denominator. The clipped objective L̃ itself is
    computed by :func:`_compute_clipped_objective` and is used only by
    the line-search.

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
        Denominator-stabilisation bound for ``μ_j (1 - μ_j)``.
    likelihood : Likelihood, optional
        Per-sentence observation model supplying the residual ``R_j`` and
        the precision weight ``w_j``. ``None`` (default) constructs a
        :class:`~src.models.likelihood.BinomialLikelihood`, so every
        pre-existing call site is byte-for-byte unchanged.

    Returns
    -------
    grad : Tensor of shape ``(k,)``.
        Epsilon-stabilised ``∇_θ L̃(θ)`` (see note above).
    H_fisher : Tensor of shape ``(k, k)``.
        Fisher-type precision (Binomial: expected information; Beta-
        Binomial: observed information), denominator stabilised the same
        way.
    """
    if len(all_K) != len(all_m) or len(all_K) != len(all_z_tokens):
        raise ValueError(
            "all_z_tokens, all_K, all_m must have the same length; "
            f"got {len(all_z_tokens)}, {len(all_K)}, {len(all_m)}"
        )

    if likelihood is None:
        likelihood = BinomialLikelihood(eps=eps)

    diff = theta - mu_0
    grad = -(Sigma_0_inv @ diff)
    H = Sigma_0_inv

    boundary_count = 0
    total_count = 0

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

        total_count += 1
        mu_raw_val = float(mu_raw.detach().item())
        if mu_raw_val < eps or mu_raw_val > 1.0 - eps:
            boundary_count += 1

        weights = pi_j * (1.0 - pi_j)                         # (L_j,)
        g_j = (weights.unsqueeze(1) * z_j).mean(dim=0)        # (k,)

        R_j = likelihood.score_mu(K_j, m_j, mu_clamped)       # residual
        w_j = likelihood.fisher_weight(K_j, m_j, mu_clamped)  # precision weight

        grad = grad + R_j * g_j
        H = H + w_j * torch.outer(g_j, g_j)

    _last_diagnostics["boundary_count"] = boundary_count
    _last_diagnostics["total_sentences"] = total_count
    _last_diagnostics["boundary_fraction"] = (
        boundary_count / total_count if total_count > 0 else 0.0
    )
    # Phase 10-1: surface the live phi / rho when a Beta-Binomial is in use.
    _last_diagnostics["rho"] = float(likelihood.rho)
    phi_attr = getattr(likelihood, "phi", None)
    _last_diagnostics["phi"] = (
        float(phi_attr.detach().item()) if torch.is_tensor(phi_attr) else None
    )

    return grad, H


def _compute_clipped_objective(
    theta: torch.Tensor,
    all_z_tokens: List[torch.Tensor],
    all_K: torch.Tensor,
    all_m: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    eps: float = 1e-6,
    likelihood: Optional[Likelihood] = None,
) -> torch.Tensor:
    """Scalar clipped log-posterior ``L̃(θ)`` for the line search.

    ``L̃(θ) = -0.5 (θ-μ_0)ᵀ Σ_0⁻¹ (θ-μ_0) - Σ_j neg_log_pmf_j``. Sentences
    with ``m_j = 0`` are skipped (contribute zero). With a Binomial
    likelihood this is bit-identical to the historical clipped binomial
    log-posterior (the combinatorial constant was, and remains, dropped).

    Parameters
    ----------
    theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
        See :func:`_compute_grad_and_fisher`.
    likelihood : Likelihood, optional
        Observation model. ``None`` (default) constructs a
        :class:`~src.models.likelihood.BinomialLikelihood`.

    Returns
    -------
    Tensor of shape ``()``.
    """
    if len(all_K) != len(all_m) or len(all_K) != len(all_z_tokens):
        raise ValueError(
            "all_z_tokens, all_K, all_m must have the same length; "
            f"got {len(all_z_tokens)}, {len(all_K)}, {len(all_m)}"
        )

    if likelihood is None:
        likelihood = BinomialLikelihood(eps=eps)

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

        if isinstance(likelihood, BinomialLikelihood):
            # Bit-identity guard (HARD CONSTRAINT 1): accumulate the two log
            # terms exactly as the pre-10-1 code did, i.e. ``(obj + A) + B``.
            # The generic ``obj - neg_log_pmf`` computes ``obj + (A + B)``, a
            # valid IEEE reassociation that differs at 1 ULP and can flip a
            # near-tie line-search accept/reject -> a different float32 MAP.
            obj = obj + K_j * torch.log(mu_clamped) + (m_j - K_j) * torch.log(
                1.0 - mu_clamped
            )
        else:
            obj = obj - likelihood.neg_log_pmf(K_j, m_j, mu_clamped)

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
    likelihood: Optional[Likelihood] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shared loop body for the autograd-tracked and detached variants."""
    k = mu_0.shape[0]
    if Sigma_0_inv.shape != (k, k):
        raise ValueError(
            f"Sigma_0_inv must be ({k}, {k}); got {tuple(Sigma_0_inv.shape)}"
        )

    validate_binomial_counts(all_K, all_m, context="fisher_scoring_map")

    if likelihood is None:
        likelihood = BinomialLikelihood(eps=eps)

    theta = mu_0.clone()
    lam = float(lambda_init)
    eye = torch.eye(k, device=mu_0.device, dtype=mu_0.dtype)

    prev_obj = _compute_clipped_objective(
        theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps, likelihood
    )

    for it in range(num_iters):
        grad, H = _compute_grad_and_fisher(
            theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps, likelihood
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
            theta_new, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps, likelihood
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
        theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps, likelihood
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
    likelihood: Optional[Likelihood] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Damped Fisher-scoring MAP for the latent factuality model.

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
    likelihood : Likelihood, optional
        Observation model (Phase 10-1). ``None`` (default) uses a
        Binomial likelihood, reproducing the pre-10-1 behaviour exactly.

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
        likelihood=likelihood,
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
    likelihood: Optional[Likelihood] = None,
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
            likelihood=likelihood,
        )
    return theta, H
