"""Main Bayesian sentence-level UQ model (binomial observation).

Phase 3-2. Wraps the feature extractor (Phase 2-1) and the damped
Fisher-scoring inner loop (Phase 3-1) into the outer-level model that
the bilevel trainer (Phase 4-1) optimises end-to-end. See
``research_document_v8.md`` Parts II, III, VII and
``prompts/phase_3_2_bayesian_main.md`` for the spec.

Observation model (binomial, replaces the v7 Bernoulli)::

    π_ℓ(θ)  = σ(θᵀ z_ℓ)                               # per-token latent factuality
    μ_j(θ)  = (1 / L_j) Σ_{ℓ ∈ s_j} π_ℓ(θ)             # sentence factuality
    K_j | θ, m_j  ~  Binomial(m_j, μ_j(θ)),   m_j = atom count

Outer loss (sum, not mean, to stay consistent with prior scaling)::

    L_outer(ψ) = Σ_{j: m_j > 0} [ -K_j log μ̃_j - (m_j - K_j) log(1 - μ̃_j) ],
    μ̃_j = clip(μ_j(θ̂(ψ)), ε, 1 - ε)

The MAP ``θ̂`` is computed differentiably through the unrolled
Fisher-scoring loop, so gradients flow into
``ψ = (W, α, μ_0, log σ_0)`` of :class:`SentenceUQParams`.

Design notes
------------
* ``compute_map`` exposes both the autograd-tracked and detached Fisher
  variants behind a single ``differentiable`` flag.
* ``compute_loss`` skips ``m_j = 0`` sentences (CLAUDE.md rule 8).
* ``predict`` is a placeholder here — the full predictive inference
  (4-level uncertainty decomposition, token attribution, probit
  shrinkage) lives in ``src/inference/predict.py`` (Phase 3-3).
* :func:`verify_local_pd` validates the local Laplace approximation by
  checking that both the Fisher-type precision *and* the negative true
  Hessian of the clipped log-posterior are positive-definite at θ̂. It
  costs ``O(k²)`` backward passes (``torch.autograd.functional.hessian``)
  and should be called every ~5 epochs, not every step.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.features.extractor import SentenceUQParams
from src.models.fisher_scoring import (
    _compute_clipped_objective,
    _compute_grad_and_fisher,
    fisher_scoring_map,
    fisher_scoring_map_detached,
)


__all__ = ["BayesianSentenceUQ", "verify_local_pd"]


class BayesianSentenceUQ(nn.Module):
    """Outer-level binomial-likelihood Bayesian UQ model.

    Wraps :class:`SentenceUQParams` and the Fisher-scoring inner loop so
    the bilevel trainer (Phase 4-1) can backpropagate the outer
    likelihood loss through ``θ̂(ψ)`` and update ``ψ``.

    Parameters
    ----------
    feature_params : SentenceUQParams
        Learnable feature-extractor parameters ψ. Owns the prior
        ``(μ_0, log σ_0)`` consumed by the inner Fisher loop.
    num_fisher_iters : int, optional
        Iterations of the damped Fisher-scoring inner loop. Default 10
        — keep small to bound the memory cost of the unrolled backward
        pass (CLAUDE.md rule 9 / Phase 3-1 spec).
    eps : float, optional
        Clipping bound for ``μ_j`` to avoid log singularities. Default
        ``1e-6``.
    """

    def __init__(
        self,
        feature_params: SentenceUQParams,
        num_fisher_iters: int = 10,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not isinstance(feature_params, SentenceUQParams):
            raise TypeError(
                "feature_params must be a SentenceUQParams instance; "
                f"got {type(feature_params).__name__}"
            )
        if num_fisher_iters <= 0:
            raise ValueError(
                f"num_fisher_iters must be positive, got {num_fisher_iters}"
            )
        if eps <= 0.0 or eps >= 0.5:
            raise ValueError(f"eps must lie in (0, 0.5); got {eps}")

        self.feature_params = feature_params
        self.num_fisher_iters = int(num_fisher_iters)
        self.eps = float(eps)

    # ------------------------------------------------------------------
    # MAP estimation
    # ------------------------------------------------------------------

    def compute_map(
        self,
        all_z_tokens: List[torch.Tensor],
        all_K: torch.Tensor,
        all_m: torch.Tensor,
        differentiable: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Damped Fisher-scoring MAP ``θ̂`` and Fisher-type precision.

        Reads the prior ``(μ_0, Σ_0⁻¹)`` from ``self.feature_params`` and
        delegates to :func:`fisher_scoring_map` (autograd-tracked) or
        :func:`fisher_scoring_map_detached` (inference).

        Parameters
        ----------
        all_z_tokens : list of N tensors of shape ``(L_j, k)``.
            Per-sentence per-token features from Phase 2-1.
        all_K : Tensor of shape ``(N,)``, integer dtype.
            Supported-atom counts.
        all_m : Tensor of shape ``(N,)``, integer dtype. ``m_j = 0`` rows
            are skipped inside the Fisher loop.
        differentiable : bool, optional
            ``True`` (default) → unrolled autograd-tracked Fisher pass
            (used by the outer training loop). ``False`` → ``no_grad``
            inference-only pass.

        Returns
        -------
        theta_hat : Tensor of shape ``(k,)``.
        H_fisher : Tensor of shape ``(k, k)``.
            Fisher-type precision at ``θ̂``.
        """
        mu_0 = self.feature_params.mu_0
        Sigma_0_inv = self.feature_params.get_Sigma_0_inv()

        if differentiable:
            return fisher_scoring_map(
                all_z_tokens=all_z_tokens,
                all_K=all_K,
                all_m=all_m,
                mu_0=mu_0,
                Sigma_0_inv=Sigma_0_inv,
                num_iters=self.num_fisher_iters,
                eps=self.eps,
            )
        return fisher_scoring_map_detached(
            all_z_tokens=all_z_tokens,
            all_K=all_K,
            all_m=all_m,
            mu_0=mu_0,
            Sigma_0_inv=Sigma_0_inv,
            num_iters=self.num_fisher_iters,
            eps=self.eps,
        )

    # ------------------------------------------------------------------
    # Outer training loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        all_z_tokens: List[torch.Tensor],
        all_K: torch.Tensor,
        all_m: torch.Tensor,
    ) -> torch.Tensor:
        """Binomial negative log-likelihood at ``θ̂(ψ)`` (sum, m_j>0 only).

        Math::

            L = Σ_{j: m_j > 0} [ -K_j log μ̃_j - (m_j - K_j) log(1 - μ̃_j) ],
            μ̃_j = clip( (1/L_j) Σ_ℓ σ(θ̂ᵀ z_ℓ), ε, 1 - ε )

        Because ``θ̂`` is produced by the differentiable
        :meth:`compute_map`, gradients propagate to
        ``ψ = (W, α, μ_0, log σ_0)`` via the outer ``loss.backward()``.

        Sentences with ``m_j = 0`` are skipped (CLAUDE.md rule 8). Sum
        scaling (not mean) is chosen so the loss magnitude matches the
        prior penalty inside the MAP objective.

        Parameters
        ----------
        all_z_tokens, all_K, all_m
            See :meth:`compute_map`.

        Returns
        -------
        Tensor of shape ``()`` — scalar outer loss.
        """
        if len(all_K) != len(all_m) or len(all_K) != len(all_z_tokens):
            raise ValueError(
                "all_z_tokens, all_K, all_m must have the same length; "
                f"got {len(all_z_tokens)}, {len(all_K)}, {len(all_m)}"
            )

        theta_hat, _ = self.compute_map(
            all_z_tokens, all_K, all_m, differentiable=True
        )

        dtype = theta_hat.dtype
        device = theta_hat.device
        total_loss = torch.zeros((), dtype=dtype, device=device)

        for j in range(len(all_K)):
            m_j_int = (
                int(all_m[j].item()) if torch.is_tensor(all_m[j]) else int(all_m[j])
            )
            if m_j_int == 0:
                continue

            z_j = all_z_tokens[j]
            if z_j.dim() != 2 or z_j.shape[1] != theta_hat.shape[0]:
                raise ValueError(
                    f"all_z_tokens[{j}] must be (L_j, k={theta_hat.shape[0]}); "
                    f"got shape {tuple(z_j.shape)}"
                )

            K_j = all_K[j].to(dtype)
            m_j = all_m[j].to(dtype)

            pi_j = torch.sigmoid(z_j @ theta_hat)
            mu_clamped = torch.clamp(pi_j.mean(), self.eps, 1.0 - self.eps)

            total_loss = total_loss + (
                -K_j * torch.log(mu_clamped)
                - (m_j - K_j) * torch.log(1.0 - mu_clamped)
            )

        return total_loss

    # ------------------------------------------------------------------
    # Predictive inference — Phase 3-3 stub
    # ------------------------------------------------------------------

    def predict(
        self,
        z_tokens: torch.Tensor,
        m_j: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Post-training predictive inference (Phase 3-3 implements this).

        The full implementation — 4-level uncertainty decomposition
        (latent / ratio / count / strict), token-level attribution,
        probit shrinkage — lives in ``src/inference/predict.py`` as the
        :class:`Predictor` class, which consumes a stored
        ``(θ̂, Σ̂)``. This stub is here only so the public method name
        is reserved by Phase 3-2 as the spec requires.

        Parameters
        ----------
        z_tokens : Tensor of shape ``(L_j, k)``.
            Per-token features for a single sentence.
        m_j : int, optional
            Atomic-fact count. Needed for ratio-/count-level
            decomposition; latent-level epistemic is reported when
            ``m_j is None``.

        Raises
        ------
        NotImplementedError
            Always — use ``src.inference.predict.Predictor`` instead.
        """
        raise NotImplementedError(
            "BayesianSentenceUQ.predict is implemented in Phase 3-3 "
            "(src/inference/predict.py — Predictor class). It needs the "
            "Laplace posterior (theta_hat, Sigma_hat) which Phase 3-2 "
            "does not yet store."
        )


# ---------------------------------------------------------------------------
# Laplace validity check
# ---------------------------------------------------------------------------


def verify_local_pd(
    theta_hat: torch.Tensor,
    all_z_tokens: List[torch.Tensor],
    all_K: torch.Tensor,
    all_m: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    clip_eps: float = 1e-6,
    pd_tol: float = 1e-8,
    eps: Optional[float] = None,
) -> Dict[str, float]:
    """Check positive-definiteness of both precision matrices at ``θ̂``.

    Laplace's approximation ``p(θ|D) ≈ N(θ̂, Σ̂)`` is locally valid only
    when the posterior is locally log-concave at the MAP, i.e. when the
    negative Hessian of the (clipped) log-posterior is positive
    definite. We check two surrogates:

    1. **Fisher-type precision** — the matrix returned by
       :func:`_compute_grad_and_fisher`. Cheap (closed form) and PSD by
       construction.
    2. **True precision** ``-∇²_θ L̃(θ̂)`` — exact, via
       ``torch.autograd.functional.hessian`` on
       :func:`_compute_clipped_objective`. Expensive (``O(k²)`` backward
       passes), hence the spec recommends invoking this every ~5 epochs.

    Phase 7-3 fix 8: the log-clip ``ε`` and the eigenvalue positivity
    threshold are conceptually different — the former is a numerical
    stabiliser inside the objective and gradient, the latter is the
    decision rule for "positive definite". They are now exposed as
    separate ``clip_eps`` and ``pd_tol`` knobs.

    Parameters
    ----------
    theta_hat : Tensor of shape ``(k,)``.
        Point at which to evaluate the Hessian.
    all_z_tokens : list of N tensors of shape ``(L_j, k)``.
    all_K : Tensor of shape ``(N,)``, integer dtype.
    all_m : Tensor of shape ``(N,)``, integer dtype.
    mu_0 : Tensor of shape ``(k,)``.
    Sigma_0_inv : Tensor of shape ``(k, k)``.
    clip_eps : float, optional
        Log-stability clip ``ε`` forwarded to
        :func:`_compute_grad_and_fisher` and
        :func:`_compute_clipped_objective`. Default ``1e-6``.
    pd_tol : float, optional
        Positivity threshold on the smallest eigenvalue. A precision
        matrix is considered PD when ``min_eig > pd_tol``. Default
        ``1e-8`` — looser than ``clip_eps`` because near-zero eigenvalues
        are an algorithmic concern about Laplace validity, not a
        numerical-overflow concern.
    eps : float, optional
        Legacy single-knob alias. When passed, it sets ``clip_eps`` to
        the given value and leaves ``pd_tol`` at its default. Retained
        for callers that have not yet adopted the split signature.

    Returns
    -------
    dict with keys:
        - ``fisher_min_eig`` (float): smallest eigenvalue of the
          Fisher-type precision.
        - ``true_min_eig`` (float): smallest eigenvalue of
          ``-∇²_θ L̃(θ̂)``.
        - ``fisher_pd`` (bool): ``fisher_min_eig > pd_tol``.
        - ``true_pd`` (bool): ``true_min_eig > pd_tol``.
        - ``laplace_valid_local`` (bool): both PD.
    """
    if eps is not None:
        clip_eps = float(eps)

    theta_d = theta_hat.detach()
    z_d = [z.detach() for z in all_z_tokens]
    mu_0_d = mu_0.detach()
    Sigma_0_inv_d = Sigma_0_inv.detach()

    # ---- Fisher-type precision (no autograd needed) ----
    with torch.no_grad():
        _, H_fisher = _compute_grad_and_fisher(
            theta_d, z_d, all_K, all_m, mu_0_d, Sigma_0_inv_d, clip_eps
        )
    H_fisher_sym = 0.5 * (H_fisher + H_fisher.T)
    fisher_eigs = torch.linalg.eigvalsh(H_fisher_sym)
    fisher_min_eig = float(fisher_eigs.min().item())

    # ---- True precision = -Hessian(L̃) at θ̂ ----
    def objective_fn(t: torch.Tensor) -> torch.Tensor:
        return _compute_clipped_objective(
            t, z_d, all_K, all_m, mu_0_d, Sigma_0_inv_d, clip_eps
        )

    H_true = torch.autograd.functional.hessian(objective_fn, theta_d)
    true_precision = -H_true
    true_prec_sym = 0.5 * (true_precision + true_precision.T)
    true_eigs = torch.linalg.eigvalsh(true_prec_sym)
    true_min_eig = float(true_eigs.min().item())

    fisher_pd = fisher_min_eig > pd_tol
    true_pd = true_min_eig > pd_tol

    return {
        "fisher_min_eig": fisher_min_eig,
        "true_min_eig": true_min_eig,
        "fisher_pd": fisher_pd,
        "true_pd": true_pd,
        "laplace_valid_local": fisher_pd and true_pd,
    }
