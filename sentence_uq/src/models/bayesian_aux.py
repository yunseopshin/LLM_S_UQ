"""Auxiliary Bayesian regression head for the sentence-level UQ model.

Phase 4-2. Implements a closed-form logit-transformed Bayesian Gaussian
regression that maps a sentence-level aggregate feature ``z_j`` to the
target uncertainty ``U_j^* ∈ [0, 1]`` supplied by an offline expensive
method (e.g. semantic entropy, LUQ). See ``research_document_v8.md``
Part VIII and ``prompts/phase_4_2_aux_model.md`` for the spec.

Model
-----
Logit transformation guarantees the prediction stays in ``[0, 1]`` while
preserving Gaussian conjugacy::

    V_j := logit(U_j^*)  ~  N(θᵀ z_j, σ²),
    θ                    ~  N(μ_0, Σ_0)

Exact conjugate posterior (Part VIII §8.2)::

    Σ_N⁻¹ = Σ_0⁻¹ + (1/σ²) Zᵀ Z,
    θ_N   = Σ_N (Σ_0⁻¹ μ_0 + (1/σ²) Zᵀ V)

Sufficient statistics: ``T_1 = Zᵀ Z``, ``T_2 = Zᵀ V``.

Predictive (logit space, Part VIII §8.3)::

    V_* | z_*  ~  N(θ_Nᵀ z_*,  σ² + z_*ᵀ Σ_N z_*)
                  ^^^^^^^^^^^   ^^^^   ^^^^^^^^^^^^
                  logit_mean    aleatoric_logit
                                       epistemic_logit (over θ)

Probability-space prediction: ``Û_* = σ(θ_Nᵀ z_*)``.

Design notes
------------
* **Closed form** — no autograd needed; the class lives outside the
  bilevel trainer of Phase 4-1.
* **Numerical safety** — :func:`safe_logit` clips ``U`` away from the
  ``{0, 1}`` boundary (Part VIII §8.1 ε-smoothing).
* **Sufficient statistics first** — :meth:`fit` materialises ``T_1``,
  ``T_2`` and exposes ``Σ_N``, ``θ_N`` as attributes, so callers can
  inspect / re-derive the posterior or do online updates.
* **Noise variance** — passed in by the user, or estimated post-hoc
  with :meth:`estimate_noise_variance` (residual-based with ``N - k``
  degrees of freedom). The class deliberately does not refit on its
  own noise estimate to keep behaviour explicit.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


__all__ = ["safe_logit", "BayesianLogitRegression"]


def safe_logit(u: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Numerically stable logit ``log(u / (1 - u))`` with boundary clipping.

    Clips ``u`` into ``[eps, 1 - eps]`` so the transformation stays
    finite on the boundary values that often appear in ratio-style
    targets (``U_j^* = 0`` or ``1``). This matches the ε-smoothing
    recipe of Part VIII §8.1.

    Parameters
    ----------
    u : Tensor of any shape, dtype floating.
        Values to transform, expected to lie in ``[0, 1]``.
    eps : float, optional
        Lower / upper clip distance from the boundary. Must satisfy
        ``0 < eps < 0.5``. Defaults to ``1e-3``.

    Returns
    -------
    Tensor of the same shape and dtype as ``u``.
    """
    if not torch.is_tensor(u):
        raise TypeError(f"u must be a torch.Tensor; got {type(u).__name__}")
    if not (0.0 < eps < 0.5):
        raise ValueError(f"eps must lie in (0, 0.5); got {eps}")

    u_clipped = u.clamp(min=eps, max=1.0 - eps)
    return torch.log(u_clipped) - torch.log1p(-u_clipped)


class BayesianLogitRegression:
    """Closed-form logit-transformed Bayesian Gaussian regression.

    Parameters
    ----------
    feature_dim : int
        Dimension ``k`` of ``z_j``. Must be positive.
    prior_mu : Tensor of shape ``(k,)``, optional
        Prior mean ``μ_0``. Defaults to zeros.
    prior_sigma : float or Tensor of shape ``(k,)``, optional
        Isotropic / diagonal prior std. The prior covariance is
        ``Σ_0 = diag(prior_sigma**2)``. Defaults to ``1.0``.
    noise_sigma : float, optional
        Likelihood noise std ``σ``. Defaults to ``0.1``. Can be
        overridden post-fit with :meth:`set_noise_sigma` after running
        :meth:`estimate_noise_variance`.

    Attributes
    ----------
    feature_dim : int
        ``k``.
    prior_mu : Tensor of shape ``(k,)``.
    prior_Sigma : Tensor of shape ``(k, k)``.
    prior_Sigma_inv : Tensor of shape ``(k, k)``.
    noise_sigma : float
    theta_N : Tensor of shape ``(k,)`` or ``None``
        Posterior mean. ``None`` before :meth:`fit`.
    Sigma_N : Tensor of shape ``(k, k)`` or ``None``
        Posterior covariance. ``None`` before :meth:`fit`.
    T1 : Tensor of shape ``(k, k)`` or ``None``
        Sufficient statistic ``Zᵀ Z`` from the most recent :meth:`fit`.
    T2 : Tensor of shape ``(k,)`` or ``None``
        Sufficient statistic ``Zᵀ V`` from the most recent :meth:`fit`.
    """

    def __init__(
        self,
        feature_dim: int,
        prior_mu: Optional[torch.Tensor] = None,
        prior_sigma: float | torch.Tensor = 1.0,
        noise_sigma: float = 0.1,
    ) -> None:
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError(
                f"feature_dim must be a positive int; got {feature_dim}"
            )
        if float(noise_sigma) <= 0.0:
            raise ValueError(
                f"noise_sigma must be positive; got {noise_sigma}"
            )

        self.feature_dim = int(feature_dim)

        if prior_mu is None:
            prior_mu_t = torch.zeros(self.feature_dim, dtype=torch.float64)
        else:
            prior_mu_t = prior_mu.to(torch.float64).reshape(-1)
            if prior_mu_t.shape != (self.feature_dim,):
                raise ValueError(
                    f"prior_mu must have shape ({self.feature_dim},); "
                    f"got {tuple(prior_mu_t.shape)}"
                )
        self.prior_mu = prior_mu_t

        if torch.is_tensor(prior_sigma):
            ps = prior_sigma.to(torch.float64).reshape(-1)
            if ps.shape == (1,):
                ps = ps.expand(self.feature_dim).contiguous()
            if ps.shape != (self.feature_dim,):
                raise ValueError(
                    f"prior_sigma tensor must broadcast to shape "
                    f"({self.feature_dim},); got {tuple(ps.shape)}"
                )
        else:
            ps_val = float(prior_sigma)
            if ps_val <= 0.0:
                raise ValueError(
                    f"prior_sigma must be positive; got {ps_val}"
                )
            ps = torch.full(
                (self.feature_dim,), ps_val, dtype=torch.float64
            )
        if torch.any(ps <= 0.0):
            raise ValueError("all entries of prior_sigma must be positive")
        self.prior_Sigma = torch.diag(ps * ps)
        self.prior_Sigma_inv = torch.diag(1.0 / (ps * ps))

        self.noise_sigma = float(noise_sigma)

        self.theta_N: Optional[torch.Tensor] = None
        self.Sigma_N: Optional[torch.Tensor] = None
        self.T1: Optional[torch.Tensor] = None
        self.T2: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, Z: torch.Tensor, U_star: torch.Tensor) -> "BayesianLogitRegression":
        """Compute the conjugate posterior ``(θ_N, Σ_N)`` in closed form.

        Math (Part VIII §8.2)::

            V       = safe_logit(U^*)
            T_1     = Zᵀ Z
            T_2     = Zᵀ V
            Σ_N⁻¹   = Σ_0⁻¹ + (1/σ²) · T_1
            θ_N     = Σ_N (Σ_0⁻¹ μ_0 + (1/σ²) · T_2)

        Parameters
        ----------
        Z : Tensor of shape ``(N, feature_dim)``.
            Sentence-level aggregate features.
        U_star : Tensor of shape ``(N,)`` with values in ``[0, 1]``.
            Target uncertainty from the offline expensive method.

        Returns
        -------
        self
            For call chaining.
        """
        Z_t, V = self._prepare_training_inputs(Z, U_star)

        sigma2 = self.noise_sigma * self.noise_sigma
        T1 = Z_t.T @ Z_t                                                # (k, k)
        T2 = Z_t.T @ V                                                  # (k,)

        Sigma_N_inv = self.prior_Sigma_inv + T1 / sigma2
        Sigma_N_inv = 0.5 * (Sigma_N_inv + Sigma_N_inv.T)               # symmetrise
        Sigma_N = self._safe_inverse(Sigma_N_inv)

        rhs = self.prior_Sigma_inv @ self.prior_mu + T2 / sigma2
        theta_N = Sigma_N @ rhs

        self.T1 = T1
        self.T2 = T2
        self.Sigma_N = Sigma_N
        self.theta_N = theta_N
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, z_new: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predictive distribution for one or more new sentences.

        For each row ``z_*`` of ``z_new`` returns the logit-space mean
        and variance plus the probability-space point estimate::

            logit_mean      = θ_Nᵀ z_*
            epistemic_logit = z_*ᵀ Σ_N z_*
            aleatoric_logit = σ²
            logit_var       = epistemic_logit + aleatoric_logit
            p_factual       = σ(logit_mean)

        Parameters
        ----------
        z_new : Tensor of shape ``(feature_dim,)`` or ``(M, feature_dim)``.

        Returns
        -------
        dict with keys (all 1-D tensors of length ``M`` when a batch is
        passed, scalar tensors for a single sample):
            ``p_factual``       : σ(θ_Nᵀ z_*)
            ``logit_mean``      : θ_Nᵀ z_*
            ``logit_var``       : σ² + z_*ᵀ Σ_N z_*
            ``epistemic_logit`` : z_*ᵀ Σ_N z_*
            ``aleatoric_logit`` : σ² broadcast to match the batch
        """
        if self.theta_N is None or self.Sigma_N is None:
            raise RuntimeError(
                "BayesianLogitRegression.predict called before .fit(...)"
            )

        z = z_new.to(torch.float64)
        squeeze = False
        if z.dim() == 1:
            if z.shape[0] != self.feature_dim:
                raise ValueError(
                    f"z_new must have last dim {self.feature_dim}; "
                    f"got {tuple(z.shape)}"
                )
            z = z.unsqueeze(0)
            squeeze = True
        elif z.dim() == 2:
            if z.shape[1] != self.feature_dim:
                raise ValueError(
                    f"z_new must have last dim {self.feature_dim}; "
                    f"got {tuple(z.shape)}"
                )
        else:
            raise ValueError(
                f"z_new must be 1-D or 2-D; got shape {tuple(z.shape)}"
            )

        logit_mean = z @ self.theta_N                                   # (M,)
        epistemic = ((z @ self.Sigma_N) * z).sum(dim=1).clamp_min(0.0)  # (M,)
        sigma2 = self.noise_sigma * self.noise_sigma
        aleatoric = torch.full_like(epistemic, sigma2)
        logit_var = epistemic + aleatoric
        p_factual = torch.sigmoid(logit_mean)

        out = {
            "p_factual": p_factual,
            "logit_mean": logit_mean,
            "logit_var": logit_var,
            "epistemic_logit": epistemic,
            "aleatoric_logit": aleatoric,
        }
        if squeeze:
            out = {k: v.squeeze(0) for k, v in out.items()}
        return out

    # ------------------------------------------------------------------
    # Noise variance estimation
    # ------------------------------------------------------------------

    def estimate_noise_variance(
        self, Z: torch.Tensor, U_star: torch.Tensor
    ) -> float:
        """Residual-based estimator ``σ² = (1/(N-k)) Σ (V_j - θ_Nᵀ z_j)²``.

        Uses the *current* posterior mean ``θ_N`` (caller is expected
        to have run :meth:`fit` first, or to repeat fit→estimate→fit
        in an outer loop if they want a self-consistent estimate).

        Parameters
        ----------
        Z : Tensor of shape ``(N, feature_dim)``.
        U_star : Tensor of shape ``(N,)`` with values in ``[0, 1]``.

        Returns
        -------
        float
            Estimated noise variance. The class is left unchanged; the
            caller can update ``noise_sigma`` via :meth:`set_noise_sigma`.

        Raises
        ------
        RuntimeError
            If ``.fit(...)`` has not been called.
        ValueError
            If ``N <= feature_dim`` (no residual degrees of freedom).
        """
        if self.theta_N is None:
            raise RuntimeError(
                "estimate_noise_variance requires a fitted posterior; "
                "call .fit(Z, U_star) first."
            )

        Z_t, V = self._prepare_training_inputs(Z, U_star)
        N = Z_t.shape[0]
        k = self.feature_dim
        if N <= k:
            raise ValueError(
                f"residual-based σ² needs N > k; got N={N}, k={k}"
            )

        residuals = V - Z_t @ self.theta_N
        sigma2 = float((residuals * residuals).sum().item() / (N - k))
        return sigma2

    def set_noise_sigma(self, noise_sigma: float) -> None:
        """Override ``σ``. Refit if you want the posterior to reflect it.

        Parameters
        ----------
        noise_sigma : float
            New noise std. Must be positive.
        """
        if float(noise_sigma) <= 0.0:
            raise ValueError(
                f"noise_sigma must be positive; got {noise_sigma}"
            )
        self.noise_sigma = float(noise_sigma)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_training_inputs(
        self, Z: torch.Tensor, U_star: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if Z.dim() != 2:
            raise ValueError(
                f"Z must be 2-D (N, feature_dim); got shape {tuple(Z.shape)}"
            )
        if Z.shape[1] != self.feature_dim:
            raise ValueError(
                f"Z must have feature_dim {self.feature_dim}; "
                f"got {tuple(Z.shape)}"
            )
        if U_star.dim() != 1:
            raise ValueError(
                f"U_star must be 1-D (N,); got shape {tuple(U_star.shape)}"
            )
        if U_star.shape[0] != Z.shape[0]:
            raise ValueError(
                f"Z and U_star must share the leading dim; got "
                f"{tuple(Z.shape)} and {tuple(U_star.shape)}"
            )

        Z_t = Z.to(torch.float64)
        V = safe_logit(U_star.to(torch.float64))
        return Z_t, V

    @staticmethod
    def _safe_inverse(matrix: torch.Tensor) -> torch.Tensor:
        """Symmetrise + adaptively-jittered inverse for a near-PSD matrix."""
        sym = 0.5 * (matrix + matrix.T)
        k = sym.shape[0]
        eye = torch.eye(k, dtype=sym.dtype, device=sym.device)
        jitter = 0.0
        for _ in range(8):
            try:
                return torch.linalg.inv(sym + jitter * eye)
            except RuntimeError:
                jitter = 1e-10 if jitter == 0.0 else jitter * 10.0
        raise RuntimeError(
            "Failed to invert posterior precision even with jitter up "
            f"to {jitter:.2e}; check feature scaling / prior_sigma."
        )
