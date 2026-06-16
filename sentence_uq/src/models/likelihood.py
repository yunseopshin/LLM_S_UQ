"""Per-sentence observation likelihoods for the Bayesian sentence-level UQ model.

Phase 10-1. Defines a tiny :class:`Likelihood` interface plus two
config-selectable peer implementations that supply the per-sentence
scalars consumed by the Fisher-scoring inner loop
(``src.models.fisher_scoring``), the outer NLL
(``src.models.bayesian_main``), and the predictive decomposition
(``src.inference.predict``):

* :class:`BinomialLikelihood` -- reproduces the pre-existing epsilon
  stabilised Binomial arithmetic *exactly* (bit-identical) so the
  Binomial path is byte-for-byte unchanged.
* :class:`BetaBinomialLikelihood` -- a mean/dispersion Beta-Binomial that
  reduces to the Binomial in the limit ``phi -> inf``.

Model (mean-dispersion parameterization, Phase 10-1 section 1)
-------------------------------------------------------------
Per sentence ``j`` with sentence mean ``mu_j = mu_j(theta)``, atom count
``m_j``, supported count ``K_j``, and a *global* concentration
``phi > 0``::

    p_j ~ Beta(a_j, b_j),  a_j = phi * mu_j,  b_j = phi * (1 - mu_j)
    K_j | p_j ~ Binomial(m_j, p_j)

so ``K_j`` is marginally Beta-Binomial. Writing ``rho = 1 / (phi + 1)``::

    E[U_j | theta]   = mu_j
    Var[U_j | theta] = mu_j (1 - mu_j) / m_j * (1 + (m_j - 1) * rho)

The mean equals the Binomial mean; only the conditional variance gains
the overdispersion factor ``1 + (m_j - 1) * rho``. As ``phi -> inf``
(``rho -> 0``) this collapses to the Binomial.

Interface contract
------------------
All methods operate elementwise on tensors of *already-clamped* ``mu``
(``mu in [eps, 1 - eps]``), float ``K``, and float ``m`` (scalars or
1-D). They are autograd-friendly (no ``.detach()``, no in-place ops on
tracked tensors), so both the unrolled inner loop and the outer NLL can
backprop into ``log_phi``.

The combinatorial constant ``log C(m, K)`` is dropped from
:meth:`neg_log_pmf` in *both* likelihoods. It is independent of ``mu``
and ``phi``, so it changes neither the MAP (the line search compares
objectives at fixed ``(K, m)``), the gradients w.r.t. ``psi`` / ``theta``
/ ``log_phi``, nor the fitted ``phi``. Dropping it (a) keeps the Binomial
path bit-identical with the pre-existing ``fisher_scoring`` /
``compute_loss`` code (which never included it), and (b) makes the
``phi -> inf`` limit of the Beta-Binomial ``neg_log_pmf`` coincide
*exactly* with the Binomial ``neg_log_pmf`` used here.

Design note (Fisher weight)
---------------------------
The Binomial path uses the *expected* Fisher weight
``m / (mu (1 - mu))``. The Beta-Binomial uses the *observed* information
``-d^2 log P / d mu^2`` because no closed-form expected Fisher exists for
the Beta-Binomial. The two coincide in the ``phi -> inf`` limit at the
optimum, where ``K approx m * mu``.
"""

from __future__ import annotations

from typing import Optional

import torch


__all__ = [
    "Likelihood",
    "BinomialLikelihood",
    "BetaBinomialLikelihood",
    "make_likelihood",
]


class Likelihood:
    """Per-sentence scalar interface shared by the inner loop, the outer
    NLL, and the predictive decomposition.

    Implementations operate elementwise on 1-D (or 0-D) tensors of clamped
    ``mu``, float ``K``, and float ``m``. ``mu`` is assumed already clamped
    to ``[eps, 1 - eps]`` by the caller.
    """

    def score_mu(
        self, K: torch.Tensor, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """Residual ``r_j = d log P_j / d mu_j`` that multiplies ``g_j``."""
        raise NotImplementedError

    def fisher_weight(
        self, K: torch.Tensor, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """Precision weight ``w_j`` multiplying ``g_j g_j^T`` (``> 0``)."""
        raise NotImplementedError

    def neg_log_pmf(
        self, K: torch.Tensor, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """Per-sentence ``-log P_j`` (combinatorial constant dropped)."""
        raise NotImplementedError

    def log_prob_all_correct(
        self, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        """``log P(A_j = 1) = log P(K_j = m_j)`` (strict factuality)."""
        raise NotImplementedError

    @property
    def rho(self) -> float:
        """Within-sentence correlation used by the predictive decomposition."""
        raise NotImplementedError


class BinomialLikelihood(Likelihood):
    """Bit-identical re-expression of the pre-existing Binomial arithmetic.

    Reproduces the epsilon-stabilised expressions in
    ``src.models.fisher_scoring`` *verbatim*: with ``mu`` already clamped
    to ``[eps, 1 - eps]`` by the caller, the denominator
    ``mu * (1 - mu)`` is used directly (no second clamp), so feeding this
    likelihood through the inner loop yields outputs identical to the
    historical hard-coded path.

    ::

        score_mu             = (K - m * mu) / (mu * (1 - mu))
        fisher_weight        = m / (mu * (1 - mu))
        neg_log_pmf          = -(K * log(mu) + (m - K) * log(1 - mu))
        log_prob_all_correct = m * log(mu)
        rho                  = 0.0
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = float(eps)

    def score_mu(
        self, K: torch.Tensor, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        return (K - m * mu) / (mu * (1.0 - mu))

    def fisher_weight(
        self, K: torch.Tensor, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        return m / (mu * (1.0 - mu))

    def neg_log_pmf(
        self, K: torch.Tensor, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        return -(K * torch.log(mu) + (m - K) * torch.log(1.0 - mu))

    def log_prob_all_correct(
        self, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        return m * torch.log(mu)

    @property
    def rho(self) -> float:
        return 0.0


class BetaBinomialLikelihood(Likelihood):
    """Mean/dispersion Beta-Binomial peer (Phase 10-1 section 2).

    ``phi`` is held as a live tensor (typically ``exp(params.log_phi)``) so
    gradients flow back to ``log_phi`` through every scalar. With
    ``a = phi * mu``, ``b = phi * (1 - mu)``, ``Psi0 = digamma``,
    ``Psi1 = polygamma(1, .)`` and
    ``log B(x, y) = lgamma(x) + lgamma(y) - lgamma(x + y)``::

        score_mu      = phi * (Psi0(K + a) - Psi0(a)
                               - Psi0(m - K + b) + Psi0(b))
        fisher_weight = phi^2 * (Psi1(a) - Psi1(K + a)
                                 + Psi1(b) - Psi1(m - K + b))      # observed info
        neg_log_pmf   = -(log B(K + a, m - K + b) - log B(a, b))
        log_prob_all_correct = lgamma(m + a) - lgamma(a)
                               + lgamma(phi) - lgamma(m + phi)
        rho           = 1 / (phi + 1)

    ``phi -> inf`` limits (verified in tests)::

        score_mu      -> (K - m mu) / (mu (1 - mu))           # Binomial score
        fisher_weight -> K / mu^2 + (m - K) / (1 - mu)^2       # observed info
                         (= m / (mu (1 - mu)) only at K = m mu)
        neg_log_pmf   -> Binomial neg_log_pmf (constant dropped)
        log_prob_all_correct -> m log mu

    Parameters
    ----------
    phi : torch.Tensor
        Global concentration ``phi > 0`` (scalar tensor; gradient-carrying).
    eps : float, optional
        Clamp bound applied to ``mu`` before any gamma call. Defaults to
        ``1e-6`` to match the inner-loop default. Clamping an
        already-clamped ``mu`` is idempotent, so this never perturbs the
        production path.
    """

    def __init__(self, phi: torch.Tensor, eps: float = 1e-6) -> None:
        if not torch.is_tensor(phi):
            phi = torch.as_tensor(phi, dtype=torch.float32)
        self.phi = phi
        self.eps = float(eps)

    # -- internal helpers ------------------------------------------------

    def _prep(
        self, K: torch.Tensor, m: torch.Tensor, mu: torch.Tensor
    ) -> tuple:
        """Promote ``(K, m, mu, phi)`` to float64, clamp ``mu``, build a/b.

        Returns ``(K, m, mu_clamped, a, b, phi, work_dtype)`` with
        ``a = phi * mu``, ``b = phi * (1 - mu)``, every value in float64.

        The gamma arguments are computed in **float64** even when the
        caller is float32, because ``digamma`` / ``polygamma`` / ``lgamma``
        of a large ``phi * mu`` suffer catastrophic cancellation in float32
        (e.g. ``digamma(a) - digamma(K + a)`` for ``a ~ 1e4+`` loses most of
        its significant digits). The float32 inner loop can push ``phi``
        large when overdispersion is weak, so this keeps the residual /
        Fisher scalars accurate there. ``work_dtype`` (the promotion of
        ``phi.dtype`` and ``mu.dtype``) is returned so each method can cast
        its result back, preserving the inner-loop dtype and autograd graph.
        """
        if torch.is_tensor(mu) and mu.is_floating_point():
            work_dtype = torch.promote_types(self.phi.dtype, mu.dtype)
        elif self.phi.is_floating_point():
            work_dtype = self.phi.dtype
        else:
            work_dtype = torch.float32

        cd = torch.float64

        def _cast(x: torch.Tensor) -> torch.Tensor:
            return x.to(cd) if torch.is_tensor(x) else torch.as_tensor(x, dtype=cd)

        K = _cast(K)
        m = _cast(m)
        mu = _cast(mu)
        phi = self.phi.to(cd)
        mu_c = torch.clamp(mu, self.eps, 1.0 - self.eps)
        a = phi * mu_c
        b = phi * (1.0 - mu_c)
        return K, m, mu_c, a, b, phi, work_dtype

    # -- interface -------------------------------------------------------

    def score_mu(
        self, K: torch.Tensor, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        K, m, _, a, b, phi, wd = self._prep(K, m, mu)
        r = phi * (
            torch.digamma(K + a)
            - torch.digamma(a)
            - torch.digamma(m - K + b)
            + torch.digamma(b)
        )
        return r.to(wd)

    def fisher_weight(
        self, K: torch.Tensor, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        K, m, _, a, b, phi, wd = self._prep(K, m, mu)
        # Observed information -d^2 log P / d mu^2; positive because the
        # trigamma Psi1 is positive and strictly decreasing.
        w = (phi * phi) * (
            torch.polygamma(1, a)
            - torch.polygamma(1, K + a)
            + torch.polygamma(1, b)
            - torch.polygamma(1, m - K + b)
        )
        return w.to(wd)

    def neg_log_pmf(
        self, K: torch.Tensor, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        K, m, _, a, b, phi, wd = self._prep(K, m, mu)
        log_b_post = _log_beta(K + a, m - K + b)
        log_b_prior = _log_beta(a, b)
        return (-(log_b_post - log_b_prior)).to(wd)

    def log_prob_all_correct(
        self, m: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        # P(A_j = 1) = P(K_j = m_j) for the Beta-Binomial.
        _, m, _, a, _, phi, wd = self._prep(0.0, m, mu)
        out = (
            torch.lgamma(m + a)
            - torch.lgamma(a)
            + torch.lgamma(phi)
            - torch.lgamma(m + phi)
        )
        return out.to(wd)

    @property
    def rho(self) -> float:
        return float((1.0 / (self.phi + 1.0)).item())


def _log_beta(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``log B(x, y) = lgamma(x) + lgamma(y) - lgamma(x + y)``."""
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)


def make_likelihood(
    likelihood: str,
    phi: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Likelihood:
    """Construct a :class:`Likelihood` from a config string.

    Parameters
    ----------
    likelihood : str
        ``"binomial"`` or ``"beta_binomial"``.
    phi : torch.Tensor, optional
        Required (live, gradient-carrying) concentration for
        ``"beta_binomial"``; ignored for ``"binomial"``.
    eps : float, optional
        Clamp bound forwarded to the constructed likelihood.

    Returns
    -------
    Likelihood
    """
    if likelihood == "beta_binomial":
        if phi is None:
            raise ValueError(
                "make_likelihood('beta_binomial') requires a phi tensor"
            )
        return BetaBinomialLikelihood(phi=phi, eps=eps)
    if likelihood == "binomial":
        return BinomialLikelihood(eps=eps)
    raise ValueError(
        f"unknown likelihood '{likelihood}'; expected "
        "'binomial' or 'beta_binomial'"
    )
