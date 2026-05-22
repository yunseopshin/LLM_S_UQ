"""Predictive inference for the Bayesian sentence-level UQ model.

Phase 3-3. Consumes a trained Laplace posterior ``p(θ | D) ≈ N(θ̂, Σ̂)``
together with the feature extractor ``ψ`` and produces, for every test
sentence, the four-level uncertainty decomposition (latent / ratio /
count / strict) of ``research_document_v8.md`` Parts IV and V.

Mathematical contract
---------------------
Given ``z_ℓ = φ_ψ(h_ℓ) ∈ R^k`` for ``ℓ = 1, …, L_*`` and a sentence's
atomic-fact count ``m_*`` (optional)::

    π̂_ℓ        = σ(θ̂ᵀ z_ℓ)                                  (1)
    μ̂          = (1 / L_*) Σ_ℓ π̂_ℓ                          (2)
    g_ℓ        = π̂_ℓ (1 - π̂_ℓ) z_ℓ                          (3)
    ĝ          = (1 / L_*) Σ_ℓ g_ℓ                           (4)

Latent epistemic (delta method, §4.3)::

    Epi_μ      = ĝᵀ Σ̂ ĝ                                      (5)

Ratio level (§4.2 / §4.3, given ``m_*``)::

    Aleatoric_U = max(0, [μ̂(1-μ̂) - Epi_μ] / m_*)              (6)
    Total_U     = Aleatoric_U + Epi_μ                          (7)

Count level (§4.4, given ``m_*``)::

    Epi_K       = m_*² · Epi_μ                                 (8)
    Aleatoric_K = m_* · max(0, μ̂(1-μ̂) - Epi_μ)                (9)

Strict factuality (§4.4, given ``m_*``)::

    p(A_*=1)   = μ̂^{m_*}                                     (10a)
    (MC form) = E_θ[μ(θ)^{m_*}]                              (10b)

Token-level attribution (§4.5)::

    Attr_ℓ      = (1 / L_*) g_ℓᵀ Σ̂ ĝ        (signed, Σ_ℓ = Epi_μ)
    LocalEpi_ℓ  = [π̂_ℓ(1-π̂_ℓ)]² · z_ℓᵀ Σ̂ z_ℓ          (always ≥ 0)

Probit-shrinkage Bayesian predictive (§5.2)::

    π̃_ℓ        = σ(θ̂ᵀ z_ℓ / sqrt(1 + (π/8) · z_ℓᵀ Σ̂ z_ℓ))
    μ̃          = (1 / L_*) Σ_ℓ π̃_ℓ

Design notes
------------
* Computation runs in fp32 throughout (CLAUDE.md rule 10) regardless of
  the dtype of the cached hidden states / features.
* When ``m_j is None`` the ratio / count / strict-factuality fields are
  set to ``None`` and only latent-level quantities are reported, matching
  the spec's "report only latent-level Epi_μ" rule.
* :class:`BatchPredictor` exposes a single forward pass over a list of
  sentences and vectorises the dense matmuls (``Σ̂`` is shared).
* The MC variant samples ``θ^(s) ~ N(θ̂, Σ̂)`` via a Cholesky factor of
  the symmetrised posterior covariance with adaptive jitter for
  numerical PD safety.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.features.extractor import (
    SentenceUQParams,
    extract_sentence_token_features,
)


__all__ = [
    "Predictor",
    "BatchPredictor",
    "save_trained_model",
    "load_trained_model",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stable_cholesky(matrix: torch.Tensor, max_tries: int = 6) -> torch.Tensor:
    """Cholesky factor of a (near-)PSD covariance with adaptive jitter.

    Symmetrises the input, then tries ``L = chol(matrix + j I)`` for
    ``j ∈ {0, 1e-8, 1e-7, …}`` until success.

    Parameters
    ----------
    matrix : Tensor of shape ``(k, k)``.
    max_tries : int
        Maximum number of jitter doublings before raising.

    Returns
    -------
    Tensor of shape ``(k, k)`` — lower-triangular Cholesky factor.
    """
    k = matrix.shape[0]
    sym = 0.5 * (matrix + matrix.T)
    eye = torch.eye(k, dtype=sym.dtype, device=sym.device)
    jitter = 0.0
    for attempt in range(max_tries):
        try:
            return torch.linalg.cholesky(sym + jitter * eye)
        except RuntimeError:
            jitter = 1e-8 if jitter == 0.0 else jitter * 10.0
    raise RuntimeError(
        f"Cholesky failed after {max_tries} jitter increases (last={jitter})"
    )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor:
    """Post-training predictive inference for a single sentence.

    Wraps the trained Laplace posterior ``(θ̂, Σ̂)`` and the feature
    extractor ``ψ`` and exposes the four-level uncertainty decomposition
    plus token-level attribution and Monte-Carlo verification.

    Parameters
    ----------
    theta_hat : Tensor of shape ``(k,)``.
        MAP from the Phase 3-1 Fisher-scoring inner loop.
    Sigma_hat : Tensor of shape ``(k, k)``.
        Laplace posterior covariance — inverse of the Fisher-type
        precision at ``θ̂``. Caller is responsible for inversion.
    feature_params : SentenceUQParams
        Trained feature-extractor parameters ψ.
    use_probit_shrinkage : bool, optional
        When ``True`` the strict-factuality plug-in uses the probit-
        shrunk mean ``μ̃`` instead of ``μ̂``. The probit-shrunk
        sentence mean is always returned in ``p_factual_probit``.
        Default ``False``.
    """

    def __init__(
        self,
        theta_hat: torch.Tensor,
        Sigma_hat: torch.Tensor,
        feature_params: SentenceUQParams,
        use_probit_shrinkage: bool = False,
    ) -> None:
        if not isinstance(feature_params, SentenceUQParams):
            raise TypeError(
                "feature_params must be a SentenceUQParams instance; "
                f"got {type(feature_params).__name__}"
            )
        if theta_hat.dim() != 1:
            raise ValueError(
                f"theta_hat must be 1-D (k,); got shape {tuple(theta_hat.shape)}"
            )
        k = theta_hat.shape[0]
        if Sigma_hat.shape != (k, k):
            raise ValueError(
                f"Sigma_hat must be ({k}, {k}); got {tuple(Sigma_hat.shape)}"
            )
        if k != feature_params.feature_dim:
            raise ValueError(
                f"theta_hat dim k={k} != feature_params.feature_dim="
                f"{feature_params.feature_dim}"
            )

        self.theta_hat = theta_hat.detach().to(torch.float32)
        self.Sigma_hat = Sigma_hat.detach().to(torch.float32)
        self.feature_params = feature_params
        self.use_probit_shrinkage = bool(use_probit_shrinkage)
        self.k = k

    # ------------------------------------------------------------------
    # Single-sentence predictive
    # ------------------------------------------------------------------

    def predict_sentence(
        self,
        z_tokens: torch.Tensor,
        m_j: Optional[int] = None,
    ) -> Dict[str, Optional[Union[float, torch.Tensor]]]:
        """Four-level decomposition + token-level attribution for one sentence.

        Implements equations (1)–(10) of the module docstring.

        Parameters
        ----------
        z_tokens : Tensor of shape ``(L_j, k)``.
            Per-token features ``z_ℓ`` for the sentence.
        m_j : int, optional
            Atomic-fact count. When ``None``, the ratio / count / strict
            fields in the returned dict are ``None``.

        Returns
        -------
        dict with keys:
            ``mu_hat``                : float
            ``p_factual_probit``      : float
            ``epi_mu``                : float
            ``aleatoric_U``           : float | None
            ``total_U``               : float | None
            ``epi_K``                 : float | None
            ``aleatoric_K``           : float | None
            ``p_strict_factual``      : float | None
            ``token_pi``              : Tensor of shape ``(L_j,)``
            ``token_attr``            : Tensor of shape ``(L_j,)``
            ``token_local_epi``       : Tensor of shape ``(L_j,)``
        """
        if z_tokens.dim() != 2:
            raise ValueError(
                f"z_tokens must be 2-D (L_j, k); got shape {tuple(z_tokens.shape)}"
            )
        L_j, k = z_tokens.shape
        if k != self.k:
            raise ValueError(
                f"z_tokens last dim {k} != theta_hat dim {self.k}"
            )
        if L_j == 0:
            raise ValueError("z_tokens must contain at least one token (L_j >= 1)")
        if m_j is not None and m_j < 0:
            raise ValueError(f"m_j must be non-negative, got {m_j}")

        with torch.no_grad():
            z = z_tokens.to(torch.float32)
            theta = self.theta_hat
            Sigma = self.Sigma_hat

            # ---- (1)–(4): per-token / sentence stats ----
            logits = z @ theta                                   # (L_j,)
            pi = torch.sigmoid(logits)                           # (L_j,)
            mu_hat = pi.mean()                                   # ()
            w = pi * (1.0 - pi)                                  # (L_j,)
            g_tokens = w.unsqueeze(1) * z                        # (L_j, k)
            g_hat = g_tokens.mean(dim=0)                         # (k,)

            # ---- (5): latent-level epistemic ----
            Sg = Sigma @ g_hat                                   # (k,)
            epi_mu = float((g_hat @ Sg).clamp_min(0.0).item())

            # ---- Token attribution Attr_ℓ = (1/L) g_ℓᵀ Σ ĝ ----
            token_attr = (g_tokens @ Sg) / float(L_j)            # (L_j,)

            # ---- LocalEpi_ℓ = w_ℓ² · z_ℓᵀ Σ z_ℓ ----
            Sz = z @ Sigma                                       # (L_j, k)
            zSz = (Sz * z).sum(dim=1)                            # (L_j,)
            token_local_epi = (w * w) * zSz                      # (L_j,)
            token_local_epi = token_local_epi.clamp_min(0.0)

            # ---- Probit-shrinkage (§5.2) ----
            shrink = torch.sqrt(1.0 + (math.pi / 8.0) * zSz.clamp_min(0.0))
            pi_probit = torch.sigmoid(logits / shrink)           # (L_j,)
            mu_probit = pi_probit.mean()                         # ()

            mu_hat_f = float(mu_hat.item())
            mu_probit_f = float(mu_probit.item())

            mu_var = mu_hat_f * (1.0 - mu_hat_f)
            mu_var_probit = mu_probit_f * (1.0 - mu_probit_f)

            out: Dict[str, Optional[Union[float, torch.Tensor]]] = {
                "mu_hat": mu_hat_f,
                "p_factual_probit": mu_probit_f,
                "epi_mu": epi_mu,
                "aleatoric_U": None,
                "total_U": None,
                "epi_K": None,
                "aleatoric_K": None,
                "p_strict_factual": None,
                "token_pi": pi.contiguous(),
                "token_attr": token_attr.contiguous(),
                "token_local_epi": token_local_epi.contiguous(),
            }

            if m_j is not None:
                m_j_int = int(m_j)
                if m_j_int == 0:
                    # ratio / count / strict are undefined for m_j = 0
                    return out

                # Use unclipped Bernoulli variance for the *parenthetical*,
                # then clip at zero per the well-definedness caveat (§4.3).
                inner = max(0.0, mu_var - epi_mu)
                aleatoric_U = inner / float(m_j_int)
                total_U = aleatoric_U + epi_mu
                epi_K = float(m_j_int) ** 2 * epi_mu
                aleatoric_K = float(m_j_int) * inner

                base_mu = mu_probit_f if self.use_probit_shrinkage else mu_hat_f
                base_mu = min(max(base_mu, 0.0), 1.0)
                p_strict = base_mu ** m_j_int

                out["aleatoric_U"] = float(aleatoric_U)
                out["total_U"] = float(total_U)
                out["epi_K"] = float(epi_K)
                out["aleatoric_K"] = float(aleatoric_K)
                out["p_strict_factual"] = float(p_strict)

            return out

    # ------------------------------------------------------------------
    # End-to-end from hidden states
    # ------------------------------------------------------------------

    def predict_from_hidden_states(
        self,
        hidden_states: torch.Tensor,
        entropy: torch.Tensor,
        top1: torch.Tensor,
        token_range: Tuple[int, int],
        m_j: Optional[int] = None,
    ) -> Dict[str, Optional[Union[float, torch.Tensor]]]:
        """High-level wrapper: feature extraction + :meth:`predict_sentence`.

        Parameters
        ----------
        hidden_states : Tensor of shape ``(T, num_layers, hidden_dim)``.
            Layer-wise hidden states for the whole generation.
        entropy : Tensor of shape ``(T,)``.
            Cached per-token predictive entropy (Phase 1-3).
        top1 : Tensor of shape ``(T,)``.
            Cached per-token top-1 probability (Phase 1-3).
        token_range : tuple ``(start, end)``
            Half-open token-index interval of the sentence.
        m_j : int, optional
            See :meth:`predict_sentence`.

        Returns
        -------
        dict
            Same schema as :meth:`predict_sentence`.
        """
        with torch.no_grad():
            z = extract_sentence_token_features(
                hidden_states,
                entropy,
                top1,
                token_range=token_range,
                params=self.feature_params,
            )
        return self.predict_sentence(z, m_j=m_j)

    # ------------------------------------------------------------------
    # Monte-Carlo verification
    # ------------------------------------------------------------------

    def predict_mc_epistemic(
        self,
        z_tokens: torch.Tensor,
        num_samples: int = 100,
        m_j: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Optional[float]]:
        """Monte-Carlo verification of the delta-method decomposition.

        Samples ``θ^(s) ~ N(θ̂, Σ̂)`` (Cholesky with adaptive jitter for
        PD safety) and computes ``μ_*(θ^(s)) = (1/L) Σ_ℓ σ(θ^(s)ᵀ z_ℓ)``
        for each sample. The latent-level MC epistemic is the sample
        variance of ``{μ_*(θ^(s))}``. When ``m_j`` is given, the law of
        total variance gives ratio- and count-level MC estimates:

            Aleatoric_U^MC = E[μ(1-μ)] / m_*
            Epistemic_U^MC = Var[μ]
            Aleatoric_K^MC = m_* · E[μ(1-μ)]
            Epistemic_K^MC = m_*² · Var[μ]
            p(A_*=1)^MC    = E[μ^{m_*}]      (posterior-averaged)

        Parameters
        ----------
        z_tokens : Tensor of shape ``(L_j, k)``.
        num_samples : int, optional
            Number of θ samples. Default 100.
        m_j : int, optional
            Atomic-fact count (see :meth:`predict_sentence`).
        generator : torch.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        dict with keys:
            ``mc_mu_mean``        : float
            ``mc_epi_mu``         : float (sample variance of μ)
            ``mc_aleatoric_U``    : float | None
            ``mc_total_U``        : float | None
            ``mc_epi_K``          : float | None
            ``mc_aleatoric_K``    : float | None
            ``mc_p_strict_factual`` : float | None
        """
        if z_tokens.dim() != 2:
            raise ValueError(
                f"z_tokens must be 2-D (L_j, k); got shape {tuple(z_tokens.shape)}"
            )
        if z_tokens.shape[1] != self.k:
            raise ValueError(
                f"z_tokens last dim {z_tokens.shape[1]} != theta_hat dim {self.k}"
            )
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")

        with torch.no_grad():
            z = z_tokens.to(torch.float32)
            L_chol = _stable_cholesky(self.Sigma_hat)            # (k, k)

            noise = torch.randn(
                self.k, num_samples,
                generator=generator,
                dtype=torch.float32,
                device=z.device,
            )
            # θ samples: (k, S) — broadcasted column-wise.
            theta_samples = self.theta_hat.unsqueeze(1) + L_chol @ noise

            # μ samples: (L_j, S) → (S,)
            logits = z @ theta_samples                           # (L_j, S)
            pi_samples = torch.sigmoid(logits)                   # (L_j, S)
            mu_samples = pi_samples.mean(dim=0)                  # (S,)

            mc_mu_mean = float(mu_samples.mean().item())
            # Use unbiased=False to keep the estimator non-negative
            # bounded; matches the population variance interpretation.
            mc_epi_mu = float(mu_samples.var(unbiased=False).item())

            out: Dict[str, Optional[float]] = {
                "mc_mu_mean": mc_mu_mean,
                "mc_epi_mu": mc_epi_mu,
                "mc_aleatoric_U": None,
                "mc_total_U": None,
                "mc_epi_K": None,
                "mc_aleatoric_K": None,
                "mc_p_strict_factual": None,
            }

            if m_j is not None and int(m_j) > 0:
                m_j_int = int(m_j)
                bern_var = float(
                    (mu_samples * (1.0 - mu_samples)).mean().item()
                )
                aleatoric_U = bern_var / float(m_j_int)
                aleatoric_K = float(m_j_int) * bern_var
                epi_K = float(m_j_int) ** 2 * mc_epi_mu
                total_U = aleatoric_U + mc_epi_mu
                p_strict = float(
                    mu_samples.clamp(0.0, 1.0).pow(m_j_int).mean().item()
                )
                out["mc_aleatoric_U"] = aleatoric_U
                out["mc_total_U"] = total_U
                out["mc_epi_K"] = epi_K
                out["mc_aleatoric_K"] = aleatoric_K
                out["mc_p_strict_factual"] = p_strict

            return out


# ---------------------------------------------------------------------------
# BatchPredictor
# ---------------------------------------------------------------------------


class BatchPredictor:
    """Vectorised predictive inference over a list of sentences.

    Re-uses a single :class:`Predictor` and processes a list of
    ``z_tokens`` tensors (each ``(L_j, k)``) with per-sentence optional
    ``m_j``. Because ``Σ̂`` is shared across all sentences, the dense
    matmuls inside :meth:`Predictor.predict_sentence` are already
    O(L_j · k²); the batch layer keeps the API ergonomic while ensuring
    we hit the cached ``Σ̂`` only once per call.

    Parameters
    ----------
    predictor : Predictor
        Trained single-sentence predictor.
    """

    def __init__(self, predictor: Predictor) -> None:
        if not isinstance(predictor, Predictor):
            raise TypeError(
                "predictor must be a Predictor instance; "
                f"got {type(predictor).__name__}"
            )
        self.predictor = predictor

    def predict(
        self,
        all_z_tokens: List[torch.Tensor],
        all_m: Optional[List[Optional[int]]] = None,
    ) -> List[Dict[str, Optional[Union[float, torch.Tensor]]]]:
        """Run :meth:`Predictor.predict_sentence` for every input sentence.

        Parameters
        ----------
        all_z_tokens : list of N tensors of shape ``(L_j, k)``.
        all_m : list of N (int | None), optional
            Per-sentence atomic-fact counts. ``None`` for a given index
            means "report latent-level only" for that sentence. If the
            whole argument is omitted, all sentences get ``m_j = None``.

        Returns
        -------
        list of dicts, each as in :meth:`Predictor.predict_sentence`.
        """
        N = len(all_z_tokens)
        if all_m is None:
            all_m = [None] * N
        if len(all_m) != N:
            raise ValueError(
                f"all_m has length {len(all_m)} but all_z_tokens has {N}"
            )

        results: List[Dict[str, Optional[Union[float, torch.Tensor]]]] = []
        for z, m in zip(all_z_tokens, all_m):
            results.append(self.predictor.predict_sentence(z, m_j=m))
        return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_trained_model(
    path: Union[str, Path],
    theta_hat: torch.Tensor,
    Sigma_hat: torch.Tensor,
    feature_params: SentenceUQParams,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the Laplace posterior and feature-extractor parameters.

    The output ``.pt`` payload contains:

        {
            "theta_hat":              (k,) fp32 Tensor on CPU,
            "Sigma_hat":              (k, k) fp32 Tensor on CPU,
            "feature_params_state_dict": ψ.state_dict() (CPU tensors),
            "feature_params_config":  {hidden_dim, num_layers, projection_dim},
            "extra":                  arbitrary user-supplied dict (optional),
        }

    Parameters
    ----------
    path : str | Path
        Destination ``.pt`` path. Parent directory is created if missing.
    theta_hat, Sigma_hat, feature_params
        See :class:`Predictor`.
    extra : dict, optional
        Arbitrary user metadata (e.g. training config, model name).
    """
    if not isinstance(feature_params, SentenceUQParams):
        raise TypeError(
            "feature_params must be a SentenceUQParams instance; "
            f"got {type(feature_params).__name__}"
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "theta_hat": theta_hat.detach().to("cpu", torch.float32).contiguous(),
        "Sigma_hat": Sigma_hat.detach().to("cpu", torch.float32).contiguous(),
        "feature_params_state_dict": {
            k: v.detach().to("cpu") for k, v in feature_params.state_dict().items()
        },
        "feature_params_config": {
            "hidden_dim": int(feature_params.hidden_dim),
            "num_layers": int(feature_params.num_layers),
            "projection_dim": int(feature_params.projection_dim),
        },
        "extra": dict(extra) if extra is not None else {},
    }
    torch.save(payload, out_path)


def load_trained_model(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> Dict[str, Any]:
    """Inverse of :func:`save_trained_model`.

    Parameters
    ----------
    path : str | Path
        Source ``.pt`` path.
    map_location : str | torch.device, optional
        Forwarded to :func:`torch.load`. Default ``"cpu"``.

    Returns
    -------
    dict with keys:
        ``theta_hat``      : Tensor of shape ``(k,)``
        ``Sigma_hat``      : Tensor of shape ``(k, k)``
        ``feature_params`` : :class:`SentenceUQParams` (with loaded state)
        ``extra``          : dict — whatever was passed to ``save_trained_model``
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"trained-model file not found: {src}")

    payload = torch.load(src, map_location=map_location, weights_only=False)
    cfg = payload["feature_params_config"]
    feature_params = SentenceUQParams(
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        projection_dim=int(cfg["projection_dim"]),
    )
    feature_params.load_state_dict(payload["feature_params_state_dict"])
    return {
        "theta_hat": payload["theta_hat"],
        "Sigma_hat": payload["Sigma_hat"],
        "feature_params": feature_params,
        "extra": payload.get("extra", {}),
    }
