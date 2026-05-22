"""Debugging and diagnostic utilities for the Bayesian sentence-level UQ model.

Phase 7-2. Notebook-friendly helpers that surface common failure modes
spotted in the research document (§XV.3) and in CLAUDE.md "Critical
Guidelines": vanishing/exploding gradients on ψ, degenerate learned
layer mixture α, Fisher-scoring stalls, ``μ_j`` saturating the clip
boundary, and skewed ``m_j`` distributions.

See ``prompts/phase_7_2_debug.md`` for the spec and
``research_document_v8.md`` Parts III, VI, VII, XV for the math.

Each function is **diagnostic only** — it logs/plots/returns numbers,
never mutates state and never silences exceptions. All matplotlib calls
go through the ``Agg`` backend by default so this module is safe to
import in headless environments and from inside Jupyter (the user can
override the backend before importing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import torch
import torch.nn.functional as F

from src.features.extractor import SentenceUQParams
from src.models.fisher_scoring import (
    _compute_clipped_objective,
    _compute_grad_and_fisher,
)


__all__ = [
    "check_gradient_flow",
    "visualize_feature_distribution",
    "diagnose_fisher_scoring",
    "sanity_check_boundary_fraction",
    "check_m_j_distribution",
]


# Han et al. (2025) report layer 14 (1-indexed, out of Llama-3-8B's 32) as
# the empirically strongest single-layer probe. Used as a reference annotation
# in :func:`visualize_feature_distribution`.
_HAN_OPTIMAL_LAYER_1INDEXED = 14


# ---------------------------------------------------------------------------
# 1. Gradient-flow check on ψ
# ---------------------------------------------------------------------------


def check_gradient_flow(
    loss: torch.Tensor,
    params: SentenceUQParams,
) -> Dict[str, Optional[float]]:
    """Print and return the gradient norm for each component of ψ.

    Intended to be called immediately after ``loss.backward()`` during
    bilevel training (Phase 4-1). Reports the L2 norm of ``.grad`` for
    ``W``, ``alpha``, ``mu_0``, ``log_sigma_0`` and emits a warning for
    any parameter whose ``.grad`` is ``None`` — a strong signal that the
    unrolled Fisher loop dropped that parameter from the autograd graph
    (CLAUDE.md rule 9).

    Parameters
    ----------
    loss : Tensor of shape ``()``.
        The scalar outer loss. Passed for API symmetry only — the
        caller must have already invoked ``loss.backward()``. The tensor
        is inspected for ``requires_grad`` to flag callers that forgot.
    params : SentenceUQParams
        The feature-extractor parameter module ψ.

    Returns
    -------
    dict
        Maps each component name (``"W"``, ``"alpha"``, ``"mu_0"``,
        ``"log_sigma_0"``) to its grad L2 norm as a Python float, or
        ``None`` if the gradient is missing.
    """
    if not isinstance(params, SentenceUQParams):
        raise TypeError(
            "params must be a SentenceUQParams instance; "
            f"got {type(params).__name__}"
        )
    if not torch.is_tensor(loss):
        raise TypeError(f"loss must be a torch.Tensor; got {type(loss).__name__}")
    if not loss.requires_grad and loss.grad_fn is None:
        warnings.warn(
            "[check_gradient_flow] loss has no grad_fn — did you call "
            "loss.backward() already, or is the loss detached?",
            stacklevel=2,
        )

    components: Dict[str, torch.Tensor] = {
        "W": params.W.weight,
        "alpha": params.alpha,
        "mu_0": params.mu_0,
        "log_sigma_0": params.log_sigma_0,
    }

    norms: Dict[str, Optional[float]] = {}
    print("[check_gradient_flow] grad norms for ψ = (W, α, μ_0, log σ_0):")
    for name, p in components.items():
        if p.grad is None:
            warnings.warn(
                f"[check_gradient_flow] {name}.grad is None — gradient did "
                "not flow to this parameter. Check the differentiable Fisher "
                "unroll (CLAUDE.md rule 9).",
                stacklevel=2,
            )
            print(f"  {name:>12s}: None")
            norms[name] = None
        else:
            n = float(p.grad.detach().norm().item())
            print(f"  {name:>12s}: {n:.6e}")
            norms[name] = n

    return norms


# ---------------------------------------------------------------------------
# 2. Feature / layer-mixture visualisation
# ---------------------------------------------------------------------------


def visualize_feature_distribution(
    feature_params: SentenceUQParams,
    sample_hidden_states: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Plot projected-feature histograms and the learned layer mixture.

    Produces a two-panel figure:

    * **Top** — per-dimension histogram of the projected features
      ``W · h_ℓ^agg`` where ``h_ℓ^agg = Σ_l softmax(α)_l · h_ℓ^(l)``,
      for the supplied sample of hidden states. Useful for spotting
      collapsed or saturated projection dimensions.
    * **Bottom** — bar chart of the softmaxed layer weights
      ``softmax(α) ∈ R^{L_layers}``. A dashed reference line marks
      Han et al.'s reported optimal probing layer (layer 14, 1-indexed,
      from the 32-layer Llama-3-8B baseline) so the learned mixture
      can be compared at a glance.

    The reference line is only rendered when the supplied
    ``feature_params.num_layers`` is large enough to contain layer 14
    (otherwise the underlying probing setup differs and the comparison
    is meaningless).

    Parameters
    ----------
    feature_params : SentenceUQParams
        The trained feature-extractor parameters ψ.
    sample_hidden_states : Tensor of shape ``(T, num_layers, hidden_dim)``.
        A representative sample of token hidden states. May arrive as
        fp16; computation is promoted to fp32.
    save_path : str | Path, optional
        If given, the figure is written to this path (parent dirs are
        created). The figure is returned in either case.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not isinstance(feature_params, SentenceUQParams):
        raise TypeError(
            "feature_params must be a SentenceUQParams instance; "
            f"got {type(feature_params).__name__}"
        )
    if not torch.is_tensor(sample_hidden_states):
        raise TypeError(
            "sample_hidden_states must be a torch.Tensor; "
            f"got {type(sample_hidden_states).__name__}"
        )
    if sample_hidden_states.dim() != 3:
        raise ValueError(
            "sample_hidden_states must be (T, num_layers, hidden_dim); "
            f"got shape {tuple(sample_hidden_states.shape)}"
        )
    T, L, D = sample_hidden_states.shape
    if L != feature_params.num_layers:
        raise ValueError(
            f"sample_hidden_states has num_layers={L} but feature_params "
            f"expects {feature_params.num_layers}"
        )
    if D != feature_params.hidden_dim:
        raise ValueError(
            f"sample_hidden_states has hidden_dim={D} but feature_params "
            f"expects {feature_params.hidden_dim}"
        )

    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    with torch.no_grad():
        h = sample_hidden_states.to(torch.float32)
        w = F.softmax(feature_params.alpha.detach().to(torch.float32), dim=0)
        h_agg = torch.einsum("l,tld->td", w, h)                       # (T, d)
        h_proj = feature_params.W(h_agg)                              # (T, p)

    proj = h_proj.detach().cpu().numpy()
    weights = w.detach().cpu().numpy()
    p = proj.shape[1]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.0, 8.0), gridspec_kw={"height_ratios": [3.0, 2.0]}
    )

    # --- Top panel: per-dimension projected-feature histograms ---
    # Overlaid step histograms keep the plot legible even for p ~ 64.
    bins = 40
    for d in range(p):
        ax_top.hist(
            proj[:, d],
            bins=bins,
            histtype="step",
            linewidth=0.6,
            alpha=0.4,
        )
    ax_top.set_xlabel("projected feature value  (W · h_aggregated)")
    ax_top.set_ylabel("count")
    ax_top.set_title(
        f"Projected feature distribution (p={p} dims, T={T} tokens)"
    )
    ax_top.grid(alpha=0.3)

    # --- Bottom panel: learned layer weights softmax(α) ---
    layer_idx = list(range(1, len(weights) + 1))  # 1-indexed for plot labels
    ax_bot.bar(layer_idx, weights, color="C0", alpha=0.8, edgecolor="C0")
    ax_bot.set_xlabel("selected layer index (1-indexed within selection)")
    ax_bot.set_ylabel("softmax(α)")
    ax_bot.set_title("Learned layer mixture")
    ax_bot.set_xticks(layer_idx)
    ax_bot.grid(alpha=0.3, axis="y")

    if feature_params.num_layers >= _HAN_OPTIMAL_LAYER_1INDEXED:
        ax_bot.axvline(
            x=_HAN_OPTIMAL_LAYER_1INDEXED,
            linestyle="--",
            linewidth=1.5,
            color="C3",
            label=f"Han et al. optimal: layer {_HAN_OPTIMAL_LAYER_1INDEXED}",
        )
        ax_bot.legend(loc="upper right", fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)

    return fig


# ---------------------------------------------------------------------------
# 3. Fisher-scoring diagnostic loop
# ---------------------------------------------------------------------------


def diagnose_fisher_scoring(
    all_z_tokens: List[torch.Tensor],
    all_K: torch.Tensor,
    all_m: torch.Tensor,
    mu_0: torch.Tensor,
    Sigma_0_inv: torch.Tensor,
    eps: float = 1e-6,
    num_iters: int = 15,
    lambda_init: float = 1e-4,
) -> Dict[str, Any]:
    """Run a verbose damped Fisher-scoring inner loop and report diagnostics.

    Re-implements the iteration body from
    :func:`src.models.fisher_scoring._fisher_scoring_core` under
    ``torch.no_grad`` so each iteration can additionally log the
    smallest eigenvalue of the Fisher-type precision (a Laplace-validity
    signal) and the gradient norm. The damping schedule is identical to
    the production loop (success → ``λ ← max(λ/2, 1e-8)``; failure →
    ``λ ← 10 λ``; abort if ``λ > 1e10``).

    A non-convergence warning is emitted when the final gradient norm is
    not noticeably smaller than the initial one, or when the loop
    aborted via the damping cap.

    Also reports the ``m_j = 0`` skip count and the empirical
    ``m_j`` summary statistics — CLAUDE.md rule 8 demands these stay
    rare.

    Parameters
    ----------
    all_z_tokens : list of N tensors of shape ``(L_j, k)``.
    all_K : Tensor of shape ``(N,)``, integer dtype.
    all_m : Tensor of shape ``(N,)``, integer dtype. ``m_j = 0`` rows skipped.
    mu_0 : Tensor of shape ``(k,)``.
    Sigma_0_inv : Tensor of shape ``(k, k)``.
    eps : float, optional
        Clipping bound for ``μ_j``. Default ``1e-6``.
    num_iters : int, optional
        Max iterations (mirrors ``fisher_scoring_map``). Default 15.
    lambda_init : float, optional
        Initial damping. Default ``1e-4``.

    Returns
    -------
    dict with keys:
        ``theta_hat`` : Tensor of shape ``(k,)``.
        ``H_fisher_final`` : Tensor of shape ``(k, k)``.
        ``objectives`` : list[float], one per accepted iteration.
        ``grad_norms`` : list[float], one per iteration (pre-step).
        ``H_min_eigs`` : list[float], smallest eigenvalue of H per iteration.
        ``converged`` : bool, heuristic — final grad norm < 1% of initial.
        ``num_m_zero`` : int, count of ``m_j = 0`` sentences.
        ``m_summary``  : dict, min/max/mean/median of ``m_j``.
    """
    if len(all_K) != len(all_m) or len(all_K) != len(all_z_tokens):
        raise ValueError(
            "all_z_tokens, all_K, all_m must have the same length; "
            f"got {len(all_z_tokens)}, {len(all_K)}, {len(all_m)}"
        )

    k = mu_0.shape[0]
    if Sigma_0_inv.shape != (k, k):
        raise ValueError(
            f"Sigma_0_inv must be ({k}, {k}); got {tuple(Sigma_0_inv.shape)}"
        )

    # m_j summary (skip count + distribution).
    m_float = all_m.detach().to(torch.float64).cpu()
    num_m_zero = int((m_float == 0).sum().item())
    m_summary = {
        "min": float(m_float.min().item()) if m_float.numel() else float("nan"),
        "max": float(m_float.max().item()) if m_float.numel() else float("nan"),
        "mean": float(m_float.mean().item()) if m_float.numel() else float("nan"),
        "median": float(m_float.median().item()) if m_float.numel() else float("nan"),
    }
    print(
        f"[diagnose_fisher_scoring] N={len(all_m)} sentences, "
        f"m_j=0 (skipped)={num_m_zero}, "
        f"m_j summary: min={m_summary['min']:.1f} max={m_summary['max']:.1f} "
        f"mean={m_summary['mean']:.2f} median={m_summary['median']:.1f}"
    )

    objectives: List[float] = []
    grad_norms: List[float] = []
    H_min_eigs: List[float] = []

    initial_grad_norm: Optional[float] = None
    aborted = False

    with torch.no_grad():
        theta = mu_0.clone()
        lam = float(lambda_init)
        eye = torch.eye(k, device=mu_0.device, dtype=mu_0.dtype)

        prev_obj = _compute_clipped_objective(
            theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
        )
        print(f"[diagnose_fisher_scoring] initial objective={prev_obj.item():.6f}")

        for it in range(num_iters):
            grad, H = _compute_grad_and_fisher(
                theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
            )
            H_sym = 0.5 * (H + H.T)
            eigs = torch.linalg.eigvalsh(H_sym)
            min_eig = float(eigs.min().item())
            gn = float(grad.norm().item())

            grad_norms.append(gn)
            H_min_eigs.append(min_eig)
            if initial_grad_norm is None:
                initial_grad_norm = gn

            try:
                delta = torch.linalg.solve(H + lam * eye, grad)
            except RuntimeError:
                lam *= 10.0
                print(
                    f"[diagnose_fisher_scoring] iter {it}: linear solve failed, "
                    f"lam->{lam:.2e}  (grad_norm={gn:.4e}, H_min_eig={min_eig:.4e})"
                )
                if lam > 1e10:
                    aborted = True
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
                objectives.append(float(new_obj.item()))
                print(
                    f"[diagnose_fisher_scoring] iter {it}: "
                    f"obj={new_obj.item():.6f}  grad_norm={gn:.4e}  "
                    f"H_min_eig={min_eig:.4e}  lam={lam:.2e}"
                )
            else:
                lam *= 10.0
                print(
                    f"[diagnose_fisher_scoring] iter {it}: rejected "
                    f"(obj={new_obj.item():.6f} < prev={prev_obj.item():.6f})  "
                    f"grad_norm={gn:.4e}  H_min_eig={min_eig:.4e}  lam->{lam:.2e}"
                )
                if lam > 1e10:
                    aborted = True
                    break

        _, H_final = _compute_grad_and_fisher(
            theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps
        )

    final_grad_norm = grad_norms[-1] if grad_norms else float("inf")
    converged = (
        not aborted
        and initial_grad_norm is not None
        and final_grad_norm < 0.01 * initial_grad_norm
    )
    if aborted:
        warnings.warn(
            "[diagnose_fisher_scoring] aborted via damping cap (λ > 1e10). "
            "Consider tightening the prior (reduce prior_sigma) or "
            "increasing lambda_init.",
            stacklevel=2,
        )
    elif not converged:
        warnings.warn(
            "[diagnose_fisher_scoring] inner loop did not visibly converge: "
            f"final grad norm {final_grad_norm:.4e} vs initial "
            f"{initial_grad_norm if initial_grad_norm is not None else float('nan'):.4e}. "
            "Increase num_iters, tighten prior, or raise lambda_init.",
            stacklevel=2,
        )

    return {
        "theta_hat": theta,
        "H_fisher_final": H_final,
        "objectives": objectives,
        "grad_norms": grad_norms,
        "H_min_eigs": H_min_eigs,
        "converged": bool(converged),
        "num_m_zero": num_m_zero,
        "m_summary": m_summary,
    }


# ---------------------------------------------------------------------------
# 4. Boundary-fraction sanity check
# ---------------------------------------------------------------------------


def sanity_check_boundary_fraction(
    all_z_tokens: List[torch.Tensor],
    all_K: torch.Tensor,
    all_m: torch.Tensor,
    theta_hat: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Report the fraction of sentences whose ``μ_j`` saturates the clip.

    Recomputes ``μ_j(θ̂) = (1/L_j) Σ_ℓ σ(θ̂ᵀ z_ℓ)`` for every sentence
    with ``m_j > 0`` and checks whether ``μ_j ≤ ε`` or ``μ_j ≥ 1 - ε``.
    A high boundary fraction (>5 %) typically means the prior is too
    loose and θ̂ is being pulled to extreme regions where the binomial
    likelihood saturates — the recommended remediation is to tighten
    the prior (reduce ``σ_0``).

    Also returns the per-sentence raw ratio ``U_j = K_j / m_j`` paired
    with ``μ̂_j`` so the caller can build a calibration scatter plot
    (the spec mentions "distribution of U_j vs μ̂_j (scatter)").

    Parameters
    ----------
    all_z_tokens : list of N tensors of shape ``(L_j, k)``.
    all_K : Tensor of shape ``(N,)``, integer dtype.
    all_m : Tensor of shape ``(N,)``, integer dtype. ``m_j = 0`` rows skipped.
    theta_hat : Tensor of shape ``(k,)``.
        MAP estimate.
    eps : float, optional
        Clipping bound used during training. Default ``1e-6``.

    Returns
    -------
    dict with keys:
        ``mu_hat``         : Tensor of shape ``(N_used,)``, the
                              recomputed ``μ_j(θ̂)`` for ``m_j > 0`` rows.
        ``U_j``            : Tensor of shape ``(N_used,)``, ``K_j / m_j``.
        ``boundary_frac``  : float, fraction at either boundary.
        ``low_frac``       : float, fraction with ``μ_j ≤ ε``.
        ``high_frac``      : float, fraction with ``μ_j ≥ 1 - ε``.
        ``n_used``         : int, sentences considered (``m_j > 0``).
        ``recommend_tighter_prior`` : bool, ``boundary_frac > 0.05``.
    """
    if len(all_K) != len(all_m) or len(all_K) != len(all_z_tokens):
        raise ValueError(
            "all_z_tokens, all_K, all_m must have the same length; "
            f"got {len(all_z_tokens)}, {len(all_K)}, {len(all_m)}"
        )
    if theta_hat.dim() != 1:
        raise ValueError(
            f"theta_hat must be 1-D (k,); got shape {tuple(theta_hat.shape)}"
        )

    mu_list: List[float] = []
    U_list: List[float] = []

    with torch.no_grad():
        theta_d = theta_hat.detach().to(torch.float32)
        for j in range(len(all_K)):
            m_j_int = int(all_m[j].item()) if torch.is_tensor(all_m[j]) else int(all_m[j])
            if m_j_int == 0:
                continue
            z_j = all_z_tokens[j].detach().to(torch.float32)
            if z_j.dim() != 2 or z_j.shape[1] != theta_d.shape[0]:
                raise ValueError(
                    f"all_z_tokens[{j}] must be (L_j, k={theta_d.shape[0]}); "
                    f"got shape {tuple(z_j.shape)}"
                )
            pi_j = torch.sigmoid(z_j @ theta_d)
            mu_j = float(pi_j.mean().item())
            K_j = float(all_K[j].item()) if torch.is_tensor(all_K[j]) else float(all_K[j])
            mu_list.append(mu_j)
            U_list.append(K_j / m_j_int)

    n_used = len(mu_list)
    if n_used == 0:
        warnings.warn(
            "[sanity_check_boundary_fraction] no sentences with m_j > 0; "
            "nothing to check.",
            stacklevel=2,
        )
        return {
            "mu_hat": torch.empty(0),
            "U_j": torch.empty(0),
            "boundary_frac": 0.0,
            "low_frac": 0.0,
            "high_frac": 0.0,
            "n_used": 0,
            "recommend_tighter_prior": False,
        }

    mu_tensor = torch.tensor(mu_list, dtype=torch.float64)
    U_tensor = torch.tensor(U_list, dtype=torch.float64)
    low_mask = mu_tensor <= eps
    high_mask = mu_tensor >= 1.0 - eps
    low_frac = float(low_mask.float().mean().item())
    high_frac = float(high_mask.float().mean().item())
    boundary_frac = low_frac + high_frac
    recommend = boundary_frac > 0.05

    print(
        f"[sanity_check_boundary_fraction] N_used={n_used}, "
        f"boundary={boundary_frac * 100:.2f}% "
        f"(low={low_frac * 100:.2f}%, high={high_frac * 100:.2f}%)"
    )
    print(
        f"[sanity_check_boundary_fraction] U_j vs μ̂_j: "
        f"U mean={float(U_tensor.mean().item()):.3f}, "
        f"μ̂ mean={float(mu_tensor.mean().item()):.3f}"
    )
    if recommend:
        warnings.warn(
            f"[sanity_check_boundary_fraction] boundary fraction "
            f"{boundary_frac * 100:.2f}% exceeds 5% — consider tightening "
            "the prior (reduce prior_sigma / log_sigma_0) to keep μ_j off "
            "the clip.",
            stacklevel=2,
        )

    return {
        "mu_hat": mu_tensor,
        "U_j": U_tensor,
        "boundary_frac": boundary_frac,
        "low_frac": low_frac,
        "high_frac": high_frac,
        "n_used": n_used,
        "recommend_tighter_prior": recommend,
    }


# ---------------------------------------------------------------------------
# 5. m_j-distribution check
# ---------------------------------------------------------------------------


def check_m_j_distribution(all_m: torch.Tensor) -> Dict[str, Any]:
    """Summary statistics and skewness check on the per-sentence atom counts.

    Per ``§XV.3`` and the Phase 7-2 spec, two pathologies matter:

    * **Excess ``m_j = 0`` sentences** — these are skipped in every
      likelihood-related computation (CLAUDE.md rule 8); a large count
      means the annotation pipeline is producing many unverifiable
      sentences and the effective sample size is much smaller than ``N``.
    * **Highly skewed ``m_j``** — a handful of sentences with very large
      ``m_j`` can dominate the binomial likelihood and crowd out the
      tail. We flag this when ``max(m) / median(m) > 10`` (over the
      ``m_j > 0`` subset) which is a coarse but cheap proxy for the
      α-weighting concern in §XV.3.

    Parameters
    ----------
    all_m : Tensor of shape ``(N,)``, integer dtype.

    Returns
    -------
    dict with keys:
        ``min``, ``max``, ``mean``, ``median`` : float, full-sample
            summary statistics.
        ``num_m_zero`` : int, count of ``m_j = 0`` sentences.
        ``frac_m_zero`` : float, ``num_m_zero / N``.
        ``skew_ratio`` : float, ``max / median`` over ``m_j > 0``
            (``inf`` if ``median == 0``; ``nan`` if no ``m_j > 0`` rows).
        ``dominance_warning`` : bool, ``True`` when the distribution
            looks dominated by a few high-``m_j`` outliers.
    """
    if not torch.is_tensor(all_m):
        raise TypeError(f"all_m must be a torch.Tensor; got {type(all_m).__name__}")
    if all_m.numel() == 0:
        raise ValueError("all_m must be non-empty")
    if all_m.dim() != 1:
        raise ValueError(f"all_m must be 1-D; got shape {tuple(all_m.shape)}")

    m = all_m.detach().to(torch.float64).cpu()
    N = int(m.numel())
    m_min = float(m.min().item())
    m_max = float(m.max().item())
    m_mean = float(m.mean().item())
    m_median = float(m.median().item())
    num_m_zero = int((m == 0).sum().item())
    frac_m_zero = num_m_zero / N

    nonzero = m[m > 0]
    if nonzero.numel() == 0:
        skew_ratio = float("nan")
        dominance = False
    else:
        nz_median = float(nonzero.median().item())
        nz_max = float(nonzero.max().item())
        if nz_median == 0.0:
            skew_ratio = float("inf")
        else:
            skew_ratio = nz_max / nz_median
        dominance = skew_ratio > 10.0

    print(
        f"[check_m_j_distribution] N={N}, "
        f"min={m_min:.1f} max={m_max:.1f} mean={m_mean:.2f} median={m_median:.1f}"
    )
    print(
        f"[check_m_j_distribution] m_j=0: {num_m_zero}/{N} "
        f"({frac_m_zero * 100:.2f}%);  "
        f"max/median over m_j>0: {skew_ratio:.2f}"
    )

    if num_m_zero > 0:
        warnings.warn(
            f"[check_m_j_distribution] {num_m_zero} sentences have m_j=0 "
            "and will be skipped in the likelihood (CLAUDE.md rule 8). "
            "Check the annotation pipeline if this count looks excessive.",
            stacklevel=2,
        )
    if dominance:
        warnings.warn(
            f"[check_m_j_distribution] m_j distribution is highly skewed "
            f"(max/median = {skew_ratio:.2f} > 10) — a few high-m_j "
            "sentences may dominate the binomial objective (§XV.3); "
            "consider the α-weighting ablation.",
            stacklevel=2,
        )

    return {
        "min": m_min,
        "max": m_max,
        "mean": m_mean,
        "median": m_median,
        "num_m_zero": num_m_zero,
        "frac_m_zero": frac_m_zero,
        "skew_ratio": skew_ratio,
        "dominance_warning": bool(dominance),
    }
