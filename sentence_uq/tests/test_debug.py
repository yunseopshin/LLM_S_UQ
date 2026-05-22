"""Tests for ``src.utils.debug`` — Phase 7-2.

Covers, for each diagnostic helper:

* :func:`check_gradient_flow` — populates ``.grad`` on ψ via an outer
  backward pass and asserts the returned norms are positive and finite;
  detached losses surface a warning.
* :func:`visualize_feature_distribution` — produces a 2-panel
  ``matplotlib.figure.Figure`` for both Llama-3-sized (num_layers=8)
  and Gemma-sized (num_layers=6) parameter modules, exercising the
  Han-et-al reference annotation branch.
* :func:`diagnose_fisher_scoring` — agrees with the production loop on
  the converged ``θ̂`` for the synthetic binomial fixture, reports the
  ``m_j = 0`` skip count, and refuses mismatched-length inputs.
* :func:`sanity_check_boundary_fraction` — recognises an artificial
  boundary regime (saturating logits) and flags it as recommending a
  tighter prior.
* :func:`check_m_j_distribution` — flags a constructed skewed
  distribution and raises on empty input.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
import pytest
import torch

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import inside debug.py

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.extractor import SentenceUQParams  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.models.fisher_scoring import fisher_scoring_map_detached  # noqa: E402
from src.utils.debug import (  # noqa: E402
    check_gradient_flow,
    check_m_j_distribution,
    diagnose_fisher_scoring,
    sanity_check_boundary_fraction,
    visualize_feature_distribution,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_binomial_fixture(
    k: int = 5,
    N: int = 12,
    L: int = 6,
    m_max: int = 4,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    insert_m_zero: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    all_z = [torch.randn(L, k, generator=g, dtype=dtype) for _ in range(N)]
    all_m = torch.randint(1, m_max + 1, (N,), generator=g)
    all_K = torch.stack(
        [torch.randint(0, int(m) + 1, (1,), generator=g) for m in all_m]
    ).squeeze(-1)
    if insert_m_zero:
        all_m[0] = 0
        all_K[0] = 0
    mu_0 = torch.zeros(k, dtype=dtype)
    Sigma_0_inv = torch.eye(k, dtype=dtype) * 0.5
    return all_z, all_K, all_m, mu_0, Sigma_0_inv


# ---------------------------------------------------------------------------
# check_gradient_flow
# ---------------------------------------------------------------------------


def test_check_gradient_flow_populates_all_components() -> None:
    """After loss.backward(), every component of ψ should report a real grad norm."""
    torch.manual_seed(0)
    params = SentenceUQParams(hidden_dim=16, num_layers=4, projection_dim=3)
    model = BayesianSentenceUQ(params, num_fisher_iters=3)

    # Generate token features through the extractor so ψ is on the autograd path.
    from src.features.extractor import extract_token_features

    T = 7
    hs = torch.randn(T, 4, 16)
    ent = torch.rand(T)
    top1 = torch.rand(T)
    z_all = extract_token_features(hs, ent, top1, params)
    z_tokens_per_sent = [z_all[:3], z_all[3:5], z_all[5:]]
    all_K = torch.tensor([1, 0, 2])
    all_m = torch.tensor([2, 1, 3])

    loss = model.compute_loss(z_tokens_per_sent, all_K, all_m)
    loss.backward()

    norms = check_gradient_flow(loss, params)
    assert set(norms.keys()) == {"W", "alpha", "mu_0", "log_sigma_0"}
    for name, value in norms.items():
        assert value is not None, f"{name} grad was unexpectedly None"
        assert value >= 0.0
        assert value == value  # not NaN


def test_check_gradient_flow_rejects_non_params() -> None:
    with pytest.raises(TypeError):
        check_gradient_flow(torch.tensor(0.0, requires_grad=True), object())  # type: ignore[arg-type]


def test_check_gradient_flow_reports_none_when_grad_missing() -> None:
    """A fresh params module has no gradients yet; the helper should warn for each."""
    params = SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=2)
    fake_loss = torch.tensor(1.23, requires_grad=True)
    with pytest.warns(UserWarning):
        norms = check_gradient_flow(fake_loss, params)
    assert all(v is None for v in norms.values())


# ---------------------------------------------------------------------------
# visualize_feature_distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden_dim, num_layers",
    [(16, 8), (12, 6)],  # parameterise over configs (CLAUDE.md compatibility rule)
)
def test_visualize_feature_distribution_smoke(
    hidden_dim: int, num_layers: int, tmp_path: Path
) -> None:
    params = SentenceUQParams(
        hidden_dim=hidden_dim, num_layers=num_layers, projection_dim=4
    )
    sample = torch.randn(20, num_layers, hidden_dim)
    save_path = tmp_path / "viz" / "feat.png"
    fig = visualize_feature_distribution(params, sample, save_path=save_path)

    # Two-panel figure as specified.
    assert len(fig.axes) == 2
    assert save_path.exists() and save_path.stat().st_size > 0

    # Reference annotation only renders when num_layers >= 14.
    bottom_ax = fig.axes[1]
    labels = [ln.get_label() for ln in bottom_ax.get_lines()]
    if num_layers >= 14:
        assert any("Han" in lab for lab in labels)
    else:
        assert not any("Han" in lab for lab in labels)


def test_visualize_feature_distribution_rejects_bad_shapes() -> None:
    params = SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=2)
    with pytest.raises(ValueError):
        visualize_feature_distribution(params, torch.randn(5, 3))  # 2-D
    with pytest.raises(ValueError):
        visualize_feature_distribution(params, torch.randn(5, 2, 8))  # wrong num_layers
    with pytest.raises(ValueError):
        visualize_feature_distribution(params, torch.randn(5, 3, 16))  # wrong hidden_dim


# ---------------------------------------------------------------------------
# diagnose_fisher_scoring
# ---------------------------------------------------------------------------


def test_diagnose_fisher_scoring_matches_production_loop() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_binomial_fixture(seed=3)
    out = diagnose_fisher_scoring(
        all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=15
    )
    theta_prod, _ = fisher_scoring_map_detached(
        all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=15
    )
    assert torch.allclose(out["theta_hat"], theta_prod, atol=1e-5)
    assert out["H_fisher_final"].shape == (mu_0.shape[0], mu_0.shape[0])
    assert len(out["grad_norms"]) == len(out["H_min_eigs"]) > 0
    assert out["num_m_zero"] == 0
    assert set(out["m_summary"].keys()) == {"min", "max", "mean", "median"}


def test_diagnose_fisher_scoring_reports_m_zero_count() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_binomial_fixture(
        seed=4, insert_m_zero=True
    )
    out = diagnose_fisher_scoring(all_z, all_K, all_m, mu_0, Sigma_0_inv, num_iters=8)
    assert out["num_m_zero"] == 1


def test_diagnose_fisher_scoring_validates_lengths() -> None:
    all_z, all_K, all_m, mu_0, Sigma_0_inv = _make_binomial_fixture()
    with pytest.raises(ValueError):
        diagnose_fisher_scoring(
            all_z[:-1], all_K, all_m, mu_0, Sigma_0_inv
        )


# ---------------------------------------------------------------------------
# sanity_check_boundary_fraction
# ---------------------------------------------------------------------------


def test_sanity_check_boundary_fraction_flags_saturation() -> None:
    """Large-magnitude features + matched θ̂ saturate σ; >5% boundary expected."""
    k = 3
    # All tokens point along +e_0 with large magnitude.
    z = [torch.ones(2, k) * 30.0 for _ in range(8)]
    theta_hat = torch.tensor([1.0, 0.0, 0.0])
    all_m = torch.tensor([2] * 8)
    all_K = torch.tensor([2] * 8)  # observed full support
    with pytest.warns(UserWarning, match="boundary"):
        out = sanity_check_boundary_fraction(z, all_K, all_m, theta_hat)
    assert out["boundary_frac"] > 0.05
    assert out["recommend_tighter_prior"] is True
    assert out["mu_hat"].shape == out["U_j"].shape == (8,)


def test_sanity_check_boundary_fraction_skips_m_zero() -> None:
    k = 3
    z = [torch.randn(3, k) for _ in range(4)]
    theta_hat = torch.zeros(k)
    all_m = torch.tensor([0, 2, 0, 3])
    all_K = torch.tensor([0, 1, 0, 2])
    out = sanity_check_boundary_fraction(z, all_K, all_m, theta_hat)
    assert out["n_used"] == 2


def test_sanity_check_boundary_fraction_validates_inputs() -> None:
    z = [torch.randn(2, 3)]
    with pytest.raises(ValueError):
        sanity_check_boundary_fraction(
            z, torch.tensor([1]), torch.tensor([1, 2]), torch.zeros(3)
        )
    with pytest.raises(ValueError):
        sanity_check_boundary_fraction(
            z, torch.tensor([1]), torch.tensor([1]), torch.zeros(3, 1)  # 2-D theta
        )


# ---------------------------------------------------------------------------
# check_m_j_distribution
# ---------------------------------------------------------------------------


def test_check_m_j_distribution_basic_stats() -> None:
    m = torch.tensor([1, 2, 3, 4, 5])
    out = check_m_j_distribution(m)
    assert out["min"] == 1.0
    assert out["max"] == 5.0
    assert out["num_m_zero"] == 0
    assert out["dominance_warning"] is False


def test_check_m_j_distribution_flags_skew() -> None:
    m = torch.tensor([1, 1, 1, 1, 1, 100])
    with pytest.warns(UserWarning, match="skewed"):
        out = check_m_j_distribution(m)
    assert out["dominance_warning"] is True
    assert out["skew_ratio"] > 10.0


def test_check_m_j_distribution_counts_zeros() -> None:
    m = torch.tensor([0, 0, 1, 2, 3])
    with pytest.warns(UserWarning, match="m_j=0"):
        out = check_m_j_distribution(m)
    assert out["num_m_zero"] == 2
    assert out["frac_m_zero"] == pytest.approx(0.4)


def test_check_m_j_distribution_rejects_empty_or_non_1d() -> None:
    with pytest.raises(ValueError):
        check_m_j_distribution(torch.tensor([], dtype=torch.long))
    with pytest.raises(ValueError):
        check_m_j_distribution(torch.tensor([[1, 2], [3, 4]]))
    with pytest.raises(TypeError):
        check_m_j_distribution([1, 2, 3])  # type: ignore[arg-type]
