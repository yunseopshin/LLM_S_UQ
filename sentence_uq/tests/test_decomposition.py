"""Tests for ``src.inference.predict`` — Phase 3-3.

Covers the invariants and equivalences from
``prompts/phase_3_3_predict.md`` and ``research_document_v8.md``
Parts IV & V:

* Basic invariants — ``mu_hat ∈ [0, 1]``, ``epi_mu >= 0``,
  ``aleatoric_U >= 0`` (after clipping), ``token_local_epi >= 0``.
* Token attribution — ``sum_ℓ token_attr ≈ epi_mu`` (Theorem 2).
* Bernoulli special case (``m_j = 1``) — ``aleatoric_U`` matches the
  v7 ``Total − Epi`` form.
* Large ``m_j`` — ``aleatoric_U`` shrinks like ``1 / m_j``.
* MC vs linear — the delta-method epistemic agrees with the MC estimate
  when the posterior is concentrated (small Σ̂), and may diverge when
  the posterior is wide (expected — non-linearity of σ kicks in).
* ``m_j = None`` — ratio / count / strict fields are ``None`` and the
  call does not raise.
* ``BatchPredictor`` — matches per-sentence :class:`Predictor` results.
* ``save_trained_model`` / ``load_trained_model`` — round-trip exactness.
* ``predict_from_hidden_states`` — agrees with the manually-extracted
  feature path.
* Probit shrinkage — ``μ̃`` lies between 0.5 and ``μ̂`` when ``Σ̂`` grows
  (shrinkage toward the prior mean of σ).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.extractor import (  # noqa: E402
    SentenceUQParams,
    extract_sentence_token_features,
)
from src.inference.predict import (  # noqa: E402
    BatchPredictor,
    Predictor,
    load_trained_model,
    save_trained_model,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_params(k: int = 6, hidden_dim: int = 8, num_layers: int = 3) -> SentenceUQParams:
    """SentenceUQParams configured so feature_dim == k."""
    params = SentenceUQParams(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        projection_dim=k - 2,
    )
    assert params.feature_dim == k
    return params


def _make_predictor(
    k: int = 6,
    sigma_scale: float = 0.1,
    seed: int = 0,
    hidden_dim: int = 8,
    num_layers: int = 3,
    use_probit_shrinkage: bool = False,
) -> Predictor:
    g = torch.Generator().manual_seed(seed)
    params = _make_params(k=k, hidden_dim=hidden_dim, num_layers=num_layers)
    theta_hat = torch.randn(k, generator=g)
    # Random PSD Σ via A Aᵀ + eps I.
    A = torch.randn(k, k, generator=g)
    Sigma_hat = sigma_scale * (A @ A.T + 0.05 * torch.eye(k))
    return Predictor(
        theta_hat=theta_hat,
        Sigma_hat=Sigma_hat,
        feature_params=params,
        use_probit_shrinkage=use_probit_shrinkage,
    )


def _make_z_tokens(L: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(L, k, generator=g)


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------


def test_init_validates_shapes() -> None:
    params = _make_params(k=6)
    theta = torch.randn(6)
    Sigma = torch.eye(6)

    Predictor(theta, Sigma, params)  # ok

    with pytest.raises(TypeError):
        Predictor(theta, Sigma, "not params")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Predictor(torch.randn(6, 1), Sigma, params)  # theta not 1-D
    with pytest.raises(ValueError):
        Predictor(theta, torch.eye(5), params)  # Sigma wrong size
    # k mismatch with feature_dim
    params_other = _make_params(k=4, hidden_dim=8, num_layers=3)
    with pytest.raises(ValueError):
        Predictor(theta, Sigma, params_other)


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_predict_sentence_basic_invariants(seed: int) -> None:
    k = 6
    pred = _make_predictor(k=k, seed=seed)
    z = _make_z_tokens(L=5, k=k, seed=seed + 100)

    out = pred.predict_sentence(z, m_j=4)

    assert 0.0 <= out["mu_hat"] <= 1.0
    assert 0.0 <= out["p_factual_probit"] <= 1.0
    assert out["epi_mu"] >= 0.0
    assert out["aleatoric_U"] is not None and out["aleatoric_U"] >= 0.0
    assert out["total_U"] is not None and out["total_U"] >= 0.0
    assert out["epi_K"] is not None and out["epi_K"] >= 0.0
    assert out["aleatoric_K"] is not None and out["aleatoric_K"] >= 0.0
    assert out["p_strict_factual"] is not None
    assert 0.0 <= out["p_strict_factual"] <= 1.0
    assert torch.all(out["token_pi"] >= 0) and torch.all(out["token_pi"] <= 1)
    assert torch.all(out["token_local_epi"] >= 0)


def test_token_attr_sums_to_epi_mu() -> None:
    """Theorem 2: Σ_ℓ Attr_ℓ ≡ Epi_μ (exact, not approximate)."""
    k = 6
    pred = _make_predictor(k=k, seed=7)
    z = _make_z_tokens(L=11, k=k, seed=8)

    out = pred.predict_sentence(z, m_j=5)
    total_attr = float(out["token_attr"].sum().item())
    assert total_attr == pytest.approx(out["epi_mu"], rel=1e-5, abs=1e-6)


def test_total_U_equals_aleatoric_plus_epi() -> None:
    pred = _make_predictor(k=6, seed=9)
    z = _make_z_tokens(L=7, k=6, seed=10)
    out = pred.predict_sentence(z, m_j=3)
    assert out["total_U"] == pytest.approx(
        out["aleatoric_U"] + out["epi_mu"], rel=1e-6, abs=1e-8
    )


# ---------------------------------------------------------------------------
# Bernoulli special case
# ---------------------------------------------------------------------------


def test_bernoulli_special_case_matches_v7_form() -> None:
    """m_j = 1 → aleatoric_U = max(0, μ̂(1-μ̂) - epi_μ), the v7 form."""
    k = 6
    pred = _make_predictor(k=k, seed=11)
    z = _make_z_tokens(L=6, k=k, seed=12)

    out = pred.predict_sentence(z, m_j=1)
    mu = out["mu_hat"]
    expected = max(0.0, mu * (1.0 - mu) - out["epi_mu"])
    assert out["aleatoric_U"] == pytest.approx(expected, rel=1e-6, abs=1e-8)
    # Total = epi + aleatoric in the v7 decomposition too.
    assert out["total_U"] == pytest.approx(expected + out["epi_mu"], abs=1e-8)


# ---------------------------------------------------------------------------
# Large m_j shrinkage
# ---------------------------------------------------------------------------


def test_aleatoric_U_shrinks_with_m_j() -> None:
    """Aleatoric_U ∝ 1 / m_j by construction (eq. 6)."""
    k = 6
    pred = _make_predictor(k=k, seed=13)
    z = _make_z_tokens(L=8, k=k, seed=14)

    out_small = pred.predict_sentence(z, m_j=2)
    out_big = pred.predict_sentence(z, m_j=200)

    assert out_big["aleatoric_U"] < out_small["aleatoric_U"]
    # Exact ratio: aleatoric_U(m_a) / aleatoric_U(m_b) = m_b / m_a
    # when the numerator (μ(1-μ) - epi_μ) is positive (clipped case
    # would invalidate the ratio — skip if so).
    inner = out_small["mu_hat"] * (1.0 - out_small["mu_hat"]) - out_small["epi_mu"]
    if inner > 1e-6:
        ratio = out_small["aleatoric_U"] / out_big["aleatoric_U"]
        assert ratio == pytest.approx(200.0 / 2.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Count level
# ---------------------------------------------------------------------------


def test_count_level_relations() -> None:
    """Epi_K = m² · Epi_μ and Aleatoric_K = m · max(0, μ(1-μ) - Epi_μ)."""
    pred = _make_predictor(k=6, seed=15)
    z = _make_z_tokens(L=5, k=6, seed=16)
    for m in [1, 3, 8]:
        out = pred.predict_sentence(z, m_j=m)
        assert out["epi_K"] == pytest.approx(m * m * out["epi_mu"], rel=1e-6)
        expected_ak = m * max(
            0.0, out["mu_hat"] * (1.0 - out["mu_hat"]) - out["epi_mu"]
        )
        assert out["aleatoric_K"] == pytest.approx(expected_ak, rel=1e-6, abs=1e-8)


# ---------------------------------------------------------------------------
# Strict factuality
# ---------------------------------------------------------------------------


def test_p_strict_factual_plugin() -> None:
    """Default (use_probit_shrinkage=False): p(A=1) = μ̂^m."""
    pred = _make_predictor(k=6, seed=17, use_probit_shrinkage=False)
    z = _make_z_tokens(L=4, k=6, seed=18)
    out = pred.predict_sentence(z, m_j=5)
    assert out["p_strict_factual"] == pytest.approx(out["mu_hat"] ** 5, rel=1e-6)


def test_p_strict_factual_probit_when_enabled() -> None:
    """use_probit_shrinkage=True swaps to (μ̃)^m for the plug-in."""
    pred = _make_predictor(k=6, seed=19, use_probit_shrinkage=True)
    z = _make_z_tokens(L=4, k=6, seed=20)
    out = pred.predict_sentence(z, m_j=5)
    assert out["p_strict_factual"] == pytest.approx(
        out["p_factual_probit"] ** 5, rel=1e-6
    )


# ---------------------------------------------------------------------------
# m_j = None
# ---------------------------------------------------------------------------


def test_m_j_none_returns_none_for_ratio_count_strict() -> None:
    pred = _make_predictor(k=6, seed=21)
    z = _make_z_tokens(L=4, k=6, seed=22)
    out = pred.predict_sentence(z, m_j=None)
    for key in ("aleatoric_U", "total_U", "epi_K", "aleatoric_K", "p_strict_factual"):
        assert out[key] is None
    # Latent-level still present.
    assert out["mu_hat"] is not None
    assert out["epi_mu"] is not None
    assert out["token_attr"].shape == (4,)


def test_m_j_zero_skips_ratio_count_strict() -> None:
    """m_j = 0 sentences have undefined ratio / count / strict — return None."""
    pred = _make_predictor(k=6, seed=23)
    z = _make_z_tokens(L=4, k=6, seed=24)
    out = pred.predict_sentence(z, m_j=0)
    for key in ("aleatoric_U", "total_U", "epi_K", "aleatoric_K", "p_strict_factual"):
        assert out[key] is None


# ---------------------------------------------------------------------------
# MC vs linear
# ---------------------------------------------------------------------------


def test_mc_matches_linear_when_sigma_small() -> None:
    """Small Σ̂ → delta method is tight; MC and linear epi_μ agree."""
    k = 6
    # Very small Σ → the linear approximation is essentially exact.
    pred = _make_predictor(k=k, sigma_scale=1e-4, seed=25)
    z = _make_z_tokens(L=6, k=k, seed=26)

    out = pred.predict_sentence(z, m_j=4)
    g = torch.Generator().manual_seed(27)
    mc = pred.predict_mc_epistemic(z, num_samples=4000, m_j=4, generator=g)

    # mu_hat should match MC mean.
    assert mc["mc_mu_mean"] == pytest.approx(out["mu_hat"], abs=2e-3)
    # Relative tolerance is loose because variance is O(1e-8) here.
    if out["epi_mu"] > 1e-10:
        assert mc["mc_epi_mu"] == pytest.approx(out["epi_mu"], rel=0.5, abs=1e-7)


def test_mc_returns_none_when_m_j_none() -> None:
    pred = _make_predictor(k=6, seed=28)
    z = _make_z_tokens(L=4, k=6, seed=29)
    g = torch.Generator().manual_seed(30)
    mc = pred.predict_mc_epistemic(z, num_samples=64, m_j=None, generator=g)
    assert mc["mc_epi_mu"] >= 0.0
    for key in (
        "mc_aleatoric_U",
        "mc_total_U",
        "mc_epi_K",
        "mc_aleatoric_K",
        "mc_p_strict_factual",
    ):
        assert mc[key] is None


def test_mc_with_m_j_relations() -> None:
    """Sanity: MC count-level satisfies epi_K = m² · epi_μ."""
    pred = _make_predictor(k=6, seed=31)
    z = _make_z_tokens(L=5, k=6, seed=32)
    g = torch.Generator().manual_seed(33)
    mc = pred.predict_mc_epistemic(z, num_samples=256, m_j=4, generator=g)
    assert mc["mc_epi_K"] == pytest.approx(16.0 * mc["mc_epi_mu"], rel=1e-6)
    assert mc["mc_total_U"] == pytest.approx(
        mc["mc_aleatoric_U"] + mc["mc_epi_mu"], abs=1e-8
    )


def test_mc_can_diverge_for_large_sigma() -> None:
    """Large Σ̂ → linear and MC may disagree; both should still be finite."""
    pred = _make_predictor(k=6, sigma_scale=2.0, seed=34)
    z = _make_z_tokens(L=5, k=6, seed=35)
    g = torch.Generator().manual_seed(36)
    out = pred.predict_sentence(z, m_j=3)
    mc = pred.predict_mc_epistemic(z, num_samples=512, m_j=3, generator=g)
    assert torch.isfinite(torch.tensor(out["epi_mu"]))
    assert torch.isfinite(torch.tensor(mc["mc_epi_mu"]))
    # Both >= 0.
    assert out["epi_mu"] >= 0.0
    assert mc["mc_epi_mu"] >= 0.0


# ---------------------------------------------------------------------------
# Probit shrinkage
# ---------------------------------------------------------------------------


def test_probit_equals_plugin_when_sigma_is_zero() -> None:
    """Σ̂ = 0 → no shrinkage → π̃ ≡ π̂ → μ̃ = μ̂."""
    k = 6
    params = _make_params(k=k)
    theta = torch.randn(k)
    Sigma = torch.zeros(k, k)
    pred = Predictor(theta, Sigma, params)
    z = _make_z_tokens(L=5, k=k, seed=37)
    out = pred.predict_sentence(z, m_j=2)
    assert out["p_factual_probit"] == pytest.approx(out["mu_hat"], abs=1e-6)
    assert out["epi_mu"] == pytest.approx(0.0, abs=1e-8)


def test_probit_shrinks_toward_half() -> None:
    """Large Σ̂ → π̃ closer to 0.5 than π̂ for every confident token."""
    k = 6
    params = _make_params(k=k)
    # Confident logits (large magnitude) so σ(logit) is near 0 or 1.
    theta = torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    Sigma = 10.0 * torch.eye(k)
    pred = Predictor(theta, Sigma, params)
    # z aligned with theta_0 so logits are large.
    z = torch.zeros(4, k)
    z[:, 0] = 1.0
    out = pred.predict_sentence(z, m_j=None)
    # μ̂ should be very close to 1; μ̃ should be strictly smaller (toward 0.5).
    assert out["mu_hat"] > 0.99
    assert out["p_factual_probit"] < out["mu_hat"]
    assert out["p_factual_probit"] > 0.5


# ---------------------------------------------------------------------------
# predict_from_hidden_states
# ---------------------------------------------------------------------------


def test_predict_from_hidden_states_matches_manual_extraction() -> None:
    k = 6
    hidden_dim = 8
    num_layers = 3
    pred = _make_predictor(
        k=k, hidden_dim=hidden_dim, num_layers=num_layers, seed=40
    )
    T = 12
    g = torch.Generator().manual_seed(41)
    hidden = torch.randn(T, num_layers, hidden_dim, generator=g)
    ent = torch.randn(T, generator=g).abs()
    top1 = torch.rand(T, generator=g)
    rng = (3, 9)

    z = extract_sentence_token_features(
        hidden, ent, top1, token_range=rng, params=pred.feature_params
    )
    out_manual = pred.predict_sentence(z, m_j=4)
    out_e2e = pred.predict_from_hidden_states(hidden, ent, top1, rng, m_j=4)

    assert out_manual["mu_hat"] == pytest.approx(out_e2e["mu_hat"], rel=1e-6)
    assert out_manual["epi_mu"] == pytest.approx(out_e2e["epi_mu"], rel=1e-6)
    assert torch.allclose(out_manual["token_pi"], out_e2e["token_pi"], atol=1e-6)


# ---------------------------------------------------------------------------
# BatchPredictor
# ---------------------------------------------------------------------------


def test_batch_predictor_matches_per_sentence_calls() -> None:
    k = 6
    pred = _make_predictor(k=k, seed=42)
    batch = BatchPredictor(pred)

    z_list = [_make_z_tokens(L=L, k=k, seed=100 + i) for i, L in enumerate([3, 5, 8])]
    m_list = [2, None, 4]

    batched = batch.predict(z_list, m_list)
    expected = [pred.predict_sentence(z, m_j=m) for z, m in zip(z_list, m_list)]

    assert len(batched) == 3
    for got, exp in zip(batched, expected):
        assert got["mu_hat"] == pytest.approx(exp["mu_hat"], rel=1e-7)
        assert got["epi_mu"] == pytest.approx(exp["epi_mu"], rel=1e-7)
        assert (got["aleatoric_U"] is None) == (exp["aleatoric_U"] is None)
        assert torch.allclose(got["token_attr"], exp["token_attr"])


def test_batch_predictor_default_m_list_is_all_none() -> None:
    pred = _make_predictor(k=6, seed=43)
    batch = BatchPredictor(pred)
    z_list = [_make_z_tokens(L=L, k=6, seed=200 + i) for i, L in enumerate([4, 6])]
    out = batch.predict(z_list)
    assert len(out) == 2
    for d in out:
        assert d["aleatoric_U"] is None
        assert d["p_strict_factual"] is None


def test_batch_predictor_length_mismatch_raises() -> None:
    pred = _make_predictor(k=6, seed=44)
    batch = BatchPredictor(pred)
    z_list = [_make_z_tokens(L=4, k=6, seed=300)]
    with pytest.raises(ValueError):
        batch.predict(z_list, [1, 2])


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    k = 6
    pred = _make_predictor(k=k, seed=50)
    z = _make_z_tokens(L=5, k=k, seed=51)
    out_orig = pred.predict_sentence(z, m_j=3)

    save_path = tmp_path / "trained.pt"
    save_trained_model(
        save_path,
        theta_hat=pred.theta_hat,
        Sigma_hat=pred.Sigma_hat,
        feature_params=pred.feature_params,
        extra={"note": "round-trip test"},
    )

    loaded = load_trained_model(save_path)
    assert torch.allclose(loaded["theta_hat"], pred.theta_hat)
    assert torch.allclose(loaded["Sigma_hat"], pred.Sigma_hat)
    assert loaded["extra"] == {"note": "round-trip test"}

    pred2 = Predictor(
        theta_hat=loaded["theta_hat"],
        Sigma_hat=loaded["Sigma_hat"],
        feature_params=loaded["feature_params"],
    )
    out_new = pred2.predict_sentence(z, m_j=3)
    assert out_new["mu_hat"] == pytest.approx(out_orig["mu_hat"], rel=1e-6)
    assert out_new["epi_mu"] == pytest.approx(out_orig["epi_mu"], rel=1e-6)
    assert out_new["p_strict_factual"] == pytest.approx(
        out_orig["p_strict_factual"], rel=1e-6
    )


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    pred = _make_predictor(k=6, seed=52)
    nested = tmp_path / "a" / "b" / "c" / "model.pt"
    save_trained_model(
        nested,
        theta_hat=pred.theta_hat,
        Sigma_hat=pred.Sigma_hat,
        feature_params=pred.feature_params,
    )
    assert nested.exists()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_trained_model(tmp_path / "does_not_exist.pt")


# ---------------------------------------------------------------------------
# Multi-config (model-agnostic) sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden_dim,num_layers,projection_dim",
    [(8, 3, 4), (16, 5, 8)],
)
def test_predictor_is_model_agnostic(
    hidden_dim: int, num_layers: int, projection_dim: int
) -> None:
    """Predictor must not hardcode any specific (hidden_dim, num_layers)."""
    k = projection_dim + 2
    g = torch.Generator().manual_seed(60)
    params = SentenceUQParams(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        projection_dim=projection_dim,
    )
    theta = torch.randn(k, generator=g)
    A = torch.randn(k, k, generator=g)
    Sigma = 0.05 * (A @ A.T + torch.eye(k))
    pred = Predictor(theta, Sigma, params)

    T = 8
    hidden = torch.randn(T, num_layers, hidden_dim, generator=g)
    ent = torch.randn(T, generator=g).abs()
    top1 = torch.rand(T, generator=g)
    out = pred.predict_from_hidden_states(hidden, ent, top1, (1, 6), m_j=3)
    assert out["token_pi"].shape == (5,)
    assert 0.0 <= out["mu_hat"] <= 1.0
