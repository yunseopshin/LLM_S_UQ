"""Tests for ``src.features`` — Phase 1-3 cached scalars and Phase 2-1 extractor.

Covers (Phase 1-3 ``cached_scalars``):
- analytical correctness of ``compute_token_entropy_and_top1`` (uniform,
  one-hot, mixed),
- numerical stability (fp16 input, ``-inf`` logits, large magnitudes),
- input validation (non-2D, non-floating dtype),
- directory-level caching: recursive discovery, deterministic ``idx``
  ordering, payload schema, empty-generation handling,
- ``load_scalars`` round-trip and error handling.

Covers (Phase 2-1 ``extractor``):
- ``SentenceUQParams`` parameter shapes, ``feature_dim``, prior matrices,
- ``extract_token_features`` correctness, autograd through W / α / hidden
  states, fp16-input handling,
- ``extract_sentence_token_features`` slicing semantics,
- ``extract_sentence_aggregate_feature`` mean / std / last layout, ``L_j=1``
  edge case,
- parameterization over multiple model configs to catch hardcoded
  dimensions (model-agnostic guarantee from CLAUDE.md).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.cached_scalars import (  # noqa: E402
    cache_scalars_for_directory,
    compute_token_entropy_and_top1,
    load_scalars,
)
from src.features.extractor import (  # noqa: E402
    SentenceUQParams,
    extract_sentence_aggregate_feature,
    extract_sentence_token_features,
    extract_token_features,
)


# ---------------------------------------------------------------------------
# compute_token_entropy_and_top1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vocab", [4, 32, 128])
def test_uniform_logits_give_max_entropy(vocab: int) -> None:
    """Uniform distribution: H = log V, top1 = 1/V."""
    logits = torch.zeros(3, vocab)
    entropy, top1 = compute_token_entropy_and_top1(logits)
    assert entropy.shape == (3,)
    assert top1.shape == (3,)
    assert entropy.dtype == torch.float32
    assert top1.dtype == torch.float32
    expected_H = math.log(vocab)
    assert torch.allclose(entropy, torch.full((3,), expected_H), atol=1e-5)
    assert torch.allclose(top1, torch.full((3,), 1.0 / vocab), atol=1e-6)


def test_one_hot_logits_give_zero_entropy() -> None:
    """A confident distribution: H ≈ 0, top1 ≈ 1."""
    logits = torch.full((2, 5), -1e4)
    logits[0, 3] = 1e4
    logits[1, 0] = 1e4
    entropy, top1 = compute_token_entropy_and_top1(logits)
    assert torch.allclose(entropy, torch.zeros(2), atol=1e-5)
    assert torch.allclose(top1, torch.ones(2), atol=1e-6)


def test_known_two_class_distribution() -> None:
    """Bernoulli(0.5, 0.5) → H = log 2; Bernoulli(0.25, 0.75) → mixed."""
    # logits encoding probs (0.5, 0.5): equal logits.
    row0 = torch.tensor([0.0, 0.0])
    # logits encoding probs (0.25, 0.75): log-ratio = log 3 → logits diff log 3.
    row1 = torch.tensor([0.0, math.log(3.0)])
    logits = torch.stack([row0, row1])
    entropy, top1 = compute_token_entropy_and_top1(logits)

    p = 0.25
    expected_H1 = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert torch.allclose(
        entropy, torch.tensor([math.log(2.0), expected_H1]), atol=1e-6
    )
    assert torch.allclose(top1, torch.tensor([0.5, 0.75]), atol=1e-6)


def test_handles_negative_inf_logits_without_nan() -> None:
    """``-inf`` logits → p=0 there; the 0·log0 = 0 convention must hold."""
    logits = torch.tensor([[0.0, 0.0, float("-inf")]])
    entropy, top1 = compute_token_entropy_and_top1(logits)
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(top1).all()
    # Effective distribution is (0.5, 0.5, 0) → H = log 2, top1 = 0.5.
    assert torch.allclose(entropy, torch.tensor([math.log(2.0)]), atol=1e-6)
    assert torch.allclose(top1, torch.tensor([0.5]), atol=1e-6)


def test_fp16_input_is_handled_and_returns_fp32() -> None:
    """Generation stores logits in fp16; the cache must compute in fp32."""
    logits_fp16 = torch.zeros(4, 8, dtype=torch.float16)
    entropy, top1 = compute_token_entropy_and_top1(logits_fp16)
    assert entropy.dtype == torch.float32
    assert top1.dtype == torch.float32
    assert torch.allclose(entropy, torch.full((4,), math.log(8.0)), atol=1e-4)


def test_large_magnitude_logits_are_stable() -> None:
    """log_softmax handles the offset; no overflow / NaN with huge logits."""
    logits = torch.full((1, 1024), 1e4)
    logits[0, 0] = 1e4 + 50.0  # still dominant after stable softmax
    entropy, top1 = compute_token_entropy_and_top1(logits)
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(top1).all()
    # Top class dominates → entropy ≈ 0, top1 ≈ 1.
    assert entropy.item() < 1e-3
    assert top1.item() > 1.0 - 1e-3


def test_top1_matches_argmax_probability() -> None:
    """top1 must equal max(softmax(logits)) along the vocab axis."""
    torch.manual_seed(0)
    logits = torch.randn(10, 16)
    _, top1 = compute_token_entropy_and_top1(logits)
    expected = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
    assert torch.allclose(top1, expected, atol=1e-6)


def test_entropy_is_non_negative() -> None:
    """Predictive entropy is non-negative for any valid distribution."""
    torch.manual_seed(1)
    logits = torch.randn(50, 32) * 5.0
    entropy, _ = compute_token_entropy_and_top1(logits)
    assert (entropy >= -1e-6).all()


def test_validates_input_shape() -> None:
    with pytest.raises(ValueError):
        compute_token_entropy_and_top1(torch.zeros(8))
    with pytest.raises(ValueError):
        compute_token_entropy_and_top1(torch.zeros(2, 3, 4))


def test_validates_input_dtype() -> None:
    with pytest.raises(ValueError):
        compute_token_entropy_and_top1(torch.zeros(2, 4, dtype=torch.long))


# ---------------------------------------------------------------------------
# cache_scalars_for_directory + load_scalars
# ---------------------------------------------------------------------------


def _write_fake_generation(
    path: Path, T: int, vocab: int, seed: int = 0
) -> torch.Tensor:
    """Write a minimal Phase 1-1 ``.pt`` payload and return the logits."""
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, vocab, generator=gen, dtype=torch.float32).to(torch.float16)
    token_ids = torch.randint(0, vocab, (T,), generator=gen, dtype=torch.long)
    payload = {
        "logits": logits,
        "token_ids": token_ids,
        # Other Phase 1-1 fields are not required by the cache step.
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return logits


def test_cache_roundtrip_flat_layout(tmp_path: Path) -> None:
    gen_dir = tmp_path / "gen"
    cache_dir = tmp_path / "cache"

    logits_a = _write_fake_generation(gen_dir / "a.pt", T=4, vocab=16, seed=1)
    logits_b = _write_fake_generation(gen_dir / "b.pt", T=7, vocab=16, seed=2)
    logits_c = _write_fake_generation(gen_dir / "c.pt", T=3, vocab=16, seed=3)

    result = cache_scalars_for_directory(gen_dir, cache_dir, progress=False)
    assert result["cached"] == 3
    assert result["errors"] == []

    # Files are written with zero-padded idx in sorted order: a.pt, b.pt, c.pt.
    for idx in range(3):
        assert (cache_dir / f"{idx:05d}.pt").exists()

    # Sanity-check idx 1 (b.pt) by recomputing.
    expected_H, expected_top1 = compute_token_entropy_and_top1(logits_b)
    cached = load_scalars(1, cache_dir)
    assert cached["source_path"] == "b.pt"
    assert cached["entropy"].dtype == torch.float32
    assert cached["top1_prob"].dtype == torch.float32
    assert cached["token_ids"].dtype == torch.long
    assert torch.allclose(cached["entropy"], expected_H, atol=1e-6)
    assert torch.allclose(cached["top1_prob"], expected_top1, atol=1e-6)
    assert cached["entropy"].shape == (7,)


def test_cache_recurses_into_subdirectories(tmp_path: Path) -> None:
    """LongFact stores ``{topic}/{prompt_idx:03d}.pt`` — must be discovered."""
    gen_dir = tmp_path / "gen"
    cache_dir = tmp_path / "cache"

    _write_fake_generation(gen_dir / "chem" / "000.pt", T=2, vocab=8, seed=10)
    _write_fake_generation(gen_dir / "chem" / "001.pt", T=2, vocab=8, seed=11)
    _write_fake_generation(gen_dir / "physics" / "000.pt", T=2, vocab=8, seed=12)

    result = cache_scalars_for_directory(gen_dir, cache_dir, progress=False)
    assert result["cached"] == 3

    # Sorted by POSIX relpath: chem/000, chem/001, physics/000.
    expected_sources = ["chem/000.pt", "chem/001.pt", "physics/000.pt"]
    for idx, src in enumerate(expected_sources):
        loaded = load_scalars(idx, cache_dir)
        assert loaded["source_path"] == src


def test_cache_idx_is_stable_across_runs(tmp_path: Path) -> None:
    """Two runs with the same layout must produce the same idx assignment."""
    gen_dir = tmp_path / "gen"
    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"

    _write_fake_generation(gen_dir / "z.pt", T=2, vocab=8, seed=20)
    _write_fake_generation(gen_dir / "a.pt", T=2, vocab=8, seed=21)
    _write_fake_generation(gen_dir / "m.pt", T=2, vocab=8, seed=22)

    cache_scalars_for_directory(gen_dir, cache_a, progress=False)
    cache_scalars_for_directory(gen_dir, cache_b, progress=False)

    for idx in range(3):
        ra = load_scalars(idx, cache_a)
        rb = load_scalars(idx, cache_b)
        assert ra["source_path"] == rb["source_path"]
        assert torch.equal(ra["token_ids"], rb["token_ids"])


def test_cache_handles_empty_generation(tmp_path: Path) -> None:
    """Generation that emitted 0 tokens still gets a cache entry."""
    gen_dir = tmp_path / "gen"
    cache_dir = tmp_path / "cache"
    vocab = 8

    payload = {
        "logits": torch.empty(0, vocab, dtype=torch.float16),
        "token_ids": torch.empty(0, dtype=torch.long),
    }
    (gen_dir).mkdir(parents=True)
    torch.save(payload, gen_dir / "empty.pt")

    result = cache_scalars_for_directory(gen_dir, cache_dir, progress=False)
    assert result["cached"] == 1
    assert result["errors"] == []

    loaded = load_scalars(0, cache_dir)
    assert loaded["entropy"].shape == (0,)
    assert loaded["top1_prob"].shape == (0,)
    assert loaded["token_ids"].shape == (0,)
    assert loaded["entropy"].dtype == torch.float32


def test_cache_reports_errors_without_aborting(tmp_path: Path) -> None:
    """A corrupt file must not stop the rest of the directory from caching."""
    gen_dir = tmp_path / "gen"
    cache_dir = tmp_path / "cache"
    gen_dir.mkdir(parents=True)

    # Mismatched logits / token_ids length → ValueError, caught and recorded.
    torch.save(
        {
            "logits": torch.zeros(3, 4, dtype=torch.float16),
            "token_ids": torch.zeros(2, dtype=torch.long),
        },
        gen_dir / "bad.pt",
    )
    _write_fake_generation(gen_dir / "good.pt", T=2, vocab=4, seed=99)

    result = cache_scalars_for_directory(gen_dir, cache_dir, progress=False)
    assert result["cached"] == 1
    assert len(result["errors"]) == 1
    src, msg = result["errors"][0]
    assert src == "bad.pt"
    assert "length mismatch" in msg

    # The good file got idx=1 (after 'bad.pt' in sorted order); the bad file
    # leaves a hole at idx=0.
    good = load_scalars(1, cache_dir)
    assert good["source_path"] == "good.pt"
    assert not (cache_dir / "00000.pt").exists()


def test_cache_creates_cache_dir(tmp_path: Path) -> None:
    gen_dir = tmp_path / "gen"
    cache_dir = tmp_path / "deep" / "nested" / "cache"
    _write_fake_generation(gen_dir / "x.pt", T=1, vocab=4, seed=0)
    result = cache_scalars_for_directory(gen_dir, cache_dir, progress=False)
    assert result["cached"] == 1
    assert cache_dir.is_dir()


def test_cache_raises_for_missing_generations_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cache_scalars_for_directory(
            tmp_path / "does_not_exist", tmp_path / "cache", progress=False
        )


def test_load_scalars_validates_idx(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_scalars(-1, tmp_path)


def test_load_scalars_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_scalars(42, tmp_path)


# ---------------------------------------------------------------------------
# Phase 2-1 — extractor
# ---------------------------------------------------------------------------


# A handful of model configs to verify the implementation is model-agnostic
# (CLAUDE.md "Model Compatibility"). The deliberate choices: Llama-like wide,
# small (Gemma-7B-ish), Gemma-2-9B-ish.
_MODEL_CONFIGS = [
    pytest.param(4096, 8, id="Llama-like (d=4096, L=8)"),
    pytest.param(2048, 6, id="small (d=2048, L=6)"),
    pytest.param(3584, 10, id="Gemma-like (d=3584, L=10)"),
]


def _make_inputs(
    T: int, num_layers: int, hidden_dim: int, *, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    h = torch.randn(T, num_layers, hidden_dim, generator=gen)
    # Entropy is non-negative, top1 ∈ (0, 1) — match the cached-scalar contract.
    ent = torch.rand(T, generator=gen) * 2.0
    top1 = torch.rand(T, generator=gen).clamp(min=1e-3, max=1.0 - 1e-3)
    return h, ent, top1


# --- SentenceUQParams -------------------------------------------------------


@pytest.mark.parametrize("hidden_dim,num_layers", _MODEL_CONFIGS)
def test_params_have_expected_shapes(hidden_dim: int, num_layers: int) -> None:
    p = 64
    params = SentenceUQParams(
        hidden_dim=hidden_dim, num_layers=num_layers, projection_dim=p
    )
    assert params.W.weight.shape == (p, hidden_dim)
    assert params.W.bias is None
    assert params.alpha.shape == (num_layers,)
    assert params.mu_0.shape == (p + 2,)
    assert params.log_sigma_0.shape == (p + 2,)
    assert params.feature_dim == p + 2

    # Default initialization: α, μ_0, log σ_0 all zero.
    assert torch.equal(params.alpha, torch.zeros(num_layers))
    assert torch.equal(params.mu_0, torch.zeros(p + 2))
    assert torch.equal(params.log_sigma_0, torch.zeros(p + 2))


def test_params_custom_projection_dim() -> None:
    params = SentenceUQParams(hidden_dim=64, num_layers=4, projection_dim=16)
    assert params.feature_dim == 18
    assert params.W.weight.shape == (16, 64)


def test_params_reject_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        SentenceUQParams(hidden_dim=0, num_layers=4)
    with pytest.raises(ValueError):
        SentenceUQParams(hidden_dim=64, num_layers=0)
    with pytest.raises(ValueError):
        SentenceUQParams(hidden_dim=64, num_layers=4, projection_dim=0)


def test_params_have_no_default_dims() -> None:
    """``hidden_dim`` / ``num_layers`` must be explicit (CLAUDE.md rule)."""
    with pytest.raises(TypeError):
        SentenceUQParams()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        SentenceUQParams(hidden_dim=64)  # type: ignore[call-arg]


def test_sigma_0_default_is_identity() -> None:
    """With ``log σ_0 = 0`` both prior matrices equal ``I_k``."""
    params = SentenceUQParams(hidden_dim=32, num_layers=4, projection_dim=8)
    k = params.feature_dim
    eye = torch.eye(k)
    assert torch.allclose(params.get_Sigma_0(), eye, atol=1e-6)
    assert torch.allclose(params.get_Sigma_0_inv(), eye, atol=1e-6)


def test_sigma_0_responds_to_log_sigma() -> None:
    """``Σ_0 = diag(exp(2 log σ_0))`` and they invert each other on the diagonal."""
    params = SentenceUQParams(hidden_dim=32, num_layers=4, projection_dim=8)
    k = params.feature_dim
    with torch.no_grad():
        params.log_sigma_0.copy_(torch.linspace(-1.0, 1.0, k))

    Sigma = params.get_Sigma_0()
    Sigma_inv = params.get_Sigma_0_inv()
    expected_diag = torch.exp(2.0 * params.log_sigma_0)
    assert torch.allclose(torch.diagonal(Sigma), expected_diag, atol=1e-6)
    assert torch.allclose(
        torch.diagonal(Sigma_inv), 1.0 / expected_diag, atol=1e-6
    )
    # Off-diagonal zeros.
    off = Sigma - torch.diag(torch.diagonal(Sigma))
    assert torch.allclose(off, torch.zeros_like(off))
    # They are mutual inverses.
    assert torch.allclose(Sigma @ Sigma_inv, torch.eye(k), atol=1e-5)


# --- extract_token_features -------------------------------------------------


@pytest.mark.parametrize("hidden_dim,num_layers", _MODEL_CONFIGS)
def test_feature_dim_matches_projection_plus_two(
    hidden_dim: int, num_layers: int
) -> None:
    projection_dim = 32
    params = SentenceUQParams(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        projection_dim=projection_dim,
    )
    h, ent, top1 = _make_inputs(T=5, num_layers=num_layers, hidden_dim=hidden_dim)
    z = extract_token_features(h, ent, top1, params)
    assert z.shape == (5, projection_dim + 2)
    assert z.shape[1] == params.feature_dim


def test_last_two_columns_are_entropy_and_top1() -> None:
    params = SentenceUQParams(hidden_dim=16, num_layers=3, projection_dim=4)
    h, ent, top1 = _make_inputs(T=6, num_layers=3, hidden_dim=16, seed=7)
    z = extract_token_features(h, ent, top1, params)
    assert torch.allclose(z[:, -2], ent.to(torch.float32), atol=1e-6)
    assert torch.allclose(z[:, -1], top1.to(torch.float32), atol=1e-6)


def test_token_features_match_manual_computation() -> None:
    """Cross-check the einsum aggregation against an explicit loop."""
    params = SentenceUQParams(hidden_dim=8, num_layers=5, projection_dim=3)
    with torch.no_grad():
        params.alpha.copy_(torch.tensor([1.0, -0.5, 0.0, 0.7, -1.2]))
    h, ent, top1 = _make_inputs(T=4, num_layers=5, hidden_dim=8, seed=11)

    z = extract_token_features(h, ent, top1, params)

    w = F.softmax(params.alpha, dim=0)
    h_agg_manual = torch.zeros(4, 8)
    for l in range(5):
        h_agg_manual = h_agg_manual + w[l] * h[:, l, :]
    h_proj_manual = params.W(h_agg_manual)
    z_manual = torch.cat(
        [h_proj_manual, ent.unsqueeze(1), top1.unsqueeze(1)], dim=1
    )
    assert torch.allclose(z, z_manual, atol=1e-5)


def test_alpha_softmax_invariant_to_constant_shift() -> None:
    """``α`` and ``α + c`` yield the same softmax weights and thus same z."""
    params_a = SentenceUQParams(hidden_dim=8, num_layers=4, projection_dim=3)
    params_b = SentenceUQParams(hidden_dim=8, num_layers=4, projection_dim=3)
    with torch.no_grad():
        params_b.W.weight.copy_(params_a.W.weight)
        params_a.alpha.copy_(torch.tensor([0.3, -0.7, 1.1, 0.0]))
        params_b.alpha.copy_(params_a.alpha + 5.0)

    h, ent, top1 = _make_inputs(T=3, num_layers=4, hidden_dim=8, seed=2)
    z_a = extract_token_features(h, ent, top1, params_a)
    z_b = extract_token_features(h, ent, top1, params_b)
    assert torch.allclose(z_a, z_b, atol=1e-5)


def test_works_with_single_layer() -> None:
    """num_layers=1 must not break the einsum / softmax."""
    params = SentenceUQParams(hidden_dim=12, num_layers=1, projection_dim=5)
    h, ent, top1 = _make_inputs(T=4, num_layers=1, hidden_dim=12, seed=3)
    z = extract_token_features(h, ent, top1, params)
    # softmax of a single logit is 1 → h_agg equals the only layer.
    h_proj_manual = params.W(h[:, 0, :])
    expected = torch.cat(
        [h_proj_manual, ent.unsqueeze(1), top1.unsqueeze(1)], dim=1
    )
    assert torch.allclose(z, expected, atol=1e-6)


def test_accepts_fp16_hidden_states_and_returns_fp32() -> None:
    """Generation stores ``h^(l)`` in fp16; the extractor must promote."""
    params = SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=4)
    h32, ent, top1 = _make_inputs(T=5, num_layers=3, hidden_dim=8, seed=4)
    z16 = extract_token_features(h32.to(torch.float16), ent, top1, params)
    z32 = extract_token_features(h32, ent, top1, params)
    assert z16.dtype == torch.float32
    # Tolerances reflect the fp16 round-trip on the inputs.
    assert torch.allclose(z16, z32, atol=1e-2)


def test_gradients_flow_through_W_alpha_and_hidden_states() -> None:
    params = SentenceUQParams(hidden_dim=8, num_layers=4, projection_dim=3)
    h, ent, top1 = _make_inputs(T=6, num_layers=4, hidden_dim=8, seed=5)
    h = h.detach().requires_grad_(True)

    z = extract_token_features(h, ent, top1, params)
    loss = z.sum()
    loss.backward()

    assert params.W.weight.grad is not None
    assert torch.isfinite(params.W.weight.grad).all()
    assert params.W.weight.grad.abs().sum() > 0

    assert params.alpha.grad is not None
    assert torch.isfinite(params.alpha.grad).all()
    # Softmax gradient sums to zero in expectation, but components must be
    # non-trivial as long as W is non-zero and layers differ.
    assert params.alpha.grad.abs().sum() > 0

    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert h.grad.abs().sum() > 0

    # The prior parameters do not participate in z and must have no gradient.
    assert params.mu_0.grad is None
    assert params.log_sigma_0.grad is None


def test_extract_token_features_validates_shapes() -> None:
    params = SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=4)
    h, ent, top1 = _make_inputs(T=4, num_layers=3, hidden_dim=8)
    with pytest.raises(ValueError):
        extract_token_features(h[:, 0], ent, top1, params)  # 2-D h
    with pytest.raises(ValueError):
        extract_token_features(h[..., :7], ent, top1, params)  # wrong d
    with pytest.raises(ValueError):
        extract_token_features(h[:, :2], ent, top1, params)  # wrong L
    with pytest.raises(ValueError):
        extract_token_features(h, ent[:-1], top1, params)
    with pytest.raises(ValueError):
        extract_token_features(h, ent, top1[:-1], params)


# --- extract_sentence_token_features ----------------------------------------


def test_sentence_slice_matches_global_slice() -> None:
    params = SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=4)
    h, ent, top1 = _make_inputs(T=10, num_layers=3, hidden_dim=8, seed=6)
    z_all = extract_token_features(h, ent, top1, params)
    z_sent = extract_sentence_token_features(h, ent, top1, (3, 7), params)
    assert z_sent.shape == (4, params.feature_dim)
    assert torch.allclose(z_sent, z_all[3:7], atol=1e-6)


def test_sentence_slice_handles_single_token() -> None:
    params = SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=4)
    h, ent, top1 = _make_inputs(T=5, num_layers=3, hidden_dim=8, seed=8)
    z_sent = extract_sentence_token_features(h, ent, top1, (2, 3), params)
    assert z_sent.shape == (1, params.feature_dim)


def test_sentence_slice_validates_range() -> None:
    params = SentenceUQParams(hidden_dim=8, num_layers=3, projection_dim=4)
    h, ent, top1 = _make_inputs(T=5, num_layers=3, hidden_dim=8)
    with pytest.raises(ValueError):
        extract_sentence_token_features(h, ent, top1, (-1, 3), params)
    with pytest.raises(ValueError):
        extract_sentence_token_features(h, ent, top1, (3, 2), params)
    with pytest.raises(ValueError):
        extract_sentence_token_features(h, ent, top1, (0, 6), params)
    with pytest.raises(TypeError):
        extract_sentence_token_features(h, ent, top1, (0.0, 3), params)  # type: ignore[arg-type]


# --- extract_sentence_aggregate_feature -------------------------------------


def test_aggregate_feature_shape_and_layout() -> None:
    """Layout is [mean(z), std(z), z_last], each ∈ R^k → total 3k."""
    k = 6
    z = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 0.0, 1.0, 2.0, 7.0, 8.0],
            [2.0, 4.0, 2.0, 0.0, 6.0, 4.0],
        ]
    )
    agg = extract_sentence_aggregate_feature(z)
    assert agg.shape == (3 * k,)

    expected_mean = z.mean(dim=0)
    expected_std = z.std(dim=0, unbiased=False)
    expected_last = z[-1]

    assert torch.allclose(agg[:k], expected_mean, atol=1e-6)
    assert torch.allclose(agg[k : 2 * k], expected_std, atol=1e-6)
    assert torch.allclose(agg[2 * k :], expected_last, atol=1e-6)


def test_aggregate_feature_single_token_uses_zero_std() -> None:
    """``L_j = 1``: std must be 0 (not NaN); mean and last equal the row."""
    z = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
    agg = extract_sentence_aggregate_feature(z)
    k = z.shape[1]
    assert agg.shape == (3 * k,)
    assert torch.allclose(agg[:k], z[0])
    assert torch.allclose(agg[k : 2 * k], torch.zeros(k))
    assert torch.allclose(agg[2 * k :], z[0])
    assert torch.isfinite(agg).all()


def test_aggregate_feature_validates_input() -> None:
    with pytest.raises(ValueError):
        extract_sentence_aggregate_feature(torch.zeros(5))  # 1-D
    with pytest.raises(ValueError):
        extract_sentence_aggregate_feature(torch.zeros(0, 4))  # L_j = 0


def test_aggregate_feature_is_differentiable() -> None:
    """Gradients must flow — used by the Phase 4-2 auxiliary model."""
    z = torch.randn(4, 5, requires_grad=True)
    agg = extract_sentence_aggregate_feature(z)
    agg.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


@pytest.mark.parametrize("hidden_dim,num_layers", _MODEL_CONFIGS)
def test_end_to_end_sentence_pipeline(
    hidden_dim: int, num_layers: int
) -> None:
    """Combine slicing + aggregation for a realistic per-model sanity check."""
    projection_dim = 32
    params = SentenceUQParams(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        projection_dim=projection_dim,
    )
    h, ent, top1 = _make_inputs(
        T=12, num_layers=num_layers, hidden_dim=hidden_dim, seed=42
    )

    z_sent = extract_sentence_token_features(h, ent, top1, (4, 9), params)
    assert z_sent.shape == (5, projection_dim + 2)

    agg = extract_sentence_aggregate_feature(z_sent)
    assert agg.shape == (3 * (projection_dim + 2),)
    assert torch.isfinite(agg).all()
