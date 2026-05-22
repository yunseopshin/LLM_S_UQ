"""
Unit tests for src/features/cached_scalars.py (Phase 1-3).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# compute_token_entropy_and_top1
# ---------------------------------------------------------------------------

class TestComputeTokenEntropyAndTop1:
    def test_output_shapes(self):
        from src.features.cached_scalars import compute_token_entropy_and_top1

        T, V = 10, 100
        logits = torch.randn(T, V)
        entropy, top1 = compute_token_entropy_and_top1(logits)
        assert entropy.shape == (T,)
        assert top1.shape == (T,)

    def test_output_dtype_fp32(self):
        from src.features.cached_scalars import compute_token_entropy_and_top1

        logits = torch.randn(5, 50, dtype=torch.float16)
        entropy, top1 = compute_token_entropy_and_top1(logits)
        assert entropy.dtype == torch.float32
        assert top1.dtype == torch.float32

    def test_entropy_nonnegative(self):
        from src.features.cached_scalars import compute_token_entropy_and_top1

        logits = torch.randn(20, 200)
        entropy, _ = compute_token_entropy_and_top1(logits)
        assert (entropy >= 0).all(), "Entropy must be non-negative"

    def test_top1_prob_in_0_1(self):
        from src.features.cached_scalars import compute_token_entropy_and_top1

        logits = torch.randn(20, 200)
        _, top1 = compute_token_entropy_and_top1(logits)
        assert (top1 >= 0).all() and (top1 <= 1).all()

    def test_uniform_distribution_max_entropy(self):
        """Uniform logits → entropy should equal log(V)."""
        from src.features.cached_scalars import compute_token_entropy_and_top1

        T, V = 4, 1000
        logits = torch.zeros(T, V)  # uniform after softmax
        entropy, top1 = compute_token_entropy_and_top1(logits)

        expected_entropy = torch.tensor(V).float().log()
        assert torch.allclose(entropy, expected_entropy.expand(T), atol=1e-4)
        assert torch.allclose(top1, torch.full((T,), 1.0 / V), atol=1e-5)

    def test_peaked_distribution_low_entropy(self):
        """One dominant logit → near-zero entropy, top1 ≈ 1."""
        from src.features.cached_scalars import compute_token_entropy_and_top1

        T, V = 3, 500
        logits = torch.full((T, V), -100.0)
        logits[:, 0] = 100.0  # spike at index 0

        entropy, top1 = compute_token_entropy_and_top1(logits)
        assert (entropy < 1e-3).all()
        assert torch.allclose(top1, torch.ones(T), atol=1e-4)

    def test_fp16_input_accepted(self):
        """Should accept fp16 logits without error."""
        from src.features.cached_scalars import compute_token_entropy_and_top1

        logits = torch.randn(8, 50, dtype=torch.float16)
        entropy, top1 = compute_token_entropy_and_top1(logits)
        assert not torch.isnan(entropy).any()
        assert not torch.isnan(top1).any()

    def test_no_nan_with_zero_logits(self):
        """0*log(0) should be handled gracefully (nansum)."""
        from src.features.cached_scalars import compute_token_entropy_and_top1

        logits = torch.zeros(5, 100)
        entropy, top1 = compute_token_entropy_and_top1(logits)
        assert not torch.isnan(entropy).any()


# ---------------------------------------------------------------------------
# cache_scalars_for_directory + load_scalars
# ---------------------------------------------------------------------------

def _make_fake_generation(T: int = 10, V: int = 200) -> dict:
    return {
        "token_ids": torch.randint(0, V, (T,)),
        "logits": torch.randn(T, V, dtype=torch.float16),
        "hidden_states": torch.randn(T, 2, 64, dtype=torch.float16),
        "text": "fake text",
    }


class TestCacheScalarsForDirectory:
    def test_creates_output_files(self):
        from src.features.cached_scalars import cache_scalars_for_directory

        with tempfile.TemporaryDirectory() as gen_dir, \
             tempfile.TemporaryDirectory() as cache_dir:

            # Write 3 fake generation files
            for i in range(3):
                torch.save(_make_fake_generation(), Path(gen_dir) / f"{i:05d}.pt")

            cache_scalars_for_directory(gen_dir, cache_dir)

            for i in range(3):
                assert (Path(cache_dir) / f"{i:05d}.pt").exists()

    def test_output_keys(self):
        from src.features.cached_scalars import cache_scalars_for_directory

        with tempfile.TemporaryDirectory() as gen_dir, \
             tempfile.TemporaryDirectory() as cache_dir:

            torch.save(_make_fake_generation(), Path(gen_dir) / "00000.pt")
            cache_scalars_for_directory(gen_dir, cache_dir)

            data = torch.load(Path(cache_dir) / "00000.pt", weights_only=False)
            assert set(data.keys()) >= {"entropy", "top1_prob", "token_ids"}

    def test_output_shapes_match_token_count(self):
        from src.features.cached_scalars import cache_scalars_for_directory

        T = 15
        with tempfile.TemporaryDirectory() as gen_dir, \
             tempfile.TemporaryDirectory() as cache_dir:

            torch.save(_make_fake_generation(T=T), Path(gen_dir) / "00000.pt")
            cache_scalars_for_directory(gen_dir, cache_dir)

            data = torch.load(Path(cache_dir) / "00000.pt", weights_only=False)
            assert data["entropy"].shape == (T,)
            assert data["top1_prob"].shape == (T,)
            assert data["token_ids"].shape == (T,)

    def test_resume_skips_existing(self):
        """Files already in cache_dir should not be overwritten."""
        from src.features.cached_scalars import cache_scalars_for_directory

        with tempfile.TemporaryDirectory() as gen_dir, \
             tempfile.TemporaryDirectory() as cache_dir:

            for i in range(2):
                torch.save(_make_fake_generation(), Path(gen_dir) / f"{i:05d}.pt")

            # First run
            cache_scalars_for_directory(gen_dir, cache_dir)
            mtime0 = (Path(cache_dir) / "00000.pt").stat().st_mtime

            # Second run — file should not be touched
            cache_scalars_for_directory(gen_dir, cache_dir)
            mtime1 = (Path(cache_dir) / "00000.pt").stat().st_mtime

            assert mtime0 == mtime1

    def test_empty_directory_no_error(self):
        from src.features.cached_scalars import cache_scalars_for_directory

        with tempfile.TemporaryDirectory() as gen_dir, \
             tempfile.TemporaryDirectory() as cache_dir:
            cache_scalars_for_directory(gen_dir, cache_dir)  # should not raise


# ---------------------------------------------------------------------------
# load_scalars
# ---------------------------------------------------------------------------

class TestLoadScalars:
    def test_load_round_trip(self):
        from src.features.cached_scalars import cache_scalars_for_directory, load_scalars

        T = 12
        with tempfile.TemporaryDirectory() as gen_dir, \
             tempfile.TemporaryDirectory() as cache_dir:

            gen = _make_fake_generation(T=T)
            torch.save(gen, Path(gen_dir) / "00007.pt")
            cache_scalars_for_directory(gen_dir, cache_dir)

            data = load_scalars(7, cache_dir)
            assert data["entropy"].shape == (T,)
            assert data["top1_prob"].shape == (T,)
            assert torch.equal(data["token_ids"], gen["token_ids"])
