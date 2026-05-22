"""
Unit tests for src/data/generation.py (Phase 1-1).

Uses GPT-2 (small, CPU-friendly) to avoid needing the full Llama model.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpt2_model_and_tokenizer():
    """Load tiny GPT-2 for testing (downloads ~548 MB on first run)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        output_hidden_states=True,
        weights_only=False,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Test: load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    @pytest.fixture(scope="class")
    def cpu_model_and_tok(self):
        """Load GPT-2 on CPU to avoid CUDA compatibility issues in test environments."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", device_map="cpu", output_hidden_states=True, weights_only=False
        )
        model.eval()
        return model, tokenizer

    def test_returns_model_and_tokenizer(self, cpu_model_and_tok):
        model, tokenizer = cpu_model_and_tok
        assert model is not None
        assert tokenizer is not None

    def test_model_in_eval_mode(self, cpu_model_and_tok):
        model, _ = cpu_model_and_tok
        assert not model.training

    def test_model_supports_hidden_states(self, cpu_model_and_tok):
        """Forward pass should return hidden_states tuple."""
        model, tokenizer = cpu_model_and_tok
        ids = tokenizer.encode("hello", return_tensors="pt")
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        assert out.hidden_states is not None
        assert len(out.hidden_states) > 1


# ---------------------------------------------------------------------------
# Test: generate_with_hidden_states
# ---------------------------------------------------------------------------

class TestGenerateWithHiddenStates:
    @pytest.fixture(scope="class")
    def model_and_tok(self):
        return _gpt2_model_and_tokenizer()

    def test_output_keys(self, model_and_tok):
        from src.data.generation import generate_with_hidden_states

        model, tokenizer = model_and_tok
        result = generate_with_hidden_states(
            model, tokenizer, "Hello", max_new_tokens=5, selected_layers=[0, 1]
        )
        assert set(result.keys()) == {"text", "token_ids", "hidden_states", "logits"}

    def test_text_is_string(self, model_and_tok):
        from src.data.generation import generate_with_hidden_states

        model, tokenizer = model_and_tok
        result = generate_with_hidden_states(
            model, tokenizer, "Hello", max_new_tokens=5, selected_layers=[0, 1]
        )
        assert isinstance(result["text"], str)

    def test_token_ids_shape(self, model_and_tok):
        from src.data.generation import generate_with_hidden_states

        model, tokenizer = model_and_tok
        result = generate_with_hidden_states(
            model, tokenizer, "Hello", max_new_tokens=10, selected_layers=[0, 1]
        )
        T = result["token_ids"].shape[0]
        assert result["token_ids"].ndim == 1
        assert T <= 10

    def test_hidden_states_shape(self, model_and_tok):
        from src.data.generation import generate_with_hidden_states

        model, tokenizer = model_and_tok
        selected_layers = [0, 1, 2]
        result = generate_with_hidden_states(
            model, tokenizer, "Hello world", max_new_tokens=8, selected_layers=selected_layers
        )
        T = result["token_ids"].shape[0]
        hs = result["hidden_states"]
        assert hs.shape == (T, len(selected_layers), model.config.hidden_size), (
            f"Expected ({T}, {len(selected_layers)}, {model.config.hidden_size}), got {hs.shape}"
        )

    def test_hidden_states_dtype_fp16(self, model_and_tok):
        from src.data.generation import generate_with_hidden_states

        model, tokenizer = model_and_tok
        result = generate_with_hidden_states(
            model, tokenizer, "Hello", max_new_tokens=5, selected_layers=[0, 1]
        )
        assert result["hidden_states"].dtype == torch.float16

    def test_logits_shape(self, model_and_tok):
        from src.data.generation import generate_with_hidden_states

        model, tokenizer = model_and_tok
        result = generate_with_hidden_states(
            model, tokenizer, "Hello", max_new_tokens=7, selected_layers=[0]
        )
        T = result["token_ids"].shape[0]
        assert result["logits"].shape == (T, model.config.vocab_size)

    def test_logits_dtype_fp16(self, model_and_tok):
        from src.data.generation import generate_with_hidden_states

        model, tokenizer = model_and_tok
        result = generate_with_hidden_states(
            model, tokenizer, "Hello", max_new_tokens=5, selected_layers=[0]
        )
        assert result["logits"].dtype == torch.float16

    def test_token_ids_and_hidden_states_length_match(self, model_and_tok):
        from src.data.generation import generate_with_hidden_states

        model, tokenizer = model_and_tok
        result = generate_with_hidden_states(
            model, tokenizer, "Test prompt.", max_new_tokens=12, selected_layers=[0, 6]
        )
        T = result["token_ids"].shape[0]
        assert result["hidden_states"].shape[0] == T
        assert result["logits"].shape[0] == T

    def test_respects_max_new_tokens(self, model_and_tok):
        from src.data.generation import generate_with_hidden_states

        model, tokenizer = model_and_tok
        max_new = 3
        result = generate_with_hidden_states(
            model, tokenizer, "Hello", max_new_tokens=max_new, selected_layers=[0]
        )
        assert result["token_ids"].shape[0] <= max_new


# ---------------------------------------------------------------------------
# Test: save_generation / round-trip
# ---------------------------------------------------------------------------

class TestSaveGeneration:
    def test_save_and_reload(self):
        from src.data.generation import generate_with_hidden_states, save_generation

        model, tokenizer = _gpt2_model_and_tokenizer()
        result = generate_with_hidden_states(
            model, tokenizer, "Hello", max_new_tokens=5, selected_layers=[0, 1]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            save_generation(result, path)
            assert path.exists()

            loaded = torch.load(path, weights_only=False)
            assert loaded["text"] == result["text"]
            assert torch.equal(loaded["token_ids"], result["token_ids"])
            assert torch.equal(loaded["hidden_states"], result["hidden_states"])
            assert torch.equal(loaded["logits"], result["logits"])


# ---------------------------------------------------------------------------
# Test: batch_generate
# ---------------------------------------------------------------------------

class TestBatchGenerate:
    def test_creates_pt_files_and_metadata(self):
        from src.data.generation import batch_generate

        model, tokenizer = _gpt2_model_and_tokenizer()
        prompts = ["Tell me about Einstein.", "Tell me about Curie."]
        entities = ["Einstein", "Curie"]

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                save_dir=tmpdir,
                selected_layers=[0, 1],
                max_new_tokens=5,
                entities=entities,
            )

            for idx in range(len(prompts)):
                assert (Path(tmpdir) / f"{idx:05d}.pt").exists()

            import json
            with open(Path(tmpdir) / "metadata.json") as f:
                meta = json.load(f)
            assert len(meta) == len(prompts)
            assert meta[0]["entity"] == "Einstein"
            assert meta[1]["entity"] == "Curie"

    def test_resume_skips_existing(self):
        """Running batch_generate twice should not re-generate already-saved entries."""
        from src.data.generation import batch_generate, generate_with_hidden_states, save_generation

        model, tokenizer = _gpt2_model_and_tokenizer()
        prompts = ["Hello.", "World."]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-save index 0 manually
            result = generate_with_hidden_states(
                model, tokenizer, prompts[0], max_new_tokens=3, selected_layers=[0]
            )
            save_generation(result, Path(tmpdir) / "00000.pt")
            import json
            with open(Path(tmpdir) / "metadata.json", "w") as f:
                json.dump([{"idx": 0, "prompt": prompts[0]}], f)

            # Patch generate_with_hidden_states to count calls
            call_count = {"n": 0}
            original = generate_with_hidden_states

            def counting_generate(*args, **kwargs):
                call_count["n"] += 1
                return original(*args, **kwargs)

            import src.data.generation as gen_module
            original_fn = gen_module.generate_with_hidden_states
            gen_module.generate_with_hidden_states = counting_generate

            try:
                batch_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    save_dir=tmpdir,
                    selected_layers=[0],
                    max_new_tokens=3,
                )
            finally:
                gen_module.generate_with_hidden_states = original_fn

            # Only index 1 should have been generated
            assert call_count["n"] == 1
