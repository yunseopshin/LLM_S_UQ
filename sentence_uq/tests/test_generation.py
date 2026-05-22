"""Tests for ``src.data.generation`` — Phase 1-1 generation utilities.

We avoid loading real HuggingFace models by injecting a small fake transformer
(``FakeCausalLM``) that mimics the surface area used by ``generation.py``:

- ``.config`` with ``hidden_size`` / ``num_hidden_layers`` / ``vocab_size``
- ``parameters()`` yielding a tensor (for device discovery)
- ``__call__(input_ids, past_key_values=None, use_cache=True,
  output_hidden_states=True)`` returning an object with ``logits``,
  ``hidden_states`` (tuple of length ``num_hidden_layers + 1``) and
  ``past_key_values``.

Tests are parameterised over multiple ``(hidden_dim, num_hidden_layers)``
configurations to guard against hardcoded assumptions (CLAUDE.md rule).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.generation import (  # noqa: E402
    apply_chat_template_if_available,
    auto_select_layers,
    batch_generate,
    generate_with_hidden_states,
    make_prompt,
    resolve_selected_layers,
    save_generation,
    write_dataset_metadata,
)


# ---------------------------------------------------------------------------
# Fake model / tokenizer
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Whitespace tokenizer with a built-in chat template + EOS."""

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.chat_template = "{user}"

    def apply_chat_template(
        self, messages, tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        # Tag the prompt so tests can verify the template path was taken.
        return "<chat>" + messages[-1]["content"]

    def _encode_str(self, text: str) -> list[int]:
        # Map characters to a small deterministic vocab (avoid id 0 == pad,
        # avoid id 1 == eos).
        return [(ord(c) % (self.vocab_size - 2)) + 2 for c in text][:32] or [2]

    def __call__(
        self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False
    ) -> dict[str, torch.Tensor]:
        ids = torch.tensor([self._encode_str(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(t) + 32) for t in token_ids if int(t) != self.eos_token_id)

    def convert_tokens_to_ids(self, tok: str) -> int:
        return -1  # no Llama-3 <|eot_id|> in the fake vocab


class FakeCausalLM(torch.nn.Module):
    """Minimal causal LM with the HF interface used by ``generation.py``.

    - Linear "embedding" + a stack of ``num_hidden_layers`` identity-ish layers
      that perturb the hidden state slightly so per-layer states differ.
    - A ``lm_head`` projecting to ``vocab_size``.
    - Returns ``hidden_states`` as a tuple of length ``num_hidden_layers + 1``
      (embedding output at index 0, then one per transformer block).
    - ``past_key_values`` carries running ``seq_len`` for asserting cache use.
    """

    def __init__(self, hidden_dim: int, num_hidden_layers: int, vocab_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            vocab_size=vocab_size,
            output_hidden_states=True,
            _name_or_path="fake-model",
        )
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_hidden_layers)]
        )
        # Initialise close to identity so outputs stay finite and distinguishable.
        for i, layer in enumerate(self.layers):
            with torch.no_grad():
                layer.weight.copy_(torch.eye(hidden_dim) + 0.01 * (i + 1))
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[dict] = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
    ) -> SimpleNamespace:
        assert output_hidden_states, "tests assume output_hidden_states=True"
        h = self.embed(input_ids)  # (1, L, D)
        hidden_states: list[torch.Tensor] = [h]
        for layer in self.layers:
            h = torch.tanh(layer(h))
            hidden_states.append(h)
        logits = self.lm_head(h)
        prev_len = int(past_key_values["seq_len"]) if past_key_values else 0
        new_past = {"seq_len": prev_len + int(input_ids.shape[1])}
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states),
            past_key_values=new_past,
        )


_CONFIGS = [
    (16, 4, 32),   # tiny
    (32, 6, 48),   # different dims (catches hardcoded-shape regressions)
]


# ---------------------------------------------------------------------------
# auto_select_layers
# ---------------------------------------------------------------------------


def test_auto_select_layers_matches_spec_examples() -> None:
    assert auto_select_layers(32, 8) == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    assert auto_select_layers(28, 8) == [0, 4, 8, 12, 16, 20, 24, 28]


@pytest.mark.parametrize("n,target", [(4, 2), (6, 3), (16, 4), (40, 8), (42, 8)])
def test_auto_select_layers_invariants(n: int, target: int) -> None:
    layers = auto_select_layers(n, target)
    # Contains both endpoints, strictly increasing, all in range.
    assert layers[0] == 0
    assert layers[-1] == n
    assert all(0 <= li <= n for li in layers)
    assert layers == sorted(set(layers))


def test_auto_select_layers_validates_inputs() -> None:
    with pytest.raises(ValueError):
        auto_select_layers(0, 8)
    with pytest.raises(ValueError):
        auto_select_layers(8, 1)


def test_resolve_selected_layers_passthrough_validates() -> None:
    assert resolve_selected_layers(8, None) == auto_select_layers(8)
    assert resolve_selected_layers(8, [0, 4, 8]) == [0, 4, 8]
    assert resolve_selected_layers(8, [8, 0, 4]) == [0, 4, 8]  # sorted
    with pytest.raises(ValueError):
        resolve_selected_layers(8, [0, 0, 4])
    with pytest.raises(ValueError):
        resolve_selected_layers(8, [0, 9])
    with pytest.raises(ValueError):
        resolve_selected_layers(8, [-1, 4])


# ---------------------------------------------------------------------------
# make_prompt + chat template
# ---------------------------------------------------------------------------


def test_make_prompt_factscore_bio_uses_template() -> None:
    item = {"dataset": "factscore_bio", "entity": "Marie Curie", "prompt": "ignored"}
    assert make_prompt(item) == "Tell me a bio of Marie Curie."


def test_make_prompt_longfact_passthrough() -> None:
    item = {"dataset": "longfact", "prompt": "Describe X in detail.", "topic": "x"}
    assert make_prompt(item) == "Describe X in detail."


def test_make_prompt_unknown_dataset_falls_back_to_prompt() -> None:
    item = {"prompt": "raw prompt"}
    assert make_prompt(item) == "raw prompt"


def test_apply_chat_template_uses_template_when_available() -> None:
    tok = FakeTokenizer()
    out = apply_chat_template_if_available(tok, "hello")
    assert out == "<chat>hello"


def test_apply_chat_template_falls_back_when_absent() -> None:
    class Bare:
        chat_template = None

    out = apply_chat_template_if_available(Bare(), "hello")
    assert out == "hello"


# ---------------------------------------------------------------------------
# generate_with_hidden_states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hidden_dim,num_layers,vocab", _CONFIGS)
def test_generate_shapes_and_dtypes(hidden_dim: int, num_layers: int, vocab: int) -> None:
    torch.manual_seed(0)
    model = FakeCausalLM(hidden_dim, num_layers, vocab)
    tok = FakeTokenizer(vocab_size=vocab)
    selected = resolve_selected_layers(num_layers, None)

    rec = generate_with_hidden_states(
        model,
        tok,
        prompt="hi there",
        selected_layers=selected,
        max_new_tokens=5,
        do_sample=False,  # greedy → deterministic, easier to inspect
    )

    T = rec["token_ids"].shape[0]
    assert 0 < T <= 5
    assert rec["hidden_states"].shape == (T, len(selected), hidden_dim)
    assert rec["logits"].shape == (T, vocab)
    assert rec["hidden_states"].dtype == torch.float16
    assert rec["logits"].dtype == torch.float16
    assert rec["token_ids"].dtype == torch.long
    assert rec["prompt"] == "hi there"
    # Chat template path was taken (FakeTokenizer prefixes with "<chat>").
    assert rec["prompt_text"].startswith("<chat>")
    assert isinstance(rec["text"], str)


def test_generate_validates_selected_layer_range() -> None:
    model = FakeCausalLM(16, 4, 32)
    tok = FakeTokenizer(vocab_size=32)
    with pytest.raises(ValueError):
        generate_with_hidden_states(
            model, tok, "hi", selected_layers=[0, 999], max_new_tokens=2,
        )


def test_generate_greedy_is_deterministic() -> None:
    model = FakeCausalLM(16, 4, 32)
    tok = FakeTokenizer(vocab_size=32)
    a = generate_with_hidden_states(
        model, tok, "abc", selected_layers=[0, 4], max_new_tokens=4, do_sample=False,
    )
    b = generate_with_hidden_states(
        model, tok, "abc", selected_layers=[0, 4], max_new_tokens=4, do_sample=False,
    )
    assert torch.equal(a["token_ids"], b["token_ids"])


def test_generate_stops_on_eos() -> None:
    """If the model is forced to emit EOS, generation halts early."""

    class AlwaysEOS(FakeCausalLM):
        def forward(self, *args, **kwargs):  # type: ignore[override]
            out = super().forward(*args, **kwargs)
            logits = out.logits.clone()
            logits[...] = -1e9
            logits[..., 1] = 1e9  # eos_token_id
            return SimpleNamespace(
                logits=logits,
                hidden_states=out.hidden_states,
                past_key_values=out.past_key_values,
            )

    model = AlwaysEOS(16, 4, 32)
    tok = FakeTokenizer(vocab_size=32)
    assert tok.eos_token_id == 1
    rec = generate_with_hidden_states(
        model, tok, "abc", selected_layers=[0], max_new_tokens=8, do_sample=False,
    )
    # First sampled token is EOS → 0 generated tokens, finished=True.
    assert rec["token_ids"].numel() == 0
    assert rec["finished"] is True
    assert rec["hidden_states"].shape == (0, 1, 16)
    assert rec["logits"].shape == (0, 32)


# ---------------------------------------------------------------------------
# save_generation + batch_generate + metadata
# ---------------------------------------------------------------------------


def test_save_generation_roundtrip(tmp_path: Path) -> None:
    model = FakeCausalLM(16, 4, 32)
    tok = FakeTokenizer(vocab_size=32)
    selected = [0, 2, 4]
    rec = generate_with_hidden_states(
        model, tok, "hi", selected_layers=selected, max_new_tokens=3, do_sample=False,
    )
    model_info = {
        "model_name": "fake-model",
        "hidden_dim": 16,
        "num_hidden_layers": 4,
        "vocab_size": 32,
    }
    out = tmp_path / "x.pt"
    save_generation(
        rec, out, model_info=model_info, selected_layers=selected,
        dataset_tag="factscore_bio", meta={"entity": "Marie Curie"},
    )
    loaded = torch.load(out, weights_only=False)
    assert loaded["model_config"] == {
        "name": "fake-model",
        "hidden_dim": 16,
        "num_hidden_layers": 4,
        "vocab_size": 32,
        "selected_layers": [0, 2, 4],
    }
    assert loaded["selected_layers"] == [0, 2, 4]
    assert loaded["dataset"] == "factscore_bio"
    assert loaded["meta"] == {"entity": "Marie Curie"}
    assert torch.equal(loaded["token_ids"], rec["token_ids"])
    assert loaded["hidden_states"].shape == rec["hidden_states"].shape


def test_batch_generate_writes_files_per_dataset(tmp_path: Path) -> None:
    model = FakeCausalLM(16, 4, 32)
    tok = FakeTokenizer(vocab_size=32)
    selected = [0, 2, 4]
    model_info = {
        "model_name": "fake-model",
        "hidden_dim": 16,
        "num_hidden_layers": 4,
        "vocab_size": 32,
    }
    items = [
        {"dataset": "factscore_bio", "entity": "Marie Curie",
         "prompt": "Tell me a bio of Marie Curie.", "prompt_idx": 0},
        {"dataset": "factscore_bio", "entity": "Ada Lovelace",
         "prompt": "Tell me a bio of Ada Lovelace.", "prompt_idx": 1},
        {"dataset": "longfact", "topic": "chemistry",
         "prompt": "Tell me about catalysis.", "prompt_idx": 3},
    ]
    fs_dir = tmp_path / "gen" / "factscore_bio"
    lf_dir = tmp_path / "gen" / "longfact"
    res = batch_generate(
        items, model=model, tokenizer=tok, model_info=model_info,
        selected_layers=selected, factscore_dir=fs_dir, longfact_dir=lf_dir,
        max_new_tokens=3, do_sample=False, progress=False,
    )
    assert res["generated"] == 3
    assert res["skipped"] == 0
    assert res["errors"] == []
    assert (fs_dir / "Marie_Curie.pt").exists()
    assert (fs_dir / "Ada_Lovelace.pt").exists()
    assert (lf_dir / "chemistry" / "003.pt").exists()


def test_batch_generate_skips_existing(tmp_path: Path) -> None:
    model = FakeCausalLM(16, 4, 32)
    tok = FakeTokenizer(vocab_size=32)
    model_info = {
        "model_name": "fake-model", "hidden_dim": 16,
        "num_hidden_layers": 4, "vocab_size": 32,
    }
    items = [
        {"dataset": "factscore_bio", "entity": "X",
         "prompt": "Tell me a bio of X.", "prompt_idx": 0},
    ]
    fs_dir = tmp_path / "gen" / "factscore_bio"
    lf_dir = tmp_path / "gen" / "longfact"
    res1 = batch_generate(
        items, model=model, tokenizer=tok, model_info=model_info,
        selected_layers=[0, 2, 4], factscore_dir=fs_dir, longfact_dir=lf_dir,
        max_new_tokens=2, do_sample=False, progress=False,
    )
    res2 = batch_generate(
        items, model=model, tokenizer=tok, model_info=model_info,
        selected_layers=[0, 2, 4], factscore_dir=fs_dir, longfact_dir=lf_dir,
        max_new_tokens=2, do_sample=False, progress=False,
    )
    assert res1["generated"] == 1 and res1["skipped"] == 0
    assert res2["generated"] == 0 and res2["skipped"] == 1


def test_write_dataset_metadata(tmp_path: Path) -> None:
    out = tmp_path / "gen" / "longfact"
    model_info = {
        "model_name": "fake-model", "hidden_dim": 32,
        "num_hidden_layers": 6, "vocab_size": 48,
    }
    items = [
        {"dataset": "longfact", "topic": "t", "prompt": "p", "prompt_idx": 0},
        {"dataset": "longfact", "topic": "t", "prompt": "p", "prompt_idx": 1},
    ]
    path = write_dataset_metadata(
        out, model_info=model_info, selected_layers=[0, 3, 6],
        generation_config={"max_new_tokens": 2}, dataset_tag="longfact",
        items=items,
    )
    assert path.exists()
    import json
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["dataset"] == "longfact"
    assert meta["model"]["hidden_dim"] == 32
    assert meta["selected_layers"] == [0, 3, 6]
    assert meta["num_items"] == 2
