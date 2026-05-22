"""Tests for ``src.data.sentence_split`` — Phase 1-2 sentence splitting + token mapping.

We test against two backends:

- A **real** fast HuggingFace tokenizer (``gpt2``) — exercises the
  ``return_offsets_mapping`` path that runs in production with Llama / Mistral / etc.
- A small **fake slow** tokenizer — exercises the incremental-decode fallback
  used when ``return_offsets_mapping`` is unavailable.

Both share the actual ``en_core_web_sm`` spaCy pipeline; auto-install of
the model is the user's responsibility (``requirements.txt`` lists spacy and
the project's CLAUDE.md describes the auto-download path).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.sentence_split import (  # noqa: E402
    load_spacy_model,
    map_sentences_to_tokens,
    process_generation,
    split_into_sentences,
)


# ---------------------------------------------------------------------------
# spaCy fixture (shared across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nlp() -> Any:
    """Load en_core_web_sm once for the whole module."""
    try:
        return load_spacy_model("en")
    except Exception as exc:  # pragma: no cover - env-specific
        pytest.skip(f"spaCy en_core_web_sm not available: {exc}")


# ---------------------------------------------------------------------------
# Real GPT-2 tokenizer (fast, supports return_offsets_mapping)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_tokenizer() -> Any:
    """A real fast HuggingFace tokenizer — same offset API as Llama / Mistral."""
    try:
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover - dependency check
        pytest.skip("transformers not available")
    try:
        return AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not load gpt2 tokenizer: {exc}")


# ---------------------------------------------------------------------------
# Fake slow tokenizer (no offset_mapping → forces the incremental-decode path)
# ---------------------------------------------------------------------------


class _CharTokenizer:
    """One character == one token. Slow tokenizer with no offset support.

    - ``__call__`` does NOT honour ``return_offsets_mapping`` (raises
      ``NotImplementedError`` if asked) so the production code drops to the
      incremental-decode fallback.
    - ``decode(ids[:k])`` returns exactly the first ``k`` characters of the
      original text, which makes assertions about the fallback path easy.
    """

    def __init__(self, text: str) -> None:
        self._text = text
        # id i (1-indexed; reserve 0 as a sentinel) → character i-1
        self._id_to_char = {i + 1: c for i, c in enumerate(text)}
        self._char_to_id = {(i, c): i + 1 for i, c in enumerate(text)}

    def encode(self, text: str) -> list[int]:
        assert text == self._text, "fixture only encodes its own text"
        return [i + 1 for i in range(len(text))]

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
    ) -> dict[str, Any]:
        if return_offsets_mapping:
            raise NotImplementedError("slow tokenizer: no offset mapping")
        return {"input_ids": self.encode(text)}

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self._id_to_char[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# split_into_sentences
# ---------------------------------------------------------------------------


def test_split_two_sentences(nlp: Any) -> None:
    text = "Hello world. This is a test."
    sents = split_into_sentences(text, nlp)
    assert len(sents) == 2
    assert sents[0]["text"].strip() == "Hello world."
    assert sents[1]["text"].strip() == "This is a test."
    # Char ranges actually slice back to the same substrings
    for s in sents:
        assert text[s["char_start"]:s["char_end"]] == s["text"]


def test_split_single_sentence(nlp: Any) -> None:
    text = "Just one statement here"
    sents = split_into_sentences(text, nlp)
    assert len(sents) == 1
    assert sents[0]["char_start"] == 0
    assert sents[0]["char_end"] == len(text)


def test_split_empty_text(nlp: Any) -> None:
    assert split_into_sentences("", nlp) == []


def test_split_drops_whitespace_only_sentences(nlp: Any) -> None:
    # spaCy occasionally splits on stray newlines; whitespace-only spans must
    # never be returned (they would later get an empty token range anyway).
    sents = split_into_sentences("Real sentence.   ", nlp)
    assert all(s["text"].strip() for s in sents)


# ---------------------------------------------------------------------------
# map_sentences_to_tokens — fast path (real tokenizer)
# ---------------------------------------------------------------------------


def test_map_with_real_tokenizer_recovers_sentence_text(
    nlp: Any, gpt2_tokenizer: Any
) -> None:
    """Decoded token ranges must round-trip to the original sentence strings."""
    text = "Hello world. This is a test."
    ids = gpt2_tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids = torch.tensor(ids, dtype=torch.long)

    sents = split_into_sentences(text, nlp)
    ranges = map_sentences_to_tokens(sents, token_ids, gpt2_tokenizer)

    assert len(ranges) == len(sents) == 2
    for s, (a, b) in zip(sents, ranges):
        assert 0 <= a < b <= token_ids.numel()
        decoded = gpt2_tokenizer.decode(
            token_ids[a:b].tolist(), skip_special_tokens=True
        )
        # Decoded slice contains the sentence content (modulo leading space
        # that BPE attaches to mid-text tokens).
        assert s["text"].strip() in decoded.strip() or decoded.strip() in s["text"].strip()
        # The two sentences together must contain the visible characters.
    # Concatenating the decoded slices reconstructs the full text.
    decoded_full = "".join(
        gpt2_tokenizer.decode(token_ids[a:b].tolist(), skip_special_tokens=True)
        for a, b in ranges
    )
    assert decoded_full.strip() == text.strip()


def test_map_ranges_are_non_overlapping_and_ordered(
    nlp: Any, gpt2_tokenizer: Any
) -> None:
    text = "First fact about cats. Second fact about dogs. Third fact."
    ids = gpt2_tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids = torch.tensor(ids, dtype=torch.long)

    sents = split_into_sentences(text, nlp)
    ranges = map_sentences_to_tokens(sents, token_ids, gpt2_tokenizer)

    assert len(ranges) == len(sents)
    prev_end = 0
    for a, b in ranges:
        assert a >= prev_end, "token ranges must be ordered and non-overlapping"
        assert b > a, "non-empty range expected for normal sentences"
        prev_end = b
    assert prev_end == token_ids.numel(), "all tokens must be assigned"


# ---------------------------------------------------------------------------
# map_sentences_to_tokens — fallback path (slow tokenizer)
# ---------------------------------------------------------------------------


def test_map_with_slow_tokenizer_uses_incremental_decode(nlp: Any) -> None:
    """A tokenizer with no offset API must still produce correct ranges."""
    text = "Hello world. This is a test."
    tok = _CharTokenizer(text)
    token_ids = torch.tensor(tok.encode(text), dtype=torch.long)

    sents = split_into_sentences(text, nlp)
    ranges = map_sentences_to_tokens(sents, token_ids, tok)

    assert len(ranges) == 2
    # Char tokenizer = 1 char per token, so token_range matches char_start /
    # char_end exactly (modulo where whitespace tokens land per the
    # "preceding sentence" rule).
    # Sentence 0 covers chars [0, 12), sentence 1 covers [13, 28). The space
    # at index 12 is pure whitespace → preceding sentence → ends up in S0.
    assert ranges[0] == (0, 13)
    assert ranges[1] == (13, 28)


def test_map_empty_inputs() -> None:
    """No sentences and/or no tokens must not crash."""

    class _Dummy:
        def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
            return ""

    assert map_sentences_to_tokens([], torch.zeros(0, dtype=torch.long), _Dummy()) == []
    sents = [{"text": "x", "char_start": 0, "char_end": 1}]
    assert map_sentences_to_tokens(sents, torch.zeros(0, dtype=torch.long), _Dummy()) == [(0, 0)]


# ---------------------------------------------------------------------------
# process_generation
# ---------------------------------------------------------------------------


def test_process_generation_attaches_token_ranges(
    nlp: Any, gpt2_tokenizer: Any
) -> None:
    text = "Hello world. This is a test."
    ids = gpt2_tokenizer(text, add_special_tokens=False)["input_ids"]
    rec = {"text": text, "token_ids": torch.tensor(ids, dtype=torch.long)}

    out = process_generation(rec, gpt2_tokenizer, nlp)
    assert "sentences" in out
    sents = out["sentences"]
    assert len(sents) == 2
    for s in sents:
        assert {"text", "char_start", "char_end", "token_range"} <= set(s.keys())
        a, b = s["token_range"]
        assert isinstance(a, int) and isinstance(b, int)
        assert 0 <= a < b <= len(ids)


def test_process_generation_filters_empty_ranges(nlp: Any) -> None:
    """Sentences with no tokens assigned to them must be dropped."""

    class _NoTextTokenizer:
        """Decodes to empty string regardless of ids → no sentences will be split."""

        def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
            return ""

        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            return {"input_ids": [], "offset_mapping": []}

    rec = {"text": "", "token_ids": torch.tensor([], dtype=torch.long)}
    out = process_generation(rec, _NoTextTokenizer(), nlp)
    assert out["sentences"] == []


def test_process_generation_single_sentence(
    nlp: Any, gpt2_tokenizer: Any
) -> None:
    text = "A solitary statement"
    ids = gpt2_tokenizer(text, add_special_tokens=False)["input_ids"]
    rec = {"text": text, "token_ids": torch.tensor(ids, dtype=torch.long)}
    out = process_generation(rec, gpt2_tokenizer, nlp)
    assert len(out["sentences"]) == 1
    a, b = out["sentences"][0]["token_range"]
    assert (a, b) == (0, len(ids))


def test_process_generation_token_range_decodes_to_sentence(
    nlp: Any, gpt2_tokenizer: Any
) -> None:
    """Stronger version of the round-trip — each sentence's token slice decodes
    to text that contains the sentence's non-whitespace content."""
    text = (
        "Marie Curie was a physicist and chemist. "
        "She won two Nobel Prizes. "
        "She lived in Paris."
    )
    ids = gpt2_tokenizer(text, add_special_tokens=False)["input_ids"]
    rec = {"text": text, "token_ids": torch.tensor(ids, dtype=torch.long)}
    out = process_generation(rec, gpt2_tokenizer, nlp)

    assert len(out["sentences"]) == 3
    for s in out["sentences"]:
        a, b = s["token_range"]
        decoded = gpt2_tokenizer.decode(
            rec["token_ids"][a:b].tolist(), skip_special_tokens=True
        ).strip()
        # The decoded slice should match the sentence content modulo
        # surrounding whitespace.
        assert decoded == s["text"].strip()
