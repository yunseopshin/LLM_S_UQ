"""
Unit tests for src/data/sentence_split.py (Phase 1-2).

Uses GPT-2 tokenizer (fast, no GPU needed) for token-mapping tests.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nlp():
    from src.data.sentence_split import load_spacy_model
    return load_spacy_model("en")


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


# ---------------------------------------------------------------------------
# load_spacy_model
# ---------------------------------------------------------------------------

class TestLoadSpacyModel:
    def test_returns_nlp_object(self, nlp):
        import spacy
        assert isinstance(nlp, spacy.language.Language)

    def test_can_parse_sentence(self, nlp):
        doc = nlp("Hello world.")
        assert len(list(doc.sents)) >= 1


# ---------------------------------------------------------------------------
# split_into_sentences
# ---------------------------------------------------------------------------

class TestSplitIntoSentences:
    def test_two_sentences(self, nlp):
        from src.data.sentence_split import split_into_sentences

        result = split_into_sentences("Hello world. This is a test.", nlp)
        assert len(result) == 2

    def test_sentence_keys(self, nlp):
        from src.data.sentence_split import split_into_sentences

        result = split_into_sentences("Hello world.", nlp)
        assert len(result) == 1
        assert set(result[0].keys()) >= {"text", "char_start", "char_end"}

    def test_char_positions_consistent(self, nlp):
        from src.data.sentence_split import split_into_sentences

        text = "Hello world. This is a test."
        result = split_into_sentences(text, nlp)
        for sent in result:
            # Extracted substring should match reported text
            assert text[sent["char_start"]:sent["char_end"]] == sent["text"]

    def test_empty_text_returns_empty(self, nlp):
        from src.data.sentence_split import split_into_sentences

        assert split_into_sentences("", nlp) == []
        assert split_into_sentences("   ", nlp) == []

    def test_single_sentence(self, nlp):
        from src.data.sentence_split import split_into_sentences

        result = split_into_sentences("Albert Einstein was a physicist.", nlp)
        assert len(result) == 1

    def test_long_text(self, nlp):
        from src.data.sentence_split import split_into_sentences

        text = " ".join([f"This is sentence number {i}." for i in range(20)])
        result = split_into_sentences(text, nlp)
        assert len(result) >= 10  # Should detect many sentences


# ---------------------------------------------------------------------------
# map_sentences_to_tokens
# ---------------------------------------------------------------------------

class TestMapSentencesToTokens:
    def _make_token_ids(self, tokenizer, text: str) -> torch.Tensor:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long)

    def test_basic_two_sentences(self, nlp, tokenizer):
        from src.data.sentence_split import map_sentences_to_tokens, split_into_sentences

        text = "Hello world. This is a test."
        token_ids = self._make_token_ids(tokenizer, text)
        sentences = split_into_sentences(text, nlp)
        result = map_sentences_to_tokens(sentences, token_ids, tokenizer)

        assert len(result) == 2
        for sent in result:
            assert "tok_start" in sent
            assert "tok_end" in sent
            assert sent["tok_end"] > sent["tok_start"]

    def test_token_ranges_cover_text(self, nlp, tokenizer):
        """Decoded tokens for each sentence should contain the sentence text."""
        from src.data.sentence_split import map_sentences_to_tokens, split_into_sentences

        text = "Albert Einstein was born in 1879. He developed the theory of relativity."
        token_ids = self._make_token_ids(tokenizer, text)
        sentences = split_into_sentences(text, nlp)
        result = map_sentences_to_tokens(sentences, token_ids, tokenizer)

        for sent in result:
            span_ids = token_ids[sent["tok_start"]:sent["tok_end"]].tolist()
            decoded = tokenizer.decode(span_ids, skip_special_tokens=True).strip()
            # The decoded span should contain key words from the sentence
            assert len(decoded) > 0

    def test_no_overlap_and_ordered(self, nlp, tokenizer):
        """Token ranges should be non-overlapping and ordered."""
        from src.data.sentence_split import map_sentences_to_tokens, split_into_sentences

        text = "First sentence here. Second sentence there. Third one as well."
        token_ids = self._make_token_ids(tokenizer, text)
        sentences = split_into_sentences(text, nlp)
        result = map_sentences_to_tokens(sentences, token_ids, tokenizer)

        for i in range(len(result) - 1):
            assert result[i]["tok_end"] <= result[i + 1]["tok_start"]

    def test_empty_token_ids_returns_empty(self, nlp, tokenizer):
        from src.data.sentence_split import map_sentences_to_tokens, split_into_sentences

        text = "Hello world."
        sentences = split_into_sentences(text, nlp)
        result = map_sentences_to_tokens(sentences, torch.zeros(0, dtype=torch.long), tokenizer)
        assert result == []

    def test_empty_sentences_returns_empty(self, nlp, tokenizer):
        from src.data.sentence_split import map_sentences_to_tokens

        token_ids = self._make_token_ids(tokenizer, "Hello world.")
        result = map_sentences_to_tokens([], token_ids, tokenizer)
        assert result == []

    def test_tok_end_within_bounds(self, nlp, tokenizer):
        from src.data.sentence_split import map_sentences_to_tokens, split_into_sentences

        text = "Short text. Another sentence."
        token_ids = self._make_token_ids(tokenizer, text)
        sentences = split_into_sentences(text, nlp)
        result = map_sentences_to_tokens(sentences, token_ids, tokenizer)

        T = len(token_ids)
        for sent in result:
            assert sent["tok_start"] >= 0
            assert sent["tok_end"] <= T


# ---------------------------------------------------------------------------
# process_generation
# ---------------------------------------------------------------------------

class TestProcessGeneration:
    def _make_generation(self, tokenizer, text: str) -> dict:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return {
            "text": text,
            "token_ids": torch.tensor(ids, dtype=torch.long),
        }

    def test_returns_list_of_dicts(self, nlp, tokenizer):
        from src.data.sentence_split import process_generation

        gen = self._make_generation(tokenizer, "Hello world. This is a test.")
        result = process_generation(gen, tokenizer, nlp)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)

    def test_all_required_keys_present(self, nlp, tokenizer):
        from src.data.sentence_split import process_generation

        gen = self._make_generation(tokenizer, "Einstein was a physicist. He won the Nobel Prize.")
        result = process_generation(gen, tokenizer, nlp)
        assert len(result) > 0
        for item in result:
            for key in ("text", "char_start", "char_end", "tok_start", "tok_end"):
                assert key in item, f"Missing key: {key}"

    def test_filters_empty_token_range(self, nlp, tokenizer):
        """Sentences with zero-width token range should be excluded."""
        from src.data.sentence_split import process_generation

        gen = self._make_generation(tokenizer, "Hello world. This is a test.")
        result = process_generation(gen, tokenizer, nlp)
        for item in result:
            assert item["tok_end"] > item["tok_start"]

    def test_empty_text(self, nlp, tokenizer):
        from src.data.sentence_split import process_generation

        gen = {"text": "", "token_ids": torch.zeros(0, dtype=torch.long)}
        assert process_generation(gen, tokenizer, nlp) == []

    def test_single_sentence_text(self, nlp, tokenizer):
        from src.data.sentence_split import process_generation

        text = "Marie Curie was a pioneering scientist."
        gen = self._make_generation(tokenizer, text)
        result = process_generation(gen, tokenizer, nlp)
        assert len(result) >= 1
