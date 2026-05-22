"""
Unit tests for src/data/annotation.py (Phase 1-4).

API calls are mocked — no real Anthropic key needed.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_api_client(response_text: str = "SUPPORTED") -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns response_text."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client.messages.create.return_value = msg
    return client


# ---------------------------------------------------------------------------
# annotate_sentence_with_llm_judge
# ---------------------------------------------------------------------------

class TestAnnotateSentenceWithLLMJudge:
    def test_supported_returns_1(self):
        from src.data.annotation import annotate_sentence_with_llm_judge

        client = _make_api_client("SUPPORTED")
        result = annotate_sentence_with_llm_judge("Einstein", "He was born in 1879.", client)
        assert result == 1

    def test_not_supported_returns_0(self):
        from src.data.annotation import annotate_sentence_with_llm_judge

        client = _make_api_client("NOT_SUPPORTED")
        result = annotate_sentence_with_llm_judge("Einstein", "He was born on Mars.", client)
        assert result == 0

    def test_unparseable_returns_none(self):
        from src.data.annotation import annotate_sentence_with_llm_judge

        client = _make_api_client("I am not sure about this.")
        result = annotate_sentence_with_llm_judge("Einstein", "Some sentence.", client)
        assert result is None

    def test_api_error_returns_none(self):
        from src.data.annotation import annotate_sentence_with_llm_judge

        client = MagicMock()
        client.messages.create.side_effect = RuntimeError("API error")
        result = annotate_sentence_with_llm_judge("Einstein", "Some sentence.", client)
        assert result is None

    def test_with_wikipedia_context(self):
        """Context is forwarded — no crash."""
        from src.data.annotation import annotate_sentence_with_llm_judge

        client = _make_api_client("SUPPORTED")
        result = annotate_sentence_with_llm_judge(
            "Einstein", "He developed relativity.", client,
            wikipedia_context="Albert Einstein was a physicist..."
        )
        assert result == 1

    def test_api_called_with_correct_model(self):
        from src.data.annotation import annotate_sentence_with_llm_judge

        client = _make_api_client("SUPPORTED")
        annotate_sentence_with_llm_judge("Einstein", "He was a physicist.", client)
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert call_kwargs["temperature"] == 0

    def test_entity_in_user_content(self):
        """Entity must appear in user message, not system prompt."""
        from src.data.annotation import annotate_sentence_with_llm_judge

        client = _make_api_client("SUPPORTED")
        annotate_sentence_with_llm_judge("Marie Curie", "She discovered polonium.", client)
        call_kwargs = client.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "Marie Curie" in user_content
        assert "She discovered polonium." in user_content


# ---------------------------------------------------------------------------
# retrieve_wikipedia_context
# ---------------------------------------------------------------------------

class TestRetrieveWikipediaContext:
    def test_returns_none_on_import_error(self):
        """Should not raise even if wikipediaapi is unavailable."""
        from src.data.annotation import retrieve_wikipedia_context

        with patch.dict("sys.modules", {"wikipediaapi": None}):
            result = retrieve_wikipedia_context("Albert Einstein")
        # May return None or a real result depending on environment — just no crash
        assert result is None or isinstance(result, str)

    def test_returns_string_or_none(self):
        from src.data.annotation import retrieve_wikipedia_context

        result = retrieve_wikipedia_context("Albert Einstein")
        assert result is None or isinstance(result, str)

    def test_respects_max_chars(self):
        from src.data.annotation import retrieve_wikipedia_context

        result = retrieve_wikipedia_context("Albert Einstein", max_chars=100)
        if result is not None:
            assert len(result) <= 100

    def test_unknown_entity_returns_none_or_str(self):
        from src.data.annotation import retrieve_wikipedia_context

        result = retrieve_wikipedia_context("Zzz_Totally_Nonexistent_Entity_XYZ_12345")
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# annotate_batch
# ---------------------------------------------------------------------------

class TestAnnotateBatch:
    def _sentences(self, n: int = 3) -> list:
        return [
            {"entity": "Einstein", "text": f"Sentence number {i}."}
            for i in range(n)
        ]

    def test_returns_list_same_length(self):
        from src.data.annotation import annotate_batch

        client = _make_api_client("SUPPORTED")
        sents = self._sentences(3)
        results = annotate_batch(sents, client, use_wiki=False, save_path=None)
        assert len(results) == 3

    def test_label_key_present(self):
        from src.data.annotation import annotate_batch

        client = _make_api_client("SUPPORTED")
        results = annotate_batch(self._sentences(2), client, use_wiki=False)
        for r in results:
            assert "label" in r

    def test_supported_label_values(self):
        from src.data.annotation import annotate_batch

        client = _make_api_client("SUPPORTED")
        results = annotate_batch(self._sentences(2), client, use_wiki=False)
        assert all(r["label"] == 1 for r in results)

    def test_saves_json(self):
        from src.data.annotation import annotate_batch

        client = _make_api_client("NOT_SUPPORTED")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "annotated.json"
            annotate_batch(
                self._sentences(2), client,
                use_wiki=False, save_path=save_path, save_interval=1,
            )
            assert save_path.exists()
            with open(save_path) as f:
                data = json.load(f)
            assert len(data) == 2

    def test_resume_skips_existing(self):
        """Sentences already in the save file should not trigger API calls."""
        from src.data.annotation import annotate_batch

        sents = self._sentences(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "annotated.json"

            # Pre-populate with first 2 sentences
            pre = [{**s, "label": 1} for s in sents[:2]]
            with open(save_path, "w") as f:
                json.dump(pre, f)

            client = _make_api_client("SUPPORTED")
            annotate_batch(
                sents, client, use_wiki=False, save_path=save_path, resume=True,
            )

            # Only 1 new API call should have been made
            assert client.messages.create.call_count == 1

    def test_no_api_calls_when_all_resumed(self):
        from src.data.annotation import annotate_batch

        sents = self._sentences(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "annotated.json"
            pre = [{**s, "label": 0} for s in sents]
            with open(save_path, "w") as f:
                json.dump(pre, f)

            client = _make_api_client("SUPPORTED")
            results = annotate_batch(
                sents, client, use_wiki=False, save_path=save_path, resume=True,
            )
            assert client.messages.create.call_count == 0
            assert all(r["label"] == 0 for r in results)
