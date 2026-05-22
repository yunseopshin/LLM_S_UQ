"""Tests for ``src.data.annotation`` — Phase 1-4.

Uses a fully mocked auxiliary-LM client (no network calls). Wikipedia
retrieval is exercised by monkey-patching the HTTP helper so the test
suite stays offline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data import annotation  # noqa: E402
from src.data.annotation import (  # noqa: E402
    RateLimiter,
    annotate_batch,
    annotate_record,
    annotate_sentence,
    decompose_to_atomic_facts,
    is_meaningful_sentence,
    judge_atomic_fact,
    retrieve_knowledge,
)


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------


class _MockClient:
    """Recordable, scripted ApiClient stand-in for tests.

    Routes prompts by simple substring match → output. The handler keys are
    checked in insertion order; the first match wins. Unrecognised prompts
    raise so a missing fixture cannot quietly skip a code path.
    """

    def __init__(self, routes: list[tuple[str, str]]) -> None:
        self.routes = routes
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        for needle, reply in self.routes:
            if needle in prompt:
                return reply
        raise AssertionError(f"No mock route matched prompt:\n{prompt[:400]}")


# ---------------------------------------------------------------------------
# Sentence filtering
# ---------------------------------------------------------------------------


def test_is_meaningful_sentence_filters_short_and_boilerplate() -> None:
    assert is_meaningful_sentence("Albert Einstein was born in 1879.")
    assert not is_meaningful_sentence("")
    assert not is_meaningful_sentence("Sure!")
    assert not is_meaningful_sentence("ok.")
    assert not is_meaningful_sentence("Here are some facts about him:")
    assert not is_meaningful_sentence("???")
    # Three short but factual words pass.
    assert is_meaningful_sentence("Einstein won Nobel.")


# ---------------------------------------------------------------------------
# Decomposition + revision + subjectivity filter
# ---------------------------------------------------------------------------


def test_decompose_runs_full_pipeline() -> None:
    client = _MockClient(
        [
            (
                "Break the SENTENCE",
                "- He won the Nobel Prize in 1921.\n- He developed relativity.\n",
            ),
            ("Rewrite the FACT", "REVISED: Einstein won the Nobel Prize in 1921."),
            (
                "OBJECTIVE factual claim",
                "ANSWER: OBJECTIVE",
            ),
        ]
    )
    facts = decompose_to_atomic_facts(
        "He won the Nobel Prize in 1921 and developed relativity.",
        "Albert Einstein",
        client,
        response_context="Albert Einstein. He won the Nobel Prize in 1921.",
    )
    # Both facts pass revision (mock revises every fact to the same string,
    # so dedupe should leave a single survivor) and subjectivity filter.
    assert len(facts) == 1
    assert facts[0] == "Einstein won the Nobel Prize in 1921."

    # The decomposition prompt + 2 revisions + 1 dedup'd subjectivity check.
    kinds = [c["prompt"].split("\n", 1)[0] for c in client.calls]
    assert kinds[0].startswith("Break the SENTENCE")
    assert all(c["temperature"] == 0.0 for c in client.calls)


def test_decompose_drops_subjective_claims() -> None:
    client = _MockClient(
        [
            (
                "Break the SENTENCE",
                "- Einstein was a kind man.\n- Einstein was a German-born physicist.\n",
            ),
            ("Rewrite the FACT", "REVISED: kept-as-is"),
            ("Einstein was a kind man", "ANSWER: SUBJECTIVE"),
            ("Einstein was a German-born physicist", "ANSWER: OBJECTIVE"),
        ]
    )
    # Skip revision so the original strings survive into the subjectivity filter.
    facts = decompose_to_atomic_facts(
        "Einstein was a kind man and a German-born physicist.",
        "Albert Einstein",
        client,
        revise=False,
    )
    assert facts == ["Einstein was a German-born physicist."]


def test_decompose_skips_meaningless_sentences() -> None:
    client = _MockClient([])
    assert decompose_to_atomic_facts("Sure!", "Albert Einstein", client) == []
    assert client.calls == []  # no API hit for filtered sentences


def test_decompose_handles_none_output() -> None:
    client = _MockClient([("Break the SENTENCE", "NONE")])
    assert (
        decompose_to_atomic_facts(
            "It was a long time ago in a galaxy far away.",
            "Star Wars",
            client,
        )
        == []
    )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def test_judge_supported() -> None:
    client = _MockClient([("Decide whether the FACT is supported", "ANSWER: SUPPORTED")])
    assert judge_atomic_fact("Einstein won the Nobel.", "context", client) == 1


def test_judge_not_supported() -> None:
    client = _MockClient(
        [("Decide whether the FACT is supported", "ANSWER: NOT_SUPPORTED")]
    )
    assert judge_atomic_fact("Einstein lived on Mars.", "context", client) == 0


def test_judge_empty_context_is_zero() -> None:
    client = _MockClient([])
    assert judge_atomic_fact("any fact", "", client) == 0
    assert client.calls == []


def test_judge_format_fallback() -> None:
    """Robust to LMs that ignore the strict ANSWER: format."""
    client = _MockClient([("Decide whether the FACT", "Yes, this is supported.")])
    assert judge_atomic_fact("fact", "context", client) == 1


# ---------------------------------------------------------------------------
# Retrieval (Wikipedia API stubbed)
# ---------------------------------------------------------------------------


def test_retrieve_factscore_bio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_fetch(title: str, *, timeout: float = 20.0) -> Optional[str]:
        calls.append(title)
        if title == "Albert Einstein":
            return "Albert Einstein was a German-born theoretical physicist."
        return None

    monkeypatch.setattr(annotation, "_wiki_fetch_extract", fake_fetch)

    text = retrieve_knowledge(
        "Albert Einstein", "factscore_bio", cache_dir=tmp_path
    )
    assert "German-born theoretical physicist" in text
    assert calls == ["Albert Einstein"]

    # Second call: served from cache (no new HTTP hit).
    calls.clear()
    text2 = retrieve_knowledge(
        "Albert Einstein", "factscore_bio", cache_dir=tmp_path
    )
    assert text == text2
    assert calls == []


def test_retrieve_longfact_searches_and_concatenates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_search(query: str, *, limit: int = 3, timeout: float = 20.0) -> list[str]:
        return ["Topic A", "Topic B"]

    extracts = {
        "Topic A": "First article body.",
        "Topic B": "Second article body.",
    }

    def fake_fetch(title: str, *, timeout: float = 20.0) -> Optional[str]:
        return extracts.get(title)

    monkeypatch.setattr(annotation, "_wiki_search_titles", fake_search)
    monkeypatch.setattr(annotation, "_wiki_fetch_extract", fake_fetch)

    text = retrieve_knowledge(
        "chemistry",
        "longfact",
        cache_dir=tmp_path,
        extra_query="periodic table",
    )
    assert "Topic A" in text and "First article body." in text
    assert "Topic B" in text and "Second article body." in text


def test_retrieve_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError):
        retrieve_knowledge("x", "unknown_dataset")


# ---------------------------------------------------------------------------
# annotate_sentence & rate limiter wiring
# ---------------------------------------------------------------------------


def test_annotate_sentence_end_to_end() -> None:
    client = _MockClient(
        [
            (
                "Break the SENTENCE",
                "- Einstein was born in 1879.\n- Einstein developed relativity.",
            ),
            (
                "Rewrite the FACT",
                "REVISED: same",
            ),
            ("OBJECTIVE factual claim", "ANSWER: OBJECTIVE"),
            ("Decide whether the FACT", "ANSWER: SUPPORTED"),
        ]
    )
    result = annotate_sentence(
        "Einstein was born in 1879 and developed relativity.",
        "Albert Einstein",
        "factscore_bio",
        client,
        knowledge_context="Stubbed Wikipedia context.",
    )
    # Both atomic facts get revised to the same string → dedup → one survivor.
    assert result["m_j"] == 1
    assert result["K_j"] == 1
    assert result["claims"][0]["label"] == 1
    assert result["sentence"].startswith("Einstein was born")


def test_annotate_sentence_skips_when_filter_rejects() -> None:
    client = _MockClient([])
    assert annotate_sentence(
        "Sure!",
        "Albert Einstein",
        "factscore_bio",
        client,
        knowledge_context="ctx",
    ) == {
        "sentence": "Sure!",
        "m_j": 0,
        "K_j": 0,
        "claims": [],
    }


def test_rate_limiter_invoked_for_every_call(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _MockClient(
        [
            ("Break the SENTENCE", "- f1.\n"),
            ("Rewrite the FACT", "REVISED: f1."),
            ("OBJECTIVE factual claim", "ANSWER: OBJECTIVE"),
            ("Decide whether the FACT", "ANSWER: SUPPORTED"),
        ]
    )
    calls: list[float] = []

    def fake_wait(self: RateLimiter) -> None:
        calls.append(0.0)

    monkeypatch.setattr(RateLimiter, "wait", fake_wait, raising=True)
    rl = RateLimiter(rps=10.0)

    annotate_sentence(
        "Einstein was born in 1879.",
        "Albert Einstein",
        "factscore_bio",
        client,
        knowledge_context="ctx",
        rate_limiter=rl,
    )
    # 1 decompose + 1 revise + 1 subjectivity + 1 judge = 4 calls.
    assert len(calls) == len(client.calls) == 4


# ---------------------------------------------------------------------------
# annotate_record / annotate_batch
# ---------------------------------------------------------------------------


def _make_factscore_record() -> dict[str, Any]:
    return {
        "dataset": "factscore_bio",
        "entity": "Albert Einstein",
        "text": "Einstein was born in 1879. He developed relativity.",
        "sentences": [
            {
                "text": "Einstein was born in 1879.",
                "char_start": 0,
                "char_end": 26,
                "token_range": (0, 5),
            },
            {
                "text": "He developed relativity.",
                "char_start": 27,
                "char_end": 51,
                "token_range": (5, 9),
            },
        ],
    }


def _annotation_routes() -> list[tuple[str, str]]:
    return [
        ("Break the SENTENCE", "- Atom 1.\n"),
        ("Rewrite the FACT", "REVISED: Atom 1."),
        ("OBJECTIVE factual claim", "ANSWER: OBJECTIVE"),
        ("Decide whether the FACT", "ANSWER: SUPPORTED"),
    ]


def test_annotate_record_aggregates_totals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        annotation,
        "retrieve_knowledge",
        lambda *a, **kw: "stubbed ctx",
    )
    client = _MockClient(_annotation_routes())
    ann = annotate_record(_make_factscore_record(), "factscore_bio", client)

    assert ann["total_m"] == 2
    assert ann["total_K"] == 2
    assert len(ann["sentences"]) == 2
    for sent in ann["sentences"]:
        assert sent["m_j"] == 1
        assert sent["K_j"] == 1
        assert sent["claims"][0]["text"] == "Atom 1."
        assert sent["claims"][0]["label"] == 1


def test_annotate_batch_writes_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        annotation,
        "retrieve_knowledge",
        lambda *a, **kw: "stubbed ctx",
    )
    routes = _annotation_routes()
    client = _MockClient(routes)
    records = [_make_factscore_record()]
    out_dir = tmp_path / "factscore_bio"

    res = annotate_batch(
        records,
        dataset_type="factscore_bio",
        api_client=client,
        out_dir=out_dir,
        progress=False,
    )
    assert res["annotated"] == 1
    assert res["skipped"] == 0
    assert (out_dir / "Albert_Einstein.json").exists()
    combined = json.loads((out_dir / "annotated.json").read_text())
    assert isinstance(combined, list) and len(combined) == 1
    assert combined[0]["entity"] == "Albert Einstein"

    # Re-run: existing per-record file should be skipped.
    client2 = _MockClient(routes)
    res2 = annotate_batch(
        records,
        dataset_type="factscore_bio",
        api_client=client2,
        out_dir=out_dir,
        progress=False,
    )
    assert res2["annotated"] == 0
    assert res2["skipped"] == 1
    assert client2.calls == []  # nothing re-annotated


def test_annotate_batch_longfact_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        annotation,
        "retrieve_knowledge",
        lambda *a, **kw: "stubbed ctx",
    )
    client = _MockClient(_annotation_routes())
    record = {
        "dataset": "longfact",
        "topic": "chemistry",
        "prompt": "Explain the periodic table.",
        "prompt_idx": 7,
        "text": "The periodic table organises elements.",
        "sentences": [
            {
                "text": "The periodic table organises elements.",
                "char_start": 0,
                "char_end": 38,
                "token_range": (0, 6),
            }
        ],
    }
    out_dir = tmp_path / "longfact"
    res = annotate_batch(
        [record],
        dataset_type="longfact",
        api_client=client,
        out_dir=out_dir,
        progress=False,
    )
    assert res["annotated"] == 1
    target = out_dir / "chemistry" / "007.json"
    assert target.exists()
    combined = json.loads((out_dir / "annotated.json").read_text())
    assert combined[0]["topic"] == "chemistry"
    assert combined[0]["prompt_idx"] == 7
