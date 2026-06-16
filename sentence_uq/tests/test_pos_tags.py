"""Tests for Phase 11-0 POS alignment + saturation diagnostic (read-only).

- :func:`compute_token_pos` aligns spaCy word POS onto the Llama BPE token axis
  using the same offset machinery as sentence splitting. We exercise it with a
  real fast tokenizer (``gpt2``) and a fake tokenizer that forces the
  whitespace -> ``"SPACE"`` branch.
- :func:`diagnose_saturation_by_pos` is checked on a tiny synthetic batch:
  gates equal ``pi*(1-pi)``, ``Epi`` values are non-negative, and ``|C_j|==0``
  sentences are excluded from the mask metrics without raising.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.saturation_diag import diagnose_saturation_by_pos  # noqa: E402
from src.data.pos_tags import (  # noqa: E402
    SPACE_TAG,
    compute_token_pos,
    is_content,
)
from src.data.sentence_split import load_spacy_model  # noqa: E402
from src.features.extractor import SentenceUQParams  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nlp() -> Any:
    """Load en_core_web_sm once for the whole module."""
    try:
        return load_spacy_model("en")
    except Exception as exc:  # pragma: no cover - env-specific
        pytest.skip(f"spaCy en_core_web_sm not available: {exc}")


@pytest.fixture(scope="module")
def gpt2_tokenizer() -> Any:
    """A real fast tokenizer with the same offset API as Llama / Mistral."""
    try:
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover
        pytest.skip("transformers not available")
    try:
        return AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"could not load gpt2 tokenizer: {exc}")


# ---------------------------------------------------------------------------
# compute_token_pos — alignment with a real tokenizer
# ---------------------------------------------------------------------------


def _char_to_pos(text: str, nlp: Any) -> List[Any]:
    char_pos: List[Any] = [None] * len(text)
    for tok in nlp(text):
        for c in range(tok.idx, min(tok.idx + len(tok.text), len(text))):
            char_pos[c] = tok.pos_
    return char_pos


def test_compute_token_pos_length_and_alignment(nlp: Any, gpt2_tokenizer: Any) -> None:
    text = "Vovin was a professor at UC Davis."
    enc = gpt2_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]
    token_ids = torch.tensor(ids, dtype=torch.long)

    pos = compute_token_pos(text, token_ids, gpt2_tokenizer, nlp)

    # (1) length equals T
    assert len(pos) == len(ids)
    assert all(isinstance(p, str) for p in pos)

    # (2) every token matches an independent char->pos lookup at its anchor
    char_pos = _char_to_pos(text, nlp)
    for i, (a, b) in enumerate(offs):
        seg = text[a:b]
        if not seg.strip():
            assert pos[i] == SPACE_TAG
            continue
        anchor = next(c for c in range(a, b) if not text[c].isspace())
        expected = char_pos[anchor]
        assert pos[i] == (expected if expected is not None else SPACE_TAG)


def test_compute_token_pos_hand_checked_words(nlp: Any, gpt2_tokenizer: Any) -> None:
    """PROPN/NOUN/DET/ADP/PUNCT land on the right BPE tokens; subwords inherit."""
    text = "Vovin was a professor at UC Davis."
    enc = gpt2_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    pos = compute_token_pos(text, torch.tensor(enc["input_ids"]), gpt2_tokenizer, nlp)
    offs = enc["offset_mapping"]

    def tags_for_word(word_start: int, word_end: int) -> List[str]:
        """All token tags whose first non-ws char lies in [word_start, word_end)."""
        out = []
        for i, (a, b) in enumerate(offs):
            if not text[a:b].strip():
                continue
            anchor = next(c for c in range(a, b) if not text[c].isspace())
            if word_start <= anchor < word_end:
                out.append(pos[i])
        return out

    assert tags_for_word(0, 5) == ["PROPN"] * len(tags_for_word(0, 5))  # Vovin
    assert all(t == "PROPN" for t in tags_for_word(0, 5)) and tags_for_word(0, 5)
    assert all(t == "DET" for t in tags_for_word(10, 11))    # a
    # professor: may be one BPE token or split into pieces; ALL must be NOUN.
    prof_tags = tags_for_word(12, 21)
    assert prof_tags and all(t == "NOUN" for t in prof_tags)
    assert all(t == "ADP" for t in tags_for_word(22, 24))    # at
    assert all(t == "PUNCT" for t in tags_for_word(33, 34))  # .


# ---------------------------------------------------------------------------
# compute_token_pos — whitespace -> SPACE (fake offset tokenizer)
# ---------------------------------------------------------------------------


class _FakeOffsetTokenizer:
    """Fast-style tokenizer returning a fixed (ids, offset_mapping) for ``text``."""

    def __init__(self, ids: List[int], offsets: List[tuple]) -> None:
        self._ids = ids
        self._offsets = offsets

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
    ) -> dict:
        out = {"input_ids": list(self._ids)}
        if return_offsets_mapping:
            out["offset_mapping"] = list(self._offsets)
        return out


def test_compute_token_pos_whitespace_is_space(nlp: Any) -> None:
    # "A  B": token 1 spans the two spaces -> must be SPACE.
    text = "A  B"
    ids = [1, 2, 3]
    offsets = [(0, 1), (1, 3), (3, 4)]
    tok = _FakeOffsetTokenizer(ids, offsets)

    pos = compute_token_pos(text, torch.tensor(ids), tok, nlp)
    assert len(pos) == 3
    assert pos[1] == SPACE_TAG
    assert pos[0] != SPACE_TAG and pos[2] != SPACE_TAG


def test_compute_token_pos_empty() -> None:
    class _Dummy:
        def __call__(self, *a: Any, **k: Any) -> dict:
            return {"input_ids": []}

    assert compute_token_pos("anything", torch.zeros(0, dtype=torch.long), _Dummy(), None) == []


def test_is_content_default_and_custom() -> None:
    assert is_content("PROPN")
    assert is_content("NOUN")
    assert not is_content("DET")
    assert not is_content(SPACE_TAG)
    assert is_content("VERB", {"PROPN", "VERB"})
    assert not is_content("NOUN", {"PROPN", "NUM"})


# ---------------------------------------------------------------------------
# diagnose_saturation_by_pos — synthetic batch
# ---------------------------------------------------------------------------


def _record(source_id: str, T: int, hidden_dim: int, num_layers: int) -> dict:
    return {
        "dataset": "factscore_bio",
        "source_id": source_id,
        "token_range": (0, T),
        "K_j": 1,
        "m_j": 2,
        "hidden_states": torch.randn(T, num_layers, hidden_dim, dtype=torch.float16),
        "entropy": torch.rand(T, dtype=torch.float32),
        "top1": torch.rand(T, dtype=torch.float32),
    }


def test_diagnose_gate_equals_pi_times_one_minus_pi(nlp: Any) -> None:
    """Single deterministic token: by_class mean_gate must equal pi*(1-pi)."""
    params = SentenceUQParams(hidden_dim=1, num_layers=1, projection_dim=1)
    with torch.no_grad():
        params.W.weight.copy_(torch.tensor([[2.0]]))
        params.alpha.zero_()  # softmax over 1 layer -> weight 1.0
    params.eval()

    rec = {
        "dataset": "factscore_bio",
        "source_id": "ent",
        "token_range": (0, 1),
        "K_j": 1,
        "m_j": 1,
        "hidden_states": torch.tensor([[[3.0]]], dtype=torch.float16),  # (1,1,1)
        "entropy": torch.tensor([0.5], dtype=torch.float32),
        "top1": torch.tensor([0.9], dtype=torch.float32),
    }
    # z = [W*h_agg, ent, top1] = [2*3, 0.5, 0.9] = [6.0, 0.5, 0.9], k=3
    theta = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32)  # logit = 0.6
    Sigma = torch.eye(3, dtype=torch.float32)
    pos_by_source = {("factscore_bio", "ent"): ["NOUN"]}

    res = diagnose_saturation_by_pos(
        [rec], params, theta, Sigma, pos_by_source, sat_threshold=0.05,
    )
    pi = float(torch.sigmoid(torch.tensor(0.6)))
    expected_gate = pi * (1.0 - pi)
    assert res["by_class"]["content"]["mean_gate"] == pytest.approx(
        expected_gate, abs=1e-4
    )
    # Single content token, Epi values finite and non-negative.
    s = res["per_sentence"][0]
    assert s["Epi_unif"] >= 0.0 and s["Epi_mask"] >= 0.0
    assert s["n_content"] == 1


def test_diagnose_excludes_zero_content_sentences(nlp: Any) -> None:
    hidden_dim, num_layers = 3, 2
    params = SentenceUQParams(
        hidden_dim=hidden_dim, num_layers=num_layers, projection_dim=2
    )  # k = 4
    params.eval()
    k = params.feature_dim
    theta = torch.zeros(k, dtype=torch.float32)  # pi = 0.5, gate = 0.25 everywhere
    Sigma = torch.eye(k, dtype=torch.float32)

    rec_a = _record("A", 4, hidden_dim, num_layers)  # has content
    rec_b = _record("B", 3, hidden_dim, num_layers)  # no content
    pos_by_source = {
        ("factscore_bio", "A"): ["NOUN", "DET", "PROPN", "ADP"],
        ("factscore_bio", "B"): ["DET", "ADP", "PUNCT"],
    }

    res = diagnose_saturation_by_pos(
        [rec_a, rec_b], params, theta, Sigma, pos_by_source, sat_threshold=0.05,
    )

    # theta = 0 -> every gate is 0.25; nothing is saturated.
    assert res["by_class"]["content"]["mean_gate"] == pytest.approx(0.25, abs=1e-5)
    assert res["by_class"]["function"]["mean_gate"] == pytest.approx(0.25, abs=1e-5)
    assert res["overall_sat_fraction"] == pytest.approx(0.0, abs=1e-9)

    # All Epi values non-negative.
    for s in res["per_sentence"]:
        assert s["Epi_unif"] >= 0.0

    # Record B (|C_j| == 0) is excluded from the mask metrics, not crashed on.
    by_id = {s["source_id"]: s for s in res["per_sentence"]}
    assert by_id["B"]["n_content"] == 0
    assert by_id["B"]["Epi_mask"] is None
    assert by_id["B"]["ratio"] is None
    assert by_id["B"]["mu_mask"] is None
    assert by_id["A"]["n_content"] == 2
    assert by_id["A"]["Epi_mask"] is not None

    # Degeneracy guard counts B; ratio stats only over the one valid sentence.
    assert res["degeneracy"]["n_C0"] == 1
    assert res["epi"]["ratio"]["n"] == 1
