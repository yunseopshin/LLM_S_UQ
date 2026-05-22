"""Sentence splitting + token-range mapping for Bayesian sentence-level UQ.

Phase 1-2. Takes a generation record produced by :mod:`src.data.generation`
(text plus the generated ``token_ids``) and returns, for each sentence,

- its surface text and character span in ``text``,
- the half-open token range ``[tok_start, tok_end)`` into ``token_ids``.

Why this matters
----------------
Phase 3 needs, per sentence, the *token indices* that belong to it so that
``μ_j(θ) = (1/L_j) Σ_{ℓ∈s_j} σ(θᵀ z_ℓ)`` is well-defined (CLAUDE.md Core Math).
Hidden states and logits are stored along the same token axis, so
``token_range`` is the bridge between spaCy sentence boundaries and the
per-token features.

Method
------
1. spaCy (``en_core_web_sm`` by default) splits the decoded text into
   sentences, each carrying ``(char_start, char_end)``.
2. To get a *per-token* ``(char_start, char_end)`` we re-encode the text with
   ``return_offsets_mapping=True`` (only fast HuggingFace tokenizers expose
   this). If re-encoding yields the same length as the stored ``token_ids``,
   the offsets are used directly. Otherwise — or if the tokenizer is slow —
   we fall back to incremental decoding: ``decode(token_ids[:i+1])`` grows
   by exactly the ``i``-th token's substring.
3. Each token is then assigned to a sentence by looking up the sentence
   index of its first **non-whitespace** character. This gives:

   - whitespace-only tokens between sentences → preceding sentence;
   - subword tokens that straddle a sentence boundary → preceding sentence
     (per CLAUDE.md "If a sentence boundary falls mid-subword, assign the
     token to the preceding sentence").

Notes
-----
- BPE tokenization (Llama / GPT-2 / etc.) does not respect word boundaries,
  so naive ``text.split()`` would not produce ranges aligned with
  ``token_ids``. We always work in character space.
- Sentences whose final token range is empty (``tok_start == tok_end``) are
  dropped by :func:`process_generation`; downstream code never has to guard
  against ``L_j = 0``.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Optional, Sequence

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# spaCy model loader
# ---------------------------------------------------------------------------


_SPACY_MODEL_NAMES: dict[str, str] = {
    "en": "en_core_web_sm",
}


def load_spacy_model(lang: str = "en") -> Any:
    """Load the spaCy pipeline for ``lang`` (default English), auto-installing if needed.

    Parameters
    ----------
    lang : str
        Two-letter language code. Currently only ``"en"`` is mapped explicitly
        (``en_core_web_sm``); other codes fall back to ``f"{lang}_core_news_sm"``.

    Returns
    -------
    spacy.Language
        Loaded pipeline with the sentencizer / parser enabled.

    Raises
    ------
    RuntimeError
        If the model is missing *and* auto-install via
        ``python -m spacy download <name>`` fails.

    Notes
    -----
    We import ``spacy`` lazily so that importing this module does not require
    spaCy when only :func:`map_sentences_to_tokens` is used.
    """
    import spacy

    model_name = _SPACY_MODEL_NAMES.get(lang, f"{lang}_core_news_sm")
    try:
        return spacy.load(model_name)
    except OSError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name]
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"spaCy model {model_name!r} is not installed and auto-install "
                f"failed: {exc}"
            ) from exc
        return spacy.load(model_name)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------


def split_into_sentences(text: str, nlp: Any) -> list[dict[str, Any]]:
    """Split ``text`` into sentences with character spans.

    Parameters
    ----------
    text : str
        The decoded generation text.
    nlp : spacy.Language
        A loaded spaCy pipeline (see :func:`load_spacy_model`).

    Returns
    -------
    list of dict
        ``[{"text": str, "char_start": int, "char_end": int}, ...]`` in
        document order. Empty / whitespace-only sentences are dropped so
        ``char_end > char_start`` and ``"text".strip()`` is non-empty for
        every returned dict.
    """
    if not text:
        return []

    doc = nlp(text)
    out: list[dict[str, Any]] = []
    for sent in doc.sents:
        s_text = sent.text
        if not s_text or not s_text.strip():
            continue
        out.append(
            {
                "text": s_text,
                "char_start": int(sent.start_char),
                "char_end": int(sent.end_char),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Per-token character offsets
# ---------------------------------------------------------------------------


def _token_ids_to_list(token_ids: Any) -> list[int]:
    """Coerce ``token_ids`` (Tensor / ndarray / list) into a plain ``list[int]``."""
    if isinstance(token_ids, Tensor):
        return token_ids.detach().to("cpu").tolist()
    if hasattr(token_ids, "tolist"):
        return list(token_ids.tolist())
    return [int(t) for t in token_ids]


def _offsets_via_reencoding(
    text: str, token_ids: list[int], tokenizer: Any
) -> Optional[list[tuple[int, int]]]:
    """Try the fast path: re-encode ``text`` and reuse its offset mapping.

    Returns ``None`` if the tokenizer is slow, lacks ``return_offsets_mapping``
    support, or yields an id sequence whose length differs from
    ``len(token_ids)``. Token-id equality is *not* required — only length —
    because tokenizers occasionally re-tokenize edge tokens (e.g. leading
    whitespace) differently when fed decoded text.
    """
    try:
        enc = tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
    except (TypeError, ValueError, NotImplementedError):
        return None

    offsets = enc.get("offset_mapping") if isinstance(enc, dict) else None
    if offsets is None:
        offsets = getattr(enc, "offset_mapping", None)
    re_ids = enc.get("input_ids") if isinstance(enc, dict) else None
    if re_ids is None:
        re_ids = getattr(enc, "input_ids", None)

    if offsets is None or re_ids is None:
        return None
    if len(re_ids) != len(token_ids):
        return None

    return [(int(a), int(b)) for a, b in offsets]


def _offsets_via_incremental_decode(
    token_ids: list[int], tokenizer: Any
) -> list[tuple[int, int]]:
    """Compute per-token char ranges by repeatedly decoding growing prefixes.

    For each ``i``, ``decode(token_ids[:i+1])`` extends ``decode(token_ids[:i])``
    by exactly the substring contributed by token ``i``. We therefore record
    ``(len(prev), len(curr))`` — that pair is the token's char range.

    When decoding normalises an earlier character (very rare with
    ``skip_special_tokens=True`` on standard BPE tokenizers), we walk the
    longest common prefix and use that as the start. Length-difference falls
    out automatically as the token's "added" span.

    Complexity: O(T²) decoding cost; only used when the fast path is
    unavailable or disagrees on length, so the slow path rarely runs.
    """
    offsets: list[tuple[int, int]] = []
    prev = ""
    for i in range(len(token_ids)):
        curr = tokenizer.decode(token_ids[: i + 1], skip_special_tokens=True)
        if curr.startswith(prev):
            offsets.append((len(prev), len(curr)))
        else:
            common = 0
            for j in range(min(len(prev), len(curr))):
                if prev[j] == curr[j]:
                    common = j + 1
                else:
                    break
            offsets.append((common, len(curr)))
        prev = curr
    return offsets


def _first_non_whitespace(text: str, start: int, end: int) -> int:
    """Return the index of the first non-whitespace char in ``text[start:end]``.

    Falls back to ``start`` for whitespace-only ranges (or empty ranges) so the
    caller can still look up a sentence index.
    """
    if end <= start:
        return start
    for k in range(start, end):
        if not text[k].isspace():
            return k
    return start


def _assign_tokens_to_sentences(
    offsets: Sequence[tuple[int, int]],
    sentences: Sequence[dict[str, Any]],
    text: str,
) -> list[int]:
    """For each token, return the index of the sentence it belongs to.

    Rule (per CLAUDE.md): use the first non-whitespace character of the token
    to look up the sentence. If that character falls in no sentence (e.g. the
    token is pure whitespace sitting between sentences), assign to the most
    recent preceding sentence. Tokens preceding every sentence (rare — leading
    whitespace before the first sentence) go to sentence 0.

    Returns a list of length ``len(offsets)``; values lie in
    ``[0, len(sentences))``.
    """
    n = len(sentences)
    if n == 0:
        return [-1] * len(offsets)

    starts = [s["char_start"] for s in sentences]
    ends = [s["char_end"] for s in sentences]

    out: list[int] = []
    for c_start, c_end in offsets:
        anchor = _first_non_whitespace(text, c_start, c_end)
        sent_idx: Optional[int] = None
        for j in range(n):
            if starts[j] <= anchor < ends[j]:
                sent_idx = j
                break
        if sent_idx is None:
            # Anchor falls in a gap. Find the latest sentence ending at or
            # before the anchor — that's the "preceding" sentence.
            for j in range(n - 1, -1, -1):
                if ends[j] <= anchor:
                    sent_idx = j
                    break
            if sent_idx is None:
                sent_idx = 0  # token precedes every sentence
        out.append(sent_idx)
    return out


def map_sentences_to_tokens(
    sentences: Sequence[dict[str, Any]],
    token_ids: Any,
    tokenizer: Any,
) -> list[tuple[int, int]]:
    """Compute the token range ``[tok_start, tok_end)`` for each sentence.

    Parameters
    ----------
    sentences : sequence of dict
        Output of :func:`split_into_sentences` — each dict has ``text``,
        ``char_start``, ``char_end``.
    token_ids : LongTensor (T,) | sequence of int
        The generated token ids from :mod:`src.data.generation`. Only their
        decoded character spans are used here.
    tokenizer
        A HuggingFace tokenizer. Fast tokenizers go through
        :func:`_offsets_via_reencoding`; everything else uses the incremental
        decode fallback.

    Returns
    -------
    list of (int, int)
        One ``(tok_start, tok_end)`` per input sentence, in order. Empty
        sentences (no tokens assigned) return ``(0, 0)`` — the caller
        (:func:`process_generation`) filters them out.

    Notes
    -----
    A token is bound to a sentence by the sentence index of its first
    non-whitespace character; see :func:`_assign_tokens_to_sentences`.
    """
    id_list = _token_ids_to_list(token_ids)
    T = len(id_list)
    n = len(sentences)

    if T == 0 or n == 0:
        return [(0, 0)] * n

    text = tokenizer.decode(id_list, skip_special_tokens=True)

    offsets = _offsets_via_reencoding(text, id_list, tokenizer)
    if offsets is None:
        offsets = _offsets_via_incremental_decode(id_list, tokenizer)

    assignments = _assign_tokens_to_sentences(offsets, sentences, text)

    ranges: list[tuple[int, int]] = [(0, 0)] * n
    starts: list[Optional[int]] = [None] * n
    ends: list[Optional[int]] = [None] * n
    for tok_idx, sent_idx in enumerate(assignments):
        if sent_idx < 0:
            continue
        if starts[sent_idx] is None:
            starts[sent_idx] = tok_idx
        ends[sent_idx] = tok_idx + 1

    for j in range(n):
        if starts[j] is not None and ends[j] is not None:
            ranges[j] = (int(starts[j]), int(ends[j]))
    return ranges


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


def process_generation(
    generation_result: dict[str, Any],
    tokenizer: Any,
    nlp: Any,
) -> dict[str, Any]:
    """Split a generation's text into sentences and attach token ranges.

    Parameters
    ----------
    generation_result : dict
        A record matching the schema produced by
        :func:`src.data.generation.generate_with_hidden_states` /
        :func:`save_generation`. Only ``"text"`` and ``"token_ids"`` are
        consulted.
    tokenizer
        HuggingFace tokenizer (same one used during generation; required for
        correct offsets).
    nlp : spacy.Language
        Loaded spaCy pipeline (see :func:`load_spacy_model`).

    Returns
    -------
    dict
        ``{"sentences": [{"text", "char_start", "char_end", "token_range"}, ...]}``
        where ``token_range`` is a ``(tok_start, tok_end)`` tuple of plain ints.
        Sentences whose token range is empty are filtered out.
    """
    text: str = generation_result.get("text", "") or ""
    token_ids = generation_result.get("token_ids")
    if token_ids is None:
        return {"sentences": []}

    sentences = split_into_sentences(text, nlp)
    if not sentences:
        return {"sentences": []}

    ranges = map_sentences_to_tokens(sentences, token_ids, tokenizer)

    out: list[dict[str, Any]] = []
    for s, (a, b) in zip(sentences, ranges):
        if b > a:
            out.append(
                {
                    "text": s["text"],
                    "char_start": int(s["char_start"]),
                    "char_end": int(s["char_end"]),
                    "token_range": (int(a), int(b)),
                }
            )
    return {"sentences": out}


__all__ = [
    "load_spacy_model",
    "split_into_sentences",
    "map_sentences_to_tokens",
    "process_generation",
]
