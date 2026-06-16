"""POS tags aligned to the Llama BPE-token axis (Phase 11-0).

spaCy assigns part-of-speech tags to *word* tokens, but every feature in
this project lives on the LLM's **BPE** token axis (``hidden_states`` /
``entropy`` / ``top1`` are all ``(T, ...)`` with ``T = len(token_ids)``).
To ask "are content tokens less saturated than function tokens?" we first
need a UPOS tag *per BPE token*.

This module maps spaCy word POS onto the BPE axis using the **same**
character-offset machinery that :mod:`src.data.sentence_split` uses to map
sentence boundaries onto tokens, so the alignment is identical to the one
that produced each sentence's ``token_range``:

1. spaCy parses the decoded ``text`` -> per word ``(char_start, char_end, pos_)``.
2. Each BPE token gets a character span via the existing offset helpers
   (:func:`~src.data.sentence_split._offsets_via_reencoding`, falling back to
   :func:`~src.data.sentence_split._offsets_via_incremental_decode`).
3. Each BPE token inherits the UPOS of the spaCy word that *contains its
   first non-whitespace character* (mirrors
   :func:`~src.data.sentence_split._assign_tokens_to_sentences`). A BPE token
   split from one word (e.g. ``"prof" + "essor"``) therefore inherits that
   word's POS.
4. Whitespace-only / unmatched BPE tokens become ``"SPACE"`` (a function tag).

The one reusable export is :func:`compute_token_pos`; the later content-mask
pooling phase imports it unchanged. Plain-ASCII only.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from src.data.sentence_split import (
    _first_non_whitespace,
    _offsets_via_incremental_decode,
    _offsets_via_reencoding,
    _token_ids_to_list,
)

__all__ = [
    "DEFAULT_CONTENT_SET",
    "SPACE_TAG",
    "compute_token_pos",
    "is_content",
]

#: UPOS tags treated as "content" by default (Phase 11-0). Parameterised so the
#: diagnostic can re-run for narrower sets (``{"PROPN", "NUM"}``) or wider ones
#: (``{"PROPN", "NUM", "NOUN", "ADJ", "VERB"}``).
DEFAULT_CONTENT_SET = ("PROPN", "NUM", "NOUN", "ADJ")

#: Tag assigned to whitespace-only / unmatched BPE tokens. Never a content tag.
SPACE_TAG = "SPACE"


def is_content(pos: str, content_set: Sequence[str] = DEFAULT_CONTENT_SET) -> bool:
    """Return ``True`` iff ``pos`` is a content UPOS.

    Parameters
    ----------
    pos : str
        A UPOS tag (e.g. ``"PROPN"``, ``"DET"``, ``"SPACE"``).
    content_set : sequence of str, optional
        The set of tags counted as content. Defaults to
        :data:`DEFAULT_CONTENT_SET`. Accepts any iterable of tags so callers
        may pass a ``set``, ``tuple``, or ``list``.

    Returns
    -------
    bool
        ``pos in content_set``.
    """
    return pos in set(content_set)


def _char_to_pos(text: str, nlp: Any) -> List[Optional[str]]:
    """Build a per-character UPOS lookup over ``text``.

    Returns a list of length ``len(text)`` where entry ``c`` is the UPOS of the
    spaCy word covering character ``c`` (via ``token.idx`` .. ``token.idx +
    len(token.text)``), or ``None`` for characters in no word span (inter-word
    whitespace). This mirrors the "first non-whitespace char looks up its
    container" rule of
    :func:`~src.data.sentence_split._assign_tokens_to_sentences`, but the
    container here is a spaCy *word* rather than a sentence.

    Parameters
    ----------
    text : str
        The decoded generation text.
    nlp : spacy.Language
        A loaded spaCy pipeline with a POS tagger (see
        :func:`src.data.sentence_split.load_spacy_model`).
    """
    char_pos: List[Optional[str]] = [None] * len(text)
    doc = nlp(text)
    n = len(text)
    for tok in doc:
        start = int(tok.idx)
        end = start + len(tok.text)
        if end > n:
            end = n
        pos = tok.pos_
        for c in range(start, end):
            char_pos[c] = pos
    return char_pos


def compute_token_pos(
    text: str,
    token_ids: Any,
    tokenizer: Any,
    nlp: Any,
) -> List[str]:
    """Return a per-token UPOS tag aligned to the ``token_ids`` axis (length ``T``).

    Steps (mirror :mod:`src.data.sentence_split`):

    1. spaCy-parse ``text``; collect ``(char_start, char_end, pos_)`` per word
       and fold them into a per-character lookup (:func:`_char_to_pos`).
    2. Compute a ``(char_start, char_end)`` span per BPE token via the existing
       offset helpers (re-encoding fast path, incremental-decode fallback).
    3. Assign each BPE token the UPOS of the spaCy word containing its first
       **non-whitespace** character. A subword piece inherits its word's POS.
    4. Whitespace-only / unmatched BPE tokens -> :data:`SPACE_TAG`.

    Parameters
    ----------
    text : str
        Decoded generation text. For records produced by
        :mod:`src.data.generation`, ``text == tokenizer.decode(token_ids,
        skip_special_tokens=True)``, so the offsets and the spaCy parse share a
        coordinate system.
    token_ids : LongTensor ``(T,)`` | sequence of int
        The generated token ids whose POS we want.
    tokenizer : transformers tokenizer
        Same tokenizer used at generation time (needed for correct offsets).
    nlp : spacy.Language
        Loaded spaCy pipeline (POS tagger enabled).

    Returns
    -------
    list of str
        One UPOS tag per token, length ``T == len(token_ids)``. Always returns
        valid strings (never ``None``); unmatched tokens are :data:`SPACE_TAG`.
    """
    id_list = _token_ids_to_list(token_ids)
    T = len(id_list)
    if T == 0:
        return []
    if not text:
        return [SPACE_TAG] * T

    char_pos = _char_to_pos(text, nlp)
    n_chars = len(text)

    # Per-BPE-token char spans (same fast/slow paths as sentence splitting).
    offsets = _offsets_via_reencoding(text, id_list, tokenizer)
    if offsets is None:
        offsets = _offsets_via_incremental_decode(id_list, tokenizer)

    out: List[str] = []
    for c_start, c_end in offsets:
        cs = max(0, min(int(c_start), n_chars))
        ce = max(0, min(int(c_end), n_chars))
        # Whitespace-only or empty span -> function ("SPACE").
        if ce <= cs or not text[cs:ce].strip():
            out.append(SPACE_TAG)
            continue
        anchor = _first_non_whitespace(text, cs, ce)
        pos = char_pos[anchor] if 0 <= anchor < n_chars else None
        out.append(pos if pos is not None else SPACE_TAG)

    # ``offsets`` always has length T (re-encoding path is length-checked;
    # incremental decode yields one span per id), so ``out`` is length T.
    return out
