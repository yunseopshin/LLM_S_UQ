"""
Phase 1-2: Sentence splitting and token-range mapping.

Splits generated text into sentences (via spaCy) and finds the token index
range [tok_start, tok_end) in the generation's token_ids for each sentence.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Dict, List, Optional

import torch
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# 1. spaCy loader
# ---------------------------------------------------------------------------

def load_spacy_model(lang: str = "en"):
    """Load the spaCy model for the given language, auto-installing if absent.

    Args:
        lang: Language code; only "en" (→ en_core_web_sm) is currently supported.

    Returns:
        A loaded spaCy Language object.
    """
    model_name = "en_core_web_sm" if lang == "en" else f"{lang}_core_web_sm"
    try:
        import spacy
        return spacy.load(model_name)
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
        )
        import spacy
        return spacy.load(model_name)


# ---------------------------------------------------------------------------
# 2. Sentence splitting
# ---------------------------------------------------------------------------

def split_into_sentences(text: str, nlp) -> List[Dict]:
    """Split text into sentences using spaCy.

    Args:
        text: Raw text string.
        nlp: Loaded spaCy Language object.

    Returns:
        List of dicts, each with:
            "text"       : str  — sentence text
            "char_start" : int  — start char offset in *text*
            "char_end"   : int  — end char offset in *text* (exclusive)
    """
    if not text.strip():
        return []

    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        # Locate the stripped sentence within the original span
        raw = sent.text
        strip_offset = len(raw) - len(raw.lstrip())
        char_start = sent.start_char + strip_offset
        char_end = char_start + len(sent_text)
        sentences.append(
            {"text": sent_text, "char_start": char_start, "char_end": char_end}
        )
    return sentences


# ---------------------------------------------------------------------------
# 3. Token-range mapping
# ---------------------------------------------------------------------------

def map_sentences_to_tokens(
    sentences: List[Dict],
    token_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> List[Dict]:
    """Assign each sentence a half-open token range [tok_start, tok_end).

    Strategy:
        1. Try to get char→token mapping via tokenizer re-encoding with
           return_offsets_mapping=True (fast tokenizers support this).
        2. Fallback: decode token_ids one-by-one, tracking character position
           manually.  Used when the re-encoded length differs from the
           original token_ids length.

    Args:
        sentences: Output of split_into_sentences — list of dicts with
                   "text", "char_start", "char_end".
        token_ids: LongTensor (T,) — generated token ids from generation.py.
        tokenizer: Matching tokenizer.

    Returns:
        Same list of dicts, each augmented with:
            "tok_start" : int — inclusive token index
            "tok_end"   : int — exclusive token index  (tok_end > tok_start)
        Sentences for which no token can be found are dropped.
    """
    if len(sentences) == 0 or len(token_ids) == 0:
        return []

    token_ids_list = token_ids.tolist()

    # ---- Method 1: offset_mapping via re-encoding -------------------------
    full_text = tokenizer.decode(token_ids_list, skip_special_tokens=True)

    try:
        enc = tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = enc["offset_mapping"]  # list of (char_start, char_end) per token

        if len(offsets) == len(token_ids_list):
            return _map_with_offsets(sentences, offsets)

        # Length mismatch — fall through to fallback
    except Exception:
        pass  # Fast tokenizer not available; use fallback

    # ---- Method 2: manual decode fallback ---------------------------------
    offsets = _build_offsets_by_decode(token_ids_list, tokenizer)
    return _map_with_offsets(sentences, offsets)


def _map_with_offsets(
    sentences: List[Dict],
    offsets: List[tuple],
) -> List[Dict]:
    """Map sentence char ranges onto token indices using a precomputed offset list.

    Args:
        sentences: Dicts with "char_start" / "char_end".
        offsets: List of (char_start, char_end) for each token (0-indexed).

    Returns:
        Filtered list of sentence dicts with "tok_start" and "tok_end" added.
    """
    results = []
    for sent in sentences:
        s_char = sent["char_start"]
        e_char = sent["char_end"]

        tok_start = None
        tok_end = None
        for i, (t_s, t_e) in enumerate(offsets):
            # Token overlaps with sentence char range
            if t_e > s_char and t_s < e_char:
                if tok_start is None:
                    tok_start = i
                tok_end = i + 1

        if tok_start is None or tok_end is None or tok_end <= tok_start:
            continue  # No covering token found; skip sentence

        results.append({**sent, "tok_start": tok_start, "tok_end": tok_end})
    return results


def _build_offsets_by_decode(
    token_ids: List[int],
    tokenizer: PreTrainedTokenizer,
) -> List[tuple]:
    """Build per-token (char_start, char_end) offsets by decoding tokens one-by-one.

    Decodes greedily: each token is decoded in the context of all preceding
    tokens so that byte-pair merges are handled correctly.  The character
    position of token i is taken as the length increase of the cumulative
    decoded string.

    Args:
        token_ids: List of integer token ids.
        tokenizer: Matching tokenizer.

    Returns:
        List of (char_start, char_end) tuples, one per token.
    """
    offsets = []
    prev_len = 0
    for i in range(len(token_ids)):
        decoded_so_far = tokenizer.decode(token_ids[: i + 1], skip_special_tokens=True)
        cur_len = len(decoded_so_far)
        offsets.append((prev_len, cur_len))
        prev_len = cur_len
    return offsets


# ---------------------------------------------------------------------------
# 4. High-level pipeline
# ---------------------------------------------------------------------------

def process_generation(
    generation_result: Dict,
    tokenizer: PreTrainedTokenizer,
    nlp,
    min_tokens: int = 1,
    min_chars: int = 3,
) -> List[Dict]:
    """Full pipeline: text → sentences → token-mapped, filtered sentence list.

    Args:
        generation_result: Dict returned by generation.generate_with_hidden_states,
                           must contain "text" (str) and "token_ids" (LongTensor).
        tokenizer: Matching tokenizer.
        nlp: Loaded spaCy model.
        min_tokens: Minimum number of tokens a sentence must cover (inclusive).
        min_chars: Minimum character length of sentence text.

    Returns:
        List of sentence dicts:
            "text"       : str
            "char_start" : int
            "char_end"   : int
            "tok_start"  : int  — index into generation_result["token_ids"]
            "tok_end"    : int  — exclusive
    """
    text = generation_result.get("text", "")
    token_ids = generation_result.get("token_ids", torch.zeros(0, dtype=torch.long))

    sentences = split_into_sentences(text, nlp)
    mapped = map_sentences_to_tokens(sentences, token_ids, tokenizer)

    # Filter: too few tokens or too short text
    filtered = [
        s for s in mapped
        if (s["tok_end"] - s["tok_start"]) >= min_tokens
        and len(s["text"]) >= min_chars
    ]
    return filtered
