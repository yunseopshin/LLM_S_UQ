# Phase 1-2 — Sentence Splitting + Token Mapping

Implement `src/data/sentence_split.py`.

**Purpose**:
Split generated text into sentences and map each sentence to its corresponding token index range.

**Background**:
- We have token_ids (T,) from generation and the decoded text
- spaCy splits text into sentences with char_start/char_end
- Goal: find (tok_start, tok_end) range for each sentence

**Requirements**:

1. Function `load_spacy_model(lang="en")`:
   - Return spacy.load("en_core_web_sm")
   - Auto-install if missing

2. Function `split_into_sentences(text, nlp)`:
   - Input: text (str), nlp (spacy model)
   - Returns: list of dict {"text": str, "char_start": int, "char_end": int}

3. Function `map_sentences_to_tokens(sentences, token_ids, tokenizer)`:
   - Input: sentence list, token_ids (T,) LongTensor, tokenizer
   - Compute (tok_start, tok_end) range [tok_start, tok_end) for each sentence
   - Method: use return_offsets_mapping=True to get per-token (char_start, char_end),
     then match against sentence char ranges.
   - Fallback: if re-encoding produces different length than token_ids,
     decode tokens one-by-one and track char positions.
   - Returns: list of (tok_start, tok_end) tuples, in sentence order

4. Function `process_generation(generation_result, tokenizer, nlp)`:
   - High-level wrapper over split + map
   - Returns: dict with "sentences": [{"text", "char_start", "char_end", "token_range"}, ...]
   - Filter out invalid sentences (empty token ranges)

**Important**:
- Llama tokenizer is BPE-based — token boundaries ≠ word boundaries
- Handle whitespace, newlines, special characters carefully
- If a sentence boundary falls mid-subword (rare), assign the token to the preceding sentence

**Tests `tests/test_sentence_split.py`**:
- Simple text "Hello world. This is a test." → 2 sentences
- Token ranges actually correspond to correct tokens (decode and compare)
- Edge cases: empty text, single sentence
