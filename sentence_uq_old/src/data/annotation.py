"""
Phase 1-4: Factuality annotation for generated sentences.

Default approach: LLM-as-judge using the Anthropic Claude API (option B).
A stub for factscore (option A) is also provided.

Prompt injection mitigation: entity/sentence are passed as separate variables
inside the user turn, never interpolated into the system prompt.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Wikipedia context retrieval
# ---------------------------------------------------------------------------

def retrieve_wikipedia_context(entity: str, max_chars: int = 3000) -> Optional[str]:
    """Retrieve a Wikipedia summary for the given entity.

    Args:
        entity: Name of the entity (e.g. "Albert Einstein").
        max_chars: Maximum number of characters to return.

    Returns:
        Truncated Wikipedia summary string, or None on error.
    """
    try:
        import wikipediaapi  # type: ignore
        wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="LLM-S-UQ/1.0 (research project)",
        )
        page = wiki.page(entity)
        if not page.exists():
            return None
        return page.summary[:max_chars]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 2. LLM-as-judge (Option B — default)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a fact-checker. Given an entity and a sentence from a biography, "
    "determine if the sentence is factually correct. "
    "Answer with a single word: \"SUPPORTED\" or \"NOT_SUPPORTED\"."
)


def annotate_sentence_with_llm_judge(
    entity: str,
    sentence: str,
    api_client: Any,
    wikipedia_context: Optional[str] = None,
) -> Optional[int]:
    """Annotate a single sentence as factually supported (1) or not (0).

    Calls the Anthropic Claude API (claude-sonnet-4-6) with temperature=0.
    Falls back gracefully if the response cannot be parsed.

    Args:
        entity: The subject entity of the biography.
        sentence: A single sentence to fact-check.
        api_client: An initialised ``anthropic.Anthropic`` client.
        wikipedia_context: Optional reference text to provide to the judge.

    Returns:
        1  — sentence is SUPPORTED
        0  — sentence is NOT_SUPPORTED
        None — API error or unparseable response
    """
    user_parts = [
        f"Entity: {entity}",
        f"Sentence: {sentence}",
    ]
    if wikipedia_context:
        user_parts.append(f"Reference: {wikipedia_context}")

    user_content = "\n".join(user_parts)

    try:
        response = api_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            temperature=0,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        answer = response.content[0].text.strip().upper()
        if "NOT_SUPPORTED" in answer:
            return 0
        if "SUPPORTED" in answer:
            return 1
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. FActScore stub (Option A)
# ---------------------------------------------------------------------------

def annotate_sentence_with_factscore(
    entity: str,
    sentence: str,
) -> Optional[int]:
    """Stub for FActScore-based annotation (requires factscore library + OpenAI key).

    Not implemented — raises NotImplementedError.  Left as a placeholder for
    future integration.
    """
    raise NotImplementedError(
        "FActScore annotation is not yet implemented. Use annotate_sentence_with_llm_judge."
    )


# ---------------------------------------------------------------------------
# 4. Batch annotation with resume
# ---------------------------------------------------------------------------

def annotate_batch(
    processed_sentences: List[Dict],
    api_client: Any,
    use_wiki: bool = True,
    resume: bool = True,
    save_path: Optional[str | Path] = None,
    rate_limit_sleep: float = 0.5,
    save_interval: int = 100,
) -> List[Dict]:
    """Annotate a list of sentence dicts, with resume support and periodic saving.

    Each dict in ``processed_sentences`` must have:
        "entity"  : str  — biography subject
        "text"    : str  — sentence text

    Args:
        processed_sentences: List of sentence dicts to annotate.
        api_client: Initialised ``anthropic.Anthropic`` client.
        use_wiki: Whether to fetch Wikipedia context for each entity.
        resume: If True and save_path exists, skip already-annotated entries.
        save_path: JSON file to write intermediate + final results.
        rate_limit_sleep: Seconds to sleep between API calls.
        save_interval: Persist results every N newly annotated sentences.

    Returns:
        List of sentence dicts augmented with:
            "label"  : int | None  — 1 / 0 / None
    """
    save_path = Path(save_path) if save_path else None

    # Load existing annotations for resume
    annotated: Dict[str, Optional[int]] = {}
    if resume and save_path and save_path.exists():
        with open(save_path) as f:
            saved = json.load(f)
        for entry in saved:
            key = _sentence_key(entry)
            if "label" in entry:
                annotated[key] = entry["label"]

    # Wikipedia context cache (one fetch per entity)
    wiki_cache: Dict[str, Optional[str]] = {}

    results: List[Dict] = []
    newly_done = 0

    for sent in processed_sentences:
        key = _sentence_key(sent)

        if key in annotated:
            results.append({**sent, "label": annotated[key]})
            continue

        # Wikipedia context
        ctx: Optional[str] = None
        if use_wiki:
            entity = sent.get("entity", "")
            if entity not in wiki_cache:
                wiki_cache[entity] = retrieve_wikipedia_context(entity)
            ctx = wiki_cache[entity]

        label = annotate_sentence_with_llm_judge(
            entity=sent.get("entity", ""),
            sentence=sent["text"],
            api_client=api_client,
            wikipedia_context=ctx,
        )

        entry = {**sent, "label": label}
        results.append(entry)
        annotated[key] = label
        newly_done += 1

        time.sleep(rate_limit_sleep)

        if save_path and newly_done % save_interval == 0:
            _save_results(results, save_path)

    if save_path:
        _save_results(results, save_path)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentence_key(sent: Dict) -> str:
    """Stable key for deduplication / resume."""
    return f"{sent.get('entity', '')}|||{sent.get('text', '')}"


def _save_results(results: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
