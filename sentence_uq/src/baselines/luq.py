"""LUQ (Long-text Uncertainty Quantification) baseline.

Phase 5-1 baseline #3, following Zhang et al. (2024). Unlike semantic
entropy (prompt-level), LUQ produces a *per-sentence* uncertainty
score for the reference response by checking how often each sentence
is entailed by independent re-samples from the same prompt:

1. Generate ``m`` sampled responses ``Y = {y^{(1)}, …, y^{(m)}}``
   at ``temperature > 0`` for prompt ``x``.
2. For every sentence ``s_j`` in the reference response, compute the
   NLI entailment probability that *each* sample ``y^{(k)}`` supports
   ``s_j``::

       c_j^{(k)} = p_NLI(entailment | premise = y^{(k)}, hypothesis = s_j).

   The reference response is treated as a separate "anchor" answer;
   the consistency score averages over all ``m`` samples
   (and skips ``k`` where the sample is empty)::

       consistency_j = (1 / m) Σ_k c_j^{(k)},
       U(s_j)        = 1 - consistency_j.

A higher ``U`` ⇔ the claim has weak support across re-samples ⇒ more
uncertain (and empirically more likely to be non-factual).

Implementation notes
--------------------
* Shares :class:`NLIScorer` with the semantic-entropy baseline so the
  Phase 5-1 runner can keep a *single* DeBERTa model loaded — the
  Phase 5-1 spec explicitly requires this.
* The sample-generation step is factored into
  :func:`generate_luq_samples` (a thin re-export of the semantic
  entropy sampler) so callers can cache samples on disk; the
  per-sentence step then operates on cached strings only.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch

from src.baselines.semantic_entropy import (
    NLIScorer,
    generate_semantic_samples,
)


__all__ = [
    "generate_luq_samples",
    "compute_luq_for_sentences",
    "compute_luq",
]


def generate_luq_samples(
    prompt: str,
    model: Any,
    tokenizer: Any,
    *,
    num_samples: int = 10,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 256,
) -> List[str]:
    """Generate ``num_samples`` LUQ samples for ``prompt`` (alias).

    Re-exposes :func:`src.baselines.semantic_entropy.generate_semantic_samples`
    so the LUQ runner doesn't have to depend on the SE module directly.
    Keeping the two samplers identical lets the Phase 5-1 runner re-use
    a single cache on disk for both baselines.
    """
    return generate_semantic_samples(
        prompt,
        model,
        tokenizer,
        num_samples=num_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )


def _consistency_scores(
    sentence: str,
    samples: Sequence[str],
    nli_scorer: NLIScorer,
) -> float:
    """Mean entailment probability ``(1/m) Σ_k p_NLI(entail | y^(k), s_j)``.

    Empty samples (whitespace-only) are skipped from the average to
    avoid passing degenerate premises through the NLI model.
    """
    usable = [s for s in samples if s and s.strip()]
    if not usable:
        return float("nan")
    premises = list(usable)
    hypotheses = [sentence] * len(usable)
    probs = nli_scorer.entailment_prob(premises, hypotheses).cpu()
    return float(probs.to(torch.float32).mean().item())


def compute_luq_for_sentences(
    sentences: Sequence[str],
    samples: Sequence[str],
    nli_scorer: NLIScorer,
) -> List[float]:
    """Per-sentence ``U(s_j) = 1 - consistency_j`` given cached samples.

    Parameters
    ----------
    sentences : sequence of str
        Sentences from the *reference* response (Phase 1-2 output).
    samples : sequence of str
        ``m`` independently-sampled responses for the same prompt
        (Phase 5-1 cache).
    nli_scorer : :class:`NLIScorer`

    Returns
    -------
    list[float] of the same length as ``sentences``.
        Each entry is in ``[0, 1]`` (``nan`` when no sample was usable).
    """
    out: List[float] = []
    for sent in sentences:
        if not sent or not sent.strip():
            out.append(float("nan"))
            continue
        c = _consistency_scores(sent, samples, nli_scorer)
        if c != c:  # NaN
            out.append(float("nan"))
        else:
            out.append(float(1.0 - c))
    return out


def compute_luq(
    prompt: str,
    model: Any,
    tokenizer: Any,
    nli_model: NLIScorer,
    num_samples: int = 10,
    *,
    sentences: Optional[Sequence[str]] = None,
    samples: Optional[Sequence[str]] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 256,
) -> List[float]:
    """End-to-end LUQ scoring for one prompt.

    Either ``sentences`` (the reference response split into sentences,
    e.g. via Phase 1-2) must be passed, or the caller must supply a
    pre-generated sample list. Typical usage from the Phase 5-1
    runner is to pass *both* ``sentences`` and cached ``samples`` so no
    LLM call happens at scoring time::

        scores = compute_luq(prompt, None, None, nli, sentences=sents,
                             samples=cached_samples)

    Parameters
    ----------
    prompt : str
    model, tokenizer : HuggingFace LM + tokenizer (only required when
        ``samples`` is None).
    nli_model : :class:`NLIScorer`
        Loaded once globally (Phase 5-1 spec).
    num_samples : int, optional
        Defaults to 10 (spec).
    sentences : sequence of str, optional
        Reference-response sentences to score. If omitted, the
        function falls back to scoring the entire prompt response as a
        single "sentence", which mirrors LUQ's prompt-level summary.
    samples : sequence of str, optional
        If provided, skip on-the-fly sample generation.
    temperature, top_p, max_new_tokens : sampling hyperparameters.

    Returns
    -------
    list[float]
        ``U(s_j)`` for each entry in ``sentences``. When ``sentences``
        is None, the list has length one.
    """
    if samples is None:
        if model is None or tokenizer is None:
            raise ValueError(
                "compute_luq needs either `samples` or (`model`, `tokenizer`)"
            )
        samples = generate_luq_samples(
            prompt,
            model,
            tokenizer,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

    if sentences is None:
        sentences = [prompt]

    return compute_luq_for_sentences(list(sentences), list(samples), nli_model)
