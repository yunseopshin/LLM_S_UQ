"""Semantic Entropy baseline (Kuhn et al., 2023).

Phase 5-1 baseline #2. The implementation follows the reference at
https://github.com/lorenzkuhn/semantic_uncertainty:

1. Draw ``m`` samples ``y^{(1)}, …, y^{(m)}`` from ``p(y | x)`` at
   temperature ``> 0``.
2. Cluster the samples by *bidirectional* NLI entailment with a
   pre-trained DeBERTa-MNLI model (``microsoft/deberta-large-mnli``).
   Two responses live in the same cluster iff each entails the other.
3. The semantic entropy is the Shannon entropy of the empirical
   cluster distribution::

       p_c = |C_c| / m,    SE = - Σ_c p_c · log p_c.

The score is the same for every sentence in the prompt — semantic
entropy is a prompt-level uncertainty signal. The Phase 5-1 runner
broadcasts the score to every sentence of the corresponding prompt.

Implementation notes
--------------------
* The heavy generation step is factored into
  :func:`generate_semantic_samples` so the Phase 5-1 runner can cache
  samples on disk (CLAUDE.md guideline 2 — "precompute expensive
  steps offline").
* The NLI model is loaded once and reused via :class:`NLIScorer`;
  ``scripts/05_baselines.py`` shares a single instance across the
  semantic-entropy and LUQ baselines (Phase 5-1 spec — "NLI model:
  load once globally (singleton pattern)").
* All arithmetic runs in fp32 even when the NLI / LM weights live in
  half precision (CLAUDE.md rule 10).
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


__all__ = [
    "DEFAULT_NLI_MODEL",
    "NLIScorer",
    "generate_semantic_samples",
    "cluster_by_entailment",
    "compute_semantic_entropy_from_samples",
    "compute_semantic_entropy",
]


#: Default NLI checkpoint (Phase 5-1 spec).
DEFAULT_NLI_MODEL: str = "microsoft/deberta-large-mnli"


# ---------------------------------------------------------------------------
# NLI scoring (shared with LUQ)
# ---------------------------------------------------------------------------


class NLIScorer:
    """Thin wrapper around a HuggingFace MNLI classifier.

    The DeBERTa-MNLI label space is ``["contradiction", "neutral",
    "entailment"]``. This class exposes :meth:`entailment_prob` and
    :meth:`predict_label` so callers don't have to remember the label
    indices.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model id. Defaults to :data:`DEFAULT_NLI_MODEL`.
    device : str, optional
        ``"cuda"`` falls back to ``"cpu"`` when CUDA is unavailable.
    dtype : torch.dtype, optional
        Weight dtype. fp16 is fine on GPU; fp32 is safer on CPU.

    Notes
    -----
    Designed as a singleton — instantiate once per process and pass
    around. The Phase 5-1 spec is explicit on this point.
    """

    LABEL_NAMES: tuple[str, str, str] = ("contradiction", "neutral", "entailment")
    ENTAILMENT_INDEX: int = 2

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        torch_dtype = dtype if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )
        self.model.eval().to(self.device)

        id2label = getattr(self.model.config, "id2label", None) or {}
        # microsoft/deberta-large-mnli reports CONTRADICTION/NEUTRAL/ENTAILMENT
        # but the exact casing can differ between checkpoints; locate the
        # entailment index dynamically and fall back to 2 if unknown.
        ent_idx: Optional[int] = None
        for idx, label in id2label.items():
            if isinstance(label, str) and "entail" in label.lower():
                try:
                    ent_idx = int(idx)
                except (TypeError, ValueError):
                    continue
                break
        self.entailment_index = ent_idx if ent_idx is not None else self.ENTAILMENT_INDEX

    @torch.no_grad()
    def _forward(self, premises: Sequence[str], hypotheses: Sequence[str]) -> Tensor:
        if len(premises) != len(hypotheses):
            raise ValueError(
                "premises and hypotheses must have equal length; got "
                f"{len(premises)} vs {len(hypotheses)}"
            )
        if len(premises) == 0:
            return torch.zeros((0, 3), dtype=torch.float32, device=self.device)
        enc = self.tokenizer(
            list(premises),
            list(hypotheses),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits
        return F.softmax(logits.to(torch.float32), dim=-1)

    def entailment_prob(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> Tensor:
        """Return per-pair entailment probability ``p(entail | premise, hyp)``.

        Parameters
        ----------
        premises, hypotheses : sequences of equal length.

        Returns
        -------
        Tensor of shape ``(N,)`` on the scorer's device.
        """
        probs = self._forward(premises, hypotheses)
        return probs[:, self.entailment_index]

    def predict_label(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> List[int]:
        """Return argmax label index per pair."""
        probs = self._forward(premises, hypotheses)
        return probs.argmax(dim=-1).cpu().tolist()


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


def generate_semantic_samples(
    prompt: str,
    model: Any,
    tokenizer: Any,
    *,
    num_samples: int = 10,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 256,
) -> List[str]:
    """Sample ``num_samples`` decoded continuations of ``prompt``.

    Parameters
    ----------
    prompt : str
        Raw user prompt. Chat templates are applied opportunistically
        when ``tokenizer.apply_chat_template`` is available.
    model, tokenizer
        HuggingFace ``AutoModelForCausalLM`` and its tokenizer.
    num_samples : int
        ``m`` in the math above. Must be ``>= 1``.
    temperature, top_p : float
        Sampling hyperparameters. The semantic-entropy paper uses
        ``temperature = 1.0`` and no nucleus filtering.
    max_new_tokens : int

    Returns
    -------
    list[str]
        Decoded continuations (prompt stripped).
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1; got {num_samples}")
    if temperature <= 0.0:
        raise ValueError(
            f"semantic entropy requires temperature > 0; got {temperature}"
        )

    apply_chat = getattr(tokenizer, "apply_chat_template", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    if apply_chat is not None and chat_template:
        prompt_text = apply_chat(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = prompt

    enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = int(enc["input_ids"].shape[1])

    samples: List[str] = []
    with torch.no_grad():
        for _ in range(num_samples):
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=int(max_new_tokens),
                pad_token_id=getattr(tokenizer, "pad_token_id", None)
                or getattr(tokenizer, "eos_token_id", None),
            )
            new_ids = out[0, prompt_len:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            samples.append(text.strip())
    return samples


# ---------------------------------------------------------------------------
# Clustering + entropy
# ---------------------------------------------------------------------------


def cluster_by_entailment(
    samples: Sequence[str],
    nli_scorer: NLIScorer,
    *,
    threshold: float = 0.5,
) -> List[int]:
    """Group ``samples`` into semantic equivalence classes via bidirectional NLI.

    Two responses ``y_a``, ``y_b`` are in the same cluster iff
    ``argmax NLI(y_a -> y_b) == ENTAILMENT`` *and* the converse holds —
    or, when ``threshold > 0``, when both entailment probabilities
    exceed ``threshold``. The greedy single-pass procedure follows the
    reference implementation (Kuhn et al. 2023, Algorithm 1).

    Parameters
    ----------
    samples : sequence of str (length ``m``).
    nli_scorer : :class:`NLIScorer`.
    threshold : float, optional
        If ``> 0``, use a probability threshold on entailment;
        otherwise use the model's argmax label.

    Returns
    -------
    list[int] of length ``m``
        ``cluster_ids[i]`` is the integer cluster id assigned to
        ``samples[i]`` (ids are dense, starting at 0).
    """
    n = len(samples)
    if n == 0:
        return []
    cluster_ids: List[int] = [-1] * n
    representatives: List[int] = []

    for i in range(n):
        if cluster_ids[i] != -1:
            continue
        cluster_idx = len(representatives)
        cluster_ids[i] = cluster_idx
        representatives.append(i)

        rest = [j for j in range(i + 1, n) if cluster_ids[j] == -1]
        if not rest:
            continue

        premises = [samples[i]] * len(rest) + [samples[j] for j in rest]
        hypotheses = [samples[j] for j in rest] + [samples[i]] * len(rest)

        if threshold > 0.0:
            probs = nli_scorer.entailment_prob(premises, hypotheses).cpu()
            fwd = probs[: len(rest)]
            bwd = probs[len(rest):]
            agree = (fwd >= threshold) & (bwd >= threshold)
        else:
            labels = nli_scorer.predict_label(premises, hypotheses)
            fwd = torch.tensor(labels[: len(rest)])
            bwd = torch.tensor(labels[len(rest):])
            agree = (fwd == nli_scorer.entailment_index) & (
                bwd == nli_scorer.entailment_index
            )

        for k, j in enumerate(rest):
            if bool(agree[k].item()):
                cluster_ids[j] = cluster_idx

    return cluster_ids


def compute_semantic_entropy_from_samples(
    samples: Sequence[str],
    nli_scorer: NLIScorer,
    *,
    threshold: float = 0.5,
) -> float:
    """Discrete semantic entropy ``SE = -Σ p_c log p_c`` over NLI clusters.

    Parameters
    ----------
    samples : sequence of str
        Pre-generated continuations of a single prompt (``m`` samples).
    nli_scorer : :class:`NLIScorer`
    threshold : float, optional
        Forwarded to :func:`cluster_by_entailment`.

    Returns
    -------
    float
        Shannon entropy in nats. Range ``[0, log m]``; higher → more
        semantic disagreement → more uncertain.
    """
    if len(samples) == 0:
        return 0.0
    cluster_ids = cluster_by_entailment(
        samples, nli_scorer, threshold=threshold
    )
    counts: dict[int, int] = {}
    for cid in cluster_ids:
        counts[cid] = counts.get(cid, 0) + 1
    m = len(cluster_ids)
    se = 0.0
    for c in counts.values():
        p = c / m
        if p > 0.0:
            se -= p * math.log(p)
    return float(se)


def compute_semantic_entropy(
    prompt: str,
    model: Any,
    tokenizer: Any,
    nli_model: NLIScorer,
    num_samples: int = 10,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 256,
    threshold: float = 0.5,
) -> float:
    """End-to-end semantic entropy for ``prompt``.

    Convenience wrapper that combines :func:`generate_semantic_samples`
    and :func:`compute_semantic_entropy_from_samples`. Prefer the
    two-step interface when batching across many prompts so the
    expensive generation phase can be cached separately
    (CLAUDE.md guideline 2).

    Parameters
    ----------
    prompt : str
    model, tokenizer : HuggingFace LM + tokenizer.
    nli_model : :class:`NLIScorer`
        Loaded once globally (Phase 5-1 spec).
    num_samples : int, optional
        Defaults to 10 (spec).
    temperature, top_p, max_new_tokens : sampling hyperparameters.
    threshold : float, optional
        NLI entailment-probability threshold used for clustering.

    Returns
    -------
    float
        Shannon entropy over the semantic clusters.
    """
    samples = generate_semantic_samples(
        prompt,
        model,
        tokenizer,
        num_samples=num_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    return compute_semantic_entropy_from_samples(
        samples, nli_model, threshold=threshold
    )
