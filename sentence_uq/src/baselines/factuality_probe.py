"""Han et al. (2025) factuality probe — direct comparison baseline.

Phase 5-1 baseline #5. Reference code:
https://github.com/JThh/fact-probe.

The Phase 5-1 spec calls out two controlled variants:

(a) **Original** (faithful to Han et al.) — re-encode every atomic
    claim through the LLM and read off the *last-token* hidden state
    of a single layer (default ``layer 14``). Train an L1-regularised
    logistic regression on the resulting ``(h_c, y_c)`` pairs, where
    ``y_c`` is the per-claim factuality label saved by Phase 1-4. At
    inference, sentence-level scores are the **mean** of the
    claim-level predicted probabilities.

(b) **Adapted** (ablation) — use the generation-time hidden states
    cached in Phase 1-1 (no re-encoding). Take the last token of
    each sentence at the same target layer and train L1-LR on
    ``(h_j, A_j)`` with ``A_j = 1{K_j = m_j}``.

Both variants are point estimates: there is no posterior covariance
or epistemic / aleatoric split. Their job is to anchor the
comparison along the calibration axis (ECE, AUROC).

Han et al. report ``AUROC = 0.7357`` for the original variant on the
Llama-3.1-8B in-domain split — kept as a numeric sanity target for
the Phase 6 evaluation.

L1 logistic regression
----------------------
We use ``sklearn.linear_model.LogisticRegression(penalty="l1",
solver="liblinear")`` exactly as the Phase 5-1 spec dictates. The
inverse regularisation strength ``C`` defaults to ``1.0`` but is
exposed so the runner can grid-search it on the validation set.

Selected-layer mapping
----------------------
The cached generation tensors only store a subset of layers
(``model_config["selected_layers"]``). When ``target_layer`` is not
among them we fall back to the *closest* selected layer and log the
substitution — there is no reasonable way to re-derive a missing
layer offline.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


__all__ = [
    "DEFAULT_TARGET_LAYER",
    "FactualityProbeBaseline",
    "extract_adapted_features",
    "extract_original_features",
    "pick_layer_index",
]


#: Han et al. (2025) report results around layer 14 for Llama-3-8B.
DEFAULT_TARGET_LAYER: int = 14


# ---------------------------------------------------------------------------
# Layer mapping
# ---------------------------------------------------------------------------


def pick_layer_index(
    target_layer: int, selected_layers: Sequence[int]
) -> int:
    """Return the index *into* ``selected_layers`` whose value is closest to ``target_layer``.

    Phase 1-1 stores hidden states for an arbitrary subset of layers
    (``selected_layers``). The hidden-state tensor's middle axis is
    therefore indexed by the *position in* ``selected_layers`` — not
    by the absolute layer index. This helper handles the translation.

    Parameters
    ----------
    target_layer : int
        Absolute layer index requested by the user (Han et al. default
        14).
    selected_layers : sequence of int
        The list saved alongside the generation tensors.

    Returns
    -------
    int
        The position ``i`` such that ``selected_layers[i]`` minimises
        ``|selected_layers[i] - target_layer|``. Ties prefer the
        smaller absolute layer.
    """
    if not selected_layers:
        raise ValueError("selected_layers must be non-empty")
    best_pos = 0
    best_diff = abs(int(selected_layers[0]) - int(target_layer))
    for i, layer in enumerate(selected_layers):
        diff = abs(int(layer) - int(target_layer))
        if diff < best_diff:
            best_diff = diff
            best_pos = i
    return best_pos


# ---------------------------------------------------------------------------
# Feature extraction — adapted (generation-time)
# ---------------------------------------------------------------------------


def extract_adapted_features(
    hidden_states: Tensor,
    token_range: Tuple[int, int],
    layer_index: int,
) -> Tensor:
    """Last-token hidden state of a sentence at a single layer.

    Parameters
    ----------
    hidden_states : Tensor of shape ``(T, num_selected_layers, hidden_dim)``.
        Generation-time hidden states (Phase 1-1).
    token_range : tuple ``(start, end)`` — half-open.
        Token span of the sentence in the generated response.
    layer_index : int
        Position in the *selected-layers* axis (use
        :func:`pick_layer_index` to translate from an absolute layer).

    Returns
    -------
    Tensor of shape ``(hidden_dim,)`` in fp32 on CPU.
    """
    start, end = int(token_range[0]), int(token_range[1])
    if end <= start:
        raise ValueError(
            f"token_range covers zero tokens: ({start}, {end})"
        )
    T, L_lay, _ = hidden_states.shape
    if end > T:
        raise ValueError(
            f"token_range end={end} exceeds sequence length T={T}"
        )
    if not (0 <= layer_index < L_lay):
        raise ValueError(
            f"layer_index={layer_index} out of range [0, {L_lay})"
        )
    last_token = end - 1
    return hidden_states[last_token, layer_index].to(torch.float32).cpu()


# ---------------------------------------------------------------------------
# Feature extraction — original (re-encode)
# ---------------------------------------------------------------------------


def extract_original_features(
    claim_texts: Sequence[str],
    model: Any,
    tokenizer: Any,
    target_layer: int = DEFAULT_TARGET_LAYER,
    *,
    batch_size: int = 8,
) -> Tensor:
    """Re-encode each claim through the LLM and read off layer-``target_layer`` last-token states.

    Mirrors the Han et al. (2025) Stage-2 procedure: feed each atomic
    claim (already produced by Phase 1-4) to the LLM under
    teacher-forcing and pull the hidden state at ``target_layer`` for
    the *last* token of the claim.

    The model is expected to expose
    ``output_hidden_states=True``; the loader in
    :func:`src.data.generation.load_model` sets this automatically.

    Parameters
    ----------
    claim_texts : sequence of str (length ``C``).
    model, tokenizer : HuggingFace causal LM + tokenizer.
    target_layer : int, optional
        Absolute layer index. The HF hidden-states tuple has length
        ``num_hidden_layers + 1`` (index 0 = embedding); this argument
        indexes directly into it. Defaults to
        :data:`DEFAULT_TARGET_LAYER`.
    batch_size : int, optional
        Per-batch claim count. The DeBERTa-MNLI baselines run in their
        own process, so we keep this conservative.

    Returns
    -------
    Tensor of shape ``(C, hidden_dim)`` in fp32 on CPU.
    """
    if not claim_texts:
        return torch.zeros((0, 0), dtype=torch.float32)

    num_hidden = int(getattr(model.config, "num_hidden_layers", 0)) + 1
    if not (0 <= target_layer < num_hidden):
        raise ValueError(
            f"target_layer={target_layer} out of range [0, {num_hidden}) "
            f"for this model"
        )

    device = next(model.parameters()).device
    pad_id = (
        getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None)
    )
    if tokenizer.pad_token is None and pad_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    out_rows: List[Tensor] = []
    for batch_start in range(0, len(claim_texts), batch_size):
        batch = list(claim_texts[batch_start : batch_start + batch_size])
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden = out.hidden_states[target_layer]              # (B, L, D)
        # Last *non-pad* token index per row.
        if "attention_mask" in enc:
            lengths = enc["attention_mask"].sum(dim=1) - 1
        else:
            lengths = torch.full(
                (hidden.shape[0],),
                hidden.shape[1] - 1,
                dtype=torch.long,
                device=hidden.device,
            )
        idx = lengths.clamp(min=0).unsqueeze(1).unsqueeze(2)
        idx = idx.expand(-1, 1, hidden.shape[-1])             # (B, 1, D)
        last = hidden.gather(dim=1, index=idx).squeeze(1)     # (B, D)
        out_rows.append(last.to(torch.float32).cpu())

    return torch.cat(out_rows, dim=0)


# ---------------------------------------------------------------------------
# The baseline
# ---------------------------------------------------------------------------


class FactualityProbeBaseline:
    """Han et al. (2025) L1-logistic-regression factuality probe.

    Parameters
    ----------
    variant : {"original", "adapted"}, optional
        Defaults to ``"original"`` (the published Han et al. setup).
    target_layer : int, optional
        Single layer to read the hidden state from (Han et al. default
        14; see :data:`DEFAULT_TARGET_LAYER`).
    C : float, optional
        Inverse L1 regularisation strength (sklearn default 1.0).
    max_iter : int, optional
        Forwarded to sklearn ``liblinear``. Default 1000.
    random_state : int, optional
        Forwarded to sklearn.

    Aggregation
    -----------
    The original variant produces per-claim probabilities. The
    sentence-level score is the **mean** of the per-claim
    probabilities (Han et al. report mean and min; mean is the
    default in the reference repo and tends to calibrate better).
    """

    def __init__(
        self,
        variant: str = "original",
        target_layer: int = DEFAULT_TARGET_LAYER,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 0,
    ) -> None:
        if variant not in ("original", "adapted"):
            raise ValueError(
                f"variant must be 'original' or 'adapted'; got {variant!r}"
            )
        self.variant = variant
        self.target_layer = int(target_layer)
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._clf: Any = None
        self._feature_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Adapted variant
    # ------------------------------------------------------------------

    def build_adapted_dataset(
        self,
        sentence_records: Iterable[Dict[str, Any]],
        selected_layers: Sequence[int],
    ) -> Dict[str, Tensor]:
        """Per-sentence ``(H, A, m)`` for the adapted variant.

        ``A_j = 1{K_j = m_j}``; sentences with ``m_j == 0`` are
        dropped (CLAUDE.md rule 8).
        """
        layer_idx = pick_layer_index(self.target_layer, selected_layers)
        H_rows: List[Tensor] = []
        A_list: List[int] = []
        m_list: List[int] = []
        for rec in sentence_records:
            m_j = int(rec.get("m_j", 0) or 0)
            if m_j == 0:
                continue
            tr = (int(rec["token_range"][0]), int(rec["token_range"][1]))
            h = extract_adapted_features(
                hidden_states=rec["hidden_states"],
                token_range=tr,
                layer_index=layer_idx,
            )
            H_rows.append(h)
            A_list.append(1 if int(rec.get("K_j", 0)) == m_j else 0)
            m_list.append(m_j)
        if not H_rows:
            return {
                "H": torch.zeros((0, 0), dtype=torch.float32),
                "A": torch.zeros((0,), dtype=torch.long),
                "m": torch.zeros((0,), dtype=torch.long),
            }
        return {
            "H": torch.stack(H_rows, dim=0),
            "A": torch.tensor(A_list, dtype=torch.long),
            "m": torch.tensor(m_list, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    # Original variant
    # ------------------------------------------------------------------

    def build_original_dataset(
        self,
        sentence_records: Iterable[Dict[str, Any]],
        model: Any,
        tokenizer: Any,
    ) -> Dict[str, Any]:
        """Per-claim ``(H, y)`` plus a ``sentence_to_claims`` index.

        Each input record is expected to expose ``claims`` (the
        per-sentence list saved by Phase 1-4, each entry having
        ``text`` and ``label``). Records without claims (``m_j == 0``)
        are skipped.

        Returns
        -------
        dict with keys::

            "H":                  Tensor (C, hidden_dim)  per-claim features
            "y":                  LongTensor (C,)         per-claim labels
            "sentence_to_claims": list[tuple[int, int]]   half-open ranges into H
            "sentence_records":   list[dict]              the kept sentence rows
        """
        ordered_records: List[Dict[str, Any]] = []
        claim_ranges: List[Tuple[int, int]] = []
        flat_texts: List[str] = []
        flat_labels: List[int] = []

        cursor = 0
        for rec in sentence_records:
            claims = rec.get("claims") or []
            if not claims:
                continue
            start = cursor
            for c in claims:
                text = str(c.get("text", "") or "").strip()
                if not text:
                    continue
                flat_texts.append(text)
                flat_labels.append(int(c.get("label", 0) or 0))
                cursor += 1
            end = cursor
            if end == start:
                continue
            ordered_records.append(rec)
            claim_ranges.append((start, end))

        if not flat_texts:
            return {
                "H": torch.zeros((0, 0), dtype=torch.float32),
                "y": torch.zeros((0,), dtype=torch.long),
                "sentence_to_claims": [],
                "sentence_records": [],
            }

        H = extract_original_features(
            flat_texts, model, tokenizer, target_layer=self.target_layer
        )
        y = torch.tensor(flat_labels, dtype=torch.long)
        return {
            "H": H,
            "y": y,
            "sentence_to_claims": claim_ranges,
            "sentence_records": ordered_records,
        }

    # ------------------------------------------------------------------
    # Common fit / predict
    # ------------------------------------------------------------------

    def fit(self, H: Tensor, y: Tensor) -> "FactualityProbeBaseline":
        """Fit the L1-regularised logistic regression.

        Parameters
        ----------
        H : Tensor of shape ``(N, hidden_dim)``.
        y : Tensor of shape ``(N,)`` of {0, 1} labels.

        Returns
        -------
        self
        """
        from sklearn.linear_model import LogisticRegression

        if H.dim() != 2:
            raise ValueError(
                f"H must be 2-D; got shape {tuple(H.shape)}"
            )
        if y.shape != (H.shape[0],):
            raise ValueError(
                f"y must have shape (N={H.shape[0]},); got {tuple(y.shape)}"
            )

        H_np = H.detach().cpu().to(torch.float32).numpy()
        y_np = y.detach().cpu().to(torch.int64).numpy()
        if len(np.unique(y_np)) < 2:
            raise ValueError(
                "training labels are degenerate (only one class present); "
                "L1 logistic regression cannot be fit"
            )

        self._clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._clf.fit(H_np, y_np)
        self._feature_dim = int(H_np.shape[1])
        return self

    def predict_proba(self, H: Tensor) -> Tensor:
        """Return ``p(y = 1 | h)`` per row of ``H``."""
        if self._clf is None or self._feature_dim is None:
            raise RuntimeError("FactualityProbeBaseline.fit must be called first")
        if not torch.is_tensor(H):
            raise TypeError("H must be a torch.Tensor")
        if H.dim() == 1:
            H = H.unsqueeze(0)
        if H.shape[1] != self._feature_dim:
            raise ValueError(
                f"feature_dim mismatch: model fit on {self._feature_dim}; "
                f"predict received {H.shape[1]}"
            )
        H_np = H.detach().cpu().to(torch.float32).numpy()
        classes = list(self._clf.classes_)
        pos_idx = classes.index(1) if 1 in classes else len(classes) - 1
        probs = self._clf.predict_proba(H_np)[:, pos_idx]
        return torch.from_numpy(probs.astype(np.float32))

    # ------------------------------------------------------------------
    # Sentence-level aggregation (original variant only)
    # ------------------------------------------------------------------

    def aggregate_sentence_scores(
        self,
        claim_probs: Tensor,
        sentence_to_claims: Sequence[Tuple[int, int]],
        agg: str = "mean",
    ) -> Tensor:
        """Aggregate per-claim probabilities into one score per sentence.

        Parameters
        ----------
        claim_probs : Tensor of shape ``(C,)``.
            Per-claim ``p(y=1)`` from :meth:`predict_proba`.
        sentence_to_claims : sequence of half-open ranges into
            ``claim_probs``. Element ``j`` is ``(start, end)``.
        agg : {"mean", "min", "geomean"}, optional
            ``"mean"`` (default — matches the reference repo),
            ``"min"`` (most conservative), or ``"geomean"`` (product
            of per-claim probabilities, the "all-supported" heuristic).

        Returns
        -------
        Tensor of shape ``(J,)`` in fp32.
        """
        if agg not in ("mean", "min", "geomean"):
            raise ValueError(
                f"agg must be 'mean' | 'min' | 'geomean'; got {agg!r}"
            )
        probs = claim_probs.detach().cpu().to(torch.float32)
        out = torch.empty(
            len(sentence_to_claims), dtype=torch.float32
        )
        for j, (start, end) in enumerate(sentence_to_claims):
            if end <= start:
                out[j] = float("nan")
                continue
            window = probs[start:end]
            if agg == "mean":
                out[j] = float(window.mean().item())
            elif agg == "min":
                out[j] = float(window.min().item())
            else:  # geomean
                # Stable computation: exp(mean(log p)).
                eps = 1e-12
                out[j] = float(
                    torch.exp(torch.log(window.clamp_min(eps)).mean()).item()
                )
        return out
