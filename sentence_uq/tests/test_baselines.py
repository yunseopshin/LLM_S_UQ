"""Tests for ``src.baselines`` — Phase 5-1 baselines.

Covers the five baselines plus the layer-mapping helper:

- :func:`compute_token_entropy_baseline`: math + slicing + input validation
- :class:`NLIScorer` interaction via a duck-typed fake (no transformers
  download): :func:`cluster_by_entailment`,
  :func:`compute_semantic_entropy_from_samples`, :func:`compute_luq_for_sentences`
- :class:`LogisticRegressionBaseline`: feature build, strict + ratio
  targets, ``m_j = 0`` rows dropped, sample-weight augmentation in
  ``ratio`` mode
- :class:`FactualityProbeBaseline`: :func:`pick_layer_index`,
  :func:`extract_adapted_features`, fit/predict cycle on a tiny
  Han-style dataset, sentence-level aggregation (mean / min / geomean)

The expensive original variant (LLM re-encoding) and the real
DeBERTa-MNLI checkpoint are not exercised here — those are integration
concerns and would slow the unit-test loop to a crawl.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Sequence

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.baselines.factuality_probe import (  # noqa: E402
    FactualityProbeBaseline,
    extract_adapted_features,
    pick_layer_index,
)
from src.baselines.logistic_regression import (  # noqa: E402
    LogisticRegressionBaseline,
    build_sentence_features,
    collate_sentence_features,
)
from src.baselines.luq import compute_luq_for_sentences  # noqa: E402
from src.baselines.semantic_entropy import (  # noqa: E402
    cluster_by_entailment,
    compute_semantic_entropy_from_samples,
)
from src.baselines.token_entropy import compute_token_entropy_baseline  # noqa: E402


# ---------------------------------------------------------------------------
# Fake NLI scorer — duck-typed substitute for :class:`NLIScorer`
# ---------------------------------------------------------------------------


class _FakeNLI:
    """Minimal stand-in implementing ``entailment_prob`` / ``predict_label``.

    Pairs whose premise and hypothesis fall into a user-supplied
    ``support_pairs`` set get probability ``1.0``; everything else
    gets ``0.0``. This lets us test the clustering / consistency
    logic deterministically.
    """

    ENTAILMENT_INDEX = 2

    def __init__(self, support_pairs: Sequence[tuple[str, str]]):
        self._support = set(support_pairs)
        self.entailment_index = self.ENTAILMENT_INDEX

    def entailment_prob(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> torch.Tensor:
        return torch.tensor(
            [1.0 if (p, h) in self._support else 0.0
             for p, h in zip(premises, hypotheses)],
            dtype=torch.float32,
        )

    def predict_label(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> List[int]:
        return [
            self.ENTAILMENT_INDEX if (p, h) in self._support else 0
            for p, h in zip(premises, hypotheses)
        ]


# ---------------------------------------------------------------------------
# token_entropy
# ---------------------------------------------------------------------------


def test_token_entropy_mean_over_range() -> None:
    ent = torch.tensor([0.1, 0.5, 0.9, 1.3], dtype=torch.float32)
    assert math.isclose(
        compute_token_entropy_baseline(ent, (0, 2)), 0.3, rel_tol=1e-5
    )
    assert math.isclose(
        compute_token_entropy_baseline(ent, (1, 4)), (0.5 + 0.9 + 1.3) / 3,
        rel_tol=1e-5,
    )


def test_token_entropy_promotes_fp16_to_fp32() -> None:
    ent = torch.tensor([0.2, 0.6], dtype=torch.float16)
    out = compute_token_entropy_baseline(ent, (0, 2))
    assert isinstance(out, float)
    assert math.isclose(out, 0.4, abs_tol=1e-3)


def test_token_entropy_rejects_bad_inputs() -> None:
    ent = torch.zeros(5)
    with pytest.raises(ValueError):
        compute_token_entropy_baseline(ent, (3, 3))     # zero-length
    with pytest.raises(ValueError):
        compute_token_entropy_baseline(ent, (0, 99))    # past end
    with pytest.raises(ValueError):
        compute_token_entropy_baseline(ent, (-1, 2))    # negative
    with pytest.raises(ValueError):
        compute_token_entropy_baseline(torch.zeros(3, 3), (0, 2))  # 2D
    with pytest.raises(TypeError):
        compute_token_entropy_baseline([0.1, 0.2], (0, 2))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# semantic_entropy
# ---------------------------------------------------------------------------


def test_cluster_by_entailment_bidirectional_grouping() -> None:
    # Three samples, A and B are mutually entailing; C is isolated.
    samples = ["A", "B", "C"]
    support = [("A", "B"), ("B", "A")]  # A↔B only
    nli = _FakeNLI(support)
    clusters = cluster_by_entailment(samples, nli, threshold=0.5)
    # A and B should share a cluster id; C should be alone.
    assert clusters[0] == clusters[1]
    assert clusters[0] != clusters[2]
    assert len(set(clusters)) == 2


def test_cluster_by_entailment_unidirectional_does_not_merge() -> None:
    # A entails B but not the converse — must remain separate clusters.
    samples = ["A", "B"]
    nli = _FakeNLI([("A", "B")])
    clusters = cluster_by_entailment(samples, nli, threshold=0.5)
    assert clusters[0] != clusters[1]


def test_semantic_entropy_matches_cluster_distribution() -> None:
    # Two distinct semantic clusters of equal size → entropy = log 2.
    samples = ["A", "B", "C", "D"]
    nli = _FakeNLI(
        [("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")]
    )
    se = compute_semantic_entropy_from_samples(samples, nli, threshold=0.5)
    assert math.isclose(se, math.log(2), rel_tol=1e-5)

    # All samples in one cluster → entropy 0.
    nli_all = _FakeNLI([
        (a, b) for a in samples for b in samples if a != b
    ])
    se0 = compute_semantic_entropy_from_samples(samples, nli_all, threshold=0.5)
    assert math.isclose(se0, 0.0, abs_tol=1e-8)


def test_semantic_entropy_empty_returns_zero() -> None:
    nli = _FakeNLI([])
    assert compute_semantic_entropy_from_samples([], nli) == 0.0


# ---------------------------------------------------------------------------
# luq
# ---------------------------------------------------------------------------


def test_luq_consistency_score() -> None:
    sentences = ["The sky is blue.", "Cats can drive cars."]
    samples = ["The sky is blue.", "The sky is azure.", "Mars is red."]
    # First sentence is "supported" by samples 0 and 1; second by none.
    support = [
        ("The sky is blue.", "The sky is blue."),
        ("The sky is azure.", "The sky is blue."),
    ]
    nli = _FakeNLI(support)
    scores = compute_luq_for_sentences(sentences, samples, nli)
    # U = 1 - mean entailment over 3 samples = 1 - 2/3 for s0, 1 - 0 for s1.
    assert math.isclose(scores[0], 1.0 - 2.0 / 3.0, rel_tol=1e-5)
    assert math.isclose(scores[1], 1.0, rel_tol=1e-5)


def test_luq_skips_empty_sentences_and_samples() -> None:
    sentences = ["", "  ", "real claim"]
    samples = ["", "  ", "real claim"]
    nli = _FakeNLI([("real claim", "real claim")])
    scores = compute_luq_for_sentences(sentences, samples, nli)
    assert math.isnan(scores[0])
    assert math.isnan(scores[1])
    # Only one usable sample (the third); it entails the only real sentence.
    assert math.isclose(scores[2], 0.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# logistic_regression
# ---------------------------------------------------------------------------


def _toy_records(hidden_dim: int = 4, num_layers: int = 3) -> List[dict]:
    """Two sentences inside one prompt with disjoint token ranges."""
    T = 6
    h = torch.arange(T * num_layers * hidden_dim, dtype=torch.float32)
    h = h.view(T, num_layers, hidden_dim) / 10.0
    ent = torch.linspace(0.1, 0.6, T)
    top1 = torch.linspace(0.9, 0.4, T)
    return [
        {
            "dataset": "factscore_bio",
            "source_id": "demo",
            "hidden_states": h,
            "entropy": ent,
            "top1": top1,
            "token_range": (0, 3),
            "K_j": 3,
            "m_j": 3,
        },
        {
            "dataset": "factscore_bio",
            "source_id": "demo",
            "hidden_states": h,
            "entropy": ent,
            "top1": top1,
            "token_range": (3, 6),
            "K_j": 1,
            "m_j": 4,
        },
    ]


def test_build_sentence_features_shape_and_layer_mean() -> None:
    recs = _toy_records()
    feat = build_sentence_features(
        recs[0]["hidden_states"], recs[0]["entropy"], recs[0]["top1"],
        token_range=recs[0]["token_range"],
    )
    # hidden_dim (4) + entropy + top1
    assert feat.shape == (4 + 2,)
    # First-three-tokens mean ent / top1 matches the slicing.
    assert math.isclose(
        feat[-2].item(), float(recs[0]["entropy"][:3].mean()), abs_tol=1e-5
    )
    assert math.isclose(
        feat[-1].item(), float(recs[0]["top1"][:3].mean()), abs_tol=1e-5
    )


def test_logistic_regression_strict_fit_predict_cycle() -> None:
    # Build a deterministic feature → label dataset large enough to fit.
    torch.manual_seed(0)
    N, D = 80, 6
    Z = torch.randn(N, D)
    # Label depends on the sign of the first coordinate.
    K = (Z[:, 0] > 0).to(torch.long) * 3
    m = torch.full((N,), 3, dtype=torch.long)
    # Inject some m_j == 0 rows that must be dropped.
    Z = torch.cat([Z, torch.randn(5, D)], dim=0)
    K = torch.cat([K, torch.zeros(5, dtype=torch.long)], dim=0)
    m = torch.cat([m, torch.zeros(5, dtype=torch.long)], dim=0)

    clf = LogisticRegressionBaseline(target="strict", C=1.0).fit(Z, K, m)
    probs = clf.predict_proba(Z[:5])
    assert probs.shape == (5,)
    assert torch.isfinite(probs).all()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_logistic_regression_ratio_target_uses_sample_weights() -> None:
    torch.manual_seed(1)
    N, D = 60, 4
    Z = torch.randn(N, D)
    # Linear logits → factuality probability.
    logits = Z @ torch.tensor([1.5, -0.5, 0.7, 0.1])
    p = torch.sigmoid(logits)
    m = torch.full((N,), 5, dtype=torch.long)
    K = torch.binomial(m.to(torch.float32), p).to(torch.long)

    clf = LogisticRegressionBaseline(target="ratio", C=1.0).fit(Z, K, m)
    probs = clf.predict_proba(Z[:3])
    assert probs.shape == (3,)
    assert torch.isfinite(probs).all()


def test_logistic_regression_rejects_degenerate_labels() -> None:
    Z = torch.randn(10, 3)
    K = torch.zeros(10, dtype=torch.long)
    m = torch.full((10,), 2, dtype=torch.long)
    with pytest.raises(ValueError):
        LogisticRegressionBaseline(target="strict").fit(Z, K, m)


def test_collate_sentence_features_matches_individual_calls() -> None:
    recs = _toy_records()
    packed = collate_sentence_features(recs)
    assert packed["Z"].shape == (2, 4 + 2)
    assert torch.equal(packed["K"], torch.tensor([3, 1]))
    assert torch.equal(packed["m"], torch.tensor([3, 4]))


# ---------------------------------------------------------------------------
# factuality_probe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selected,target,expected_pos",
    [
        ([0, 4, 8, 12, 16, 20, 24, 28, 32], 14, 3),   # 12 is closest to 14
        ([0, 4, 8, 12, 16, 20, 24, 28, 32], 16, 4),   # exact match
        ([5, 10, 25], 14, 1),                          # 10 is closest
        ([7], 99, 0),                                  # only one option
    ],
)
def test_pick_layer_index_closest(
    selected: List[int], target: int, expected_pos: int
) -> None:
    assert pick_layer_index(target, selected) == expected_pos


def test_pick_layer_index_rejects_empty() -> None:
    with pytest.raises(ValueError):
        pick_layer_index(14, [])


def test_extract_adapted_features_takes_last_token_at_layer() -> None:
    T, L_lay, D = 5, 4, 3
    hidden = torch.arange(T * L_lay * D, dtype=torch.float32).view(T, L_lay, D)
    feat = extract_adapted_features(hidden, token_range=(1, 4), layer_index=2)
    # last_token = end - 1 = 3
    assert torch.equal(feat, hidden[3, 2].to(torch.float32))


def test_factuality_probe_adapted_fit_predict() -> None:
    torch.manual_seed(7)
    selected_layers = [0, 4, 8, 12, 16]
    layer_pos = pick_layer_index(14, selected_layers)
    assert layer_pos == 3  # 12 is closest to 14

    # Build a tiny dataset where last-token hidden states linearly separate.
    N, T, D = 40, 6, 5
    hidden = torch.randn(N, T, len(selected_layers), D)
    direction = torch.randn(D)
    labels = (hidden[:, -1, layer_pos] @ direction > 0).to(torch.long)
    # Make sure both classes are present.
    if labels.sum() in (0, N):
        labels[0] = 1 - labels[0]

    records = []
    for i in range(N):
        records.append({
            "dataset": "factscore_bio",
            "source_id": f"r{i}",
            "hidden_states": hidden[i],
            "entropy": torch.zeros(T),
            "top1": torch.ones(T),
            "token_range": (0, T),
            "K_j": int(labels[i].item()) * 4,
            "m_j": 4,
        })
    probe = FactualityProbeBaseline(variant="adapted", target_layer=14, C=10.0)
    train_pack = probe.build_adapted_dataset(records, selected_layers)
    assert train_pack["H"].shape == (N, D)

    probe.fit(train_pack["H"], train_pack["A"])
    probs = probe.predict_proba(train_pack["H"])
    assert probs.shape == (N,)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    # Easy linearly separable task → high training accuracy.
    train_acc = ((probs >= 0.5).to(torch.long) == train_pack["A"]).float().mean()
    assert float(train_acc) > 0.85


def test_factuality_probe_adapted_skips_m_zero_rows() -> None:
    selected_layers = [0, 1, 2]
    records = [
        {
            "hidden_states": torch.zeros(3, 3, 2),
            "entropy": torch.zeros(3), "top1": torch.zeros(3),
            "token_range": (0, 3), "K_j": 0, "m_j": 0,
        },
        {
            "hidden_states": torch.zeros(3, 3, 2),
            "entropy": torch.zeros(3), "top1": torch.zeros(3),
            "token_range": (0, 3), "K_j": 2, "m_j": 2,
        },
    ]
    probe = FactualityProbeBaseline(variant="adapted", target_layer=1)
    pack = probe.build_adapted_dataset(records, selected_layers)
    assert pack["H"].shape == (1, 2)
    assert int(pack["A"].item()) == 1


def test_factuality_probe_aggregate_modes() -> None:
    probe = FactualityProbeBaseline(variant="original")
    # Three sentences with 2, 1, 3 claims respectively.
    probs = torch.tensor([0.9, 0.1, 0.8, 0.6, 0.6, 0.6], dtype=torch.float32)
    ranges = [(0, 2), (2, 3), (3, 6)]
    mean = probe.aggregate_sentence_scores(probs, ranges, agg="mean")
    minv = probe.aggregate_sentence_scores(probs, ranges, agg="min")
    geo = probe.aggregate_sentence_scores(probs, ranges, agg="geomean")
    assert torch.allclose(mean, torch.tensor([0.5, 0.8, 0.6]), atol=1e-5)
    assert torch.allclose(minv, torch.tensor([0.1, 0.8, 0.6]), atol=1e-5)
    # geomean of [0.6, 0.6, 0.6] is 0.6; geomean of [0.9, 0.1] = sqrt(0.09).
    assert math.isclose(float(geo[0].item()), math.sqrt(0.09), rel_tol=1e-5)
    assert math.isclose(float(geo[2].item()), 0.6, rel_tol=1e-5)


def test_factuality_probe_predict_before_fit_raises() -> None:
    probe = FactualityProbeBaseline(variant="adapted")
    with pytest.raises(RuntimeError):
        probe.predict_proba(torch.zeros(1, 4))
