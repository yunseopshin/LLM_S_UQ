# Phase 9.7 — Strict-Factuality Readout Diagnostic (Findings)

**Date**: 2026-05-29
**Setup**: 2, production λ=1e-2 model (`results/setup_2/trained_model.pt`)
**Script**: `scripts/09f_strict_readout_diag.py` (no retraining)
**Artifact**: `document/strict_readout_diag.json`

## 0. Motivation

On strict factuality (A_j = 1{K_j = m_j}) we trail Han on AUROC/AUPRC. We
reported the strict **detection score** as the binomial all-atoms probability
`p_strict = μ̃_j ** m_j` (`predict.py:297`). This diagnostic asks whether the
gap is a *readout* choice or a *representation* deficit, by re-scoring the same
model with different strict scores (AUROC/AUPRC are rank-only).

## 1. Result (353 test sentences, frac strict-factual = 0.0765, median m_j = 3)

| strict detection score | AUROC | AUPRC |
|---|---|---|
| **μ̂  (decoupled from m)** | **0.8269** | **0.2612** |
| μ̃ probit (decoupled) | 0.8258 | 0.2612 |
| μ̂ ** m  (Ours Point, current) | 0.7844 | 0.2448 |
| μ̃ ** m  (Ours Bayesian, current report) | 0.7797 | 0.2388 |
| — *reference* — | | |
| Han `factuality_probe_adapted` | 0.8113 | 0.2659 |
| Han `factuality_probe_original` | 0.8487 | 0.3866 |
| Han `factuality_probe_original_repo` | 0.8576 | 0.4217 |

## 2. Findings

- **The strict-detection gap to Han's adapted probe is a READOUT artefact, not a
  representation deficit.** Ranking by μ̂ gives AUROC **0.827 > Han adapted 0.811**
  and AUPRC **0.261 ≈ Han adapted 0.266** — from the *same* model, **zero
  retraining**. The reported 0.780 (μ̃^m) discards ~0.047 AUROC.
- **Why μ̂^m hurts ranking.** Raising to the power m couples the factuality signal
  with sentence length m (a nuisance variable for *ranking*). Within a fixed m it
  is monotone (harmless), but across sentences it reshuffles the ranking — high-m
  sentences are pushed down regardless of content, compressing discrimination.
  μ̂^m is the right *probability* (for ECE/Brier) but the wrong *ranking* score.
- **Uniform exponent tempering is a no-op.** μ̃^(β·m) = (μ̃^m)^β is a monotone
  reparam of μ̃^m → identical AUROC/AUPRC (verified). The only lever that changes
  the ranking is *decoupling* from m (use μ̂/μ̃), not scaling the exponent.
- **Residual gap is representation, not readout.** Even with μ̂, we sit at
  0.827/0.261 — competitive with the adapted probe but **below the stock Han repo**
  (0.858/0.422, esp. on AUPRC). That remaining gap is a representation/method gap
  (see §4). NB the stock-repo sentence-strict number is granularity-inflated
  relative to its atom-level ≈0.787 (see `MEMORY` han-stock-code-baseline);
  the leakage-free, model-class-matched comparator is the adapted probe.

## 3. Implication for reporting (proposed, no retraining)

Decouple the two strict quantities:
- **Detection (AUROC, AUPRC, PRR)** → score by **μ̂** (the model's sentence
  factuality, the same quantity used for ratio-level metrics). Apples-to-apples
  with every probe baseline (all are ranked scores).
- **Calibrated strict probability (ECE, Brier, NLL)** → keep **μ̃^m** (the honest
  P(all m atoms supported)).

This is coherent (not cherry-picking): μ̂^m over-weights a nuisance variable for
ranking; the principled detector score is μ̂. Implementation: a few lines in
`04_evaluate` (`_strict_row` / strict pool) + the predictor's strict outputs.

## 4. If we want to beat the stronger Han variants (representation levers, need retraining)

Prior-fixing (§3-6) is **orthogonal** to this gap — it touches Σ̂/epistemic and
mild θ regularisation (σ₀ drifted only 1.0→0.97), not μ̂ discrimination. Real
levers for AUROC/AUPRC:
- **`projection_dim` = 64 bottleneck.** Han probes the full hidden state (≫64-d);
  our learned W compresses to 64-d before the probe → likely the main cap on
  discrimination/AUPRC. Sweep larger projection_dim.
- **α is near-uniform** (softmax 0.085–0.124 over layers 0–32, no sharp peak), so
  multi-layer aggregation may be *diluting* the most discriminative layer (Han uses
  a single best layer ≈14). Try a temperature on α / single-best-layer ablation.
- Feature richness beyond [proj, entropy, top1].
