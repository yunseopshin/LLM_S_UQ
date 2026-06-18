# Phase 11 — Consolidated Findings & Conclusions (Setup 2)

**Date**: 2026-06-16/17
**Scope**: a focused investigation triggered by "our numbers look weak across the
board vs baselines." This doc is the single source of truth for what we found,
what we changed, what we abandoned, and the recommended direction. Detailed
write-ups: `phase_11_A_readout.md`, `phase_11_B_ood_epistemic_vs_baselines.md`.

---

## 0. TL;DR

1. **Strict-AUROC was self-handicapped by a scoring asymmetry.** Ranking the
   strict event by μ̂ (like every baseline) instead of μ̂^m lifts our strict
   AUROC **0.784 → 0.827** with NO model change — now above logreg (0.803) and
   adapted (0.811). **Implemented** in `04_evaluate.py`. ✅
2. **Route A (change the sentence readout) is a dead end.** End-to-end
   `readout=last` training regressed (AUROC 0.667). The sklearn diagnostic that
   promised 0.797 was a StandardScaler artifact. Readout optionised but NOT
   promoted (default `token_mean`). ✅ shelved
3. **The Bayesian epistemic/aleatoric decomposition does NOT earn its place.**
   In-domain it collapses (correct but useless); OOD it FAILS to rank errors
   (err-AUROC **0.364, below chance**), beaten by mean entropy, logreg, and even
   our own confidence. Root cause is structural: `epi_mu` is gated by w=μ(1−μ)
   (r=0.79) → a confidence proxy, not genuine parameter uncertainty. ❌ **negative
   result**
4. **The project still has real, validated contributions** (§5) — they just are
   NOT the epistemic decomposition. Recommended: reframe around them; report epi
   as an honest negative.

---

## 1. Baseline fairness — which Han variant

Three Han variants existed; the **fairest is `factuality_probe_original_repo`**:
Han's actual `fact-probe` code, CV-selected probe, **trained on our setup_2 train
split** (`_han_on_ours/prep_data.py`) and tested on our test pool, no leakage,
and it is Han's strongest result (strict AUROC 0.858). It re-encodes atoms (the
property to preserve). `_adapted` removes re-encoding → it is an ablation of OUR
generation-time contribution, not a "Han" baseline. `_original` (our reimpl,
0.849 ≈ repo 0.858) is kept as a reimpl-validation footnote.

Han's 0.858 AUROC is **unreachable** from generation-time states (re-encoding is
worth ~+0.045 and costs ~6 orders more compute). The Han comparison rests on
strict-ECE + single-pass + competitive AUROC, NOT on beating Han's AUROC.

---

## 2. Strict-AUROC scoring fix (IMPLEMENTED) ✅

**Asymmetry found** (`04_evaluate.py:596` vs `:819`): baselines rank the strict
event by their raw sentence score μ̂; ours ranked by p_strict = μ̂^m. AUROC is a
pure ranking metric → apples-to-oranges that cost us.

| strict scoring of the SAME production model | AUROC | ECE | Brier |
|---|---|---|---|
| μ̂^m (old, self-handicapped) | 0.784 | 0.0505 | 0.068 |
| **μ̂ (fair, == baselines)** | **0.827** | — | — |
| μ̂ used for ECE (WRONG object) | — | 0.1739 | 0.104 |

- AUROC/AUPRC now rank by μ̂ (fair); ECE/Brier stay on μ̂^m (the calibrated
  strict-event probability — using μ̂ there gives 0.174 ≈ Han's failure, because
  μ̂ is P(one atom ok), not P(all m ok)). The two metrics take different objects
  by design; legacy μ̂^m AUROC kept as `AUROC_pstrict`.
- **Result**: Ours strict AUROC **0.827** (> logreg 0.803, adapted 0.811; < Han
  0.849–0.858), strict ECE **0.052** (≪ Han 0.17). Validated end-to-end into a
  scratch dir; **production `results/setup_2` not yet regenerated** (pending).
- Ratio tier was already symmetric (both use μ̂) — unaffected.

---

## 3. Route A — readout swap (SHELVED) ✅

Hypothesis: our `μ_j = mean_ℓ σ(θᵀ z_ℓ)` (project-then-pool) discards signal vs
`σ(θᵀ pool(z_ℓ))` (pool-then-project). Implemented `readout ∈ {token_mean (default,
bit-identical), mean, last, attention}` across the pipeline (346+ tests pass).

- **End-to-end `readout=last` training REGRESSED**: strict AUROC 0.667, ratio r
  0.247 (`results/setup_2_readout_last/`).
- The sklearn diagnostic's 0.797 was a StandardScaler + direct-A_j-LR artifact;
  through our real Fisher-MAP it is only 0.755. The readout change does not
  transfer the gain. **Conclusion**: Route A shelved; readout kept as an option
  but NOT promoted.
- (The strict-AUROC gain in §2 came from scoring, not from the readout.)

---

## 4. Epistemic decomposition — NEGATIVE RESULT ❌

### 4a. In-domain PRR (rejection)
The Bayesian PRR ranks by `epi_mu`, which collapses in-domain (epi_μ ~ 1.7e-3) →
ratio PRR 0.103 / strict 0.025 (bad). total_U / aleatoric do NOT fix it (≈0.118 /
0.028): aleatoric variance peaks at μ=0.5, so confidently-wrong (low-μ) sentences
get LOW variance and are not rejected. **Only confidence works**: ranking by −μ̂
gives ratio PRR **0.333** / strict **0.158** — beating/matching all baselines.
(This mirrors §2: ours should rank by the same kind of signal baselines use.)
*Not yet wired into `04_evaluate`.*

### 4b. OOD — the make-or-break test
- Pre-check (annotation-free, ID vs OOD detection): `epi_mu` is the ONLY signal
  that rises OOD (AUROC 0.589); entropy & confidence go the wrong way (the model
  is confidently fluent on OOD). Encouraging but weak, and detection ≠ error
  detection.
- **Decisive (annotated 292 OOD sentences, 222 m_j>0)** — OOD ERROR detection:

  | signal | strict err-AUROC | ratio PRR | Spearman(U) |
  |---|---|---|---|
  | **epi_mu** | **0.364** ❌ | 0.281 | +0.186 (wrong sign) |
  | **total_U** | **0.367** ❌ | 0.296 | +0.100 |
  | conf = −μ̂ | 0.625 | 0.449 | −0.238 |
  | **mean_entropy (token_entropy)** | **0.673** | 0.425 | −0.202 |
  | logreg (probe) | 0.639 | 0.447 | −0.252 |

  **epi_mu is below chance at OOD error detection** and is beaten by mean
  entropy, logreg, and our own confidence.

### 4c. Root cause (structural, not tuning)
`epi_mu = ĝᵀΣ̂ĝ`, `ĝ ∝ w·z`, `w = π(1−π)`. The readout is gated by the
confidence term w = μ(1−μ):
- pearson(epi_mu, μ(1−μ)) = **+0.791**, pearson(epi_mu, μ) = +0.652.

So `epi_mu` is effectively a mid-confidence / aleatoric proxy, not parameter
uncertainty about correctness. This corroborates Phase 9.2 (epi_logit is also a
confidence proxy). Phase 9.3's "epi rises ×2.77 OOD" is a population mean-shift
that does NOT rank which OOD predictions are wrong. **Not fixable by retraining.**

### 4d. Conclusion
The Bayesian epistemic/aleatoric decomposition — a core stated contribution —
**does not earn its place as an error-detection / selective-prediction signal**.
In-domain it collapses; OOD it fails (below chance). Documented as a negative
result.

---

## 5. What actually survives (the real contribution)

Independent of the epistemic story, the following are validated on Setup 2:

- **Binomial observation model → strict calibration.** Strict ECE **0.052** vs
  Han **0.17** (3.3×). Unique: P(all atoms correct) = μ^m is the right object;
  baselines' mean-atom-prob μ̂ is overconfident for the strict event. This is the
  cleanest win and is novel.
- **Single forward pass (no re-encoding).** ~0.09 ms/sentence vs Han's re-encode
  ~67 s — ~6 orders of magnitude, with competitive accuracy.
- **Competitive discrimination** after the §2 fix: strict AUROC 0.827, ratio MAE
  0.224 / r 0.453 — at/above the simple generation-time probes.
- **Ratio-level Binomial NLL** (ours-only, principled).

The honest framing: *a fast, single-pass, well-calibrated sentence-factuality
probability via a binomial observation model* — NOT an epistemic-uncertainty
method.

---

## 6. State of the code / artifacts

| item | status |
|---|---|
| `readout` option (extractor/trainer/predict/04_eval/03_train/config) | landed, default token_mean, tests pass |
| strict-AUROC μ̂-ranking fix + `AUROC_pstrict` (`04_evaluate.py`) | landed, validated in scratch dir |
| production `results/setup_2` regenerated with the fix | **NOT done** (pending user OK) |
| in-domain PRR confidence-ranking | **NOT wired** (diagnostic only) |
| OOD labels `data/processed/longfact/annotated.json` | created (292 sent, 222 m>0) |
| diagnostics | `scripts/diag_readout_pool.py`, `diag_readout_fishermap.py`, `diag_prr_signals.py`, `diag_ood_detection.py`, `diag_ood_error_detection.py` |
| `results/setup_2_readout_last/` | trained last-readout model (regressed; kept for reference) |

## 7. Open decisions (deferred)

- Regenerate `results/setup_2` with the scoring fix (headline tables).
- Whether to wire confidence-ranked PRR for the in-domain selective-prediction row.
- Paper reframing per §5 (drop epistemic as a selling point; keep as negative).
