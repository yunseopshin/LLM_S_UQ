# Phase 11-A — Sentence Readout Option (project-then-pool vs pool-then-project)

**Status**: DONE — readout optionised + tested. Route A (readout swap) **shelved**
(end-to-end regressed to AUROC 0.667). Real finding: strict-AUROC scoring
asymmetry (μ^m vs μ) — fixing it lifts the unchanged production model 0.784→0.827.
See §3b.
**Setup**: 2 (FActScore-Bio in-domain)
**Owner**: route-A discrimination recovery

---

## 1. Motivation — the diagnosis

In-domain strict AUROC trails simple probes built on the *same* generation-time
hidden states:

| model | readout | features | strict AUROC |
|---|---|---|---|
| Ours (production) | `mean_ℓ σ(θᵀ z_ℓ)` (project-then-pool) | 66-d z_ℓ | **0.780** |
| logistic_regression | token-mean → LR | 4096-d | 0.803 |
| factuality_probe_adapted | last-token → LR | 4096-d | 0.811 |
| Han repo (re-encode) | per-atom re-encode → LR | 4096-d | 0.858 |

The signal is already in our features, but our **project-then-pool** readout
(collapse each token to a scalar π_ℓ = σ(θᵀ z_ℓ) *before* averaging) discards
it. `scripts/diag_readout_pool.py` confirmed this by swapping ONLY the readout
on the SAME trained 66-d z_ℓ (harness validated: reproduces CSV Ours(Point)
0.78437 exactly):

| readout on our z_ℓ | strict AUROC | strict ECE |
|---|---|---|
| A. ours `mean σ(θᵀz)` (project-then-pool) | 0.784 | 0.0505 |
| C. **last-token** (pool-then-project, 66-d) | **0.797** | **0.0325** |
| B. mean (pool-then-project, 66-d) | 0.793 | 0.0628 |
| D. mean+std+last (198-d) | 0.793 | 0.0722 |

**Gap decomposition (0.784 → 0.811):** project-then-pool penalty ≈ +0.013
(A→C, same features) + 66-d compression penalty ≈ +0.014 (C 66-d 0.797 <
adapted 4096-d 0.811). Han 0.858 (re-encode) is unreachable from generation-time
states. Full rationale: memory `routeA-readout-diagnostic.md`.

**Route A** = relocate the pooling to *before* the scalar projection:
`μ_j = σ(θᵀ · pool(z_ℓ))`. Keeps binomial K_j~Bin(m_j, μ_j) + Bayesian
posterior + epistemic/aleatoric + single-pass; only the *averaging form* of the
Theorem-2 token attribution changes (replaceable by gradient-of-μ_j).

---

## 2. Design — optionised, original code preserved

The readout is a **pooling applied to z_tokens during feature preparation**,
keyed on `params.readout`. Because the Fisher-scoring core always forms
`μ_j = mean over rows of σ(θᵀ row)`, feeding it a single pooled `(1, k)` row
yields exactly `σ(θᵀ ζ_j)` — so the whole inner-loop / likelihood / predictor
stack is reused **unchanged**, and `token_mean` stays byte-for-byte identical
to the pre-11-A model.

| `readout` | rows fed to core | resulting μ_j |
|---|---|---|
| `token_mean` (default) | `(L_j, k)` z_ℓ unchanged | `mean_ℓ σ(θᵀ z_ℓ)` — **original** |
| `mean` | `(1, k)` = mean_ℓ z_ℓ | `σ(θᵀ mean_ℓ z_ℓ)` |
| `last` | `(1, k)` = z_{L_j-1} | `σ(θᵀ z_last)` |
| `attention` | `(1, k)` = Σ a_ℓ z_ℓ, a=softmax(z·v/√k) | `σ(θᵀ ζ_j)` |

- `token_mean` is the **default** everywhere → existing `trained_model.pt`,
  configs, and all 346 tests are unaffected (verified).
- `attention` registers ONE extra param `attn_v ∈ R^k` (zero-init ⇒ uniform ⇒
  equals `mean` at step 0). Registered ONLY for `attention`, so every other
  readout keeps an identical `state_dict` (same guard pattern as `log_phi`).

### Files touched
- `src/features/extractor.py` — `SentenceUQParams(readout=...)` + `attn_v` +
  `pool_token_features(z, params)`.
- `src/train/trainer.py` — `_collate` applies `pool_token_features` (train +
  eval path).
- `src/inference/predict.py` — `Predictor.predict_from_raw` pools; `save_/
  load_trained_model` round-trip the `readout` config key.
- `scripts/04_evaluate.py` — `_extract_z_tokens` pools.
- `scripts/03_train.py` — `--readout {token_mean,mean,last,attention}` + reads
  `model.readout` from config.
- `configs/default.yaml` — `model.readout: token_mean`.
- `tests/test_features.py` — 8 new tests (shapes, identity, differentiability,
  attn param registration, manual-equality, invalid-value).

### How to select (the option)
```bash
# original (unchanged):
python scripts/03_train.py --setup 2 --readout token_mean   # or omit the flag
# pool-then-project last-token (best in diagnostic):
python scripts/03_train.py --setup 2 --readout last  --results-dir results/setup_2_readout_last
# attention pool:
python scripts/03_train.py --setup 2 --readout attention --results-dir results/setup_2_readout_attn
```
Inference auto-detects the readout from the saved model config — no flag needed
at eval time.

---

## 3. Validation plan & results

- [x] All 346 existing tests pass with `token_mean` default (bit-identity).
- [x] 8 new readout unit tests pass.
- [ ] Train `readout=last` on setup_2 (same regime: 300 ep, lr 1e-3, fisher 10,
      logit_reg_lambda 1e-2) → `results/setup_2_readout_last/`. **(running)**
- [ ] Evaluate `readout=last`; compare strict AUROC / ratio / ECE vs production
      `token_mean` (0.780 / 0.0518). Target: end-to-end ≈ diagnostic's 0.797.
- [ ] If confirmed, optionally stack with wider features (projection_dim↑ /
      multi-layer) to chase the remaining ~0.014 toward simple-probe parity.

### End-to-end results
| readout | strict AUROC | strict ECE | ratio MAE | ratio r | binom NLL |
|---|---|---|---|---|---|
| token_mean (production) | 0.780 | 0.0518 | 0.224 | 0.453 | 1.327 |
| **last (end-to-end trained)** | **0.667** ❌ | 0.167 ❌ | 0.285 ❌ | 0.247 ❌ | 2.375 ❌ |

**Route A (readout swap) FAILED end-to-end.** Joint from-scratch training of W/α
under `readout=last` (production regime) collapsed to AUROC 0.667 — *worse* than
the 0.780 baseline. `results/setup_2_readout_last/`.

---

## 3b. Why it failed — and the real finding (`diag_readout_fishermap.py`)

Isolating features vs optimisation (production W/α FIXED, Fisher-MAP refits θ):

| readout (fixed prod features) | AUROC(μ^m) | AUROC(μ) |
|---|---|---|
| token_mean | 0.784 | **0.827** |
| last | 0.755 | 0.803 |

Two discoveries:

1. **The sklearn diagnostic over-promised.** Its last-token 0.797 leaned on
   `StandardScaler` + a direct L1-LR on the strict label `A_j`. Through our real
   Fisher-MAP + binomial pipeline, fixed-feature last-readout is only 0.755
   (standardisation in-pipeline made it *worse*, 0.755→0.755/ECE↑). The readout
   swap does **not** transfer the gain. Route A is shelved.

2. **The actual handicap is the strict-AUROC SCORING, not the model.** Our
   pipeline ranks the strict event by `p_strict = μ^m`; **every baseline ranks by
   raw μ** (`04_evaluate.py:596` vs `:819`). AUROC is a pure ranking metric, so
   this is apples-to-oranges and it *cost us*. Scoring the SAME production
   `token_mean` model by μ (fair, == baselines):

   | strict AUROC scoring | value | 95% CI |
   |---|---|---|
   | μ^m (current, self-handicapped) | 0.784 | [0.714, 0.848] |
   | **μ (fair, == baselines)** | **0.827** | [0.761, 0.888] |
   | baselines (rank by μ) | logreg 0.803 / adapted 0.811 / Han repo 0.858 | |

   → No retraining, no readout change: strict AUROC 0.784 → **0.827**, above the
   simple probes, within ~0.03 of Han stock. ECE/Brier stay on μ^m (0.052) where
   the calibrated event probability is the right object.

**Conclusion**: Route A is a dead end; the strict-AUROC gap was largely a
self-inflicted scoring asymmetry. Recommended action moved to the scoring fix
(score strict AUROC by μ for ours, consistent with baselines; keep μ^m for
calibration). Readout option is retained (default `token_mean`) for future
experiments but is NOT promoted.

### Why AUROC uses μ but ECE/Brier MUST stay on μ^m (not a contradiction)

AUROC and ECE measure different properties, so each takes the object it is
defined on — "consistency" is per-metric-correctness, not one shared array.

| strict scoring of production model | AUROC | ECE | Brier |
|---|---|---|---|
| μ^m  (P(all m correct) — binomial event prob) | 0.784 | **0.0505** | 0.068 |
| μ    (E[U_j] = P(one atom correct) — per-atom) | **0.827** | 0.1739 | 0.104 |

- **AUROC = ranking** (scale-free; m-dependence is pure noise → μ ranks better,
  and every baseline ranks by μ, so μ is the fair cross-method score).
- **ECE/Brier = probability calibration**; the number must equal P(A_j=1) = μ^m.
  Scoring ECE on μ calibrates a *per-atom* probability against an *all-atoms*
  event → systematic overconfidence → **0.174, essentially identical to Han's
  0.173** (Han uses raw atom-mean μ for the strict event — the very failure our
  binomial μ^m fixes). Switching our ECE to μ would discard the contribution.
- Analogy: rank by logits, calibrate on softmax — different objects, not
  inconsistent. Here μ↔μ^m is that relationship (m breaks cross-sentence
  monotonicity).
- The only *correct* unification is UP to μ^m for both (AUROC 0.784 + ECE
  0.052), preserved in the CSV as the `AUROC_pstrict` column. Never down to μ.

**Implemented** (`04_evaluate.py::_strict_row`, `rank_score` param): Ours strict
AUROC/AUPRC rank by μ̂ (fair, == baselines); Brier/ECE stay on μ̂^{m_j};
legacy μ̂^{m_j} AUROC kept as `AUROC_pstrict`. Baselines unchanged (rank_score
defaults to their μ̂). All 346 + 5 evaluate-script tests pass.

---

## 4. Honest framing notes

- Route A targets **parity with the simple generation-time probes (~0.81)**, NOT
  Han re-encode (0.858, structurally out of reach for single-pass).
- The Han-comparison headline stays **strict-ECE + single-pass + UQ**; Route A
  removes the embarrassment of trailing the *simple* probes, it does not flip
  the Han AUROC.
- Token attribution: `last`/`mean`/`attention` collapse a sentence to one pooled
  row, so per-token heatmaps degenerate. If kept, attribution must move to the
  gradient `∂μ_j/∂z_ℓ` form (still single-pass). Tracked as a follow-up.
