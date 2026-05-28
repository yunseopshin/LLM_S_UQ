# Phase 9.2 — Logit-Space Epistemic Readout

## Background & Motivation

Phase 9.1 diagnostics (`results/setup_2/epistemic_diagnostics.json`,
`results/setup_2/document/phase_9_diagnostics.md`) established that the epistemic collapse
(`epi_μ ≈ 8e-4`) is caused by **sigmoid saturation** (92% of tokens have
`π̂ < 0.05` or `> 0.95`), NOT by posterior over-concentration. The posterior
covariance `Σ̂` is healthy (trace ≈ k, eigenvalues O(1)).

The current probability-space epistemic signal:

    epi_μ = ĝᵀ Σ̂ ĝ,   where ĝ = (1/L) Σ_ℓ π̂_ℓ(1−π̂_ℓ) z_ℓ

is crushed by the `π̂(1−π̂)` Jacobian envelope → ‖ĝ‖ ≈ 0.055 → epi_μ ≈ 8e-4.

The logit-space readout bypasses this damping:

    epi_logit = z̄ᵀ Σ̂ z̄,   where z̄ = (1/L) Σ_ℓ z_ℓ

Phase 9.1 bonus analysis showed this lifts PRR-AUC from 0.139 → 0.317
(beating point estimate ~0.248 and no-skill baseline 0.198) on setup 2
test split, with Spearman(|μ̂−U|) = −0.608.

**This phase implements the logit-space readout into the production pipeline
and validates it is not merely a confidence proxy.**

---

## Prerequisites

- Phase 3-3 (`src/inference/predict.py`) — `Predictor` class exists
- Phase 6-2 (`scripts/04_evaluate.py`) — evaluation pipeline exists
- Phase 9.1 (`scripts/09_diagnose_epistemic.py`) — diagnostic script exists
- Trained model: `results/setup_2/trained_model.pt`

---

## Prompt 9.2-1: Partial Correlation Validation Script

```
Create `scripts/09b_validate_logit_epistemic.py`.

This script TESTS whether the logit-space epistemic signal (z̄ᵀ Σ̂ z̄) carries
information beyond what μ̂ already tells us, or whether it is merely a confidence
proxy correlated with μ̂.

NOTE: empirically the signal has INVERTED polarity vs. a textbook epistemic reading
— higher epi_logit goes with LOWER error and LOWER U (Spearman(epi_logit, |μ̂−U|) =
−0.608). This is exactly why the partial-correlation gate below is load-bearing:
there is a real chance it FAILS, in which case the signal is a μ̂/confidence proxy
and must not be integrated. Do not assume the outcome.

**Inputs**: same as Phase 9.1 — `results/setup_2/trained_model.pt`, test split
sentences with m_j > 0.

**Analyses to implement** (all on the test split, N=353 sentences):

1. **Partial Spearman correlation**:
   Compute Spearman(epi_logit, |μ̂ − U| | μ̂). That is: rank-transform all
   three variables (epi_logit, |μ̂ − U|, μ̂), regress epi_logit on μ̂ to get
   residuals, regress |μ̂ − U| on μ̂ to get residuals, compute Spearman between
   the two residual vectors. If the partial correlation is still meaningfully
   negative (say, |ρ| > 0.2), the signal is not merely a μ̂ proxy.

2. **Stratified PRR-AUC**:
   Split test sentences into terciles by μ̂ (low / mid / high confidence).
   Compute PRR-AUC within each tercile using epi_logit as the rejection signal.
   If logit-space epistemic provides lift in all (or most) terciles, it carries
   information beyond μ̂.

3. **Variance Inflation Factor (VIF)** — secondary check only:
   Compute VIF = 1/(1−r²) between epi_logit and μ̂. Report it, but treat the partial
   Spearman (check 1) as the PRIMARY gate. VIF < 5 only rules out r ≳ 0.89 collinearity,
   which is far too loose to settle the proxy question on its own.

4. **Comparison table**: Print a summary table:
   | Check | Value | Pass criterion | Pass? |
   |---|---|---|---|
   | Partial Spearman(epi_logit, |err| \| μ̂) | ... | |ρ| > 0.2 | ... |
   | Stratified PRR-AUC (low μ̂ tercile) | ... | > no-skill | ... |
   | Stratified PRR-AUC (mid μ̂ tercile) | ... | > no-skill | ... |
   | Stratified PRR-AUC (high μ̂ tercile) | ... | > no-skill | ... |
   | VIF(epi_logit, μ̂) | ... | < 5 | ... |

**Output artifacts**:
- `results/setup_2/logit_epistemic_validation.json` — all computed values
- `results/setup_2/logit_epistemic_validation.png` — 2×2 figure:
  - (0,0): scatter of epi_logit residuals vs |err| residuals (after μ̂ partialed out)
  - (0,1): stratified PRR-AUC bar chart (3 terciles + overall)
  - (1,0): epi_logit vs μ̂ scatter (to visualize their relationship)
  - (1,1): epi_logit vs |μ̂ − U| scatter colored by μ̂ tercile
- Print the summary table to stdout

**CLI**: `python scripts/09b_validate_logit_epistemic.py --setup 2 --device cpu`

**Implementation notes**:
- Reuse model loading and test data loading from `scripts/09_diagnose_epistemic.py`
- For PRR-AUC computation, reuse `src/evaluation/metrics.py::compute_prr`. It ranks by
  the signal and rejects the HIGHEST-signal samples first, then reports the mean quality
  (U) of the remaining sentences. Pass `epi_logit` DIRECTLY as the rejection signal — do
  NOT negate it. This is exactly how Phase 9.1 produced PRR-AUC = 0.317
  (`compute_prr(U_true, epi_logit)`): empirically high epi_logit ↔ low U, so rejecting
  high epi_logit first raises remaining quality. Do NOT flip the sign based on the
  textbook "high variance = reject first" intuition — the empirical sign is what
  reproduces 0.317.
- For no-skill baseline per tercile: mean(U) within that tercile.
- Use scipy.stats for rank transformations and correlations.
```

---

## Prompt 9.2-2: Add Logit-Space Epistemic to Predictor

```
Extend `src/inference/predict.py` to add logit-space epistemic uncertainty.

**Changes to class `Predictor`**:

1. Add method `predict_sentence` return keys (alongside existing ones):
   - "epi_logit": float — z̄ᵀ Σ̂ z̄, where z̄ = mean(z_ℓ) over tokens
   - "epi_logit_tokmean": float — (1/L) Σ_ℓ (z_ℓᵀ Σ̂ z_ℓ), per-token mean variant

   Compute these AFTER the existing epi_mu calculation, inside the same
   `torch.no_grad()` / fp32 block. Both use self.Sigma_hat (already float32 on CPU in
   the Predictor — match that dtype; there is no float64 path here). Clamp at zero for
   numerical safety, mirroring the existing epi_mu (`.clamp_min(0.0)`):
     z_bar = z_tokens.mean(dim=0)               # (k,)
     epi_logit = float((z_bar @ (self.Sigma_hat @ z_bar)).clamp_min(0.0).item())
     Sz = z_tokens @ self.Sigma_hat
     epi_logit_tokmean = float((Sz * z_tokens).sum(dim=1).clamp_min(0.0).mean().item())

2. Do NOT change any existing return values or behavior. This is purely additive.

3. Update `BatchPredictor` similarly — add the two new fields to the batch output arrays.

4. Ensure dtype safety: cast z_bar to float64 before the quadratic form if Sigma_hat
   is float64 (match existing conventions in the file).

**Changes to class `Predictor.predict_mc_epistemic`**: no changes needed — MC sampling
is probability-space only.

**Do NOT touch**: `predict_from_hidden_states`, `save_trained_model`, `load_trained_model`.

**Tests** — add to `tests/test_decomposition.py`:
- `test_epi_logit_nonneg`: epi_logit >= 0 for any input (it's a quadratic form with PSD Σ̂)
- `test_epi_logit_tokmean_ge_epi_logit`: by Jensen's inequality, the per-token mean
  should be >= the mean-token variant (equality when all tokens are identical).
  Verify on random inputs: epi_logit_tokmean >= epi_logit - 1e-7 (float tolerance)
- `test_epi_logit_deterministic`: same input → same output (no randomness)
```

---

## Prompt 9.2-3: Integrate Logit-Space Epistemic into Evaluation

```
Update `scripts/04_evaluate.py` to use the logit-space epistemic signal for
PRR (Prediction Rejection Ratio) computation.

**Changes**:

1. In `_ours_predictions` (or the function that calls `predictor.predict_sentence`
   in a loop): extract the new "epi_logit" field and add it to the returned dict
   as a new array `"epi_logit"`.

2. In the PRR computation section for the "Ours (Bayesian)" model row:
   - Currently PRR uses `epi_mu` as the rejection signal (higher = reject first).
   - Add a SECOND PRR computation using `epi_logit` as rejection signal.
   - IMPORTANT: pass `epi_logit` DIRECTLY, exactly like `epi_mu` — do NOT negate it and
     do NOT change the sort direction. `compute_prr` already rejects the highest-
     `epi_logit` sentences first. Empirically high epi_logit ↔ low U, so this is the
     direction that reproduces the Phase 9.1 value of 0.317. Negating epi_logit would
     reject the lowest-epi_logit (high-U) sentences first and push PRR-AUC BELOW the
     no-skill baseline (~0.198) — the opposite of the intended result.
   - Report both PRR-AUC values in the ratio-level CSV:
     existing column: `prr_auc` (probability-space, unchanged)
     new column: `prr_auc_logit` (logit-space epistemic)

3. In the summary table printed to stdout, add a row or annotation showing
   the logit-space PRR-AUC alongside the probability-space one for "Ours (Bayesian)".

4. Save `epi_logit` per-sentence values to the per-sentence CSV if one exists,
   alongside the existing `epi_mu`.

**Do NOT change**: baseline model evaluation (Han et al., Point estimate).
Those don't have a logit-space epistemic signal.

**Verification**: After running `python scripts/04_evaluate.py --setup 2`,
the logit-space PRR-AUC for Ours (Bayesian) should approximately match the
Phase 9.1 diagnostic value of ~0.317. A small difference is acceptable if the
diagnostic script and evaluate script use slightly different test split filtering
or PRR implementation details, but they should be within ±0.02.
```

---

## Prompt 9.2-4: Update Diagnostic Findings Document

```
Create `results/setup_2/document/phase_9_2_findings.md` documenting the validation results
(same folder as the Phase 9.1 findings).

After running prompts 9.2-1 through 9.2-3, record:

1. Whether the partial correlation check passed (with the actual value)
2. Stratified PRR-AUC results per tercile
3. The final PRR-AUC from the integrated evaluation pipeline
4. Whether tempering is formally dropped from the remediation plan
5. Next steps: OOD verification (if still planned), paper section draft

This is a findings document, not an implementation prompt. Write it after
the implementation prompts are complete and results are available.
Template:

# Phase 9.2 — Logit-Space Epistemic Readout (Findings)

**Date**: [date]
**Setup**: 2 (FActScore-Bio, in-domain)

## 1. Validation: is epi_logit a genuine epistemic signal?
[partial correlation, stratified PRR, VIF results]

## 2. Integrated evaluation results
[PRR-AUC comparison: prob-space vs logit-space vs baselines]

## 3. Decision log
- Tempering: [dropped / kept]
- Logit-space readout: [adopted / conditional]
- Next: [OOD / paper draft / ...]
```

---

## Execution Order

1. **Prompt 9.2-1** first — run the validation script. If partial correlation
   check FAILS (|ρ| < 0.2), STOP and reassess before proceeding. The logit-space
   signal may be a confidence proxy and integrating it would be misleading.
2. **Prompt 9.2-2** — extend `predict.py` (purely additive, safe to do in parallel
   with 9.2-1 if desired, but only integrate into evaluation after validation passes).
3. **Prompt 9.2-3** — integrate into evaluation pipeline.
4. **Prompt 9.2-4** — document findings.

---

## Verification Checklist

After all prompts are complete:

- [ ] Partial correlation |ρ| > 0.2 (logit-space is not merely a μ̂ proxy)
- [ ] Stratified PRR-AUC > no-skill in at least 2 of 3 terciles
- [ ] `predict_sentence` returns `epi_logit` and `epi_logit_tokmean` keys
- [ ] `epi_logit >= 0` test passes
- [ ] Jensen inequality test passes (tokmean >= zbar variant)
- [ ] `04_evaluate.py` reports `prr_auc_logit` column in ratio CSV
- [ ] Logit-space PRR-AUC ≈ 0.317 (±0.02) matching Phase 9.1 diagnostic
- [ ] No existing tests broken (`pytest tests/`)
- [ ] Findings document created with all validation results

---

## Important Constraints

- **No retraining.** All changes use existing `θ̂`, `Σ̂`, and feature parameters.
- **Purely additive.** Do not modify existing `epi_mu` computation or any existing
  return values. The probability-space signal is preserved for comparison.
- **Rejection direction.** Pass `epi_logit` to `compute_prr` DIRECTLY, exactly like
  `epi_mu` (reject highest-signal first) — do NOT negate it. Empirically high epi_logit
  ↔ low U, so the direct direction is what reproduces PRR-AUC = 0.317; negating drops it
  below the no-skill baseline. Double-check this in every PRR computation.
- **Only modify requested files.** Do not refactor unrelated code.
