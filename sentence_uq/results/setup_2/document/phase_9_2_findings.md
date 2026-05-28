# Phase 9.2 — Logit-Space Epistemic Readout (Findings)

**Date**: 2026-05-28
**Setup**: 2 (FActScore-Bio, in-domain)
**Inputs**: `results/setup_2/trained_model.pt`, test split = 353 sentences (m_j > 0)
**Script**: `scripts/09b_validate_logit_epistemic.py --setup 2`
**Artifacts**: `results/setup_2/logit_epistemic_validation.{json,png}`

---

## 1. Validation: is `epi_logit = z̄ᵀ Σ̂ z̄` a genuine epistemic signal?

**Verdict: NO — it is a μ̂ / confidence proxy. Gate FAILED.**

| Check | Value | Criterion | Pass |
|---|---|---|---|
| **Partial Spearman** ρ(epi_logit, \|μ̂−U\| \| μ̂) | **−0.072** | \|ρ\| > 0.2 | ❌ **FAIL** |
| (raw Spearman, no control) | −0.608 | — | — |
| Stratified PRR-AUC (low μ̂ tercile) | 0.1228 | > 0.072 | ✅ |
| Stratified PRR-AUC (mid μ̂ tercile) | 0.1658 | > 0.148 | ✅ (margin +0.018) |
| Stratified PRR-AUC (high μ̂ tercile) | 0.4523 | > 0.377 | ✅ |
| VIF(epi_logit, μ̂) | 1.48 | < 5 | ✅ |
| Overall PRR-AUC (epi_logit) | 0.3172 | — | — |

### Interpretation

- **The primary gate fails decisively.** Controlling for μ̂ collapses the raw
  Spearman from **−0.608 to −0.072**. Essentially all of `epi_logit`'s apparent
  relationship with error is explained by μ̂; it adds almost nothing beyond μ̂.
  The Phase 9.1 PRR gain (0.139 → 0.317) is therefore a **confidence effect**,
  not parameter uncertainty.
- **VIF passes but is misleading.** VIF=1.48 (linear r≈0.57) looks low, but the
  real dependence between `epi_logit` and μ̂ is monotonic/rank-based, captured
  only by the rank partial correlation. This empirically confirms that VIF<5 is
  an inadequate gate for the proxy question — the rank partial Spearman is the
  load-bearing check.
- **Stratified PRR passes 3/3 but with negligible margins** (mid tercile only
  +0.018). Any within-confidence-band signal is at noise level.
- Sanity: raw Spearman −0.608 and overall PRR 0.3172 reproduce the Phase 9.1
  diagnostic exactly — same data, same signal, pipeline consistent.

---

## 2. Integrated evaluation results

**Not performed.** Per the execution order in
`prompts/phase_9_2_logit_epistemic.md` (§Execution Order: "if the partial
correlation check FAILS, STOP"), Phase 9.2-2 (`predict.py` extension) and
9.2-3 (`04_evaluate.py` integration) were **halted** and not implemented. The
overall PRR-AUC of 0.3172 is recorded here as a confidence-proxy artifact, not
adopted as an epistemic metric.

---

## 3. Decision log

- **Posterior tempering (Phase 9 §5.1): DROPPED.** Phase 9.1 refuted its premise
  (Σ̂ is healthy, O(1), not over-concentrated) and a global scalar leaves PRR
  ranking invariant.
- **Logit-space readout (Phase 9.2): NOT ADOPTED as epistemic.** Failed the
  partial-correlation validation — it is a confidence proxy. Not integrated into
  the production pipeline.
- **Why both candidates fail in-domain.** `epi_μ` is genuinely epistemic in
  construction but is crushed by sigmoid saturation (Phase 9.1) and ranks below
  no-skill; `epi_logit` ranks well but is not epistemic. This is consistent with
  the Phase 9.1 finding that, in-domain, the posterior is tight in the
  data-informed directions — i.e. there is **genuinely little epistemic signal
  to extract in Setup 2**, and near-zero epistemic is the *correct* answer for
  in-domain points.
- **Next: OOD verification.** The epistemic decomposition can only be validated
  where parameter uncertainty actually exists — out of distribution. Plan: check
  whether `epi_μ` / `epi_logit` rise on out-of-domain inputs (LongFact) relative
  to in-domain Bio test, using the Setup-2-trained model. Requires generating +
  annotating an OOD pilot (LongFact data is not yet present in the repo).

---

## 4. Reproduce

```bash
python scripts/09b_validate_logit_epistemic.py --setup 2 --device cpu
```
