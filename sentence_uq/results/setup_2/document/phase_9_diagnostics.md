# Phase 9.1 — Epistemic Collapse Diagnostics (Findings)

**Date**: 2026-05-28
**Setup**: 2 (FActScore-Bio, in-domain)
**Inputs**: `results/setup_2/trained_model.pt`, test split = 353 sentences (m_j > 0), k = 66
**Method**: no retraining; `scripts/09_diagnose_epistemic.py --setup 2`
**Artifacts**:
- `results/setup_2/epistemic_diagnostics.json`
- `results/setup_2/epistemic_diag_eigenspectrum.png`
- `results/setup_2/epistemic_diag_distributions.png`
- `results/setup_2/epistemic_diag_signal_vs_error.png`

---

## 1. One-line conclusion

The epistemic collapse (`epi_μ` ≈ 8e-4) is driven by **sigmoid saturation (Factor 2)**,
**not** by posterior over-concentration (Factor 1) as hypothesized in
`prompts/phase_9_epistemic_collapse.md`. Consequently the document's first-line
remedy — **posterior tempering — is both wrong-premised and ineffective** and
should be dropped.

---

## 2. Diagnostic results

`epi_μ = ĝᵀ Σ̂ ĝ`, with `ĝ = (1/L) Σ_ℓ π̂_ℓ(1−π̂_ℓ) z_ℓ`.

| # | Diagnostic | Key numbers | Verdict |
|---|---|---|---|
| 1 | Σ̂ eigenspectrum | λ_max=1.46, λ_min=0.017, λ_mean=1.00, trace=66.2≈k, cond=86, anisotropy(λ_max/λ_mean)=1.45 | Σ̂ **healthy** — not collapsed |
| — | eigenvalue distribution | **8** dirs with λ<0.5; majority at prior scale (~1.0) | data informs only a **~8-D subspace** |
| 2 | ĝ norm & π̂ | **92% of tokens saturated** (π̂<0.05 or >0.95), π̂ median=0.000, ‖ĝ‖ mean=0.055 | **Factor 2 = primary cause** |
| 3 | upper-bound decomposition | epi_μ=8.1e-4; ‖ĝ‖²·λ_max=4.4e-3; ‖ĝ‖²·λ_mean=3.0e-3; actual/upper=0.18 | magnitude set by **‖ĝ‖²**, not Σ̂ |
| 4 | learned σ₀ | σ₀ ∈ [1.00, 1.45], mean=1.21 (init=1.0) | prior **did not tighten** (Factor 3 ruled out) |
| 5 | Fisher data vs prior | Fisher λ_max=58.2, prior Σ₀⁻¹ diag max=0.997, ratio=58× | data dominates only a **few directions** |

### Interpretation

In `Σ̂⁻¹ = Σ₀⁻¹ + Fisher`, the Fisher term is large in only ~8 directions
(λ_min = 1/58 ≈ 0.017); the other 58 directions stay at prior scale (~1.0).
The document's reasoning ("N_eff/k high → uniform collapse across all 66
directions") does not hold — the data is informative in a low-dimensional
subspace only.

The collapse is entirely in **ĝ**: with 92% of tokens at π̂ ≈ 0 or 1, the
Jacobian envelope `π̂(1−π̂)` → 0, so ‖ĝ‖ ≈ 0.055 and `epi_μ = ĝᵀΣ̂ĝ` ≈ 8e-4.
Σ̂ is O(1) and is not the bottleneck.

---

## 3. Bonus — logit-space epistemic, validated in-domain

For each test sentence we compared the current probability-space signal against
a logit-space readout `z̄ᵀ Σ̂ z̄` (no `π̂(1−π̂)` damping; z̄ = mean token feature),
ranked against the per-sentence ratio error `|μ̂ − U|`.

| Signal | Spearman(|μ̂−U|) | PRR_AUC (ratio) |
|---|---|---|
| `epi_μ` (probability space, current) | +0.494 | 0.139 |
| `z̄ᵀ Σ̂ z̄` (logit space) | −0.608 | **0.317** |
| `mean_ℓ z_ℓᵀ Σ̂ z_ℓ` (per-token mean) | −0.565 | 0.289 |
| no-skill baseline = mean(U) | — | 0.198 |
| (reference) Point estimate | — | ~0.248 |

The logit-space signal lifts PRR-AUC from **0.139 → 0.317**, beating both the
no-skill plateau (0.198) and the Point estimate (~0.248). This validates the
"swap epistemic readout to logit space" remediation **without any OOD data
generation**.

---

## 4. Implications for remediation

- **Tempering (doc §5.1 / §9.2): drop.** Σ̂ is already O(1); scaling by τ>1
  produces non-physical variances (σ² in the hundreds–thousands) with no
  principled basis, and a global scalar leaves the PRR ranking invariant
  (argsort unchanged), so it cannot fix the rejection-curve problem.
- **Logit-space epistemic (Direction 2): leading candidate.** Adding a
  logit-space readout to `predict.py` requires no retraining (Σ̂, z already
  available).
- **Open check (1):** confirm the logit-space PRR gain reflects genuine
  epistemic uncertainty and not a confidence proxy correlated with μ̂ — verify
  with a partial correlation controlling for μ̂.
- **OOD verification (original Plan B):** still valuable for the paper, but
  lower priority now that the signal is already demonstrable in-domain.

---

## 5. Reproduce

```bash
python scripts/09_diagnose_epistemic.py --setup 2 --device cpu
```
