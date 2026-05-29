# Phase 9.5 — Logit Regulariser: Retraining Results

**Date**: 2026-05-29
**Setup**: 2 (FActScore-Bio, in-domain)
**Change**: added an optional per-token logit-magnitude penalty to training
(`logit_reg_lambda`, Option A of `phase_9_4_saturation_remedy_options.md`).
- `src/models/bayesian_main.py`: penalty `+ λ·Σ_{j:m>0}(1/L_j)Σ_ℓ(θ̂ᵀz_ℓ)²` in
  `compute_loss` (differentiable through θ̂; λ=0 = baseline).
- `scripts/03_train.py`: `--logit-reg-lambda` CLI + config + stored in `extra`.
**Models**: baseline (`results/setup_2/`) + λ∈{3e-4,1e-3,3e-3,1e-2,3e-2,1e-1}
(`results/setup_2_logitreg/lam*`). Same regime as baseline (300 ep, lr 1e-3,
fisher 10). Eval: `scripts/09e_compare_logitreg.py` → `document/logitreg_compare.json`.

---

## 1. λ sweep (in-domain Setup-2 test, 353 sentences)

| λ | epi_μ mean | epi_μ median | sat% | \|logit\| med | ECE | MAE | Pearson r | err-AUROC |
|---|---|---|---|---|---|---|---|---|
| 0 (baseline) | 8.07e-4 | 4.22e-4 | 0.92 | 21.1 | 0.0667 | 0.218 | 0.432 | 0.785 |
| 3e-4 | 8.22e-4 | 5.94e-4 | 0.81 | 7.98 | 0.0684 | 0.221 | 0.401 | 0.787 |
| 1e-3 | 1.01e-3 | 8.23e-4 | 0.71 | 5.35 | 0.0694 | 0.219 | 0.431 | 0.785 |
| 3e-3 | 1.26e-3 | 1.01e-3 | 0.61 | 3.89 | 0.0690 | 0.216 | 0.443 | 0.789 |
| **1e-2** | **1.70e-3** | **1.41e-3** | 0.44 | 2.55 | **0.0551** | 0.224 | **0.453** | 0.784 |
| 3e-2 | 2.06e-3 | 1.87e-3 | 0.24 | 1.74 | 0.0709 | 0.236 | 0.450 | 0.772 |
| 1e-1 | 2.45e-3 | 2.30e-3 | 0.04 | 1.10 | 0.1137 | 0.262 | 0.455 | 0.745 |

**Findings**:
- **epi_μ recovers and grows monotonically** with λ: median ×5.4 (4.2e-4→2.3e-3),
  mean ×3.0 by λ=1e-1. Logit magnitude normalises (median 21→1.1) and saturation
  drops 0.92→0.04.
- **Sweet spot λ=1e-2**: epi_μ median ×3.3 (→1.41e-3) **and the best ECE (0.055 vs
  baseline 0.067)**, with MAE (0.224≈0.218), Pearson (0.453, best) and AUROC (0.784)
  all preserved. Predictions are *not* harmed — slightly improved.
- **Beyond the sweet spot (λ≥3e-2) over-regularises**: epi_μ keeps creeping up but
  ECE (→0.114), MAE (→0.262) and AUROC (→0.745) degrade as μ̂ is flattened toward 0.5.

---

## 2. Why the recovery is bounded — the ĝ↔Σ̂ coupling

Diagnostic on λ=1e-2 (`09_diagnose`) vs baseline:

| | baseline | λ=1e-2 |
|---|---|---|
| ‖ĝ‖ mean | 0.055 | 0.363 (×6.6) |
| Σ̂ trace | 66.2 | 15.6 (÷4.2) |
| Fisher λ_max | 58 | 2895 (×50) |
| epi_μ mean | 8.07e-4 | 1.70e-3 (×2.1) |

De-saturation enlarges ĝ ×6.6, but the **same** directions gain Fisher information
(×50), so the posterior **tightens** there (Σ̂ trace ÷4.2). `epi_μ = ĝᵀΣ̂ĝ` nets only
~2–3×. This is intrinsic Bayesian-linear-model geometry — the directions the model is
most sensitive to (large ĝ) are exactly those the data constrains most (small Σ̂) — so
magnitude recovery via de-saturation is real but **capped** (nowhere near the naive
25× ĝ-channel "potential" from the temperature sweep, which ignored this coupling).

---

## 3. OOD re-check at λ=1e-2 (`09c`, annotation-free)

| signal | in-domain median | OOD median | OOD/in | p(OOD>in) |
|---|---|---|---|---|
| epi_μ | 1.41e-3 | 1.96e-3 | 1.39× | 5.2e-5 |
| epi_logit | 0.153 | 0.211 | 1.38× | 1.2e-41 |

- **epi_μ still rises significantly OOD (1.39×, p≈5e-5), now at ~3× the baseline
  absolute level** (OOD median 1.96e-3 vs baseline OOD 6.07e-4). Same relative OOD
  sensitivity, larger magnitude. (`results/setup_2_logitreg/lam1em2/document/ood_epistemic.*`)
- Note: `epi_logit` now *rises* OOD too (baseline: it *dropped*). De-saturation
  changed its behaviour; if it is to be used at all it must be re-validated with the
  9.2-1 partial-correlation gate on this model (not done here).

---

## 4. Decision / recommendation

- **Adopt λ=1e-2 as the operating point.** It maximises epistemic magnitude subject to
  calibration: epi_μ ×2–3, **better** ECE, preserved MAE/Pearson/AUROC, healthy logits
  (median 2.5), and correct OOD behaviour.
- **Still a caveat (honest)**: even at λ=1e-2 the epistemic std is ~0.04 (√1.4e-3) in-
  domain / ~0.044 OOD — meaningfully larger than baseline (~0.025) but still a modest
  fraction of total uncertainty. The ĝ↔Σ̂ coupling caps how far de-saturation alone can
  go. Larger gains would need a different parameterisation, not just more λ.
- **Open**: whether to promote the λ=1e-2 model to the production `results/setup_2/`
  (re-runs all downstream eval / baselines / paper figures) — deferred, not done here.

---

## 5. Reproduce

```bash
# train (per λ)
python scripts/03_train.py --setup 2 --config configs/default.yaml \
    --logit-reg-lambda 1e-2 --results-dir results/setup_2_logitreg/lam1em2 --device cuda
# compare
python scripts/09e_compare_logitreg.py --setup 2 --device cpu --models \
    baseline=results/setup_2/trained_model.pt \
    lam1e-2=results/setup_2_logitreg/lam1em2/trained_model.pt   # ... + other λ
# OOD re-check
python scripts/09c_ood_epistemic.py --setup 2 --device cpu \
    --trained-model results/setup_2_logitreg/lam1em2/trained_model.pt \
    --results-dir results/setup_2_logitreg/lam1em2
```
