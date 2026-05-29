# Phase 9.6 — Promote λ=1e-2 to Production (setup_2/)

**Date**: 2026-05-29
**Setup**: 2 (FActScore-Bio, in-domain)
**Action**: resolves `phase_9_summary.md` §3 item 1 — the λ=1e-2 logit-regularised
model (Phase 9.5) is now the production `results/setup_2/` model.

---

## 1. What was done & why this scope

- **Promotion = copy, not retrain.** `results/setup_2_logitreg/lam1em2/` was trained
  on the identical Setup-2 split / regime (300 ep, lr 1e-3, fisher 10, n_train=1638).
  We copied its `trained_model.pt` + `train_history.json` + `train_summary.json` into
  `results/setup_2/`. Copying (vs re-running `03_train`) guarantees the production
  model is **bit-identical to the 9.5-validated one** — no seed drift, the headline
  numbers below match `logitreg_compare.json`/`phase_9_5` exactly.
- **Only `04_evaluate` was re-run** (`--device cpu`), reusing the existing
  `baselines.json`. **`05_baselines` was NOT re-run** — verified model-independent:
  `05_baselines.py` builds only a *dummy* `BayesianSentenceUQ(hidden_dim=8,…)` to call
  the `prepare_data` helper ("The Bayesian model itself is never trained here"), and
  never loads `trained_model.pt`. Han / token_entropy / logistic_regression baselines
  are unaffected by our λ change, so their numbers are unchanged and were preserved.
- **CPU-only**: no LLM forward pass; eval is a linear probe over cached hidden states.

## 2. Backup (reversible)

The λ=0 baseline (model + every artefact `04_evaluate` overwrites) is preserved at
`results/setup_2_logitreg/lam0_baseline/`:
`trained_model.pt`, `train_{history,summary}.json`, `final_metrics_{ratio,strict}.csv`,
`ablation_*.csv`, `alpha_distribution.{csv,png}`, `prr_curves.png`, `mc_vs_linear.png`,
`eval_summary.json`, `reliability_diagrams/`, `token_heatmaps/`.
(Note: `*.pt` is git-ignored, so the backup model lives on disk only, not in git.)
This is also the canonical `baseline` reference for the 9.5 λ-sweep (`09e`).

## 3. Before / after (Setup-2 test, 353 sentences, m_j>0)

**Ratio-level — Ours (Bayesian):**

| metric | λ=0 (was) | λ=1e-2 (now) | Δ |
|---|---|---|---|
| ECE | 0.0667 | **0.0551** | **−0.0116** ✅ |
| Pearson r | 0.4323 | 0.4526 | +0.020 ✅ |
| binomial NLL | 1.473 | 1.327 | −0.146 ✅ |
| Brier | 0.0894 | 0.0847 | −0.005 ✅ |
| MAE | 0.2181 | 0.2237 | +0.006 (≈preserved) |
| PRR_AUC (Bayes) | 0.1394 | 0.1025 | −0.037 ⚠ (see §4) |
| epi_μ mean | 8.07e-4 | 1.70e-3 | ×2.1 ✅ |

**Strict — Ours (Bayesian):** AUROC 0.7839→0.7797 (≈same), strict ECE 0.0474→0.0518.
**MC vs Linear epistemic**: Pearson **0.966** (>0.9 checklist item ✅).
Reference baselines (unchanged): Han `factuality_probe_adapted` ratio ECE 0.0952 /
strict AUROC 0.811 / strict ECE 0.0330; `factuality_probe_original_repo` strict AUROC
0.858.

## 4. Honest notes (not regressions introduced by promotion)

- **In-domain Bayesian PRR stays low** (0.139→0.103). This is the *known* in-domain
  epistemic collapse (Phase 9.2/9.3): in-domain there is little parameter uncertainty,
  so μ̂/confidence (the Point PRR, 0.248, unchanged) ranks errors, not epi_μ. Epistemic
  matters OOD, where epi_μ rises (Phase 9.3, and the λ=1e-2 OOD re-check in
  `results/setup_2_logitreg/lam1em2/document/`). Promotion did not change this picture.
- **Strict ECE: Han adapted (0.033) is still lower than ours (0.052).** Pre-existing
  (true at λ=0 too). Han's strict probe is a dedicated binary classifier; ours is a
  ratio model. The headline calibration claim rests on **ratio-level ECE**, where ours
  (0.055) clearly beats Han adapted (0.095). See `phase_9_summary` §3 item 7 on framing.
- **Core hypothesis (ratio ECE)**: Ours 0.055 < Han adapted 0.095. ✅ (Bayesian == Point
  at ratio level by construction — same μ̂; the Bayesian/Point split shows in PRR &
  strict p_strict.)

## 5. Left as-is (deliberate)

The Phase 9 **baseline diagnostics** under `results/setup_2/` —
`epistemic_diag*.png`, `epistemic_diagnostics.json`, `logit_epistemic_validation.*`,
`ood_epistemic.*`, `temperature_sweep.*` — are **λ=0 records** referenced by the
phase_9.1–9.3 docs and are intentionally **not** regenerated. The λ=1e-2 epistemic /
OOD diagnostics live in `results/setup_2_logitreg/lam1em2/`. `baselines.json`, `logs/`,
`stamps/` are untouched.

## 6. Reproduce

```bash
# (model already at results/setup_2_logitreg/lam1em2/)
cp results/setup_2_logitreg/lam1em2/{trained_model.pt,train_history.json,train_summary.json} results/setup_2/
python scripts/04_evaluate.py --setup 2 --config configs/default.yaml --device cpu
# token_heatmaps select top-epi_μ sentences → ranking changes vs λ=0; clear stale files:
find results/setup_2/token_heatmaps -name '*.png' ! -newer results/setup_2/final_metrics_ratio.csv -delete
```
