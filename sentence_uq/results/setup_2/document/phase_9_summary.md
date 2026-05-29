# Phase 9 — Epistemic Collapse: Master Summary & Decision Log

**Date**: 2026-05-29 (last updated)
**Setup**: 2 (FActScore-Bio, in-domain). Model: Llama-3-8B-Instruct probe.
This is the index for the whole Phase 9 thread. Each step has its own findings
doc; the **open decisions** are consolidated at the bottom.

---

## 1. Document & artifact index

| Doc | Step | Headline |
|---|---|---|
| `phase_9_diagnostics.md` | 9.1 diagnose | Collapse cause = **sigmoid saturation** (logit median 21, 92% saturated), Σ̂ healthy. Tempering refuted. |
| `phase_9_2_findings.md` | 9.2-1 validate | Logit-space `epi_logit` is a **μ̂/confidence proxy** (partial Spearman −0.07). Gate FAILED → not integrated. |
| `phase_9_3_ood_findings.md` | 9.3 OOD | `epi_μ` **rises OOD** (×2.77 mean, p<1e-3) → genuine epistemic; in-domain≈0 is correct. §7: temperature sweep (post-hoc fails, retraining lever validated). |
| `phase_9_4_saturation_remedy_options.md` | 9.4 design | Three retraining options compared; **Option A (logit L2 penalty)** recommended. |
| `phase_9_5_logit_reg_results.md` | 9.5 retrain | λ sweep; **λ=1e-2 sweet spot** (epi_μ ×3, ECE improved); ĝ↔Σ̂ coupling caps recovery; OOD still rises. |

**Data artifacts**: `epistemic_diagnostics.json`, `logit_epistemic_validation.json`,
`ood_epistemic.{json,png}`, `temperature_sweep.{json,png}`, `logitreg_compare.json`,
`epistemic_diag_*.png` (in `results/setup_2/`), and the λ models under
`results/setup_2_logitreg/lam*/`.
**Scripts**: `09_diagnose_epistemic.py`, `09b_validate_logit_epistemic.py`,
`09c_ood_epistemic.py`, `09d_temperature_sweep.py`, `09e_compare_logitreg.py`.
**Code changed**: `src/models/bayesian_main.py` + `scripts/03_train.py`
(`logit_reg_lambda`).

---

## 2. Settled decisions

- **Posterior tempering (orig. Phase 9 §5.1): DROPPED.** Premise refuted — Σ̂ is
  O(1), not over-concentrated; a global scalar leaves PRR ranking invariant.
- **Logit-space `epi_logit`: NOT adopted as epistemic.** Failed the partial-
  correlation gate (confidence proxy) and dropped OOD on the baseline model.
- **Root cause = sigmoid saturation**, fixed by a **logit L2 penalty** in training.
- **Operating point: λ = 1e-2.** epi_μ ×2–3, ECE improved (0.067→0.055), MAE/Pearson/
  AUROC preserved, logits normalised (median 21→2.5), OOD rise preserved (×1.39, p≈5e-5).

---

## 3. OPEN decisions / next steps (nothing actioned beyond this)

1. **Promote λ=1e-2 model to production `results/setup_2/`?** — *deferred.* Would
   require re-running all downstream eval / baselines / paper figures. (`phase_9_5` §4)
2. **Epistemic magnitude is still modest** (std ~0.04; few % of total uncertainty).
   The ĝ↔Σ̂ coupling caps de-saturation gains; larger gains need a different
   parameterisation, not just more λ. — open research question. (`phase_9_5` §2,§4;
   `phase_9_3` §4)
3. **`epi_logit` re-validation on the λ=1e-2 model** — it now *rises* OOD (it dropped
   on baseline); if it is ever to be used, re-run the 9.2-1 partial-correlation gate
   on this model. (`phase_9_5` §3)
4. **Strengthen the OOD evidence** — current OOD pilot is 15 prompts, single topic
   (computer-security), annotation-free. Optional: multi-topic OOD set; annotate the
   OOD pilot to test OOD `epi_μ` vs OOD error correlation (API cost). (`phase_9_3` §5)
5. **Paper section draft** — framing: in-domain epi≈0 is correct; epi_μ rises OOD;
   logit regularisation recovers magnitude + improves calibration. — not started.

6. **Fix the prior (μ₀, σ₀) as hyperparameters instead of learning them** — *idea,
   not implemented.* Today `mu_0` and `log_sigma_0` are learnable `nn.Parameter`s
   (`src/features/extractor.py`), i.e. empirical Bayes. Fixing them (μ₀=0, σ₀=const)
   is arguably **more honest** — a prior learned from the same data it regularises
   invites the "is that really a prior?" critique, which matters for a UQ paper.
   Connects to our findings: learned σ₀ drifted 1.0→1.2 (0.97 at λ=1e-2); freezing it
   removes a degree of freedom and makes the Σ̂ interpretation cleaner. Expected fit
   impact small (the prior params barely moved). Implementation: `register_buffer` or
   `requires_grad=False` + exclude from optimiser, gated by a `learn_prior` flag; value
   set via the existing `prior_sigma_init`. Suggested as a clean ablation
   (baseline + λ=1e-2 with frozen vs learned prior).

7. **Han baseline classifier — already linear; decide whether to also report XGBoost**
   — *fact-check done, no change needed; reporting choice open.* Finding: every Han run
   on our data used **logistic regression (linear), NOT XGBoost** —
   `_han_on_ours/test_results/...csv` → `classifier=logistic_regression`, **AUROC≈0.781**
   (best C0.5, layer14); our in-repo `src/baselines/factuality_probe.py` is also
   `LogisticRegression`. So we are *already* on the weaker linear variant, and that is
   the **principled** choice (matched model class — our method is a linear probe; Han-
   XGBoost would be an unfair nonlinear comparison). Numbers: Han-logistic 0.781 vs our
   error-detection AUROC 0.785 (≈tied). **Integrity note**: Han's paper reports XGBoost
   too, so silently omitting it reads as cherry-picking; recommend headline = Han-
   logistic (fair, linear↔linear) **and** report Han-XGBoost in an appendix/ablation as
   a stronger nonlinear reference. Lean the contribution on UQ/calibration (ECE) and the
   epistemic decomposition, not raw AUROC (where probe-class methods naturally tie).

---

## 4. Status

All experiments through 9.5 are **done and recorded**. Items in §3 are **open and not
yet actioned** — awaiting direction.
