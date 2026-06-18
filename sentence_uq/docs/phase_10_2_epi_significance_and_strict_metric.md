# Phase 10-2 — Epistemic Significance Diagnostic + Strict-Metric Decoupling

**Status**: DONE — both parts implemented, tested, and run on `results/setup_2`.
**Setup**: 2 (FActScore-Bio in-domain), existing trained model
(`training.logit_reg_lambda = 1e-2`, saturation fraction ~0.42). **No retraining.**
**Spec**: `prompts/phase_10_2_epi_significance_and_strict_metric.md`.

Two independent, cheap, additive tasks that run off the existing trained model
and cached test predictions:

- **Part A** — quantify whether `epi_mu` carries *significant* information (not
  just whether its magnitude is large), via a partial-correlation gate.
- **Part B** — fix strict-factuality reporting so that **ranking** (AUROC) and
  **calibration** (ECE/Brier) use the correct, separate quantities, plus an
  optional 1-parameter recalibration `gamma`.

---

## 1. Executive Summary

- **Part A verdict: `PARTIAL_CORR_GATE: FAIL`.** On the de-saturated setup-2
  model, `epi_mu` does **not** predict error after controlling for `mu_hat`
  (partial Spearman ρ=0.049, p=0.36; logistic coef −0.28, p=0.19, wrong sign).
  It adds no rejection value beyond the point estimate (PRR-AUC 0.136 vs 0.131).
- **Part A `MC == delta`** (median ratio 0.995, Pearson 0.99): the smallness of
  `epi_mu` is **structural**, not a first-order-approximation artefact. Per the
  spec branch this says *do not chase magnitude* — pivot the epistemic story to
  **OOD / distance-aware** (a separate phase). Corroborates Phase 11-B
  ([`phase_11_B_ood_epistemic_vs_baselines.md`](phase_11_B_ood_epistemic_vs_baselines.md)).
- **Part B** decouples the strict table: AUROC is now reported per candidate
  ranker; ECE/Brier are reported only on `mu_hat^{m_j}` (and `mu_hat^{gamma m}`),
  **never on raw `mu_hat`** (type mismatch). The headline ranker is chosen on the
  **val** split and reported on **test**: `softmin` wins, AUROC **0.826**.
- **Hard constraint honoured.** All new behaviour is additive / optional;
  `tests/test_fisher_scoring.py`, `tests/test_decomposition.py`,
  `tests/test_metrics.py` pass **unchanged**. Legacy strict columns are
  bit-identical; the 19 new columns are NaN for baselines. 357/357 tests pass.

---

## 2. What was built

| Area | Files |
|---|---|
| `partial_correlation_gate(...)` + IRLS logistic w/ Wald p-values | `src/evaluation/metrics.py` |
| `fit_strict_gamma(...)` (1-param strict-NLL recalibration on val) | `src/evaluation/metrics.py` |
| `compute_strict_factuality_metrics` refactor (ranking vs calibration) | `src/evaluation/metrics.py` |
| Significance diagnostic CLI (read-only w.r.t. the model) | `scripts/05_epi_significance.py` (new) |
| Candidate rankers, val-selected headline, gamma block, `_softmin` | `scripts/04_evaluate.py` |
| Decoupling + gate + gamma + min/softmin unit tests | `tests/test_strict_metric_decoupling.py` (new) |

All numerics run in float64 NumPy; the gate's logistic fit is a dependency-free
Newton-Raphson (IRLS) with a tiny ridge for separable designs (no statsmodels).

---

## 3. Part A — Epistemic significance diagnostic

`scripts/05_epi_significance.py` loads `trained_model.pt` + the setup-2 test
split (353 sentences, m_j>0) and computes per sentence: `mu_hat`, `epi_mu`
(delta `g_hat^T Sigma_hat g_hat`), `epi_mc` (MC latent variance of `mu`,
`num_samples=200`), `ratio_err = |U_j - mu_hat|`, `strict_wrong = 1 - A_j`.

Output: `results/setup_2/epi_significance.csv` (one row per diagnostic) +
`results/setup_2/epi_significance_epi_mu.npy` (so a second run can serve as the
conditional OOD comparison set).

### 3.1 The gate (PRIMARY)

`partial_correlation_gate` runs two checks and PASSES iff `epi_mu` has the
correct sign (higher epi → larger error) AND p<0.05 in **at least one**:

1. **Partial Spearman** between `residualize(epi_mu | mu_hat)` and
   `residualize(ratio_err | mu_hat)`, where `residualize(y | x)` regresses `y`
   on `[1, x, x^2]` and returns residuals.
2. **Logistic** `strict_wrong ~ 1 + z(mu_hat) + z(epi_mu)` (standardised);
   report the `epi_mu` coefficient, its Wald p-value and sign.

### 3.2 Results (`results/setup_2`)

| diagnostic | value | note |
|---|---|---|
| **`PARTIAL_CORR_GATE`** | **FAIL** | primary |
| partial Spearman (ratio) | ρ=0.049, p=0.36 | sign +, not significant |
| epi_mu logistic coef (strict) | −0.279, p=0.19 | wrong sign, not significant |
| PRR-AUC: epi_mu vs μ-confidence | 0.1359 vs 0.1313 | y=error → lower is better; epi adds **no** value |
| MC/delta median ratio, Pearson | 0.995, r=0.99 (ρ=0.99) | **MC == delta → structural** |
| OOD epi ratio | — | skipped (no `--ood-results-dir` supplied) |

`PRR-AUC` uses `y_true = ratio_err`, so a useful signal makes the remaining
*error* fall; `epi_mu` (0.136) does not beat the pure point-estimate confidence
baseline `-|mu - 0.5|` (0.131).

### 3.3 Interpretation / branch

Gate **FAIL** and **MC ≈ delta** → in-distribution epistemic is structurally
weak (this is the *correct* Bayesian answer when the posterior is data-informed;
near-zero `epi_mu` is not an approximation error — MC confirms it). Per the spec:
**do not chase magnitude; pivot the epistemic narrative to OOD / distance-aware**
(a separate phase). This is consistent with Phase 11-B's structural finding that
`epi_mu = g_hat^T Sigma_hat g_hat` is gated by the confidence term `w = mu(1-mu)`.

---

## 4. Part B — Strict-metric decoupling (ranking vs calibration)

**Problem fixed.** AUROC is a ranking metric (calibration-invariant); ECE/Brier
are calibration metrics. The model-consistent estimate of `P(A_j = 1)` is
`mu_hat^{m_j}`, **not** raw `mu_hat`; feeding raw `mu_hat` into strict ECE is a
type mismatch that inflates ECE. Conversely `mu_hat^{m_j}` can rank *worse* than
`mu_hat` because the `m_j` power amplifies residual per-token over-confidence. So:
**calibrate on `mu_hat^{m_j}`, but report ranking for several candidate scores.**

### 4.1 Refactored signature (backward-compatible)

```
compute_strict_factuality_metrics(A_true, p_calib, uncertainty=None, ranking_score=None)
```
- Brier/ECE computed on `p_calib` (must estimate `P(A_j=1)`).
- AUROC/AUPRC computed on `ranking_score` (defaults to `p_calib`).

**Argument-order note.** The spec lists `ranking_score` ahead of `uncertainty`.
Doing that literally would break the legacy 3-positional call
`compute_strict_factuality_metrics(A, p, uncertainty)` used by
`tests/test_metrics.py` (the old 3rd positional `uncertainty = 1 - p` would
become the ranking score and flip AUROC 1.0 → 0.0). Hard constraint 0.1
("`test_metrics.py` must pass unchanged") is binding, so `uncertainty` keeps its
historical 3rd-positional slot and `ranking_score` is passed by keyword at the
new call sites. The functional contract is exactly as specified.

### 4.2 Candidate rankers (all estimates of `P(A_j=1)`, higher → more factual)

- `score_mu      = mu_hat`                  — strong ranker, bad calibration
- `score_mu_pow  = mu_hat ** m_j`           — model-consistent; the calibration target
- `score_min     = min_l token_pi[l]`       — weakest-link, matches AND-semantics of `A_j`
- `score_softmin = -(1/beta) logsumexp(-beta * token_pi)`, `beta=10` — smooth weakest-link

The **headline ranker** is whichever has the highest AUROC on the **validation**
split (425 positive-m_j sentences); it is reported on **test** with a bootstrapped
95% CI. `gamma` is fit on val by minimising strict NLL of `mu_hat^{gamma m_j}`.

### 4.3 Results (`results/setup_2`, Ours rows)

| ranking score | AUROC | Brier | ECE |
|---|---|---|---|
| ranking = `mu` | 0.8269 | (n/a) | (n/a) |
| ranking = `mu^m` | 0.7844 | 0.0679 | 0.0505 |
| ranking = `min` | 0.7386 | (n/a) | (n/a) |
| ranking = `softmin` (β=10) | **0.8263** | (n/a) | (n/a) |
| recalibrated `mu^{γ m}` (γ̂=0.610) | 0.7844 | 0.0801 | 0.0841 |

- **Headline ranker = `softmin`**, selected on val, reported on test: AUROC
  **0.8263** (95% CI 0.759–0.884). `strict_headline_basis = val`.
- **Calibration headline = `mu^m`**: ECE **0.0505**, Brier 0.0679. ECE is never
  reported on raw `mu` (type mismatch; the column was not added).
- **`gamma` recalibration** (γ̂=0.610, fit on val strict NLL) did **not** help
  here: ECE rose 0.0505 → 0.0841, so the `mu^m` block stays the calibration
  headline. Both blocks are reported so the trade-off is visible (the γ transform
  is monotone per sentence but `m_j`-dependent across sentences, so it moves both
  ranking and calibration). A small γ<1 is consistent with the Phase 10-1
  Beta-Binomial `rho ≈ 0` (over-dispersion fit on counts came out negligible);
  γ here is fit on the strict event `A_j` directly and softens the length penalty.

The legacy headline AUROC (ranking by `mu_hat`, the already-committed
Phase 11-A fix) is unchanged at **0.827**; the stale CSV that predates that fix
showed 0.78 under `mu^m` ranking and lacked the `AUROC_pstrict` column.

---

## 5. Part C — Tests

`tests/test_strict_metric_decoupling.py` (11 tests):

1. **Back-compat** — `(A, p)` and legacy `(A, p, uncertainty)` reproduce the
   pre-refactor numbers (AUROC scored on `p`).
2. **Decoupling** — constructed case where `mu` ranks `A` perfectly but `mu^m`
   reorders: `AUROC(mu) > AUROC(mu^m)` while `ECE(mu^m) < ECE(mu)`.
3. **gamma fit** — recovers a known `gamma_true=0.4` (±0.08); `gamma→1`
   reproduces plain `mu^m`; empty val returns 1.0.
4. **partial_correlation_gate** — noise `epi` → FAIL; `epi` constructed to track
   the residual error → PASS.
5. **min/softmin** — `softmin` with large `beta` approaches `min` within
   tolerance; both lie in `[0,1]`; softmin is a smooth lower envelope.

Plus the spec-pinned `test_fisher_scoring`, `test_decomposition`, `test_metrics`,
`test_evaluate_script` all pass unchanged. Full suite: **357 passed**.

---

## 6. Verdict & next steps

- **Part A is an honest negative**: in-distribution `epi_mu` is not significant,
  and MC confirms the smallness is structural. The significance claim should be
  made with **relative framing** only if it later passes OOD (the gate is the
  reusable test). Out-of-scope levers (per-token temperature / focal de-saturation
  in Part E; distance-aware/SNGP epistemic) remain for a separate retraining phase
  if the team wants to attempt to flip the gate.
- **Part B stands on its own**: the strict table now reports ranking and
  calibration with the correct, separate objects. Headline strict AUROC **0.826**
  (`softmin`), strict ECE **0.052** on `mu^m` (≪ Han 0.17) — the binomial strict
  story is the strong, reportable result.

### Reproduce

```bash
# Part A — significance gate
python scripts/05_epi_significance.py --setup 2 --config configs/setup_2.yaml --mc-samples 200

# Part B — full evaluation incl. strict decoupling (writes the strict table)
python scripts/04_evaluate.py --setup 2 --config configs/setup_2.yaml
```

Artefacts: `results/setup_2/epi_significance.csv`,
`results/setup_2/epi_significance_epi_mu.npy`,
`results/setup_2/final_metrics_strict.csv` (19 new `strict_*` / `gamma_hat`
columns on the Ours rows).
