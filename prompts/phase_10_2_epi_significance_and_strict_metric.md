# Phase 10-2 — Epistemic Significance Diagnostic + Strict-Metric Decoupling

Two independent, cheap, **no-retraining** tasks on the current trained model
(`training.logit_reg_lambda = 1e-2`, saturation fraction ~0.42):

- **Part A**: quantify whether `epi_mu` carries *significant* information (not just whether
  its absolute magnitude is large), on the de-saturated checkpoint.
- **Part B**: fix the strict-factuality reporting so that ranking (AUROC) and calibration
  (ECE) use the correct, separate quantities, and add an optional 1-parameter recalibration.

Both are additive. **Do not retrain.** Both run off the existing trained model and the cached
test predictions in `results/setup_2/`.

---

## 0. Hard Constraints (read first)

1. **Do NOT change the Binomial training path, the inner Fisher-scoring loop, the posterior,
   or any existing numerical output.** `tests/test_fisher_scoring.py`,
   `tests/test_decomposition.py`, `tests/test_metrics.py` must pass **unchanged**.
2. All new behavior is **additive** and selected by new optional arguments / new functions.
   Existing call sites keep their current behavior when the new args are left at defaults.
3. English only. Plain-ASCII math inside code and docstrings (`mu_j`, `g_hat`, `epi_mu`,
   `p_strict`, `gamma`), no Unicode combining marks.
4. Minimal change: patch existing files where possible; only create new files for the new
   diagnostic script and its test.

---

## Part A — Epistemic Significance Diagnostic

**Goal**: answer "is `epi_mu` informative?", framed as significance, not magnitude. The hard
gate is the **partial-correlation test**: does `epi_mu` predict error *after* controlling for
`mu_hat`?

### A.1 New script `scripts/05_epi_significance.py`

Read-only w.r.t. the model. Load the trained model and the test set (Setup 2). Accept
`--experiment` and `--results_dir` (default `results/setup_2`) via argparse, mirroring
`scripts/04_evaluate.py`.

For every test sentence compute (reuse `src/inference/predict.py`):
- `mu_hat`, `epi_mu` (delta-method `g_hat^T Sigma_hat g_hat`) from `Predictor.predict_sentence`.
- `epi_mc` from `Predictor.predict_mc_epistemic(z_tokens, num_samples=200, m_j=m_j)` (latent-level
  MC variance of `mu`).
- Errors: `ratio_err = abs(U_j - mu_hat)` where `U_j = K_j / m_j`; `strict_wrong = 1 - A_j`
  where `A_j = 1{K_j == m_j}`.

Then compute and write the following four diagnostics.

**(1) Partial-correlation gate (PRIMARY).** Does `epi_mu` add predictive power beyond `mu_hat`?
- Partial Spearman: `rho_partial = spearman( residualize(epi_mu | mu_hat),
  residualize(ratio_err | mu_hat) )`, where `residualize(y | x)` regresses `y` on
  `[1, x, x^2]` and returns residuals. Report `rho_partial` and its p-value.
- Also a logistic check on the strict target: fit `strict_wrong ~ 1 + mu_hat + epi_mu`
  (standardize predictors) and report the `epi_mu` coefficient, its p-value, and sign.
- **PASS** iff `epi_mu` has the correct sign (higher `epi_mu` -> larger error) AND p < 0.05 in
  at least one of the two checks. Print `PARTIAL_CORR_GATE: PASS|FAIL` explicitly.

**(2) PRR-AUC.** Use existing `compute_prr(y_true=ratio_err, uncertainty=epi_mu)`. Report
`prr_auc`. Compare against `compute_prr(..., uncertainty=mu_hat_confidence)` where
`mu_hat_confidence = -abs(mu_hat - 0.5)` (a pure point-estimate rejection baseline) to show
`epi_mu` adds rejection value beyond the mean.

**(3) MC vs delta sanity check.** Report Pearson and Spearman correlation between `epi_mc` and
`epi_mu`, and the median ratio `epi_mc / epi_mu`. Interpretation to print:
- `MC ~= delta` (ratio ~1, high corr): smallness is real (structural), not an approximation
  artifact -> if the gate also fails, recommend pivoting the epistemic story to OOD /
  distance-aware.
- `MC >> delta` (ratio >> 1): first-order approx underestimates -> recommend reporting `epi_mc`
  as the rejection signal.

**(4) OOD ratio (conditional).** If an OOD prediction set is available (e.g. a second
`--ood_results_dir`), report `median(epi_mu_ood) / median(epi_mu_indomain)` and a
Mann-Whitney U p-value. If not available, skip with a printed note; do not fail.

### A.2 Output

- `results/setup_2/epi_significance.csv`: one row per diagnostic with the numbers above.
- Console: a compact summary table plus the explicit `PARTIAL_CORR_GATE: PASS|FAIL` line and
  the MC-vs-delta interpretation line.

### A.3 Reuse, do not reimplement

`compute_prr` and `compare_mc_vs_linear_epistemic` already exist in
`src/evaluation/metrics.py`. Use them. Add `partial_correlation_gate(...)` as a small helper in
`metrics.py` (additive) so it is unit-testable.

---

## Part B — Strict-Metric Decoupling (ranking vs calibration)

**Problem being fixed**: AUROC is a ranking metric (calibration-invariant); ECE/Brier are
calibration metrics. The model-consistent estimate of `P(A_j = 1)` is `mu_hat^{m_j}`, NOT raw
`mu_hat`. Feeding raw `mu_hat` into strict ECE is a type mismatch and inflates ECE. Conversely
`mu_hat^{m_j}` can rank worse than `mu_hat` because the `m_j` power amplifies residual
per-token over-confidence. So: **calibrate on `mu_hat^{m_j}`, but report ranking for several
candidate scores.**

### B.1 Refactor `compute_strict_factuality_metrics` (additive, backward-compatible)

Current signature:
`compute_strict_factuality_metrics(A_true, p_strict, uncertainty)` -> {AUROC, AUPRC, Brier, ECE}.

Change to:
```
compute_strict_factuality_metrics(
    A_true,
    p_calib,            # probability used for Brier/ECE; must estimate P(A_j=1)
    ranking_score=None, # score used for AUROC/AUPRC; if None, use p_calib (back-compat)
    uncertainty=None,   # if None, derived as 1 - ranking_score
)
```
- Brier/ECE computed on `p_calib` exactly as before.
- AUROC/AUPRC computed on `ranking_score` (default = `p_calib`, so existing callers and
  `tests/test_metrics.py` are unchanged: pass `p_strict` positionally and everything matches).
- AUROC orientation: `roc_auc_score(A_true, ranking_score)` (higher score -> more likely
  `A_j = 1`). Keep the existing convention.

### B.2 Candidate ranking scores in `scripts/04_evaluate.py`

For our method, compute these per-sentence ranking scores (all are estimates of `P(A_j=1)`,
higher -> more factual). `token_pi` is already returned by `predict_sentence`.
- `score_mu      = mu_hat`                         (raw mean; strong ranker, bad calibration)
- `score_mu_pow  = mu_hat ** m_j`                  (model-consistent; the calibration target)
- `score_min     = min_l token_pi[l]`              (weakest-link; matches AND-semantics of A_j)
- `score_softmin = -(1/beta) * logsumexp(-beta * token_pi)`, beta = 10.0  (smooth weakest-link)

In the strict table, report **one calibration block** (Brier/ECE) on `score_mu_pow`, and
**AUROC/AUPRC for each candidate ranker** as separate columns:
`strict_AUROC_mu`, `strict_AUROC_mupow`, `strict_AUROC_min`, `strict_AUROC_softmin`
(+ matching AUPRC). Keep bootstrapped 95% CI (existing `compute_bootstrapped_ci`) on the
**reported headline ranker**, which is whichever of the four has the highest AUROC on the
validation split (decide on val, report on test; print which one was selected).

**Do not** report ECE/Brier on raw `mu_hat`. If a column for it currently exists, drop it and
leave a one-line comment explaining why (type mismatch: `mu_hat` is not an estimate of
`P(A_j=1)`).

### B.3 Optional 1-parameter recalibration (gamma)

Add `fit_strict_gamma(mu_hat_val, m_val, A_val)` to `metrics.py`:
- Define `p(gamma) = mu_hat ** (gamma * m_j)`.
- Fit `gamma > 0` on the **validation** split by minimizing strict NLL
  `-sum[ A*log p + (1-A)*log(1-p) ]` (1-D, `scipy.optimize.minimize_scalar`, bounds e.g.
  `(1e-3, 5.0)`; clamp `p` to `[1e-6, 1-1e-6]`).
- `gamma = 1` recovers the Binomial `mu_hat^{m_j}`. `gamma < 1` softens the length penalty.
- This is fit on the strict event `A_j` directly, so it is **distinct from** the Beta-Binomial
  `rho` (which was fit on counts and came out ~0). A small `gamma` here is consistent with that
  negative result.

In `04_evaluate.py`, report a second calibration block `strict_ECE_gamma`, `strict_Brier_gamma`,
`strict_AUROC_gamma` on `mu_hat ** (gamma_hat * m_j)` and print `gamma_hat`. Note that since
the gamma transform is monotone in `mu_hat` *per sentence* but `m_j`-dependent across sentences,
it can move both calibration AND ranking — report both so the trade-off is visible.

---

## Part C — Tests

Add `tests/test_strict_metric_decoupling.py` and extend the epi diagnostic test.

1. **Back-compat**: `compute_strict_factuality_metrics(A, p_strict)` (positional, no
   `ranking_score`) returns the same numbers as before the refactor. `tests/test_metrics.py`
   passes unchanged.
2. **Decoupling**: with a constructed case where `mu_hat` ranks `A` perfectly but `mu_hat^{m}`
   does not (vary `m` so the power reorders), assert `AUROC(ranking=mu_hat) >
   AUROC(ranking=mu_hat^m)` while `ECE(p_calib=mu_hat^m) < ECE(p_calib=mu_hat)`.
3. **gamma fit**: on synthetic data generated with a known `gamma_true`, `fit_strict_gamma`
   recovers it within a tolerance; `gamma -> 1` reproduces the plain `mu_hat^{m_j}` numbers.
4. **partial_correlation_gate**: on synthetic data where `epi` is pure noise w.r.t. error,
   the gate returns FAIL; where `epi` is constructed to correlate with the residual error, it
   returns PASS.
5. **min/softmin**: `score_softmin` with large `beta` approaches `score_min` within tolerance;
   both lie in `[0,1]`.

---

## Part D — Acceptance / Go-No-Go

Run Part A on `results/setup_2`, then Part B reporting. Produce one summary the human reads:

**Part A:**
| diagnostic | value | note |
|---|---|---|
| `PARTIAL_CORR_GATE` | PASS / FAIL | primary |
| partial Spearman (ratio) | ... | sign + p |
| epi_mu logistic coef (strict) | ... | sign + p |
| PRR-AUC (epi_mu) vs (mu confidence) | ... / ... | epi adds value? |
| MC/delta median ratio, Pearson | ... | structural vs approx |
| OOD epi ratio (if available) | ... | p-value |

Branch (state in the writeup):
- gate PASS -> epistemic is significant though small: report with **relative framing**
  (PRR + OOD), no further modeling needed for the significance claim.
- gate FAIL and MC ~= delta -> in-distribution epistemic is structurally weak: pivot the
  epistemic narrative to **OOD / distance-aware** (separate phase), do not chase magnitude.
- gate FAIL but MC >> delta -> switch the reported epistemic signal to `epi_mc`.

**Part B:**
| | AUROC | Brier | ECE |
|---|---|---|---|
| ranking = mu | ... | (n/a) | (n/a) |
| ranking = mu^m | ... | ... | ... |
| ranking = min / softmin | ... | (n/a) | (n/a) |
| recalibrated mu^{gamma m} (gamma=...) | ... | ... | ... |

Headline rule: **AUROC** = best ranker selected on val; **ECE/Brier** = on `mu^m` (and on
`mu^{gamma m}` if it improves ECE without hurting AUROC). Never ECE on raw `mu`.

---

## Part E — Out of scope (do not touch this phase)

- Per-token temperature scaling of the logits and focal loss. These de-saturate `mu` globally
  and have a larger blast radius (they change `mu` everywhere, hence every downstream number).
  They are the *unifying* lever for "one cause (over-confidence), two effects" and belong in a
  separate retraining phase, to be decided after Part A tells us whether the epistemic gate
  already passes at `lambda = 1e-2`.
- Distance-aware / SNGP epistemic. Only relevant if Part A's gate FAILs and MC ~= delta.
- `src/models/likelihood.py`, the inner loop, the prior treatment of `mu_0` / `log_sigma_0`.
