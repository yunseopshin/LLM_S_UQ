# Phase 13 — Where does epistemic actually add value? (Conditional error + Interval coverage)

**Status**: TODO.
**Premise**: prior phases showed `epi_mu` fails as a standalone error/OOD ranker (Phase 11-B,
w-gating). Reason: every earlier test was solvable by the point estimate `mu_hat` alone (low
`mu_hat` => predict false), leaving no room for the *variance*. This phase asks the only
question where variance can matter: **does the uncertainty of the per-sentence factuality
probability add value where `mu_hat` is ambiguous, and is that uncertainty calibrated?**

Two settings, one narrative:
- **Setting 1 (usefulness)**: in `mu_hat`-ambiguous bands, does `epi_mu` separate correct from
  incorrect sentences *beyond* what `mu_hat` already explains?
- **Setting 2 (validity)**: are the model's binomial credible intervals on `U_j = K_j/m_j`
  calibrated, and does including epistemic widen them usefully?

**No retraining. No new generation/annotation.** Runs entirely on the existing annotated
Setup-2 test split (353 sentences with `m_j>0`) and `results/setup_2/trained_model.pt`. This is
deliberately the *least* epi-favourable arena (in-domain, data-informed posterior, smallest
`epi_mu`): a positive result here is strong; a negative is consistent with prior findings and we
pivot to the calibration framing with confidence.

---

## 0. Hard Constraints (read first)

1. **No retraining, no model change.** Load `results/setup_2/trained_model.pt`.
2. **Additive only.** New script + (if needed) additive helpers in `metrics.py`. Do NOT modify
   `04_evaluate.py`, the inner loop, the posterior, `predict.py` return values, or any baseline.
   `tests/test_fisher_scoring.py`, `test_decomposition.py`, `test_metrics.py` pass unchanged.
3. **English only, plain-ASCII math** (`epi_mu`, `mu_hat`, `w = mu(1-mu)`, `K_j`, `m_j`,
   `U_j = K_j/m_j`, `A_j`), no Unicode combining marks.
4. **Reuse** `partial_correlation_gate` (added in Phase 10-2), `compute_bootstrapped_ci`,
   `plot_reliability_diagram`, and the per-signal table format of
   `scripts/diag_ood_error_detection.py`.

Per-sentence quantities (from `Predictor.predict_sentence` + `predict_mc_epistemic`):
`mu_hat`, `epi_mu` (delta `g_hat^T Sigma_hat g_hat`), `epi_mc` (MC latent variance),
`aleatoric_U`, `total_U`; labels `U_j = K_j/m_j`, `A_j = 1{K_j=m_j}`,
`err = 1 - A_j` (binary strict error), `abs_ratio_err = |U_j - mu_hat|`.

---

## Setting 1 — Conditional error-detection in `mu_hat`-ambiguous bands

**Idea**: confident regions (`mu_hat` near 0 or 1) are already solved by the point estimate.
Restrict to ambiguous bands and ask whether `epi_mu` adds discriminative power there, **after
controlling for `mu_hat`** so we do not just rediscover `mu_hat` through the `w = mu(1-mu)`
entanglement.

### 1.1 Band set (config-driven; this is the default)

```
bands:
  asymmetric_lower_sweep: [[0.2,0.7],[0.3,0.7],[0.4,0.7],[0.5,0.7],[0.5,0.8]]
  narrow_symmetric:       [[0.4,0.6],[0.45,0.55]]   # w ~ const here => clean epi signal
  upper_overconfident:    [[0.5,0.9],[0.6,0.95]]    # confident-but-wrong hallucinations
```
- `asymmetric_lower_sweep` tests robustness to the lower cut.
- `narrow_symmetric` is the **decisive** group: within it `mu_hat` (hence `w`) is ~constant, so
  any `epi_mu` variation is genuine (`g_hat` direction + `Sigma_hat`), not a `mu_hat` proxy.
- `upper_overconfident` probes whether `epi_mu` catches confident hallucinations.

### 1.2 Per-band report (one row per band)

For sentences with `mu_hat in band`:
- `N`, `base_err_rate = mean(err)`
- `AUROC(err, epi_mu)` + bootstrap 95% CI (`compute_bootstrapped_ci`)
- `AUROC(err, epi_mc)` + CI
- `partial_corr(err, epi_mu | mu_hat)` rho + p-value + sign (reuse `partial_correlation_gate`;
  it residualises on `[1, mu_hat, mu_hat^2]`)
- contrast: `AUROC(err, mean_entropy)`, `AUROC(err, -mu_hat)`

Skip bands with `N < 25` or `base_err_rate` outside `[0.1, 0.9]` (AUROC unstable); print why.

### 1.3 Reading rule (write this into the output)

- Signal present in WIDE bands but **gone in `narrow_symmetric`** => it was `mu_hat` leaking
  through `w`; **reject** the Setting-1 claim.
- Signal **survives in `narrow_symmetric`** with `partial_corr` significant (p<0.05, higher epi
  => higher err) => genuine conditional epistemic value; **accept**.
- `upper_overconfident` is reported as a separate, secondary finding.

---

## Setting 2 — Binomial credible-interval coverage + sharpness

**Idea**: express "uncertainty of the factuality probability" as a predictive interval on
`U_j = K_j/m_j` and test (a) whether it is calibrated and (b) whether epistemic widens it
usefully. This is a calibration question, independent of error-ranking.

### 2.1 Predictive intervals — discrete binomial, NOT Gaussian +/- z*sigma

`K_j` is a bounded count; `U_j in [0,1]`. Build the predictive distribution of `K_j` (then map to
`U_j`) at nominal levels `{0.50, 0.80, 0.90, 0.95}` via **equal-tailed** quantiles of the
predictive PMF. Three variants:

- **(b) aleatoric-only** (fix theta = theta_hat): `K_j ~ Binomial(m_j, mu_hat)`.
- **(c) aleatoric + epistemic** (full posterior-predictive): MC over the posterior —
  for `s = 1..S` (S=500): sample `theta^(s) ~ N(theta_hat, Sigma_hat)`, compute `mu^(s) =
  mu_j(theta^(s))`, sample `K^(s) ~ Binomial(m_j, mu^(s))`; the empirical distribution of
  `{K^(s)}` is the predictive. (Reuse the theta-sampling already inside `predict_mc_epistemic`;
  if it does not expose K-samples, add an additive helper that samples K — do NOT change
  existing return values.)
- (a) point is degenerate (zero width); include only as a sanity reference, not a coverage row.

Headline level = **95%** (multi-level curve for support, per the design decision).

### 2.2 Metrics

- **Coverage**: fraction of sentences with `K_j` inside the nominal interval, per level, for (b)
  and (c). Plot nominal-vs-empirical **interval reliability diagram** (both variants overlaid).
- **Sharpness**: mean interval width (on the `U_j` scale) for (b) and (c). Lower is better at
  matched coverage.
- **Adaptivity (the epistemic-specific test)**: stratify by `epi_mu` tercile (low/mid/high).
  In the HIGH-epi stratum, does (c) widen relative to (b) and does that *improve* coverage toward
  nominal? Report coverage and width per tercile for both variants.
- **Decomposition honesty**: report mean `aleatoric_U` vs mean `epi_mu` overall and per band —
  in mid-`mu` the width is aleatoric-dominated; state the epi share explicitly so its
  contribution is not overstated.

### 2.3 Reading rule

- (c) coverage closer to nominal than (b) — especially in the high-epi stratum — => epistemic
  adds calibrated interval value. **accept**.
- (b) ~= (c) everywhere (expected if posterior is collapsed in-domain) => epistemic adds
  negligible interval value here; report honestly. Optional fallback (out of scope this phase):
  rerun Setting 2 on a weak-OOD set (e.g. obscure-real entities) where `epi_mu` is larger.

---

## Files

| Area | File |
|---|---|
| Main experiment (Setting 1 + Setting 2) | `scripts/13_epi_conditional_coverage.py` (new) |
| K-sample predictive helper (if needed) | additive function in `src/evaluation/metrics.py` |

Outputs under `results/setup_2/conditional_epi/`:
- `band_sweep.csv` (one row per band: N, base_err, AUROCs+CI, partial_corr, contrasts)
- `coverage.csv` (level x variant x coverage x mean_width)
- `interval_reliability.png` (nominal vs empirical, b vs c)
- `sharpness_by_epi_tercile.csv`
- `setting1_verdict.txt`, `setting2_verdict.txt` (the reading-rule outcome, explicit)

Reuse: `load_trained_model`, `Predictor.predict_sentence`/`predict_mc_epistemic`,
`partial_correlation_gate`, `compute_bootstrapped_ci`, `plot_reliability_diagram`,
`compute_token_entropy_baseline`; mirror `diag_ood_error_detection.py` for tables.

---

## Tests (`tests/test_conditional_epi.py`)

1. **Band masking**: a sentence with `mu_hat=0.52` is in `[0.4,0.6]` and `[0.5,0.7]`, not in
   `[0.6,0.95]`; empty/`N<25` bands are skipped without error.
2. **Setting-1 synthetic**: data where `epi` separates `err` within a fixed-`mu` band but is pure
   noise across bands => narrow-band AUROC ~1 and `partial_corr` significant; a w-leak construct
   (epi := mu(1-mu)) => narrow-band partial_corr NOT significant. Pins the reading rule.
3. **Predictive interval**: for `Sigma_hat -> 0`, variant (c) collapses to variant (b) within
   tolerance (no epistemic => same interval).
4. **Coverage monotonicity**: empirical coverage is non-decreasing in nominal level; widths
   non-decreasing in level; all `U` intervals lie in `[0,1]`.
5. **Adaptivity**: on a construct where high-epi sentences are under-covered by (b), variant (c)
   raises their coverage toward nominal.
6. **No-retrain guard**: saved parameters unchanged (hash); full suite passes.

---

## Acceptance / Go-No-Go

**Setting 1** — primary verdict from the `narrow_symmetric` bands:

| band | N | base_err | AUROC(epi_mu) [CI] | partial_corr (p) | AUROC(entropy) | AUROC(-mu) |
|---|---|---|---|---|---|---|
| [0.45,0.55] | ... | ... | ... | ... | ... | ... |
| [0.4,0.6] | ... | ... | ... | ... | ... | ... |

PASS iff a narrow band shows `AUROC(epi_mu)` CI excluding 0.5 AND significant `partial_corr`
(correct sign). Wide-band-only signal => mu_hat residue => not a pass.

**Setting 2** — coverage at 95% and high-epi adaptivity:

| variant | cov@95 (all) | cov@95 (high-epi) | mean width |
|---|---|---|---|
| (b) aleatoric-only | ... | ... | ... |
| (c) aleatoric+epi | ... | ... | ... |

PASS iff (c) is closer to nominal than (b), most visibly in the high-epi stratum. If (b)~=(c),
report epistemic as adding negligible in-domain interval value (honest negative).

**Both decisive.** A positive in this (worst-case in-domain) arena revives the UQ contribution
on a well-posed target: "epi is a second-order signal that helps exactly where the point
estimate is ambiguous, and yields calibrated intervals." A negative is the strongest possible
version of the prior findings and justifies pivoting cleanly to the binomial-calibration +
single-pass framing.

---

## Out of scope (do not touch this phase)

- Retraining; temperature/focal de-saturation; SNGP/distance-aware; ensemble/MC-dropout.
- Any OOD or new-entity generation (the optional weak-OOD fallback for Setting 2 is a *future*
  phase, only if Setting 2 is null and the team wants to see epi where it is larger).
- `04_evaluate.py`, inner loop, posterior, `likelihood.py`, prior treatment, `predict.py`
  return values.
