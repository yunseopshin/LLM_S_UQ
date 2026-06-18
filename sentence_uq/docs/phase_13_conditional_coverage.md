# Phase 13 — Where does epistemic add value? (Conditional error + Interval coverage)

**Status**: DONE — implemented, adversarially reviewed, tested, and run on
`results/setup_2` for both in-domain (Setup-2 test split) and weak-OOD (LongFact).
**Setup**: 2 (FActScore-Bio in-domain), existing trained model. **No retraining,
no new generation/annotation.**
**Spec**: `prompts/phase_13_epi_conditional_coverage.md`.

---

## 1. Executive Summary

Prior phases (10-2, 11-B) showed `epi_mu` fails as a standalone error/OOD ranker
because every earlier test was already solvable by the point estimate `mu_hat`
alone. Phase 13 asks the only two questions where the *variance* can matter, on
the existing annotated data with the existing model:

- **Setting 1 (usefulness)** — in `mu_hat`-ambiguous bands, does `epi_mu` separate
  correct from incorrect sentences *beyond* what `mu_hat` already explains
  (controlling for `mu_hat` via a partial correlation)?
- **Setting 2 (validity)** — are the model's binomial credible intervals on
  `U_j = K_j/m_j` calibrated, and does including epistemic widen them usefully?

**Result: both settings are honest negatives, in-domain AND on weak-OOD.**
`epi_mu` adds no conditional error-ranking power, and the posterior is effectively
collapsed (epistemic is ~3% of total predictive width) so the
aleatoric-only and aleatoric+epistemic intervals are indistinguishable. The one
positive carry-forward is calibration: the **single-pass binomial interval is
well-calibrated in-domain (cov@95 = 0.966)** and degrades to over-confidence
out-of-domain (cov@95 = 0.860). This is the strongest version of the prior
findings and motivates the paper's pivot to a single-pass binomial-calibration
framing rather than an epistemic-ranking claim.

This was a strictly **additive** change: a new script, four new functions in
`metrics.py`, and a new test file. No existing run, the inner loop, the posterior,
`predict.py` return values, `04_evaluate.py`, or any baseline was modified; the
trained model was loaded read-only (hash unchanged).

---

## 2. What was built

| Area | File |
|---|---|
| Main experiment (Setting 1 + Setting 2, in-domain + weak-OOD) | `scripts/13_epi_conditional_coverage.py` (new) |
| Binomial predictive-interval helpers (4 new functions) | `src/evaluation/metrics.py` (additive) |
| Tests | `tests/test_conditional_epi.py` (new, 12 tests) |

### 2.1 New `metrics.py` helpers (additive)

- `binomial_equal_tailed_interval(m_j, p, level)` — variant **(b)**: exact
  equal-tailed predictive interval for `K ~ Binomial(m_j, p)` via the binomial
  PMF quantiles, clamped to `[0, m_j]`, nested in level.
- `equal_tailed_interval_from_samples(samples, level)` — equal-tailed quantiles of
  a discrete predictive sample set (variant **(c)**).
- `sample_posterior_predictive_K(theta_hat, Sigma_hat, z_tokens, m_j, S, gen)` —
  variant **(c)** posterior-predictive: `theta^(s) ~ N(theta_hat, Sigma_hat)`
  (reusing `predict._stable_cholesky`), `mu^(s)` from the raw sigmoid mean,
  `K^(s) ~ Binomial(m_j, mu^(s))` via `m_j` Bernoulli trials with one
  reproducible generator. `predict.py` is left unchanged.
- `predictive_interval_coverage(lo, hi, K, m)` — empirical coverage and mean width
  on the `U = K/m` scale.

### 2.2 Per-sentence quantities (read-only from the trained model)

`mu_hat`, `epi_mu` (delta `g_hat^T Sigma_hat g_hat`), `epi_mc` (MC latent
variance), `aleatoric_U`, `total_U`; labels `U_j = K_j/m_j`, `A_j = 1{K_j=m_j}`,
`err = 1 - A_j`, `abs_ratio_err = |U_j - mu_hat|`. Predictions use the raw MAP
`mu_hat` (no probit shrinkage), so Setting-1 bands and Setting-2 variant (b)/(c)
are all defined on the same `mu`.

---

## 3. Setting 1 — conditional error-detection in `mu_hat`-ambiguous bands

For each band `mu_hat in [lo, hi]` we report `N`, `base_err_rate`,
`AUROC(err, epi_mu)` and `AUROC(err, epi_mc)` with bootstrap 95% CIs, the
partial-correlation gate (reusing `partial_correlation_gate`, which residualises
on `[1, mu_hat, mu_hat^2]`), and the contrasts `AUROC(err, mean_entropy)` /
`AUROC(err, -mu_hat)`. Bands with `N < 25` or `base_err_rate` outside `[0.1, 0.9]`
are skipped. The decisive group is `narrow_symmetric` ([0.4,0.6], [0.45,0.55]):
there `mu_hat` (hence `w = mu(1-mu)`) is ~constant, so any `epi_mu` signal is
genuine, not a `mu_hat` proxy.

**Reading rule:** PASS iff a narrow band has `AUROC(epi_mu)` CI excluding 0.5 AND
a significant **partial correlation** (the partial Spearman specifically — not the
broad OR of Spearman/logistic; see Section 6).

### 3.1 Results — narrow_symmetric bands

In-domain (Setup-2 test, 353 sentences):

| band | N | base_err | AUROC(epi_mu) [95% CI] | partial_rho (p) | partial_pass |
|---|---|---|---|---|---|
| [0.4, 0.6] | 60 | 0.817 | 0.540 [0.293, 0.763] | 0.100 (p=0.45) | 0 |
| [0.45, 0.55] | 34 | 0.824 | 0.405 [0.050, 0.765] | 0.125 (p=0.48) | 0 |

Weak-OOD (LongFact, 222 sentences):

| band | N | base_err | AUROC(epi_mu) [95% CI] | partial_rho (p) | partial_pass |
|---|---|---|---|---|---|
| [0.4, 0.6] | 46 | 0.783 | 0.411 [0.211, 0.617] | 0.206 (p=0.17) | 0 |
| [0.45, 0.55] | 20 | — | skipped (N<25) | — | — |

**Verdict (both arenas): REJECT.** No band — narrow or wide — shows a significant
conditional `epi_mu` signal after controlling for `mu_hat`. AUROCs straddle 0.5
and every partial correlation is non-significant.

---

## 4. Setting 2 — binomial credible-interval coverage + sharpness

Equal-tailed predictive intervals on `U_j` at nominal levels {0.50, 0.80, 0.90,
0.95} (headline 0.95), for two variants: **(b)** aleatoric-only
`K_j ~ Binomial(m_j, mu_hat)` and **(c)** full posterior-predictive (MC over
`theta`, S=500). We report coverage, mean width, an `epi_mu`-tercile adaptivity
table, and a decomposition-honesty block (mean `aleatoric_U` vs `epi_mu`, overall
and per band).

### 4.1 Coverage (in-domain, 353 sentences)

| level | (b) cov | (c) cov | (b) width | (c) width |
|---|---|---|---|---|
| 0.50 | 0.779 | 0.779 | 0.269 | 0.280 |
| 0.80 | 0.909 | 0.907 | 0.528 | 0.544 |
| 0.90 | 0.946 | 0.952 | 0.640 | 0.655 |
| **0.95** | **0.966** | **0.963** | 0.717 | 0.731 |

high-epi stratum @95: (b)=0.992, (c)=0.983. max coverage gap (c vs b) across
levels = 0.0057. **epi_share = 0.027** (epi_mu 1.70e-3 vs aleatoric_U 6.15e-2).

### 4.2 Coverage (weak-OOD LongFact, 222 sentences)

| level | (b) cov | (c) cov | (b) width | (c) width |
|---|---|---|---|---|
| 0.50 | 0.586 | 0.572 | 0.311 | 0.306 |
| 0.80 | 0.730 | 0.739 | 0.541 | 0.557 |
| 0.90 | 0.806 | 0.802 | 0.637 | 0.653 |
| **0.95** | **0.860** | **0.856** | 0.719 | 0.730 |

high-epi stratum @95: (b)=0.932, (c)=0.919. max coverage gap (c vs b) = 0.0135.
**epi_share = 0.029** (epi_mu 1.88e-3 vs aleatoric_U 6.21e-2).

### 4.3 Reading

- **In-domain:** the binomial intervals are well-calibrated (cov@95 = 0.966 vs
  nominal 0.95). The posterior is effectively collapsed: epistemic is only **2.7%**
  of total width, so (b) ~= (c) everywhere (max gap 0.006, ~1 sentence). The
  high-epi wiggle is sub-noise. **Verdict: NEGATIVE (honest).**
- **Weak-OOD:** the intervals now **under-cover** (cov@95 = 0.860 < 0.95) — the
  model is over-confident out-of-domain. Crucially, epistemic is *still* only
  **2.9%** of width (epi_mu barely grew, 1.70e-3 -> 1.88e-3, +11%), so (c) does
  **not** rescue the under-coverage; in the high-epi stratum (c) is even slightly
  worse. **Verdict: NEGATIVE (honest).**

The honest-negative `(b) ~= (c)` outcome takes precedence whenever the posterior
is collapsed (`epi_share < 5%`), so a sub-noise coverage wiggle is never reported
as a pass — this is the decomposition-honesty guard.

---

## 5. Acceptance / Go-No-Go outcome

| | In-domain | Weak-OOD (LongFact) |
|---|---|---|
| Setting 1 (conditional error) | **REJECT** | **REJECT** |
| Setting 2 (interval coverage) | **NEGATIVE (honest)** | **NEGATIVE (honest)** |
| epi share of total width | 2.7% | 2.9% |
| binomial cov@95 (variant b) | 0.966 (calibrated) | 0.860 (under-covers) |

**Interpretation.** Phase 13 is the deliberately least-epi-favourable arena
(in-domain, data-informed posterior, smallest `epi_mu`), and the weak-OOD rerun
was the last place epi could have helped. Both come back negative. This is the
strongest possible version of the prior findings: the epistemic *variance* adds no
usable conditional-error or interval-calibration value here. What survives is a
clean, defensible positive — **the single-pass binomial interval is itself
well-calibrated in-domain and degrades predictably (over-confidence) on OOD** —
which is the framing the paper should carry forward.

---

## 6. Adversarial review and the one real fix

A multi-agent adversarial review (4 dimensions x find->verify, 23 agents) found
0 critical and 14 confirmed findings. The one material bug, flagged independently
3 times:

- **Setting-1 verdict gated on the OR-gate, not the spec's partial correlation.**
  `partial_correlation_gate` returns `passed = spearman_pass OR logistic_pass`,
  but the spec's PASS rule requires the **partial Spearman** specifically (it
  residualises on `[1, mu, mu^2]`; the auxiliary logistic check controls for `mu`
  only linearly and is weaker against the `w = mu(1-mu)` leak the band exists to
  rule out). Fixed: the verdict now gates on `partial_pass` (the Spearman
  channel); `partial_pass`/`logistic_pass` are surfaced into `band_sweep.csv`; a
  test pins the divergent case (OR fires via logistic, verdict must NOT ACCEPT).

Other fixes: strengthened the w-leak test to a *real* (mu-driven) leak that
discriminates a broken residualiser; added verdict-function tests; per-band
decomposition honesty; `partial_sign` NaN-guard; reuse of
`compute_token_entropy_baseline`; tests for `collect_signals`, `_tercile_adaptivity`,
the base-err skip branch, and a real-run no-retrain hash guard. Five findings were
correctly rejected by the verifiers (including the intentional collapsed-posterior
precedence).

---

## 7. Outputs

In-domain: `results/setup_2/conditional_epi/`
Weak-OOD:  `results/setup_2/conditional_epi_ood_longfact/`

Each directory contains:
- `band_sweep.csv` — one row per band (Setting 1), incl. `partial_pass` /
  `logistic_pass` / `gate_passed`.
- `coverage.csv` — level x variant x coverage x mean_width (Setting 2).
- `interval_reliability.png` — nominal vs empirical coverage, (b) vs (c).
- `sharpness_by_epi_tercile.csv` — coverage/width per `epi_mu` tercile @ 0.95.
- `setting1_verdict.txt`, `setting2_verdict.txt` — arena-tagged reading-rule
  outcomes.

---

## 8. How to reproduce

```bash
conda activate ele

# In-domain (Setup-2 test split)
python scripts/13_epi_conditional_coverage.py --setup 2 --device cpu

# Weak-OOD fallback (existing LongFact annotations; no new generation/annotation)
python scripts/13_epi_conditional_coverage.py --setup 2 --device cpu \
    --ood-dataset longfact

# Tests
python -m pytest tests/test_conditional_epi.py -q
```

The `--ood-dataset` path reuses the trainer's generation/cache loaders and the
already-produced annotations; it writes to a separate
`conditional_epi_ood_<dataset>/` directory and never touches the in-domain run or
the trained model.

---

## 9. Constraints honoured

- **No retraining / read-only model** — `trained_model.pt` SHA-256 unchanged
  across all runs; verified by a hash guard test (import-time and real-run).
- **Additive only** — `04_evaluate.py`, the inner loop, the posterior,
  `predict.py` return values, `likelihood.py`, `fisher_scoring.py`, and every
  baseline are byte-identical; `trainer.py` is imported, not modified.
- `tests/test_fisher_scoring.py`, `test_decomposition.py`, `test_metrics.py` pass
  unchanged (70 tests); the 12 new Phase-13 tests pass.
- New code is plain-ASCII (no Unicode math / combining marks).
- Reuses `partial_correlation_gate`, `compute_bootstrapped_ci`,
  `compute_token_entropy_baseline`, and `predict._stable_cholesky`; mirrors the
  loader pattern of `scripts/05_epi_significance.py` /
  `scripts/diag_ood_error_detection.py`.
