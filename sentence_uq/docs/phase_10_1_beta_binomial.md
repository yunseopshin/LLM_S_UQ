# Phase 10-1 — Beta-Binomial Observation Model

This document records the design, implementation, adversarial cross-check, and
empirical validation of the **Beta-Binomial observation model** added in Phase
10-1. It is a config-selectable *peer* to the existing Binomial model: both run
on identical data through identical code paths, so any difference in results
(epistemic statistics, ECE, AUROC) is attributable solely to the likelihood.

Spec: `prompts/phase_10_1_beta_binomial.md`. Prior theory: `research_document_v8`.

---

## 1. Executive Summary

- A new `src/models/likelihood.py` defines a tiny `Likelihood` interface and two
  peer implementations, `BinomialLikelihood` and `BetaBinomialLikelihood`,
  selected by a single config switch `model.likelihood: binomial | beta_binomial`.
- The inner Fisher-scoring loop, the outer NLL, the trainer, and the predictive
  decomposition all share one code path; only the per-sentence scalars differ.
- The concentration \(\phi\) is a **global** outer parameter
  \(\psi = \{W, \alpha, \mu_0, \log\sigma_0, \log\phi\}\); the inner Laplace
  posterior remains over \(\theta\) only (its dimension, \(\hat\Sigma\), and the
  epistemic term are unchanged in *form*).
- **Hard constraint honoured:** with the Binomial likelihood the entire pipeline
  is *bit-identical* to the pre-10-1 code (gradient, Fisher, MAP \(\hat\theta\),
  loss, and `state_dict`), pinned by a float32 golden-MAP regression test.
- The Beta-Binomial **reduces to the Binomial as** \(\phi\to\infty\); this was
  verified both at the closed-form scalar level and end-to-end (Section 6).
- Two defects were found by an adversarial multi-agent cross-check and fixed
  (Section 5): a line-search objective reassociation that perturbed the float32
  MAP (P0), and float32 catastrophic cancellation in the gamma scalars at large
  \(\phi\) (P2).
- Tests: `tests/test_beta_binomial.py` (27 tests) plus the full pre-existing
  suite — **326 passed**.

---

## 2. Model and Mathematics

### 2.1 Mean–dispersion parameterization

Per sentence \(j\), introduce a latent rate and a global concentration
\(\phi > 0\):

\[
p_j \sim \mathrm{Beta}(a_j, b_j),\quad a_j = \phi\,\mu_j(\theta),\quad
b_j = \phi\,(1-\mu_j(\theta)),\qquad
K_j \mid p_j \sim \mathrm{Binomial}(m_j, p_j),
\]

so \(K_j\) is marginally Beta-Binomial. The mean structure is **unchanged**:

\[
\mu_j(\theta)=\frac{1}{L_j}\sum_{\ell\in s_j}\sigma(\theta^\top z_\ell),
\qquad
g_j=\frac{\partial\mu_j}{\partial\theta}
   =\frac{1}{L_j}\sum_{\ell\in s_j}\pi_\ell(1-\pi_\ell)\,z_\ell.
\]

With \(\rho := 1/(\phi+1)\), the key moments are

\[
\mathbb{E}[U_j\mid\theta]=\mu_j,\qquad
\mathrm{Var}[U_j\mid\theta]=\frac{\mu_j(1-\mu_j)}{m_j}\,
\big[\,1+(m_j-1)\rho\,\big].
\]

The mean equals the Binomial mean; only the conditional variance gains the
overdispersion factor \(1+(m_j-1)\rho\). As \(\phi\to\infty\) (\(\rho\to 0\))
this collapses to the Binomial.

### 2.2 Per-sentence scalars

Let \(\Psi_0 = \) `torch.digamma`, \(\Psi_1 = \) `torch.polygamma(1, .)`,
\(\log B(x,y) = \mathrm{lgamma}(x)+\mathrm{lgamma}(y)-\mathrm{lgamma}(x+y)\), and
clamp \(\mu \in [\epsilon, 1-\epsilon]\) (\(\epsilon = 10^{-6}\)) before any gamma
call. The four scalars consumed by the inner loop / outer NLL / decomposition
are (with \(a=\phi\mu\), \(b=\phi(1-\mu)\)):

**Score** \(r_j = \partial\log P_j/\partial\mu_j\) (the residual that multiplies \(g_j\)):
\[
r_j^{BB}=\phi\big[\Psi_0(K_j+a)-\Psi_0(a)-\Psi_0(m_j-K_j+b)+\Psi_0(b)\big].
\]

**Fisher weight** \(w_j\) — the *observed information*
\(-\partial^2\log P_j/\partial\mu_j^2\) (the expected Fisher has no closed form
for the Beta-Binomial):
\[
w_j^{BB}=\phi^2\big[\Psi_1(a)-\Psi_1(K_j+a)+\Psi_1(b)-\Psi_1(m_j-K_j+b)\big]>0.
\]

**Negative log-pmf** (line-search objective and outer NLL):
\[
-\log P_j = -\big[\log B(K_j+a,\;m_j-K_j+b)-\log B(a,\;b)\big].
\]

**Strict target** \(A_j=\mathbf 1[K_j=m_j]\):
\[
\log P(A_j=1)=\mathrm{lgamma}(m_j+a)-\mathrm{lgamma}(a)
              +\mathrm{lgamma}(\phi)-\mathrm{lgamma}(m_j+\phi).
\]

### 2.3 The \(\phi\to\infty\) limits (used in tests)

\[
r_j^{BB}\to\frac{K_j-m_j\mu_j}{\mu_j(1-\mu_j)},\qquad
w_j^{BB}\to\frac{K_j}{\mu_j^2}+\frac{m_j-K_j}{(1-\mu_j)^2},
\]
\[
-\log P_j \to \text{Binomial NLL},\qquad
\log P(A_j=1)\to m_j\log\mu_j.
\]

Note that \(w_j^{BB}\) tends to the **observed** information, which equals the
Binomial **expected** information \(m_j/(\mu_j(1-\mu_j))\) only at the optimum
\(K_j = m_j\mu_j\). This is the one quantity that does *not* coincide with the
Binomial off-optimum even at \(\phi=\infty\) (see Sections 4.2 and 6.1).

`m_j = 0` sentences are skipped, exactly as in the Binomial path.

---

## 3. Implementation (file by file)

| File | Change |
|---|---|
| `src/models/likelihood.py` *(new)* | `Likelihood` interface; `BinomialLikelihood` (bit-identical re-expression of the historical arithmetic); `BetaBinomialLikelihood` (digamma/polygamma/lgamma); `make_likelihood()` factory. |
| `src/features/extractor.py` | `SentenceUQParams` gains `likelihood`, `phi_init`, `learn_phi`. `log_phi = nn.Parameter(log(phi_init))` and a `phi` property are registered **only** for `beta_binomial`; the Binomial parameter set / `state_dict` is byte-for-byte unchanged. `extract_token_features`, `get_Sigma_0(_inv)`, and feature dims are untouched. |
| `src/models/fisher_scoring.py` | The three hard-coded Binomial expressions are routed through `likelihood.score_mu / fisher_weight / neg_log_pmf`. A `likelihood=None` keyword on `_compute_grad_and_fisher`, `_compute_clipped_objective`, `fisher_scoring_map`, `fisher_scoring_map_detached` defaults to `BinomialLikelihood()`. `_last_diagnostics` gains `phi` / `rho`. Damping, adaptive `lambda`, `m_j=0` skip, and differentiability are unchanged. |
| `src/models/bayesian_main.py` | `make_likelihood()` builds the likelihood from `feature_params` each forward pass with a **live** \(\phi=\exp(\log\phi)\); threaded into `compute_map` and `compute_loss` (which now accumulates `+ neg_log_pmf`). \(\hat\Sigma=(H_{\text{final}})^{-1}\) unchanged. |
| `src/inference/predict.py` | Overdispersion factor \(f=1+(m_*-1)\rho\) on the ratio/count aleatoric; strict probability via `exp(likelihood.log_prob_all_correct(...))`. `save_trained_model` / `load_trained_model` persist and restore the likelihood config (returns a reconstructed `Likelihood` carrying the fitted \(\phi\)). |
| `src/train/trainer.py` | `log_phi` is automatically in the Adam parameter group (it is a submodule parameter); `phi_hat` / `rho_hat` are logged each epoch. |
| `src/utils/dispersion.py` *(new)* | Pre-flight diagnostics: the `m_j` distribution (`frac(m==0/1/>=2)`) and a method-of-moments estimate of \(\rho\). |
| `configs/` | `model.{likelihood, phi_init, learn_phi}` added to `default.yaml`; new `setup_{1,2,3}_betabinomial.yaml` thin overrides. |
| `scripts/03_train.py`, `scripts/04_evaluate.py` | Read the likelihood block, build the params, run the dispersion pre-flight, persist `phi_hat`/`rho_hat`, and build the predictor with the reconstructed likelihood. |
| `tests/test_beta_binomial.py` *(new)* | 27 tests (Section 7). |

Out of scope and **not** modified: `src/models/bayesian_aux.py`, and the prior
hyperparameter treatment of \(\mu_0\) / \(\log\sigma_0\).

---

## 4. Design Decisions (and why)

### 4.1 The combinatorial constant \(\log\binom{m}{K}\) is dropped in *both* likelihoods

`neg_log_pmf` omits the constant \(\log\binom{m}{K}\) for both Binomial and
Beta-Binomial. It is independent of \(\mu\) and \(\phi\), so it changes neither
the MAP (the line search compares objectives at fixed \((K,m)\)), the gradients
w.r.t. \(\psi/\theta/\log\phi\), nor the fitted \(\phi\). Dropping it:

1. keeps the Binomial path **bit-identical** with the pre-10-1 code, which never
   included it; and
2. makes the \(\phi\to\infty\) limit of the Beta-Binomial `neg_log_pmf` coincide
   *exactly* with the Binomial `neg_log_pmf` used here (otherwise the two would
   differ by the constant, and the limit test would fail for \(K\notin\{0,m\}\)).

### 4.2 Observed vs expected Fisher

The Binomial path uses the **expected** Fisher weight \(m/(\mu(1-\mu))\). The
Beta-Binomial uses the **observed** information (Section 2.2) because no
closed-form expected Fisher exists. The two coincide in the \(\phi\to\infty\)
limit *only at the optimum* \(K=m\mu\). Consequently \(\hat\Sigma\) — and hence
the epistemic term \(\mathrm{Epi}_\mu=\hat g^\top\hat\Sigma\hat g\) — retains a
small, deliberate residual difference from the Binomial even at \(\phi=\infty\)
(Section 6.1). This is documented in the spec and is **not** a defect.

### 4.3 Internal float64 for the gamma scalars

`digamma`/`polygamma`/`lgamma` of a large \(\phi\mu\) suffer catastrophic
cancellation in float32 (e.g. \(\Psi_0(a)-\Psi_0(K+a)\) for \(a\sim 10^4\)).
Because the inner loop runs in float32 and can push \(\phi\) large when
overdispersion is weak, `BetaBinomialLikelihood._prep` computes every gamma
argument in **float64** and casts the result back to the caller's working dtype.
This preserves the inner-loop dtype and the autograd graph while eliminating the
cancellation. (In float32 at \(\phi=10^6\) the score error dropped from
\(\sim 1.7\) to \(\sim 3\times10^{-4}\), the analytic \(O(1/\phi)\) floor.)

### 4.4 Pinning \(\phi\) to infinity for validation

To verify the \(\phi\to\infty\) reduction one must hold \(\phi\) fixed at a large
value: set `learn_phi: false` with a large `phi_init`. With `learn_phi: true`
the outer loop would move \(\phi\) away from the limit. This is the intended,
trivial mechanism — there is no difficulty in "freezing" \(\phi\).

---

## 5. Adversarial Cross-Check and Defects Fixed

An adversarial multi-agent verification (6 independent falsification agents +
synthesis) audited the change across: bit-identity, scalar correctness vs
autograd/scipy, the \(\phi\to\infty\) limits, gradient flow into \(\log\phi\),
the predictive decomposition, and spec completeness. Five dimensions passed with
independently re-derived fp64 numbers (errors \(10^{-11}\)–\(10^{-14}\)). Two
defects were found and fixed.

### 5.1 P0 — line-search objective reassociation (bit-identity violation)

The refactor wrote the clipped objective as `obj - neg_log_pmf`
(\(= obj + (A+B)\)) instead of the historical term-wise
`obj + A + B` (\(= (obj+A)+B\)), where \(A=K\log\mu\),
\(B=(m-K)\log(1-\mu)\). This is a valid IEEE reassociation that differs at 1 ULP
in \(\sim17\%\) of terms. Because the line search accepts on
`new_obj.item() > prev_obj.item()`, a 1-ULP flip at a near-tie changes
accept/reject, alters the damping schedule, and perturbs the **float32** MAP:
83/300 random cases differed, up to \(|\Delta\theta|=2.2\times10^{-3}\)
(fp64: 3/80, \(\le 2.6\times10^{-8}\)).

The grad, Fisher, and `compute_loss` paths were already bit-identical; only the
objective accumulation was at fault. The existing suite (atol \(10^{-5}/10^{-6}\),
no frozen golden MAP) could not see this.

**Fix:** special-case `BinomialLikelihood` in `_compute_clipped_objective` to
accumulate term-wise exactly as before. **Verification:** an independent
re-implementation of the pre-10-1 core now matches the refactored path with
`torch.equal` in **0/300** float32 cases (max \(|\Delta\theta|=0\)). A
parametrized float32 golden-MAP test
(`test_fisher_map_float32_golden_matches_pre_10_1`) guards against regressions.

### 5.2 P2 — float32 cancellation at large \(\phi\)

See Section 4.3. Fixed by computing the gamma scalars in float64 internally.
Closed forms were always exactly correct (fp64 error \(\le 10^{-14}\)); only
float32 precision degraded, and only at very large \(\phi\).

### 5.3 Non-blocking notes

- `predict_mc_epistemic` (the Monte-Carlo verification path) intentionally keeps
  the Binomial law-of-total-variance form; it verifies the epistemic term, which
  is likelihood-independent, and is outside the spec §3.5 analytic decomposition.
- A hypothetical old payload with `log_phi` in `state_dict` but no `likelihood`
  config key would raise on load; this is not a real back-compat path (Binomial
  models never stored `log_phi`).

---

## 6. Empirical \(\phi\to\infty\) Validation

**Question:** does Beta-Binomial with \(\phi\to\infty\) actually reproduce the
Binomial, with all other settings held fixed?

### 6.1 Probe A — fixed \(\psi\) (isolates the likelihood)

At one identical \(\psi\), compare the MAP \(\hat\theta\), the loss, and the
test-set predictions under Binomial vs Beta-Binomial\((\phi)\):

| \(\phi\) | \(\Delta\theta\) | \(\Delta\)loss | \(\Delta\mu\) | \(\Delta p_{\text{strict}}\) | \(\Delta\)epi |
|---|---|---|---|---|---|
| 1e2 | 2.2e-2 | 2.2e-1 | 1.2e-3 | 5.0e-3 | 6.0e-4 |
| 1e3 | 1.9e-3 | 2.4e-2 | 9.0e-5 | 5.2e-4 | 4.5e-4 |
| 1e4 | 4.7e-4 | 2.0e-3 | 4.9e-5 | 3.6e-5 | 4.4e-4 |
| **1e6** | **2.5e-4** | **2.8e-4** | **4.0e-5** | **4.9e-5** | 4.4e-4 |
| 1e8 | 4.6e-4 | 5.2e-4 | 4.2e-5 | 5.4e-5 | 4.4e-4 |
| 1e12 | 3.3e-3 | 4.4e-3 | 4.1e-4 | 1.4e-3 | 4.4e-4 |

- \(\Delta\theta\), \(\Delta\)loss, \(\Delta\mu\), \(\Delta p_{\text{strict}}\)
  shrink like \(O(1/\phi)\) and bottom out around \(\phi\approx10^6\) at
  \(\sim10^{-4}\) — essentially the float32 floor of the pipeline. Past
  \(\phi\approx10^8\) the float64 digamma cancellation floor takes over, so the
  curve is U-shaped with its minimum near \(10^6\).
- **\(\Delta\)epi \(\approx 4.4\times10^{-4}\) does not vanish.** This is the
  observed-vs-expected Fisher distinction (Section 4.2): the two precisions
  coincide only exactly at \(K=m\mu\), so a small residual remains in
  \(\hat\Sigma\) (hence \(\mathrm{Epi}_\mu\)) even at \(\phi=\infty\). By design,
  not a defect.

**Conclusion:** the point-estimate / likelihood quantities (MAP, loss, \(\mu\),
strict probability) converge to the Binomial; only the Laplace covariance keeps
a deliberate residual.

### 6.2 Probe B — \(k\) identical outer steps (why full-training comparison misleads)

Comparing *fully trained* weights is the wrong probe: the bilevel optimisation is
chaotic (positive Lyapunov exponent), so any tiny step-1 gap saturates to the
attractor distance after a few epochs, regardless of \(\phi\).

| \(k\) steps | \(\phi\)=1e2 | \(\phi\)=1e6 | \(\phi\)=1e10 |
|---|---|---|---|
| 1 | 2.2e-2 | 6.5e-4 | 8.0e-4 |
| 2 | 1.7e-2 | 3.7e-3 | 7.4e-3 |
| 5 | 5.2e-2 | 3.8e-2 | 3.8e-2 |
| 15 | 1.1e-1 | 9.8e-2 | 1.0e-1 |

- \(k=1,2\): the gap shrinks with \(\phi\) (the likelihoods agree per step).
- \(k=5,15\): the gap saturates at \(\sim 0.1\) for **all** \(\phi\) — a property
  of gradient descent, not of the likelihood limit. Hence a 15-epoch full-train
  comparison plateaus and is uninformative about the \(\phi\)-limit.

### 6.3 Scalar-level limit (unit tests)

`tests/test_beta_binomial.py` additionally verifies, in float64, that at
\(\phi=10^6\): `score_mu`, `neg_log_pmf`, `log_prob_all_correct` match the
Binomial within `atol=1e-3`, and `fisher_weight` matches the observed-info closed
form (and the Binomial expected info at \(K=m\mu\)).

---

## 7. Tests

`tests/test_beta_binomial.py` (27 tests) covers spec §5:

1. **Binomial regression** — `BinomialLikelihood` injected into the inner loop
   equals the no-`likelihood` path *exactly*, and both equal a hand-coded
   transcription of the historical arithmetic (grad/Fisher `torch.equal`,
   including forced-clamp boundary cases). Plus the float32 golden-MAP pin (§5.1).
2. **\(\phi\to\infty\) limit** — §6.3.
3. **Decomposition reduction** — \(\rho=0\) reproduces the current numbers;
   \(\rho>0\) scales the ratio/count aleatoric by \(1+(m-1)\rho\) with all terms
   non-negative; strict uses the Beta-Binomial form.
4. **Gradient flow** — `gradcheck` (fp64) on the four scalars w.r.t. \(\mu\) and
   \(\phi\); an end-to-end check that one outer step yields a finite, non-zero
   gradient on `log_phi`; and an inner-loop-only check (indirect path).
5. **Positivity** — `fisher_weight > 0` for random valid \((K,m,\mu,\phi)\).
6. **m_j = 0 skip** — unaffected by the Beta-Binomial likelihood.

Plus parameter gating (no `log_phi` for binomial; registered + frozen behaviour
for beta-binomial) and a save/load round-trip.

Full suite: **326 passed** (the single warning is a pre-existing intentional
non-convergence test in `test_debug.py`).

---

## 8. How to Run

```bash
# Binomial peer (default, unchanged):
python scripts/03_train.py --setup 2 --config configs/setup_2.yaml
python scripts/04_evaluate.py --setup 2 --config configs/setup_2.yaml

# Beta-Binomial peer (identical data + training, only the likelihood differs):
python scripts/03_train.py --setup 2 --config configs/setup_2_betabinomial.yaml
python scripts/04_evaluate.py --setup 2 --config configs/setup_2_betabinomial.yaml
```

Config switch (defaults preserve current behaviour):

```yaml
model:
  likelihood: binomial      # binomial | beta_binomial
  phi_init: 50.0            # beta_binomial only: initial concentration (near-Binomial)
  learn_phi: true           # beta_binomial only: if false, phi fixed at phi_init
```

Before trusting any fitted \(\phi\), `scripts/03_train.py` prints the pre-flight
dispersion diagnostics (the `m_j` distribution and the method-of-moments
\(\rho\)); if `frac(m_j >= 2)` is small, \(\phi\) is weakly identified.

---

## 9. Interpretation Guidance (Go / No-Go)

The Beta-Binomial does **not** change the *form* of
\(\mathrm{Epi}_\mu=\hat g^\top\hat\Sigma\hat g\), so it does not fix any epistemic
collapse directly. The hypothesised mechanism is indirect: if the data support
overdispersion (\(\hat\phi\) small / \(\hat\rho\) non-trivial), the likelihood no
longer needs to push \(\mu_j\) to the boundary to match all-correct / all-wrong
sentences, so \(\pi_\ell\) saturation drops and \(\hat g\) recovers. The effect is
real only if \(\hat\phi\) is meaningfully finite. If \(\hat\phi\) stays large, the
Beta-Binomial is correctly reporting that overdispersion is **absent**, and the
search should move elsewhere (token-saliency pooling, focal / calibration loss).

When reporting a Beta-Binomial run on Setup 2, compare against the Binomial peer
on: fitted \(\hat\phi\) / \(\hat\rho\), saturation fraction, mean \(\|\hat g\|\),
mean \(\mathrm{Epi}_\mu\), ratio-level ECE, and strict-level AUROC.

---

## 10. Setup 2 Run — Binomial vs Beta-Binomial (actual results)

A full run was executed on **Setup 2** (FActScore-Bio in-domain; `N_train=1638`,
`N_test=353` with `m_j>0`) with **identical** data / split / initialisation
protocol / 300 epochs / `lr=1e-3` / `logit_reg_lambda=1e-2`. Only the likelihood
differed. `log_phi` was given a separate Adam learning rate (`optim.phi_lr=0.05`)
so the concentration could reach its data-supported level within the epoch budget
while \(\psi\) kept the binomial-peer `lr` (a fair comparison). Artifacts:
`results/setup_2_betabinomial/` (`train.log`, `phi_trajectory.csv`,
`final_metrics_{ratio,strict}.csv`, `comparison_vs_binomial.txt`).

### 10.1 Pre-flight dispersion vs fitted \(\hat\phi\)

The pre-flight method-of-moments diagnostic (pooled single mean over `m_j>=2`)
reports **strong raw overdispersion**: \(\rho_{\mathrm{MoM}}=0.356\),
\(\phi_{\mathrm{MoM}}=1.81\), `frac(m_j>=2)=0.823` (well identified). But the
*fitted* concentration tells a different story once the per-sentence mean
structure \(\mu_j(\theta)\) is learned:

\[
\hat\phi = 1792,\qquad \hat\rho = 5.6\times10^{-4}.
\]

The \(\hat\phi\) trajectory is **non-monotonic**: it dips to a minimum of
\(\phi=37.3\) at epoch 11 (early on, \(\psi\) is near init, \(\mu_j\approx0.5\)
for all sentences, so the across-sentence count variance is unexplained and
*looks* overdispersed), then climbs monotonically — \(118\) (ep 50), \(355\)
(ep 100), \(992\) (ep 200), \(1792\) (ep 300, still rising). As the mean model
improves it absorbs the apparent overdispersion, driving the Beta-Binomial back
toward the Binomial (\(\rho\to 0\)).

**Interpretation:** the raw overdispersion is a *mean-misspecification artifact*,
not genuine count-level residual overdispersion. Once \(\mu_j(\theta)\) is fit,
essentially no residual overdispersion remains. This is exactly the spec's honest
"\(\hat\phi\) stays large \(\Rightarrow\) overdispersion absent" outcome.

### 10.2 Metric comparison (test split, `n=353`)

All deltas are tiny (\(<0.004\)) and the strict-AUROC 95% CIs overlap almost
completely, so the two models are **statistically indistinguishable** here.

| Tier / metric | Binomial | Beta-Binomial | \(\Delta\) |
|---|---|---|---|
| Train final loss | 1630.95 | 1631.12 | +0.17 |
| **Ratio** MAE | 0.22368 | 0.22424 | +0.00057 |
| **Ratio** RMSE | 0.29100 | 0.29150 | +0.00050 |
| **Ratio** Pearson r | 0.45264 | 0.45023 | −0.00240 |
| **Ratio** binomial NLL | 1.32703 | 1.33006 | +0.00302 |
| **Ratio** Brier | 0.08468 | 0.08497 | +0.00029 |
| **Ratio** ECE | 0.05510 | 0.05312 | −0.00199 |
| **Strict** AUROC | 0.77971 | 0.77596 | −0.00375 |
| **Strict** AUPRC | 0.23884 | 0.23656 | −0.00228 |
| **Strict** Brier | 0.06812 | 0.06829 | +0.00017 |
| **Strict** ECE | 0.05182 | 0.05321 | +0.00139 |

(All rows are the `Ours (Bayesian)` predictor. Strict AUROC 95% CIs:
Binomial \(0.780\,[0.712,0.847]\) vs Beta-Binomial \(0.776\,[0.709,0.843]\).)

### 10.3 Conclusion

With every other setting unified, the Beta-Binomial **converges to the Binomial**
on Setup 2 (\(\hat\phi=1792\), \(\hat\rho\approx6\times10^{-4}\)); no metric moves
beyond noise. The empirical answer is that FActScore-Bio sentence counts carry no
meaningful *residual* overdispersion once the model's mean structure is fit — the
apparent raw overdispersion is mean-misspecification. Per the Go/No-Go guidance
(Section 9), this directs the search elsewhere (token-saliency pooling, focal /
calibration loss) rather than the observation model. The Beta-Binomial path is
nonetheless validated end-to-end and remains available as a config switch for
datasets that *do* exhibit residual overdispersion.
