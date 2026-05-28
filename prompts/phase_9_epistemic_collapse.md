# Phase 9 — Epistemic Collapse: Diagnosis & Remediation

## 1. Problem Statement

After the full 183-entity run (Setup 2, N_train=1638, N_test=353), **epistemic
uncertainty has collapsed to near zero**:

| quantity | value |
|---|---|
| `epi_mu_mean` (delta-method) | 0.000807 |
| `epi_mu_mean` (MC, 100 samples) | 0.000677 |
| MC vs linear Pearson r | 0.901 |
| Bayesian vs Point MAE | identical (0.2181) |
| Bayesian vs Point binomial NLL | identical (1.4728) |

The MC cross-check confirms this is **not a linearization artifact** — the
posterior genuinely has near-zero variance in the directions that matter for
prediction. Consequences:

- Bayesian and Point predictions are numerically identical on all metrics
  except a marginal ECE difference (0.0474 vs 0.0530).
- Rejection by `epi_mu` is anti-correlated with error (PRR 0.139 < Point's 0.248).
- The Bayesian uncertainty decomposition — the core contribution — is vacuous.


## 2. Root-Cause Analysis

`Epi_μ = ĝᵀ Σ̂ ĝ` is a quadratic form. It can collapse via two multiplicative
factors:

### Factor 1: Σ̂ is too small (posterior over-concentration)

The Fisher-type posterior precision is:

```
Σ̂⁻¹ = Σ₀⁻¹ + Σⱼ [ mⱼ / (μ̂ⱼ(1 − μ̂ⱼ)) ] ĝⱼ ĝⱼᵀ
```

With N=1638 training sentences and average mⱼ ≈ 3–5, the effective sample size
feeding the Fisher term is ~5000–8000. With k=66 (projection_dim=64 + 2
scalar features), N_eff/k ≈ 75–120. This ratio drives Σ̂ eigenvalues to be
extremely small, making `ĝᵀ Σ̂ ĝ ≈ 0` regardless of ĝ.

The **binomial model amplifies this**: each sentence contributes mⱼ times more
precision than a Bernoulli (m=1) model would. This is the price of the improved
calibration (ECE 0.0474 vs Bernoulli's 0.1576).

### Factor 2: ĝ is too small (sigmoid saturation / gradient vanishing)

```
ĝ = (1/L) Σ_ℓ π̂_ℓ(1 − π̂_ℓ) z_ℓ
```

If token-level predictions π̂_ℓ are confident (near 0 or 1), then
π̂(1−π̂) → 0 and ĝ shrinks regardless of Σ̂. This is a well-known pathology
of Laplace approximation for logistic models: the gradient of the sigmoid
vanishes at confident predictions, so the delta-method "sensitivity" vector
becomes tiny.

**These two factors multiply**, causing a double suppression of Epi_μ.

### Factor 3 (secondary): learned prior may have tightened

`log_sigma_0` is a learnable parameter (initialized at 0 → σ₀=1). If training
pushed it negative, `Σ₀⁻¹` grows, further concentrating the posterior. This
needs to be checked.


## 3. Diagnostic Plan

All diagnostics load `trained_model.pt` and the test data. They require no
retraining and should take < 5 minutes on CPU.

### Diagnostic 1: Σ̂ eigenspectrum

```python
trained = load_trained_model("results/setup_2/trained_model.pt")
Sigma = trained["Sigma_hat"]
eigvals = torch.linalg.eigvalsh(Sigma).numpy()

print(f"λ_max = {eigvals.max():.2e}")
print(f"λ_min = {eigvals.min():.2e}")
print(f"condition number = {eigvals.max() / max(eigvals.min(), 1e-15):.2e}")
print(f"trace(Σ̂) = {eigvals.sum():.2e}")
```

**Interpretation**: If `λ_max(Σ̂) < 1e-3`, Σ̂ alone explains the collapse.

### Diagnostic 2: ĝ norm and π̂ distribution

For each test sentence, compute:
- `||ĝ||₂` (the L2 norm of the mean gradient vector)
- The distribution of per-token `π̂_ℓ` values

```python
for z_tokens in test_z_tokens_list:
    logits = z_tokens @ theta_hat          # (L, )
    pi = torch.sigmoid(logits)             # (L, )
    g_ell = pi * (1 - pi) * z_tokens.T    # (k, L) -- note: unsqueeze as needed
    g_bar = g_ell.mean(dim=1)              # (k, )
    # collect ||g_bar||, pi values
```

**Interpretation**: If median `π̂` is in [0.1, 0.9], Factor 2 is not dominant.
If most `π̂` are > 0.95 or < 0.05, sigmoid saturation is the main driver.

### Diagnostic 3: upper-bound decomposition

```python
epi_upper = (g_bar_norm ** 2) * eigvals.max()
```

Compare `epi_upper` vs actual `epi_mu`. This separates the contribution of
||ĝ||² and λ_max(Σ̂) to the collapse.

### Diagnostic 4: learned prior (log_sigma_0)

```python
params = trained["feature_params"]
log_s = params.log_sigma_0.detach()
sigma_0 = torch.exp(log_s)
print(f"learned σ₀: min={sigma_0.min():.4f}, max={sigma_0.max():.4f}, mean={sigma_0.mean():.4f}")
print(f"learned log σ₀: min={log_s.min():.4f}, max={log_s.max():.4f}")
```

**Interpretation**: If `σ₀` has shrunk well below 1.0, the prior itself
tightened during training. Compare `Σ₀⁻¹` eigenvalues against the Fisher
data term.

### Diagnostic 5: Fisher data-term magnitude

```python
Sigma_0_inv = params.get_Sigma_0_inv()
Sigma_hat_inv = torch.linalg.inv(Sigma)
Fisher_data = Sigma_hat_inv - Sigma_0_inv
fisher_eigvals = torch.linalg.eigvalsh(Fisher_data).numpy()
print(f"Fisher data term: λ_max={fisher_eigvals.max():.2e}, λ_min={fisher_eigvals.min():.2e}")
print(f"Prior term (Σ₀⁻¹ diag max): {torch.diag(Sigma_0_inv).max():.2e}")
```

**Interpretation**: If Fisher data λ_max >> Σ₀⁻¹ diagonal, data completely
dominates the prior.


## 4. Expected Diagnosis

**Primary suspect: Factor 1 (Σ̂ over-concentration)**.

Reasoning:
- N_eff/k ≈ 75–120 is a very high ratio for a Bayesian linear model.
- The binomial mⱼ weighting in the Fisher term amplifies precision 3–5×
  relative to Bernoulli.
- This is consistent with the ablation: Bernoulli *evaluation* gives different
  metrics because it changes the strict-label definition, but the underlying
  posterior (and hence Epi_μ) is the same trained object.

**Secondary suspect: Factor 2 (sigmoid saturation)** may also contribute. The
model achieves reasonable discrimination (AUROC ~0.78), which implies many π̂
are not at 0.5 — some degree of confidence is expected. But the π(1−π) damping
means even moderate confidence (π̂ = 0.8 → damping = 0.16) reduces ĝ
significantly.

The actual bottleneck will become clear from Diagnostics 1–3.


## 5. Remediation Candidates

### 5.1 Posterior tempering (recommended first attempt)

**What**: Scale the posterior covariance by a temperature τ > 1:

```
Σ̂_tempered = τ · Σ̂
```

Equivalently, this down-weights the Fisher data term:

```
Σ̂_tempered⁻¹ = Σ₀⁻¹ + (1/τ) · Σⱼ [mⱼ / (μ̂(1−μ̂))] ĝĝᵀ
```

**Why**: This is the simplest intervention — a single scalar. It directly
addresses Factor 1 without changing the MAP θ̂ (so point predictions stay
the same). The "cold posterior" literature (Wenzel et al. 2020) documents
that Laplace posteriors are often too concentrated and benefit from tempering.

**How to implement**: In `Predictor.__init__`, multiply `self.Sigma_hat *= tau`.
Or add a `posterior_temperature` config parameter and apply it in the trainer
after computing Σ̂.

**Validation**: Sweep τ ∈ {1, 10, 100, 1000} and check:
- Does `epi_mu_mean` become meaningfully nonzero?
- Does ECE degrade? (Point estimate ECE should be unchanged.)
- Does PRR-AUC improve when ranking by tempered `epi_mu`?
- Does the MC cross-check still hold? (Tempered linear should diverge from
  un-tempered MC — this is expected, not a bug.)

**Paper framing**: "Tempered Laplace posterior" — well-established technique,
cite Wenzel et al. (2020), Ritter et al. (2018).

### 5.2 Observation-count capping

**What**: In the Fisher precision, replace mⱼ with min(mⱼ, m_cap) for some
cap (e.g. m_cap = 1 or 3).

**Why**: This directly controls how much each sentence inflates precision.
With m_cap = 1 it recovers the Bernoulli precision structure while keeping
binomial scoring at evaluation time.

**Trade-off**: Changes the MAP θ̂ (requires retraining). Less principled than
tempering — harder to justify theoretically. But can be framed as a
"regularized Fisher" approach.

### 5.3 Reduce feature dimension

**What**: Lower `projection_dim` from 64 to 16 or 8, giving k = 18 or 10.

**Why**: With k=10 and N_eff ≈ 5000, N/k ≈ 500 — still very concentrated.
This is unlikely to fix the issue on its own, and also risks losing predictive
quality.

**Verdict**: Low priority. Try tempering first.

### 5.4 Widen the prior

**What**: Increase `prior_sigma_init` (e.g. from 1.0 to 10.0 or 100.0).

**Why**: Makes Σ₀⁻¹ smaller, so the prior contributes less precision. But if
the Fisher data term already dominates by 100×, widening the prior has
negligible effect.

**Verdict**: Check Diagnostic 5 first. If data term >> prior, skip this.

### 5.5 Bernoulli pivot (last resort)

**What**: Switch the entire training pipeline to Bernoulli (m=1 for all
sentences in the Fisher precision and in the loss).

**Why**: Removes the mⱼ amplification entirely. Σ̂ will be ~3–5× larger.

**Cost**: Loses the calibration advantage of binomial (ECE worsens from 0.047
to 0.158 in the current eval-only ablation). Weakens the paper's main
contribution (sentence-level binomial modeling). Should only be considered
if tempering cannot rescue the epistemic signal.


## 6. Recommended Action Sequence

1. **Run diagnostics 1–5** (no retraining, ~5 min CPU). Output: a summary
   of Σ̂ eigenspectrum, ĝ norms, π̂ distribution, learned σ₀, Fisher vs prior.

2. **Apply posterior tempering** τ ∈ {10, 100, 1000} to the existing
   trained_model.pt. Re-run `04_evaluate.py` with tempered Σ̂. Check whether
   Epi_μ becomes meaningful and whether ECE/PRR improve.

3. **If tempering works**: add τ as a config parameter, run a τ sweep, report
   results in the paper as "tempered Laplace" ablation.

4. **If tempering alone is insufficient**: combine with observation-count
   capping (§5.2). Retrain with m_cap and tempered posterior.

5. **Bernoulli pivot only if all above fail**.


## 7. Important Note on the Bernoulli Ablation

The existing "Binomial vs Bernoulli" ablation in the notebook is
**evaluation-time only** — it changes how p_strict is computed (using m=1
instead of actual mⱼ) but uses the **same trained (θ̂, Σ̂)**. Bernoulli's
higher AUROC (0.826 vs 0.784) reflects an easier discrimination task (different
label set), not a better model. Its worse ECE (0.158 vs 0.047) reflects
miscalibration from ignoring count structure.

To fairly test a Bernoulli model, one would need to **retrain** with
Bernoulli-Fisher precision. That is a separate experiment, not a simple config
toggle.


## 8. Implementation Checklist for Claude Code

### Phase 9.1: Diagnostic script

Create `scripts/09_diagnose_epistemic.py` that:

- [ ] Loads `trained_model.pt` and test data
- [ ] Computes and prints Σ̂ eigenspectrum (Diagnostic 1)
- [ ] Computes and prints ĝ norms and π̂ distribution per test sentence (Diagnostic 2)
- [ ] Computes upper bound decomposition (Diagnostic 3)
- [ ] Prints learned log_sigma_0 / σ₀ (Diagnostic 4)
- [ ] Computes Fisher data term vs prior magnitude (Diagnostic 5)
- [ ] Saves a summary JSON to `results/setup_2/epistemic_diagnostics.json`
- [ ] Saves diagnostic plots to `results/setup_2/epistemic_diag_*.png`

### Phase 9.2: Posterior tempering

- [ ] Add `posterior_temperature: float = 1.0` to config schema
- [ ] In `04_evaluate.py`, after loading trained_model, apply
      `Sigma_hat *= posterior_temperature` before constructing Predictor
- [ ] Run evaluation sweep: τ ∈ {1, 10, 100, 1000}
- [ ] Report: epi_mu_mean, ECE, PRR_AUC, AUROC for each τ
- [ ] Save results as `results/setup_2/ablation_tempering.csv`

### Phase 9.3: Tempered notebook cells

Add cells to the results_report notebook:
- [ ] Tempering sweep table
- [ ] epi_mu distribution at best τ vs τ=1
- [ ] PRR curve comparison (tempered vs un-tempered)
