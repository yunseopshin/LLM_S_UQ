# Code Review and Mathematical Validation Notes

This document summarizes the static code review and mathematical validation of the `sentence_uq` codebase for the Bayesian sentence-level factuality uncertainty quantification project.

The review focuses on whether the implementation matches the current binomial modeling direction in `research_document_v8`, and identifies issues that should be fixed before running pilot or full experiments.

---

## 1. Executive Summary

The repository is not an empty skeleton. It already contains a fairly complete method prototype with the following components:

- data generation and hidden-state extraction,
- sentence splitting and token-range mapping,
- atomic-fact annotation into `(K_j, m_j)`,
- feature extraction from selected hidden layers plus entropy and top-1 probability,
- binomial Fisher-scoring MAP estimation,
- Bayesian main model,
- posterior predictive uncertainty decomposition,
- baselines and evaluation metrics,
- tests for several core modules.

Overall, the implementation is broadly aligned with the intended model:

\[
K_j \mid \theta, m_j \sim \mathrm{Binomial}(m_j, \mu_j(\theta)),
\qquad
\mu_j(\theta)=\frac{1}{L_j}\sum_{\ell\in s_j}\sigma(\theta^\top z_\ell).
\]

However, several issues should be fixed before interpreting experimental results. The most important ones are:

1. The stored logits appear to be one step misaligned with generated token hidden states.
2. The Fisher-scoring helper claims to compute the gradient of a clipped objective, but its gradient is not the true gradient of that clipped objective at clipping boundaries.
3. Strict factuality metrics need a clear distinction between factuality detection and factual-error detection.
4. Evaluation should validate binomial count consistency, especially `0 <= K_j <= m_j`.
5. Cache files should verify `source_path` and `token_ids` against the generation file to avoid silent misalignment.

---

## 2. Components That Match the Intended Mathematics

### 2.1 Feature Extractor

The feature extractor implements the intended feature map:

\[
h_\ell^{\mathrm{agg}} = \sum_q \alpha_q h_\ell^{(q)},
\qquad
\alpha_q = \mathrm{softmax}(a)_q,
\]

\[
z_\ell =
\begin{bmatrix}
W h_\ell^{\mathrm{agg}} \\
H_\ell \\
p^{(1)}_\ell
\end{bmatrix}
\in \mathbb{R}^{p+2}.
\]

Here:

- `W` is the learnable projection matrix,
- `alpha` is the learnable pre-softmax layer weight vector,
- `mu_0` and `log_sigma_0` parameterize the Gaussian prior over `theta`,
- entropy and top-1 probability are appended as two scalar token-level uncertainty features.

The design is model-agnostic because `hidden_dim` and `num_layers` are provided at initialization rather than hardcoded.

**Assessment:** this part is mathematically consistent with the intended model.

---

### 2.2 Binomial Observation Model

The core observation model is correctly represented as

\[
K_j \mid \theta, m_j \sim \mathrm{Binomial}(m_j, \mu_j(\theta)).
\]

The MAP objective used by the Fisher-scoring routine is documented as

\[
\widetilde{\mathcal{L}}(\theta)
=
\sum_{j:m_j>0}
\left[
K_j\log \widetilde{\mu}_j
+
(m_j-K_j)\log(1-\widetilde{\mu}_j)
\right]
-
\frac{1}{2}(\theta-\mu_0)^\top\Sigma_0^{-1}(\theta-\mu_0),
\]

where

\[
\widetilde{\mu}_j = \mathrm{clip}(\mu_j,\epsilon,1-\epsilon).
\]

The implemented Fisher-type precision is

\[
H_{\mathrm{fisher}}
=
\Sigma_0^{-1}
+
\sum_{j:m_j>0}
\frac{m_j}{\widetilde{\mu}_j(1-\widetilde{\mu}_j)}g_jg_j^\top,
\]

with

\[
g_j
=
\frac{1}{L_j}\sum_{\ell\in s_j}\pi_\ell(1-\pi_\ell)z_\ell,
\qquad
\pi_\ell = \sigma(\theta^\top z_\ell).
\]

The Fisher update is

\[
\theta \leftarrow \theta + (H_{\mathrm{fisher}}+\lambda I)^{-1}\nabla_\theta \widetilde{\mathcal{L}}(\theta).
\]

The code also correctly skips sentences with `m_j = 0`.

**Assessment:** the binomial structure is mostly correct, but the clipping-gradient issue described in Section 3.1 must be resolved.

---

### 2.3 Bayesian Main Model

The main model computes a differentiable MAP estimate `theta_hat` through the unrolled Fisher-scoring loop, then evaluates the outer binomial negative log-likelihood:

\[
\mathcal{L}_{\mathrm{outer}}(\psi)
=
\sum_{j:m_j>0}
\left[
-K_j\log\widetilde{\mu}_j(\widehat{\theta}(\psi))
-(m_j-K_j)\log(1-\widetilde{\mu}_j(\widehat{\theta}(\psi)))
\right].
\]

This is consistent with the current bilevel training design:

- inner loop: compute `theta_hat(psi)` by Fisher scoring,
- outer loop: update feature parameters `psi = (W, alpha, mu_0, log_sigma_0)` by differentiating the binomial loss.

**Assessment:** conceptually sound.

---

### 2.4 Predictive Uncertainty Decomposition

The predictive module implements the intended posterior predictive decomposition. For a new sentence, it computes

\[
\widehat{\mu}
=
\frac{1}{L_*}\sum_{\ell=1}^{L_*}\sigma(\widehat{\theta}^\top z_\ell),
\]

\[
\widehat{g}
=
\frac{1}{L_*}\sum_{\ell=1}^{L_*}\widehat{\pi}_\ell(1-\widehat{\pi}_\ell)z_\ell,
\]

and the latent epistemic uncertainty

\[
\mathrm{Epi}_\mu
=
\widehat{g}^\top\widehat{\Sigma}\widehat{g}.
\]

When `m_*` is available, it computes ratio-level uncertainty as

\[
\mathrm{Aleatoric}_U
=
\max\left\{0,\frac{\widehat{\mu}(1-\widehat{\mu})-\mathrm{Epi}_\mu}{m_*}\right\},
\]

\[
\mathrm{Total}_U
=
\mathrm{Aleatoric}_U + \mathrm{Epi}_\mu.
\]

For count-level uncertainty:

\[
\mathrm{Epi}_K=m_*^2\mathrm{Epi}_\mu,
\]

\[
\mathrm{Aleatoric}_K
=m_*\max\{0,\widehat{\mu}(1-\widehat{\mu})-\mathrm{Epi}_\mu\}.
\]

For strict factuality, the plug-in version is

\[
P(A_*=1) = \widehat{\mu}^{m_*}.
\]

The Monte Carlo version also computes

\[
\mathrm{Var}_{\theta}\{\mu(\theta)\},
\qquad
\mathbb{E}_{\theta}[\mu(\theta)(1-\mu(\theta))],
\qquad
\mathbb{E}_{\theta}[\mu(\theta)^{m_*}],
\]

which is useful for validating the delta-method approximation.

**Assessment:** this part is strong and should be used in pilot diagnostics.

---

## 3. Issues That Should Be Fixed Before Experiments

## 3.1 Clipped Objective and Analytic Gradient Are Inconsistent

### Problem

The objective uses

\[
\widetilde{\mu}_j = \mathrm{clip}(\mu_j,\epsilon,1-\epsilon).
\]

If this objective is implemented literally with `torch.clamp`, then the derivative of `\widetilde{\mu}_j` with respect to `\mu_j` is zero outside the interval `(epsilon, 1-epsilon)`.

However, the current analytic gradient uses

\[
g_j = \frac{\partial \mu_j}{\partial \theta}
\]

regardless of whether `mu_j` is inside or outside the clipping interval. Therefore, at clipping boundaries, the implemented gradient is not the true gradient of the clipped objective.

This is not just a documentation issue. It affects what the Fisher-scoring routine is actually optimizing.

### Why the current tests may not catch it

The existing gradient comparison test can pass if the synthetic samples remain inside the clipping region. The mismatch appears only when `mu_j` is forced close to 0 or 1 and the clamp is active.

### Recommended fix

There are two possible choices.

#### Option A: Use the true clipped objective gradient

If the objective is truly clipped, then likelihood-gradient contributions should be zero when `mu_raw` lies outside the unclipped interval.

Conceptual patch:

```python
mu_raw = pi_j.mean()
mu_clamped = torch.clamp(mu_raw, eps, 1.0 - eps)
interior = (mu_raw > eps) & (mu_raw < 1.0 - eps)

if interior:
    denom = mu_clamped * (1.0 - mu_clamped)
    R_j = (K_j - m_j * mu_clamped) / denom
    grad = grad + R_j * g_j
    H = H + (m_j / denom) * torch.outer(g_j, g_j)
else:
    # True clipped objective has zero likelihood derivative here.
    pass
```

This option is mathematically clean.

#### Option B: Treat clipping only as numerical stabilization

If clipping is meant only to avoid numerical singularities, then the documentation should not call the result the gradient of a clipped objective. Instead, define it as an epsilon-stabilized Fisher scoring update.

For example:

\[
R_j^{\epsilon}
=
\frac{K_j-m_j\mu_j}{\mathrm{clip}(\mu_j(1-\mu_j),\epsilon,\infty)}.
\]

This option is more pragmatic but needs careful wording in the paper and code comments.

### Recommendation

Use **Option A** unless pilot experiments show severe optimization problems. If Option B is used, explicitly report boundary fractions during training and avoid saying that the update is the true gradient of the clipped objective.

---

## 3.2 Generation Logits and Hidden States Appear One Step Misaligned

### Problem

The generation loop samples the first generated token from the prefill logits. Then, inside the loop, it feeds the sampled token back into the model, records that token's hidden state, and records the logits produced after that token is processed.

This means that for generated token `token_ids[t]`:

- `hidden_states[t]` corresponds to token `t`,
- but `logits[t]` predicts token `t+1`.

The cached entropy and top-1 probability are computed from `logits[t]`. Therefore, the feature vector currently attaches next-token uncertainty to the current token hidden state:

\[
z_t = [h_t, H(x_{t+1}\mid x_{\leq t}), p^{(1)}(x_{t+1}\mid x_{\leq t})].
\]

If the intended feature is the generation-time uncertainty of the current token, it should be

\[
z_t = [h_t, H(x_t\mid x_{<t}), p^{(1)}(x_t\mid x_{<t})].
\]

### Recommended fix

Store the logits that were used to sample the current token.

Conceptual patch:

```python
prev_logits = prefill.logits[:, -1, :]  # predicts first generated token
next_token = _sample_token(prev_logits, temperature, top_p, do_sample)

for _ in range(max_new_tokens):
    token_id_int = int(next_token.item())
    if token_id_int in eos_ids:
        finished = True
        break

    # Store the distribution that produced the current token.
    gen_logits.append(prev_logits[0].detach().to("cpu", dtype=store_dtype))

    step = model(
        input_ids=next_token.view(1, 1),
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
    )
    past_key_values = step.past_key_values

    # Store the hidden state of the current token.
    for k, layer_idx in enumerate(selected):
        h = step.hidden_states[layer_idx][0, 0, :].detach()
        gen_hidden_per_layer[k].append(h.to("cpu", dtype=store_dtype))

    gen_token_ids.append(token_id_int)

    # This predicts the next token.
    prev_logits = step.logits[:, -1, :]
    next_token = _sample_token(prev_logits, temperature, top_p, do_sample)
```

### Add a test

Add a unit test with a tiny mock causal LM to verify that:

- `logits[t]` is the distribution used to sample `token_ids[t]`,
- not the distribution used to sample `token_ids[t+1]`.

This is a high-priority fix because it affects the interpretation of entropy and top-1 features.

---

## 3.3 Strict Factuality Metrics Need a Clear Direction

### Problem

The current strict factuality target is

\[
A_j = 1\{K_j=m_j\}.
\]

The predicted strict factuality probability is

\[
p_{\mathrm{strict},j}=\widehat{\mu}_j^{m_j}.
\]

Using `AUROC(A_j, p_strict)` is correct if the task is **detecting strictly factual sentences**.

But if the table or paper describes the metric as **factual-error detection**, then the correct label and score are

\[
E_j = 1\{K_j < m_j\},
\qquad
p_{\mathrm{error},j}=1-p_{\mathrm{strict},j}.
\]

### Recommended fix

Report both directions or choose one and name it precisely.

Suggested implementation:

```python
A_true = (K == m).astype(float)
E_true = 1.0 - A_true

p_strict = mu ** m
p_error = 1.0 - p_strict

strict_auroc = roc_auc_score(A_true, p_strict)
error_auroc = roc_auc_score(E_true, p_error)
```

### Recommendation

For the paper, use one of the following names:

- `Strict factuality AUROC`: label is `A_j`, score is `p_strict`.
- `Factual error detection AUROC`: label is `E_j`, score is `p_error`.

Do not mix these two.

---

## 3.4 Validate Binomial Counts Before Evaluation

### Problem

The evaluation currently constructs the strict target using a condition equivalent to `K >= m`. Under valid binomial counts this equals `K == m`, because `K <= m`. But if an annotation bug produces `K > m`, the current code would silently treat the sentence as strictly factual.

This can hide data-processing errors.

### Recommended fix

Before any evaluation or training collation, validate:

\[
0 \leq K_j \leq m_j.
\]

Suggested implementation:

```python
if np.any(K < 0) or np.any(m < 0) or np.any(K > m):
    raise ValueError("Invalid binomial counts: require 0 <= K_j <= m_j")

A = (K == m).astype(np.float64)
```

Also add the same validation for PyTorch tensors in the trainer or model input checks.

---

## 3.5 Clarify Binomial NLL With or Without the Combinatorial Constant

### Problem

The current training and evaluation NLL use

\[
-\left[K\log\mu + (m-K)\log(1-\mu)\right].
\]

This omits the combinatorial constant

\[
\log {m \choose K}.
\]

For optimization, omitting this term is fine because it does not depend on the model parameters. But for evaluation, if the metric is called `binomial_NLL`, readers may expect the full likelihood:

\[
-\log p(K\mid m,\mu)
= -\log {m \choose K}
-K\log\mu
-(m-K)\log(1-\mu).
\]

### Recommended fix

Use two explicit metric names:

- `binomial_CE`: without the combinatorial constant,
- `binomial_NLL_full`: with the combinatorial constant.

Suggested implementation using `scipy.special.gammaln`:

```python
from scipy.special import gammaln

log_comb = gammaln(m + 1) - gammaln(K + 1) - gammaln(m - K + 1)
full_nll = -(log_comb + K * np.log(mu_safe) + (m - K) * np.log(1.0 - mu_safe))
ce = -(K * np.log(mu_safe) + (m - K) * np.log(1.0 - mu_safe))
```

For the paper, report `binomial_NLL_full` if the metric is presented as a likelihood score.

---

## 3.6 Cache Indexing Should Verify Source Consistency

### Problem

Cached scalar files are indexed by the sorted order of generation files:

```text
cache/{idx:05d}.pt
```

The trainer reconstructs the same sorted order to map generation files to cache files. This works only if the generation directory has not changed since the cache was created.

If a file is added, removed, or renamed, cache indices can silently point to the wrong generation.

### Recommended fix

The cache payload already stores `source_path` and `token_ids`. The trainer should verify both.

Suggested patch in `_load_prompt_tensors`:

```python
cache_payload = torch.load(cache_path, map_location="cpu", weights_only=False)

if cache_payload.get("source_path") != rel_path:
    raise ValueError(
        f"Cache/source mismatch: cache has {cache_payload.get('source_path')}, "
        f"expected {rel_path}"
    )

if not torch.equal(cache_payload["token_ids"], gen_payload["token_ids"]):
    raise ValueError(f"Cache token_ids mismatch for {rel_path}")
```

This is a simple but important reproducibility safeguard.

---

## 4. Medium-Priority Improvements

### 4.1 Separate `eps` and `pd_tol`

The local positive-definiteness check uses the same `eps` for both log clipping and eigenvalue thresholding. These are conceptually different.

Recommended change:

```python
def verify_local_pd(..., clip_eps=1e-6, pd_tol=1e-8):
    ...
    fisher_pd = fisher_min_eig > pd_tol
    true_pd = true_min_eig > pd_tol
```

This makes the code easier to interpret and avoids conflating numerical log safety with local curvature diagnostics.

---

### 4.2 Add Boundary-Fraction Diagnostics

If clipping or epsilon stabilization is used, track how often the model produces `mu_j` near the boundary.

Suggested diagnostic:

```python
boundary_low = (mu_raw < eps).float().mean()
boundary_high = (mu_raw > 1.0 - eps).float().mean()
boundary_fraction = boundary_low + boundary_high
```

This should be logged during training and reported in pilot diagnostics.

---

### 4.3 Improve Memory Efficiency of the Trainer

The current trainer computes a list of `z_tokens` for every sentence and then runs a full-batch Fisher-scoring MAP. This is mathematically clean but may become memory-heavy beyond the pilot stage.

Possible improvements:

1. Extract token features once per prompt and slice sentence ranges from the prompt-level feature tensor.
2. Keep full-batch Fisher scoring for the main method, but cache prompt-level features during each epoch.
3. Later, if needed, define a mini-batch approximation as a separate methodological variant.

Do not introduce mini-batch Fisher scoring without explicitly changing the mathematical formulation.

---

### 4.4 Add More Targeted Unit Tests

Recommended new tests:

1. `test_generation_logits_are_current_token_logits`
   - verify that `logits[t]` predicts `token_ids[t]`, not `token_ids[t+1]`.

2. `test_clipped_gradient_boundary_behavior`
   - force `mu_raw < eps` or `mu_raw > 1-eps`, then compare analytic gradient to autograd under the chosen convention.

3. `test_invalid_binomial_counts_raise`
   - ensure `K > m`, `K < 0`, and `m < 0` raise errors.

4. `test_cache_source_path_and_token_ids_checked`
   - verify that mismatched cache files are rejected.

5. `test_strict_vs_error_metric_direction`
   - verify that strict factuality AUROC and factual-error AUROC use opposite labels and scores.

---

## 5. Priority Checklist

### Priority 1: Fix before pilot results are trusted

- [ ] Fix generation logits and hidden-state alignment.
- [ ] Decide and implement the clipping-gradient convention.
- [ ] Add binomial count validation: `0 <= K_j <= m_j`.
- [ ] Clarify strict factuality vs factual-error detection metrics.
- [ ] Verify cache `source_path` and `token_ids` in trainer loading.

### Priority 2: Fix before main experiments

- [ ] Add full binomial NLL with combinatorial constant.
- [ ] Rename the current NLL without constant to `binomial_CE` or `binomial_NLL_no_const`.
- [ ] Separate `clip_eps` and `pd_tol`.
- [ ] Add boundary-fraction diagnostics.
- [ ] Add targeted unit tests for the above issues.

### Priority 3: Improve after pilot works

- [ ] Reduce trainer memory cost by prompt-level feature extraction.
- [ ] Add MC vs delta-method epistemic correlation to pilot reports.
- [ ] Add reliability diagrams for ratio-level and strict-level calibration.
- [ ] Add final result table template for main paper experiments.

---

## 6. Final Assessment

The current codebase is already beyond a simple skeleton. The main mathematical structure is present and mostly consistent with the intended Bayesian binomial model.

The most reliable components are:

- feature extraction,
- binomial likelihood structure,
- Fisher-type MAP estimation structure,
- posterior predictive uncertainty decomposition,
- Monte Carlo validation utilities.

The most important risks are:

1. token-level uncertainty features may be one step misaligned,
2. clipped objective and analytic gradient are not fully consistent,
3. evaluation metric direction may be ambiguous,
4. invalid binomial counts may be silently accepted,
5. cache indexing may silently misalign data.

After fixing these issues, the code should be suitable for pilot experiments. The pilot should not be used as evidence for the paper until the alignment and clipping issues are resolved.
