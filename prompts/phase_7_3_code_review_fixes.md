# Phase 7-3 — Code Review Fixes (Pre-Pilot)

Apply fixes identified by static code review (`docs/code_review_math_validation.md`).
All changes in this phase must be completed before pilot results are trusted.

**Motivation**: The code review found five issues that can silently corrupt
experimental results. This phase resolves them in priority order.

**Prerequisite**: Phases 1-1 through 7-2 implemented. Tests passing.

---

## Fix 1: Generation Logits Alignment (CRITICAL)

**File**: `src/data/generation.py`, function `generate_with_hidden_states`

**Problem**: The current loop stores `step.logits` (which predicts `token_ids[t+1]`)
alongside `hidden_states[t]`. The cached entropy and top-1 probability therefore
reflect *next-token* uncertainty, not *current-token* generation-time uncertainty.

**Required change**: Store the logits that were used to *sample* the current token
(`prev_logits`), not the logits produced *after* processing it.

**Current code** (the generation loop body):

```python
for _ in range(max_new_tokens):
    token_id_int = int(next_token.item())
    if token_id_int in eos_ids:
        finished = True
        break

    step = model(
        input_ids=next_token.view(1, 1),
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
    )
    past_key_values = step.past_key_values

    for k, layer_idx in enumerate(selected):
        h = step.hidden_states[layer_idx][0, 0, :].detach()
        gen_hidden_per_layer[k].append(h.to("cpu", dtype=store_dtype))
    gen_logits.append(
        step.logits[0, -1, :].detach().to("cpu", dtype=store_dtype)
    )
    gen_token_ids.append(token_id_int)

    next_token = _sample_token(
        step.logits[:, -1, :], temperature, top_p, do_sample
    )
```

**Target code**:

```python
# Before the loop: capture the logits that will sample the first token
prev_logits = prefill.logits[:, -1, :]  # predicts first generated token

for _ in range(max_new_tokens):
    token_id_int = int(next_token.item())
    if token_id_int in eos_ids:
        finished = True
        break

    # Store the logits that PRODUCED the current token (generation-time distribution)
    gen_logits.append(prev_logits[0].detach().to("cpu", dtype=store_dtype))

    step = model(
        input_ids=next_token.view(1, 1),
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
    )
    past_key_values = step.past_key_values

    # Hidden state of the current token (after processing)
    for k, layer_idx in enumerate(selected):
        h = step.hidden_states[layer_idx][0, 0, :].detach()
        gen_hidden_per_layer[k].append(h.to("cpu", dtype=store_dtype))

    gen_token_ids.append(token_id_int)

    # These logits predict the NEXT token — store them as prev_logits
    prev_logits = step.logits[:, -1, :]
    next_token = _sample_token(prev_logits, temperature, top_p, do_sample)
```

**Key invariant after fix**: For generated token `token_ids[t]`:
- `hidden_states[t]` = representation after processing token t (conditioning: x_{≤t})
- `logits[t]` = distribution used to sample token t (conditioning: x_{<t})
- `entropy[t]` = H(x_t | x_{<t})  — generation-time uncertainty of current token
- `top1_prob[t]` = p^(1)(x_t | x_{<t})

**Update the docstring** at the top of `generation.py` to reflect the new semantics:
Replace the note about `hidden_states[t, k, :]` being "the state used to predict
token_ids[t+1]" with a clear statement that `logits[t]` is the distribution that
*sampled* `token_ids[t]`.

**IMPORTANT — downstream invalidation**:
- All existing `.pt` generation files must be **regenerated** (logits are stored wrong).
- All cached scalar files (`data/cache/`) must be **recomputed** after regeneration.
- If regeneration is not feasible for pilot, add a version field to generation files
  and validate it at load time. But regeneration is strongly preferred.

---

## Fix 2: Fisher Scoring Docstrings — Epsilon-Stabilized Terminology

**File**: `src/models/fisher_scoring.py`

**Problem**: The module docstring and function docstrings describe the gradient as
"gradient of the clipped objective L̃(θ)". This is inaccurate: the gradient uses
unclipped μ_j in the numerator and only stabilizes the denominator. The clipped
objective is used for line-search only.

**Required changes**:

1. **Module docstring**: Replace all references to "gradient of L̃" with
   "epsilon-stabilized gradient". Keep the clipped objective description for the
   `_compute_clipped_objective` function (which IS the true clipped objective).

   Specifically, replace the `Gradient::` block in the module docstring:
   ```
   Epsilon-stabilized gradient (§7.2.2 of research_document_v8)::

       ∇L̃ ≈ -Σ_0⁻¹ (θ - μ_0) + Σ_j R_j^ε · g_j,
       R_j^ε = (K_j - m_j μ_j) / max(μ_j (1 - μ_j), ε),
       g_j   = (1 / L_j) Σ_ℓ π_ℓ (1 - π_ℓ) z_ℓ

   Note: this is NOT the true gradient of the clipped objective L̃.
   At clipping boundaries, the true clipped gradient would be zero for
   the affected sentence. The implementation instead stabilizes only the
   denominator, keeping all sentences contributing to the gradient.
   The clipped objective L̃ is used only for line-search accept/reject.
   ```

2. **`_compute_grad_and_fisher` docstring**: Add a note:
   ```
   Note: the gradient returned is epsilon-stabilized (denominator-only
   clipping), not the true gradient of the clipped objective. See §7.2.2
   of research_document_v8.md.
   ```

3. **`_compute_clipped_objective` docstring**: This one is fine as-is (it IS the
   true clipped objective). No changes needed.

---

## Fix 3: Boundary Fraction Logging

**Files**: `src/models/fisher_scoring.py`, `src/utils/debug.py`

**Problem**: There is no runtime visibility into how often epsilon clipping is active.

**Required changes**:

1. In `_compute_grad_and_fisher`, after the loop, compute and return boundary
   fraction as an optional diagnostic. To avoid changing the return signature
   (which would break callers), add a module-level diagnostic dict:

   ```python
   # At module level
   _last_diagnostics: dict = {}

   def _compute_grad_and_fisher(...):
       ...
       boundary_count = 0
       total_count = 0
       for j in range(len(all_K)):
           if int(all_m[j]) == 0:
               continue
           total_count += 1
           z_j = all_z_tokens[j]
           ...
           mu_raw = pi_j.mean()
           if mu_raw.item() < eps or mu_raw.item() > 1.0 - eps:
               boundary_count += 1
           ...
       _last_diagnostics["boundary_fraction"] = (
           boundary_count / total_count if total_count > 0 else 0.0
       )
       _last_diagnostics["boundary_count"] = boundary_count
       _last_diagnostics["total_sentences"] = total_count
       return grad, H
   ```

2. In `src/utils/debug.py`, the existing `sanity_check_boundary_fraction` function
   should also be updated to use the same eps parameter and print a warning
   if boundary fraction exceeds 5%.

3. In the trainer (`src/train/trainer.py`), log boundary fraction once per epoch
   using the diagnostic dict:
   ```python
   from src.models.fisher_scoring import _last_diagnostics
   # After fisher_scoring_map call:
   bf = _last_diagnostics.get("boundary_fraction", 0.0)
   if bf > 0.05:
       logger.warning(f"Boundary fraction {bf:.1%} exceeds 5% — consider adjusting prior scale")
   ```

---

## Fix 4: Binomial Count Validation

**Files**: `src/train/trainer.py`, `src/evaluation/evaluate.py`, `src/models/fisher_scoring.py`

**Problem**: Invalid binomial counts (K > m, K < 0, m < 0) are silently accepted,
which can hide annotation bugs.

**Required changes**:

1. Add a validation utility in `src/utils/validation.py` (new file):

   ```python
   """Data validation utilities."""

   import torch
   import numpy as np
   from typing import Union

   def validate_binomial_counts(
       K: Union[torch.Tensor, np.ndarray],
       m: Union[torch.Tensor, np.ndarray],
       context: str = "",
   ) -> None:
       """Raise ValueError if binomial counts are invalid.

       Checks: 0 <= K_j <= m_j for all j, and m_j >= 0.
       """
       prefix = f"[{context}] " if context else ""
       if isinstance(K, torch.Tensor):
           if torch.any(K < 0):
               raise ValueError(f"{prefix}Found K_j < 0")
           if torch.any(m < 0):
               raise ValueError(f"{prefix}Found m_j < 0")
           if torch.any(K > m):
               raise ValueError(f"{prefix}Found K_j > m_j")
       else:
           if np.any(K < 0):
               raise ValueError(f"{prefix}Found K_j < 0")
           if np.any(m < 0):
               raise ValueError(f"{prefix}Found m_j < 0")
           if np.any(K > m):
               raise ValueError(f"{prefix}Found K_j > m_j")
   ```

2. Call `validate_binomial_counts` at these entry points:
   - `fisher_scoring_map` — before the iteration loop begins
   - `trainer.py` — in the collation step where `all_K` and `all_m` are assembled
   - `evaluate.py` — before computing any metrics

3. In evaluation, change strict factuality target from `K >= m` to `K == m`:
   ```python
   validate_binomial_counts(K, m, context="evaluation")
   A_strict = (K == m).astype(np.float64)  # NOT K >= m
   ```

---

## Fix 5: Cache Source Path and Token ID Verification

**File**: `src/train/trainer.py`, method `_load_prompt_tensors`

**Problem**: Cache files are indexed by sorted order. If the generation directory
changes (files added/removed), cache indices silently point to wrong data.

**Required change**: After loading the cache payload, verify `source_path` and
`token_ids` against the generation file.

**Current code**:

```python
@staticmethod
def _load_prompt_tensors(
    gen_dir: Path,
    cache_dir: Path,
    rel_path: str,
    cache_idx: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    gen_path = gen_dir / rel_path
    gen_payload = torch.load(gen_path, map_location="cpu", weights_only=False)
    hidden_states = gen_payload["hidden_states"]

    cache_path = cache_dir / f"{cache_idx:05d}.pt"
    cache_payload = torch.load(
        cache_path, map_location="cpu", weights_only=False
    )
    entropy = cache_payload["entropy"]
    top1 = cache_payload["top1_prob"]

    if int(hidden_states.shape[0]) != int(entropy.shape[0]):
        raise ValueError(...)
    return hidden_states, entropy, top1
```

**Target code** — add verification after loading cache_payload:

```python
    cache_payload = torch.load(
        cache_path, map_location="cpu", weights_only=False
    )

    # Verify cache-generation consistency
    cached_source = cache_payload.get("source_path", "")
    if cached_source and cached_source != rel_path:
        raise ValueError(
            f"Cache/source mismatch at idx {cache_idx}: "
            f"cache has '{cached_source}', expected '{rel_path}'"
        )

    cached_token_ids = cache_payload.get("token_ids")
    gen_token_ids = gen_payload.get("token_ids")
    if cached_token_ids is not None and gen_token_ids is not None:
        if not torch.equal(cached_token_ids, gen_token_ids):
            raise ValueError(
                f"Cache token_ids mismatch for {rel_path} at idx {cache_idx}"
            )

    entropy = cache_payload["entropy"]
    top1 = cache_payload["top1_prob"]
    ...
```

---

## Fix 6: Strict Factuality vs Error Detection Metric Direction

**File**: `src/evaluation/metrics.py`

**Problem**: The code computes `strict_auroc` but the label/score direction is
not explicitly documented. If the table says "factual error detection AUROC"
but the code uses `A_j = 1{K_j == m_j}` with `p_strict`, the number is correct
but the name is misleading (or vice versa).

**Required changes**:

1. Compute and return **both** directions explicitly:

   ```python
   def compute_strict_metrics(K, m, mu_hat):
       """Compute strict factuality and error detection AUROCs.

       Returns dict with:
           strict_factuality_auroc: P(all facts correct) detection
           error_detection_auroc: P(any fact wrong) detection
       """
       validate_binomial_counts(K, m, context="strict_metrics")

       # Strict factuality: label=1 means all atoms supported
       A_strict = (K == m).astype(np.float64)
       p_strict = mu_hat ** m

       # Error detection: label=1 means at least one atom unsupported
       E_error = 1.0 - A_strict
       p_error = 1.0 - p_strict

       result = {}
       if len(np.unique(A_strict)) > 1:
           result["strict_factuality_auroc"] = roc_auc_score(A_strict, p_strict)
           result["error_detection_auroc"] = roc_auc_score(E_error, p_error)
       else:
           result["strict_factuality_auroc"] = float("nan")
           result["error_detection_auroc"] = float("nan")
       return result
   ```

2. In the evaluation script and any result-printing code, use the explicit
   metric names. For the paper, use "error detection AUROC" as the primary
   strict-level metric (since the goal is detecting hallucinations, not
   detecting perfect sentences).

---

## Fix 7 (Priority 2): Full Binomial NLL Metric

**File**: `src/evaluation/metrics.py`

**Required**: Add `binomial_NLL_full` alongside the existing cross-entropy metric.

```python
from scipy.special import gammaln

def binomial_nll_full(K, m, mu_hat, eps=1e-8):
    """Full binomial NLL including the combinatorial constant."""
    mu_safe = np.clip(mu_hat, eps, 1.0 - eps)
    log_comb = gammaln(m + 1) - gammaln(K + 1) - gammaln(m - K + 1)
    nll = -(log_comb + K * np.log(mu_safe) + (m - K) * np.log(1.0 - mu_safe))
    return nll.mean()

def binomial_ce(K, m, mu_hat, eps=1e-8):
    """Binomial cross-entropy (no combinatorial constant)."""
    mu_safe = np.clip(mu_hat, eps, 1.0 - eps)
    ce = -(K * np.log(mu_safe) + (m - K) * np.log(1.0 - mu_safe))
    return ce.mean()
```

Rename existing NLL metric to `binomial_ce` in the evaluation output.
Report `binomial_nll_full` in the paper if presenting as a likelihood score.

---

## Fix 8 (Priority 2): Separate clip_eps and pd_tol

**File**: `src/utils/debug.py`, function `verify_local_pd`

**Required**: Split the single `eps` parameter into `clip_eps` (for log stability)
and `pd_tol` (for eigenvalue positivity threshold).

```python
def verify_local_pd(..., clip_eps=1e-6, pd_tol=1e-8):
    ...
    fisher_pd = fisher_min_eig > pd_tol
    true_pd = true_min_eig > pd_tol
```

---

## Fix 9 (Priority 2): New Unit Tests

**File**: `tests/test_code_review_fixes.py` (new file)

Add the following tests:

1. **`test_generation_logits_are_current_token_logits`**:
   - Create a tiny deterministic causal LM (or mock).
   - Generate a short sequence.
   - Verify that `softmax(logits[t])` assigns the highest probability
     to `token_ids[t]` (not `token_ids[t+1]`) when using greedy decoding.

2. **`test_clipped_gradient_boundary_behavior`**:
   - Construct a theta that forces `mu_raw < eps` for some sentence.
   - Compute the analytic gradient from `_compute_grad_and_fisher`.
   - Verify that the boundary sentence DOES contribute to the gradient
     (confirming epsilon-stabilized, not true clipped behavior).
   - Compare with autograd through `_compute_clipped_objective` — note they
     will DIFFER at the boundary (this is expected and should be documented).

3. **`test_invalid_binomial_counts_raise`**:
   - `K > m` → ValueError
   - `K < 0` → ValueError
   - `m < 0` → ValueError

4. **`test_cache_source_path_mismatch_raises`**:
   - Create a cache file with `source_path = "wrong.pt"`.
   - Call `_load_prompt_tensors` expecting `"correct.pt"`.
   - Assert ValueError.

5. **`test_strict_vs_error_metric_direction`**:
   - Create synthetic K, m with known strict/error labels.
   - Verify `strict_factuality_auroc` and `error_detection_auroc` are both
     valid AUROCs and that `error_detection_auroc == strict_factuality_auroc`
     (they should be equal since `AUROC(1-y, 1-s) == AUROC(y, s)`).

---

## Execution Order

Run fixes in this order:

1. Fix 4 (binomial validation) — no dependencies, prevents future data bugs
2. Fix 5 (cache verification) — no dependencies, prevents silent misalignment
3. Fix 6 (metric direction) — no dependencies, clarifies evaluation
4. Fix 2 (docstring updates) — no dependencies, documentation only
5. Fix 3 (boundary fraction) — depends on Fix 2 terminology
6. Fix 1 (generation logits) — **most impactful**, do last because it
   invalidates all existing data and requires regeneration
7. Fixes 7-9 (Priority 2) — after pilot infrastructure is stable

After Fix 1 is applied:
- Delete all existing `data/generations/` files
- Delete all existing `data/cache/` files
- Rerun `scripts/01_generate_data.py` and `scripts/01b_cache_scalars.py`

---

## Verification Checklist

After all fixes are applied, confirm:

- [ ] `pytest tests/` passes (including new tests from Fix 9)
- [ ] Generation loop: `logits[t]` predicts `token_ids[t]` (greedy test)
- [ ] Fisher scoring docstrings say "epsilon-stabilized", not "clipped gradient"
- [ ] Boundary fraction is logged during training
- [ ] `K > m` raises ValueError in trainer, evaluator, and Fisher scoring
- [ ] Cache loading verifies `source_path` and `token_ids`
- [ ] Evaluation reports both `strict_factuality_auroc` and `error_detection_auroc`
- [ ] `binomial_nll_full` metric available for paper reporting
- [ ] `verify_local_pd` uses separate `clip_eps` and `pd_tol`
