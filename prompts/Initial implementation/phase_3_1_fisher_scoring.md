# Phase 3-1 — Fisher Scoring Inner Loop (Binomial)

Implement `src/models/fisher_scoring.py`.

**Mathematical definition** (see research_document_v8 Part III, VII):

Clipped objective (binomial):
  L̃(θ) = Σ_j [K_j log μ̃_j + (m_j - K_j) log(1 - μ̃_j)] - (1/2)(θ - μ_0)^T Σ_0^{-1} (θ - μ_0)
  where μ̃_j = clip(μ_j, ε, 1-ε), m_j = atomic fact count, K_j = supported count
  Note: sentences with m_j = 0 contribute ℓ_j(θ) = 0 (skip)

Gradient:
  ∇_θ L̃ = -Σ_0^{-1}(θ - μ_0) + Σ_j R_j^bin g_j
  R_j^bin = (K_j - m_j μ̃_j) / (μ̃_j (1 - μ̃_j))
  g_j = (1/L_j) Σ_ℓ π_ℓ(1-π_ℓ) z_ℓ

Fisher-type precision (m_j-weighted):
  H_fisher = Σ_0^{-1} + Σ_j (m_j / (μ̃_j(1-μ̃_j))) g_j g_j^T

Damped Fisher scoring update:
  θ ← θ + (H_fisher + λI)^{-1} ∇_θ L̃

**Key change from v7**: F_j ∈ {0,1} → (K_j, m_j) binomial counts.
  - R_j becomes R_j^bin = (K_j - m_j μ_j) / (μ_j(1-μ_j))
  - Fisher weight becomes m_j / (μ_j(1-μ_j)) instead of 1 / (μ_j(1-μ_j))
  - Bernoulli is recovered when m_j = 1, K_j ∈ {0, 1}

**Requirements**:

1. Function `_compute_grad_and_fisher(theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps=1e-6)`:
   - Input: theta (k,), all_z_tokens (list of N tensors, each (L_j, k)),
     all_K (N,) int tensor, all_m (N,) int tensor, mu_0 (k,), Sigma_0_inv (k,k), eps
   - Skip sentences with m_j = 0
   - For each sentence j: compute pi_j, mu_j (clamped), g_j, R_j^bin
   - Accumulate gradient and Fisher-type Hessian (with m_j weighting)
   - Returns: grad (k,), H_fisher (k, k)
   - **Must be differentiable** (no detach, for backward through unrolled loop)

2. Function `_compute_clipped_objective(theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps=1e-6)`:
   - Returns: scalar L̃(θ) (binomial log-likelihood, skip m_j=0)

3. Function `fisher_scoring_map(all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, num_iters=15, eps=1e-6, lambda_init=1e-4, verbose=False)`:
   - Damped Fisher scoring algorithm:
     * Initialize theta = mu_0.clone()
     * At each iteration: compute grad & H, solve for delta, check improvement
     * Adaptive damping: decrease λ on success, increase on failure
     * Stop if λ > 1e10
   - Recompute H_fisher at final theta
   - Returns: theta_hat (k,), H_fisher_final (k, k)
   - **Must be differentiable** (unrolled optimization for outer loop backward)

4. Function `fisher_scoring_map_detached(...)`:
   - Same algorithm but with torch.no_grad() — for inference only

**Important**:
- Use torch.linalg.solve for the linear system
- Fallback: if solve fails, increase λ and retry
- Keep num_iters moderate (10-15) to limit memory in unrolled backward
- All tensor operations, no in-place ops that break autograd
- Skip m_j = 0 sentences in all computations

**Tests `tests/test_fisher_scoring.py`**:
- Synthetic data (k=5, N=20 sentences): convergence check with random m_j ∈ {1..5}, K_j ∈ {0..m_j}
- Bernoulli special case: all m_j=1, K_j ∈ {0,1} → should match old Bernoulli version
- Extreme case: all K_j=0 or all K_j=m_j → MAP shifts from μ_0 appropriately
- m_j=0 sentences: verify they are skipped without error
- Gradient check: torch.autograd.gradcheck on small size
- Fisher-type PD at convergence
