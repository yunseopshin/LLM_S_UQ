# Phase 3-2 — Main Bayesian Model Class (Binomial)

Implement `src/models/bayesian_main.py`.

**Reference**: research_document_v8 Part II, III, VII.

**Key change from v7**: Observation model is Binomial(K_j | m_j, μ_j(θ)), not Bernoulli(F_j | μ_j(θ)).

**Requirements**:

1. Class `BayesianSentenceUQ(nn.Module)`:
   - `__init__(self, feature_params: SentenceUQParams, num_fisher_iters=10, eps=1e-6)`
   
   - Method `compute_map(all_z_tokens, all_K, all_m, differentiable=True)`:
     * Call fisher_scoring_map (differentiable) or fisher_scoring_map_detached
     * Returns: theta_hat, H_fisher
   
   - Method `compute_loss(all_z_tokens, all_K, all_m)`:
     * theta_hat, H_fisher = compute_map(..., differentiable=True)
     * loss = Σ_j [-K_j * log(μ̃_j) - (m_j - K_j) * log(1-μ̃_j)]  (skip m_j=0)
     * Returns: scalar loss
     * theta_hat is differentiable → backward propagates to feature_params
   
   - Method `predict(z_tokens, m_j=None)`:
     * For post-training inference (requires stored theta_hat, Sigma_hat)
     * m_j is optional — needed for ratio-level uncertainty decomposition
     * Implemented in Phase 3-3

2. Function `verify_local_pd(theta_hat, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps=1e-6)`:
   - Check both Fisher-type and true Hessian (clipped) for PD
   - True Hessian via torch.autograd.functional.hessian (binomial objective)
   - Returns: dict {
       "fisher_min_eig": float,
       "true_min_eig": float,
       "fisher_pd": bool,
       "true_pd": bool,
       "laplace_valid_local": bool
     }

**Important**:
- Loss uses sum (not mean) over sentences — consistent with prior scaling
- Skip m_j=0 sentences in both loss and gradient computation
- verify_local_pd is expensive (k^2 backward passes for autograd hessian),
  so call every 5 epochs, not every epoch
