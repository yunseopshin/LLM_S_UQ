# Phase 4-2 — Auxiliary Bayesian Regression Model

Implement `src/models/bayesian_aux.py`.

**Model** (see research_document Part VIII):
Logit-transformed Bayesian Gaussian regression.

  V_j := logit(U_j*) ~ N(θ^T z_j, σ^2)
  θ ~ N(μ_0, Σ_0)

Exact conjugate posterior:
  Σ_N^{-1} = Σ_0^{-1} + (1/σ^2) Z^T Z
  θ_N = Σ_N (Σ_0^{-1} μ_0 + (1/σ^2) Z^T V)

Predictive (logit space):
  V_* ~ N(θ_N^T z_*, σ^2 + z_*^T Σ_N z_*)

**Requirements**:

1. Function `safe_logit(u, eps=1e-3)`:
   - Clip u to [eps, 1-eps], then compute logit

2. Class `BayesianLogitRegression`:
   - `__init__(self, feature_dim, prior_mu=None, prior_sigma=1.0, noise_sigma=0.1)`
   
   - Method `fit(Z, U_star)`:
     * Z: (N, feature_dim), U_star: (N,) in [0, 1]
     * V = safe_logit(U_star)
     * Compute closed-form Σ_N, θ_N
   
   - Method `predict(z_new)`:
     * Returns: dict with p_factual, epistemic_logit, aleatoric_logit, logit mean/var
   
   - Method `estimate_noise_variance(Z, U_star)`:
     * Residual-based: σ^2 = (1/(N-k)) Σ (V_j - θ_N^T z_j)^2

3. Script `scripts/04_train_aux.py`:
   - Train auxiliary model
   - Z = sentence-level aggregate features
   - U_star = target uncertainty from expensive method (offline precomputed)

**Tests `tests/test_bayesian_aux.py`**:
- Synthetic data with known θ: recovery check
- Sufficient statistics T_1, T_2 reproduce posterior
