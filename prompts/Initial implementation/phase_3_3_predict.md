# Phase 3-3 — Predictive Inference (Binomial)

Implement `src/inference/predict.py`.

**Mathematical definition** (see research_document_v8 Part IV, V):

Given trained (theta_hat, Sigma_hat, feature_params), for a new sentence with m_* atomic facts:

  z_ℓ = φ_ψ(h_ℓ)
  π̂_ℓ = σ(θ̂^T z_ℓ)
  μ̂ = (1/L) Σ_ℓ π̂_ℓ
  ĝ = (1/L) Σ_ℓ π̂_ℓ(1-π̂_ℓ) z_ℓ

**Uncertainty decomposition** (four levels, research_document_v8 §4.2–4.4):

  Latent level (μ):
    Epi_μ = ĝ^T Σ̂ ĝ

  Ratio level (U = K/m):
    Aleatoric_U = max(0, (μ̂(1-μ̂) - Epi_μ) / m_*)
    Total_U = Aleatoric_U + Epi_μ

  Count level (K):
    Epi_K = m_*^2 · Epi_μ
    Aleatoric_K = m_* · max(0, μ̂(1-μ̂) - Epi_μ)

  Strict factuality (A = 1{K=m}):
    p(A=1) = μ̂^{m_*}   (plug-in)
    or posterior-averaged: E_θ[μ(θ)^{m_*}]  (via MC)

  Key: Aleatoric_U has 1/m_* factor — more atomic facts → less ratio noise.
  When m_* is unknown at inference time, report only latent-level Epi_μ.

Token-level (unchanged from v7):
  Attr_ℓ = (1/L) g_ℓ^T Σ̂ ĝ          (signed, sums to Epi_μ)
  LocalEpi_ℓ = [π̂_ℓ(1-π̂_ℓ)]^2 · z_ℓ^T Σ̂ z_ℓ   (always non-negative)

Probit-shrinkage (unchanged):
  π̃_ℓ = σ(θ̂^T z_ℓ / sqrt(1 + (π/8) z_ℓ^T Σ̂ z_ℓ))

**Requirements**:

1. Class `Predictor`:
   - `__init__(self, theta_hat, Sigma_hat, feature_params, use_probit_shrinkage=False)`
   
   - Method `predict_sentence(z_tokens, m_j=None)`:
     * Input: z_tokens (L, k) Tensor, m_j (int or None)
     * Returns: dict with:
       - "mu_hat": float (latent factuality probability)
       - "p_factual_probit": float (probit-shrunk)
       - "epi_mu": float (latent-level epistemic)
       - "aleatoric_U": float or None (ratio-level, requires m_j)
       - "total_U": float or None (ratio-level total, requires m_j)
       - "epi_K": float or None (count-level epistemic, requires m_j)
       - "aleatoric_K": float or None (count-level, requires m_j)
       - "p_strict_factual": float or None (μ̂^{m_j}, requires m_j)
       - "token_pi": (L,), "token_attr": (L,), "token_local_epi": (L,)
     * When m_j is None, ratio/count/strict fields are None
   
   - Method `predict_from_hidden_states(hidden_states, entropy, top1, token_range, m_j=None)`:
     * High-level wrapper including feature extraction
   
   - Method `predict_mc_epistemic(z_tokens, num_samples=100, m_j=None)`:
     * Sample θ^(s) ~ N(θ_hat, Σ_hat), compute μ_* for each
     * Returns: var(μ) as MC epistemic at latent level
     * If m_j given, also return MC estimates at ratio and count levels

2. Class `BatchPredictor`:
   - Vectorized prediction over multiple sentences (each with its own m_j)

3. Functions `save_trained_model(path, ...)` and `load_trained_model(path)`:
   - Serialize/deserialize theta_hat, Sigma_hat, feature_params state_dict

**Tests `tests/test_decomposition.py`**:
- Invariants: epi_mu >= 0, aleatoric_U >= 0 (after clipping), mu_hat ∈ [0,1]
- sum(token_attr) ≈ epi_mu (within float precision)
- all(token_local_epi >= 0)
- Bernoulli special case: m_j=1 → aleatoric_U = max(0, μ̂(1-μ̂) - epi_μ), same as v7 Total - Epi
- Large m_j → aleatoric_U shrinks toward 0 (more obs → less ratio noise)
- MC vs linear: close for small Sigma_hat, can diverge for large (expected)
- m_j=None: ratio/count/strict fields are None, no error
