# Phase 2-1 — Feature Extractor

Implement `src/features/extractor.py`.

**Mathematical definition** (see research_document Part VI):

Per-token feature:
  z_ℓ = [W · h_ℓ^agg, entropy_ℓ, top1_ℓ] ∈ R^k

where:
  h_ℓ^agg = Σ_l α_l · h_ℓ^(l)   (layer-weighted hidden state)
  α_l = softmax(α)_l

Learnable parameters ψ:
  - W ∈ R^{p × d}  (projection, d=hidden_dim → p=projection_dim)
  - α ∈ R^{L_layers}  (layer weights before softmax)
  - μ_0 ∈ R^k  (prior mean)
  - log σ_0 ∈ R^k  (log diagonal prior std)

Dimension: k = p + 2 (projection_dim + entropy + top1)

**Model-agnostic**: hidden_dim and num_layers are NOT hardcoded.
They must be provided explicitly (from model.config at runtime).

**Requirements**:

1. Class `SentenceUQParams(nn.Module)`:
   - `__init__(self, hidden_dim: int, num_layers: int, projection_dim: int = 64)`
   - **No default for hidden_dim or num_layers** — must be provided from model.config
   - Learnable parameters:
     * `W`: nn.Linear(hidden_dim, projection_dim, bias=False)
     * `alpha`: nn.Parameter(torch.zeros(num_layers))
     * `mu_0`: nn.Parameter(torch.zeros(projection_dim + 2))
     * `log_sigma_0`: nn.Parameter(torch.zeros(projection_dim + 2))
   - Property `feature_dim` → projection_dim + 2
   - Method `get_Sigma_0_inv()` → diagonal matrix from exp(-2*log_sigma_0)
   - Method `get_Sigma_0()` → diagonal matrix from exp(2*log_sigma_0)

2. Function `extract_token_features(hidden_states, entropy, top1_prob, params)`:
   - Input: hidden_states (T, num_layers, hidden_dim), entropy (T,), top1_prob (T,), params
   - Compute: w = softmax(params.alpha), h_agg via einsum, h_proj = params.W(h_agg),
     z = concat([h_proj, entropy.unsqueeze(1), top1_prob.unsqueeze(1)], dim=1)
   - Returns: (T, k) Tensor

3. Function `extract_sentence_token_features(hidden_states, entropy, top1_prob, token_range, params)`:
   - Extract features for tokens in [start, end) only
   - Returns: (L_j, k) Tensor

4. Function `extract_sentence_aggregate_feature(z_tokens)`:
   - For auxiliary model (Part VIII): concat of [mean(z), std(z), z_last]
   - Returns: (3k,) Tensor
   - Edge case: L_j = 1 → std = 0

**Tests `tests/test_features.py`**:
- Feature dim == k == projection_dim + 2
- Gradients flow through W, alpha, hidden_states
- Works with single layer
- **Parameterize over multiple model configs** to catch hardcoded assumptions:
  * Config A: hidden_dim=4096, num_layers=8 (Llama-like)
  * Config B: hidden_dim=2048, num_layers=6 (small model)
  * Config C: hidden_dim=3584, num_layers=10 (Gemma-like)
