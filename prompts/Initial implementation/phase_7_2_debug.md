# Phase 7-2 — Debugging Utilities (Binomial)

Implement `src/utils/debug.py`.

**Requirements**:

1. Function `check_gradient_flow(loss, params)`:
   - After loss.backward(), print grad norm for each parameter (W, alpha, mu_0, log_sigma_0)
   - Warn if any grad is None

2. Function `visualize_feature_distribution(feature_params, sample_hidden_states, save_path)`:
   - Plot distribution of projected features (per-dimension histogram)
   - Bar chart of learned layer weights softmax(alpha)
   - Compare alpha distribution with Han et al.'s finding (layer 14 optimal):
     annotate the plot with "Han et al. optimal: layer 14" reference line

3. Function `diagnose_fisher_scoring(all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps=1e-6)`:
   - Run fisher_scoring_map with verbose=True
   - Print per-iteration: objective, gradient norm, H min eigenvalue
   - Warn if not converging
   - Report: how many sentences have m_j=0 (skipped), distribution of m_j values

4. Function `sanity_check_boundary_fraction(all_z_tokens, all_K, all_m, theta_hat, eps=1e-6)`:
   - Report % of sentences where μ_j hits clip boundary (ε or 1-ε)
   - If >5%, recommend tightening prior
   - Also report distribution of U_j = K_j/m_j vs μ̂_j (scatter)

5. Function `check_m_j_distribution(all_m)`:
   - Print summary stats: min, max, mean, median of m_j
   - Count m_j=0 sentences (should be rare)
   - Warn if m_j distribution is highly skewed (§XV.3 dominance concern)

All functions should be notebook-compatible (matplotlib inline).

**Common issues and diagnostics**:
- Tokenizer / token_ids length mismatch → re-tokenize
- Hidden state > GPU memory → offline storage + lazy loading
- Fisher scoring not converging → reduce prior_sigma or increase lambda_init
- Val metrics degrading → overfitting, tighten prior or early stop
- verify_local_pd: true_pd=False → tighter prior
- Han et al. baseline reproduction: claim decomposition requires auxiliary LM API calls,
  budget for API costs and rate limits
- m_j=0 sentences: these are skipped in likelihood, check they're not too many
- Large m_j dominance: if a few sentences with many atomic facts dominate the objective,
  consider α-weighting ablation (§XV.3)
