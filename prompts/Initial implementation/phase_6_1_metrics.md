# Phase 6-1 — Evaluation Metrics (Binomial)

Implement `src/evaluation/metrics.py`.

**Key change from v7**: Evaluation is now two-tiered:
  - **Primary**: ratio-level (U_j = K_j/m_j) — continuous target
  - **Secondary**: strict factuality (A_j = 1{K_j = m_j}) — binary target

**Requirements**:

1. Function `compute_ratio_level_metrics(U_true, mu_hat, m_j=None)`:
   - U_true: (N,) in [0, 1] — observed factuality ratio K_j/m_j
   - mu_hat: (N,) — predicted μ̂_j
   - m_j: (N,) optional — for binomial NLL
   - Returns: {
       "MAE": float,
       "RMSE": float,
       "Pearson_r": float,
       "binomial_NLL": float (if m_j given) — Σ -[K_j log μ̂_j + (m_j-K_j) log(1-μ̂_j)] / N
     }

2. Function `compute_strict_factuality_metrics(A_true, p_strict, uncertainty)`:
   - A_true: (N,) in {0, 1} — strict factuality indicator 1{K_j = m_j}
   - p_strict: (N,) — predicted P(A_j=1) = μ̂_j^{m_j}
   - uncertainty: (N,) — higher = more uncertain (for ranking)
   - Returns: {"AUROC": float, "AUPRC": float, "Brier": float, "ECE": float}

3. Function `compute_calibration_metrics(y_true, p_pred, n_bins=10)`:
   - General-purpose calibration (works for both ratio and strict)
   - Returns: {"Brier": float, "ECE": float}
   - ECE: equal-width binning

4. Function `compute_prr(y_true, uncertainty, num_thresholds=100)`:
   - Prediction Rejection Ratio (works on both ratio and strict targets)
   - Remove highest-uncertainty samples first, measure remaining quality
   - Returns: {"rejection_rates", "remaining_quality", "prr_auc"}

5. Function `compute_bootstrapped_ci(y_true, scores, metric_fn, n_bootstrap=1000, alpha=0.05)`:
   - Bootstrap confidence intervals (following Han et al.'s reporting style)
   - Returns: {"mean", "lower", "upper"}

6. Function `plot_reliability_diagram(y_true, p_pred, n_bins=10, save_path=None, title="")`:
   - Matplotlib reliability diagram with y=x diagonal

7. Function `compare_mc_vs_linear_epistemic(predictor, test_sentences, num_mc_samples=100)`:
   - Compare linear approximation vs MC epistemic for each sentence
   - Returns: Pearson correlation, MAE, scatter plot data

8. Function `full_evaluation(predictions, K_true, m_true, uncertainties)`:
   - Compute all metrics at once (both ratio-level and strict)
   - Return pandas DataFrame

**Tests `tests/test_metrics.py`**:
- Ratio: perfect prediction U_true == mu_hat → MAE=0, Pearson=1
- Strict: perfect ranking → AUROC=1.0
- Perfect calibration: p_pred = y_true → Brier=0, ECE≈0
- Binomial NLL: known closed-form case
