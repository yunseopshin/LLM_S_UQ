# Phase 6-2 — Evaluation Script (Binomial)

Implement `scripts/04_evaluate.py`.

**Key change from v7**: Two-tiered evaluation aligned with binomial model.
  - **Primary**: ratio-level (U_j = K_j/m_j, continuous)
  - **Secondary**: strict factuality (A_j = 1{K_j = m_j}, binary)
  - **New metric**: Binomial NLL — directly tests count-aware model fit

**Full evaluation pipeline**:

1. For all test sentences, compute:
   - Our method (Main + Aux) predictions: μ̂_j, epi_μ, aleatoric_U, p_strict
   - All baseline predictions (use cached results)

2. Per method, compute **ratio-level** metrics:
   - MAE, RMSE, Pearson r (μ̂_j vs U_j)
   - Binomial NLL (only for our method — baselines don't model counts)
   - Calibration: Brier, ECE (μ̂_j vs U_j)
   - PRR with epistemic uncertainty as rejection signal

3. Per method, compute **strict factuality** metrics:
   - AUROC, AUPRC (hallucination detection: A_j = 0 vs 1)
   - Brier, ECE (p_strict vs A_j)
   - Bootstrapped 95% CI (following Han et al.)
   - Inference time (FLOPs or wall-clock)

4. Save results:
   - results/final_metrics_ratio.csv: method × ratio-level metric table
   - results/final_metrics_strict.csv: method × strict metric table
   - results/reliability_diagrams/: per-method (both ratio and strict)
   - results/prr_curves.png: all methods' PRR curves overlaid
   - results/mc_vs_linear.png: linear approx vs MC scatter plot
   - results/token_heatmaps/: example sentences with token-level attribution

5. Key ablations:
   - Bayesian vs Point estimate (Sigma on/off)
   - Uniform vs Attention weights
   - Linear approximation vs MC
   - Laplace-EB correction vs without
   - **Ours vs Factuality Probe (Han et al.)**: ECE and rejection curve are the key axes
   - **Binomial vs Bernoulli**: m_j=1 ablation — does count awareness help?
   - **Layer alpha distribution**: visualize learned alpha as bar chart,
     compare with Han et al.'s finding that layer 14 is optimal.
   - **Generation-time vs re-encoded hidden state**: compare Factuality Probe
     original (re-encode) vs adapted (generation-time)

**Expected output format**:

```
=== Ratio-Level Metrics (Primary) ===
                    MAE      RMSE     Pearson   Binom NLL  ECE
Ours (Bayesian)    0.120    0.180    0.780     1.250      0.060
Ours (Point)       0.130    0.190    0.760     N/A        0.090
Fact Probe (Han)   0.145    0.210    0.720     N/A        0.110

=== Strict Factuality Metrics (Secondary, with bootstrapped 95% CI) ===
                    AUROC    AUPRC    Brier    ECE      Time(ms)
Token Entropy      0.650    0.550    0.280    0.150    1
Fact Probe (Han)   0.735    0.640    0.245    0.130    15
Fact Probe (adapt) 0.730    0.635    0.250    0.135    10
LUQ (m=10)         0.740    0.640    0.230    0.110    5000
Semantic Entropy   0.760    0.660    0.220    0.100    5000
Log Reg            0.740    0.640    0.230    0.115    5
Ours (Main)        0.770    0.680    0.210    0.095    10
Ours (Aux)         0.765    0.670    0.215    0.100    5

=== Bayesian vs Point Ablation ===
                    AUROC    Brier    ECE      Epi Info   Binom NLL
Fact Probe (Han)   0.735    0.245    0.130    N/A        N/A
Ours (Point)       0.765    0.240    0.145    N/A        N/A
Ours (Bayesian)    0.770    0.210    0.095    Yes        1.250

=== Binomial vs Bernoulli Ablation ===
                    Binom NLL  Ratio MAE  Strict ECE
Ours (Binomial)    1.250      0.120      0.095
Ours (Bernoulli)   N/A        0.145      0.110

=== Layer Weight Analysis ===
Learned alpha distribution vs Han et al. layer 14 finding
```

Read config from configs/, accept experiment name via argparse.
