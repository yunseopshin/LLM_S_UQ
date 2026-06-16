# Phase 10-1 — Work Log / Session Summary

A concise record of the Phase 10-1 (Beta-Binomial) work: what was done, what was
found, and where each artifact lives. The full technical write-up is
[`phase_10_1_beta_binomial.md`](phase_10_1_beta_binomial.md); this file is the
high-level index.

---

## 1. What was built

Beta-Binomial observation model as a **config-selectable peer** to the Binomial,
sharing the inner loop / posterior / trainer / prediction code (only the
per-sentence scalars differ), selected by `model.likelihood: binomial |
beta_binomial`.

| Area | Files |
|---|---|
| New likelihood interface + two peers + factory | `src/models/likelihood.py` |
| Gated `log_phi` / `phi` on the params | `src/features/extractor.py` |
| Scalars routed through `likelihood` (default `None` → Binomial) | `src/models/fisher_scoring.py` |
| Live-`phi` likelihood built per forward; outer NLL | `src/models/bayesian_main.py` |
| Overdispersion factor `f = 1+(m-1)ρ`; strict via `log_prob_all_correct`; save/load likelihood | `src/inference/predict.py` |
| `phi_hat`/`rho_hat` logging; **separate `phi_lr`** Adam group | `src/train/trainer.py` |
| Pre-flight dispersion (`m_j` dist + method-of-moments ρ) | `src/utils/dispersion.py` |
| The switch + `setup_{1,2,3}_betabinomial.yaml` + `optim.phi_lr` | `configs/`, `scripts/03_train.py`, `scripts/04_evaluate.py` |
| 27 tests (spec §5) | `tests/test_beta_binomial.py` |

Status: full suite **326 passed**. Out of scope and untouched:
`src/models/bayesian_aux.py`, the `μ_0` / `log σ_0` prior treatment.

## 2. Cross-check (adversarial multi-agent) → 2 defects fixed

Six independent falsification agents + synthesis audited the change. Verdict:
**GO-WITH-FIXES**. Two defects found and fixed:

- **P0 — bit-identity violation.** The line-search objective was rewritten as
  `obj - neg_log_pmf` (`obj + (A+B)`) instead of the historical term-wise
  `(obj + A) + B`; this IEEE reassociation flipped near-tie line-search
  decisions and perturbed the **float32** MAP in 83/300 cases (up to
  `|Δθ|=2.2e-3`). Fixed by restoring term-wise accumulation for the Binomial
  path → verified **0/300** divergence vs an independent pre-10-1 re-implementation.
  Guarded by `test_fisher_map_float32_golden_matches_pre_10_1`.
- **P2 — float32 cancellation at large φ.** `digamma`/`lgamma` differences lose
  precision in float32 for large `phi*mu`; fixed by computing the gamma scalars
  in float64 internally and casting back to the working dtype.

The other five dimensions PASSED with independent fp64 numbers (errors
1e-11…1e-14): scalar correctness vs autograd/scipy, the φ→∞ limits, gradient
flow into `log_phi`, the predictive decomposition, and spec completeness.

## 3. φ → ∞ reduction (theory check)

At **fixed ψ**, Beta-Binomial(φ) → Binomial as φ grows, with the gap shrinking
~`O(1/φ)` to a float floor near φ≈1e6 (then a U-turn from float64 cancellation).
The only deliberate residual is the epistemic covariance (observed vs expected
Fisher, coinciding only at the optimum `K=mμ`). Comparing *fully trained* weights
is the wrong probe — the bilevel optimisation is chaotic, so any tiny step-1 gap
saturates to ~0.1 after a few epochs regardless of φ. (Detail: `phase_10_1_beta_binomial.md` §6.)

## 4. Setup 2 experiment — Binomial vs Beta-Binomial

Full run on GPU 1, 300 epochs, identical data/split/init/`lr`/`λ`; only the
likelihood differs (`log_phi` got `phi_lr=0.05`, ψ kept `lr=1e-3`).

- **Pre-flight (raw):** method-of-moments ρ=0.356, φ=1.81 → "overdispersion strong".
- **Fitted:** **φ̂=1792, ρ̂≈6e-4** — essentially Binomial. φ̂ trajectory is
  non-monotonic: dips to 37.3 (epoch 11), then climbs to 1792 (epoch 300, still
  rising) as μ_j(θ) absorbs the apparent overdispersion.
- **Metrics (test, n=353, all deltas < 0.004, 95% CIs overlap → indistinguishable):**

  | | Binomial | Beta-Binom | Δ |
  |---|---|---|---|
  | Ratio MAE | 0.22368 | 0.22424 | +0.0006 |
  | Ratio ECE | 0.05510 | 0.05312 | −0.0020 |
  | Strict AUROC | 0.77971 | 0.77596 | −0.0037 |
  | Strict ECE | 0.05182 | 0.05321 | +0.0014 |

- **Conclusion:** the raw overdispersion is a *mean-misspecification artifact*;
  once μ_j(θ) is fit there is no meaningful *residual* overdispersion, so the
  Beta-Binomial converges to the Binomial. Per the Go/No-Go guidance this points
  the search elsewhere (token-saliency pooling, focal / calibration loss). The
  Beta-Binomial path is validated and remains available for datasets that do
  exhibit residual overdispersion.

Artifacts: `results/setup_2_betabinomial/` — `train.log`, `phi_trajectory.csv`,
`final_metrics_{ratio,strict}.csv`, `comparison_vs_binomial.txt`,
`train_summary.json`. Run record: `phase_10_1_beta_binomial.md` §10.

## 5. How to reproduce

```bash
# Binomial peer (default):
python scripts/03_train.py    --setup 2 --config configs/setup_2.yaml
python scripts/04_evaluate.py --setup 2 --config configs/setup_2.yaml

# Beta-Binomial peer (only the likelihood differs):
CUDA_VISIBLE_DEVICES=1 python -u scripts/03_train.py --setup 2 \
    --config configs/setup_2_betabinomial.yaml
python scripts/04_evaluate.py --setup 2 --config configs/setup_2_betabinomial.yaml --no-plots
```

## 6. Open / possible next steps

- Repeat on Setup 1 (cross-domain) and Setup 3 (multi-domain) via the
  `setup_{1,3}_betabinomial.yaml` configs.
- Regenerate the comparison plots (reliability diagrams, PRR) if needed.
- Commit the Phase 10-1 changes.
- Pursue the epistemic-collapse fix elsewhere (token-saliency pooling, focal /
  calibration loss), since the observation model is not the lever here.
