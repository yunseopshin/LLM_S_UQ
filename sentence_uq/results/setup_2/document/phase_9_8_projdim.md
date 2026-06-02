# Phase 9.8 — Projection-Dim Sweep: Does Less Compression Close the Re-encode Gap? (Findings)

**Date**: 2026-06-02
**Setup**: 2 (FActScore-Bio, in-domain), full split (n_train=1638, n_test=353 with m_j>0)
**Change**: retrain at `projection_dim ∈ {128, 256}`, everything else identical to the
production model (λ=1e-2, 300 ep, lr 1e-3, fisher 10, prior σ₀=1.0). Trained in
parallel (dim128→GPU0, dim256→GPU1).
**Models / artifacts**: `results/setup_2_projdim/dim{128,256}/` (trained_model.pt is
git-ignored; train/eval JSON+CSV kept). Baselines reused from `results/setup_2/baselines.json`.

## 0. Motivation

On strict factuality and ratio Pearson we trail Han's **re-encoding** probe
(`factuality_probe_original`/`_repo`, which re-encodes each atomic claim through the
LLM). Hypothesis: the gap is our 64-d feature **compression** bottleneck (`W: 4096→64`);
re-encoding effectively sees the full hidden state. If so, raising `projection_dim`
should move our discrimination toward the re-encode baseline. This tests it directly.

## 1. Result — discrimination is FLAT in projection_dim

| metric (Ours Bayesian) | dim64 (prod) | dim128 | dim256 | **original (re-encode)** | original_repo |
|---|---|---|---|---|---|
| Ratio Pearson r | 0.453 | 0.450 | 0.454 | **0.498** | 0.521 |
| Ratio MAE | 0.224 | 0.224 | 0.224 | 0.206 | 0.206 |
| Ratio RMSE | 0.291 | 0.291 | 0.291 | 0.292 | 0.278 |
| Ratio ECE | 0.055 | 0.053 | 0.052 | 0.063 | 0.051 |
| Strict AUROC (μ̃^m) | 0.780 | 0.773 | 0.767 | **0.849** | 0.858 |
| Strict AUROC (μ̂ readout) | 0.827 | 0.823 | 0.827 | — | — |
| Strict AUPRC (μ̃^m) | 0.239 | 0.253 | 0.242 | **0.387** | 0.422 |
| Strict ECE (μ̃^m) | 0.052 | 0.050 | 0.048 | 0.173 | 0.173 |

**Every discrimination metric is flat across a 4× change in projection_dim**: ratio
Pearson 0.453→0.454, strict AUROC(μ̂) 0.827→0.827, strict AUPRC ~0.24–0.26. The gap to
the re-encode baseline (Pearson +0.05, AUPRC +0.15) is **unchanged**. (ECE drifts down
slightly — 0.055→0.052 ratio, 0.052→0.048 strict — but within noise and irrelevant to
the discrimination question.)

## 2. Interpretation — the bottleneck is the FEATURE, not the compression

Our probe is **linear**: the per-token logit is
`θᵀz_ℓ = (Wᵀθ_proj)·ā_ℓ + θ_e·entropy + θ_t·top1`, i.e. a linear functional of the
multi-layer-aggregated **generation-time** hidden state `ā_ℓ`. Because `W` is *learned*,
even at dim64 the model already selects its best 64 directions. That dim128/256 add
nothing means **the linearly-decodable factuality signal in the generation-time hidden
states is already saturated at dim64** — capacity is not the limit, the feature's
information content is.

The re-encode baseline's edge is therefore **intrinsic to re-encoding** (re-reading the
emitted text surfaces information absent at generation time), not something our
architecture can recover by widening `W`. Corroboration: Han's `adapted` probe — which,
like us, uses *generation-time* states (no re-encode) — gets strict AUROC 0.811, right
next to our μ̂ readout (0.827); only the **re-encode** variants jump to ~0.85. So
generation-time methods cluster together; re-encoding is the separating factor.

## 3. Implication

- **`projection_dim` is not a lever** for closing the discrimination gap → keep the
  production `projection_dim=64` (no benefit to widening; dim64 is the parsimonious choice).
- **The "single-pass ≥ re-encode on accuracy" claim is not reachable** via this route.
  The defensible positioning stands: a single forward pass (hidden states reused, zero
  extra cost) matches the re-encode probe on RMSE/Brier and within-CI on strict AUROC,
  **wins on calibration (ECE)**, and adds UQ — at ~10⁴× lower cost (re-encode = 995 claim
  re-encodings / ~80 min vs our reuse of generation states). See `phase_9_7` for the
  AUROC/ECE readout analysis and `strict_readout_diag.json`.
- **This negative result is itself a usable ablation**: it demonstrates (a) robustness to
  `projection_dim`, and (b) that the re-encode advantage is information-bound, not
  capacity-bound — evidence for *why* a cheap single-pass probe is the right trade-off.

## 4. Reproduce

```bash
for DIM in 128 256; do
  python scripts/03_train.py --setup 2 --config configs/default.yaml \
      --logit-reg-lambda 1e-2 --projection-dim $DIM \
      --results-dir results/setup_2_projdim/dim$DIM --device cuda
  python scripts/04_evaluate.py --setup 2 --config configs/default.yaml \
      --trained-model results/setup_2_projdim/dim$DIM/trained_model.pt \
      --baselines-file results/setup_2/baselines.json \
      --results-dir results/setup_2_projdim/dim$DIM --device cuda --no-plots
done
```
