# Phase 9.4 — Saturation Remedy: Retraining Options (Design Note)

**Date**: 2026-05-28
**Setup**: 2 (FActScore-Bio, in-domain)
**Status**: design comparison — no code changed yet, no retraining run yet.

---

## 0. Why we are here

The temperature sweep (`phase_9_3_ood_findings.md` §7, `document/temperature_sweep.*`)
established:

- **Root cause of the epistemic collapse is per-token over-confidence.** The probe's
  per-token logits `θ̂ᵀz_ℓ` have **median |logit| = 21** (p90=52, max=132); 92% of
  tokens are saturated, so `ĝ = mean π̂(1-π̂)z ≈ 0` and `epi_μ = ĝᵀΣ̂ĝ ≈ 8e-4`.
- **Post-hoc temperature cannot fix it** — the honest readout carries a `1/T²`
  penalty that beats the gradient growth.
- **But the ĝ-channel has ~25× headroom** (the un-penalised `ĝᵀΣ̂ĝ` rises to 2.05e-2
  at T=8), and **μ̂ calibration is nearly invariant to logit scale** (ECE stays
  0.06–0.07). So shrinking logits during *training* should enlarge `epi_μ` while
  preserving sentence-level predictions.

`epi_μ` benefits from **two channels** when logits shrink:
1. **ĝ channel** — less saturation → larger `π̂(1-π̂)` → larger `ĝ`.
2. **Σ̂ channel** — less extreme `μ̂` → larger `μ̂(1-μ̂)` → smaller binomial Fisher
   weight `mⱼ/(μ̂(1-μ̂))` → larger posterior covariance `Σ̂`.

This note compares three retraining interventions to reduce that over-confidence.

> **Shared constraint (CLAUDE.md rule 9)**: the regulariser must stay differentiable
> w.r.t. ψ = {W, α, μ₀, σ₀}. `θ̂` comes from the inner Fisher loop and depends on ψ;
> do **not** detach it. Options 1–2 are added inside `BayesianSentenceUQ.compute_loss`
> (`src/models/bayesian_main.py`); the outer Adam loop (`trainer.train_epoch`) is
> unchanged. A new config knob (e.g. `training.logit_reg_lambda`) exposes the strength.

---

## 1. Option A — Logit L2 penalty  *(recommended)*

**Loss**
```
L(ψ) = binomial_NLL(ψ)  +  λ · (1/N) Σ_j  (1/L_j) Σ_ℓ (θ̂ᵀ z_ℓ)²
```

**Mechanism**: directly penalises large per-token logits → π̂ pulled away from 0/1.
Hits **both** channels (ĝ and Σ̂). The penalty is a smooth quadratic with a constant
gradient slope, so it keeps pulling even from deep saturation.

**Code**: ~10 lines in `compute_loss` (logits `θ̂ᵀz` are already computed there for the
likelihood); add `λ` to config. Localised — affects only training of this model.

**Trade-off**: λ too large → underfit (every μ̂ → 0.5), MAE/NLL degrade. Needs a sweep.
Calibration expected robust (sweep evidence). λ is in *logit²* units, so a natural
grid given median logit²≈440 is small: `{0, 1e-3, 3e-3, 1e-2, 3e-2}`.

**Paper framing**: "logit / confidence regularisation" — standard, easy to justify and
ablate.

---

## 2. Option B — Token entropy regularisation

**Loss**
```
L(ψ) = binomial_NLL(ψ)  −  λ · (1/N) Σ_j (1/L_j) Σ_ℓ H(π̂_ℓ),
       H(π) = −π log π − (1−π) log(1−π)
```

**Mechanism**: rewards high per-token entropy (π̂ → 0.5). Same goal as A. Its gradient
is *largest* exactly in the saturated regime, so it strongly un-saturates extreme
tokens — arguably a better-shaped pull than L2 there.

**Code**: same insertion point as A (needs eps-clipped logs for stability, fp32).

**Trade-off**: H is bounded (≤ log2), so "how hard" it pulls is harder to reason about
than a quadratic; its non-linear interaction with the binomial NLL is less transparent.
λ grid less intuitive. Risk of flattening μ̂ toward 0.5 like A if λ too big.

**Paper framing**: "maximum-entropy / confidence-penalty regularisation" — also standard
(cf. Pereyra et al. 2017), but one more moving part to explain than A.

---

## 3. Option C — Feature normalisation (z scaling)

**Idea**: bound the logit structurally by normalising the projected feature, e.g.
`ẑ = z / ‖z‖` (or per-dim standardisation), so `|θ̂ᵀẑ| ≤ ‖θ̂‖`. With ‖z‖ median≈5.3
today, this alone cuts logits ~5×.

**Mechanism**: shrinks logits without a loss term. But it changes the **feature
definition itself**, so it touches the **Σ̂ geometry** and every downstream readout
(`epi_logit`, probit shrinkage, token attribution) and **every setup (1/2/3)**.

**Code**: modify `src/features/extractor.py`. Caveat: `z = [W·Σα h, entropy, top1]` —
the 2 scalar features must **not** be normalised away, so only the 64-d projection part
can be scaled. More invasive and broader blast radius than A/B.

**Trade-off**: larger architectural change; affects all experiments and other phases;
re-opens questions already settled (e.g. the `epi_logit` analysis would need redoing).
Hardest to scope.

**Paper framing**: "feature normalisation" — common, but reads as an architecture change
rather than a targeted UQ fix.

---

## 4. Comparison

| Criterion | A: logit L2 | B: entropy reg | C: feature norm |
|---|---|---|---|
| Reduces logit magnitude | ✅ direct | ✅ direct | ✅ structural |
| ĝ channel ↑ | ✅ | ✅ | ✅ |
| Σ̂ channel ↑ | ✅ | ✅ | ~ (geometry shifts) |
| Pull strength in saturation | constant slope | strongest | n/a |
| Code blast radius | loss only (small) | loss only (small) | extractor + all setups |
| λ interpretability | high (logit²) | medium (bounded H) | n/a (no λ) |
| Calibration risk | low (sweep evidence) | low–med | unknown (re-derive) |
| Re-opens prior analyses | no | no | yes (epi_logit, probit…) |
| Paper cleanliness | high | high | medium |

---

## 5. Recommendation

**Start with Option A (logit L2 penalty).** It is the most direct lever on the
diagnosed cause, hits both channels, is localised to `compute_loss`, has an
interpretable λ, and is the cleanest to ablate in the paper. Keep B as a fallback /
ablation if A's quadratic over-flattens μ̂. Defer C — its blast radius (feature
redefinition across all setups + re-doing settled analyses) is not justified given
A/B are expected to work.

**Proposed validation after retraining (any option)**:
- λ sweep; per λ report `epi_μ` (mean/median), in-domain ECE & MAE, frac saturated.
- Re-run the OOD check (`09c`) at the chosen λ — does OOD `epi_μ` rise *and* does the
  OOD/in-domain ratio hold or improve?
- Accept the λ that maximally raises `epi_μ` subject to ECE/MAE staying within a small
  tolerance of the current (ECE≈0.067, MAE≈0.218).

---

## 6. Not yet done

- No code changed, no retraining run yet (this is a design note).
- Retraining touches `src/models/bayesian_main.py` (+ a config knob) and needs a GPU run
  per λ. Awaiting decision on which option to implement.
