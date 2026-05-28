# Phase 9.3 — OOD Epistemic Validation (Findings)

**Date**: 2026-05-28
**Model**: Setup 2 (trained on FActScore-Bio), `results/setup_2/trained_model.pt`
**Populations**:
- in-domain = Setup-2 Bio test split (353 sentences, m_j>0)
- OOD = LongFact pilot, 15 prompts (topic: computer-security) → 292 sentences
**Script**: `scripts/09c_ood_epistemic.py --setup 2` (annotation-free)
**Artifacts**: `results/setup_2/document/ood_epistemic.{json,png}`

---

## 1. Motivation

Phase 9.1/9.2 concluded that **in-domain there is essentially no epistemic
signal to extract** — near-zero `epi_μ` is the *correct* Bayesian answer when
the posterior is data-informed. The decomposition can therefore only be
validated where parameter uncertainty actually exists: **out of distribution**.
A genuine epistemic signal must **rise** on OOD inputs; a confidence proxy will
not.

LongFact data was generated fresh for this check (15 prompts via
`01_generate_data.py --setup 3 --limit 15`, same Llama-3-8B / 9 selected layers
as Setup 2). No annotation/API was used: sentence boundaries come from the same
`process_generation` splitter the annotation pipeline uses, so OOD sentences are
defined identically to in-domain.

---

## 2. Results

Per-sentence epistemic readouts from the **same Setup-2 model**:

| Signal | pop. | mean | median | p95 | OOD/in (mean) | p(OOD>in) |
|---|---|---|---|---|---|---|
| **epi_μ** = ĝᵀΣ̂ĝ | in-domain | 8.07e-4 | 4.22e-4 | 3.04e-3 | — | — |
| (probability space) | OOD | **2.23e-3** | 6.07e-4 | 4.78e-3 | **2.77×** | **9.7e-4** |
| **epi_logit** = z̄ᵀΣ̂z̄ | in-domain | 8.86 | 4.71 | 30.0 | — | — |
| (logit space) | OOD | 5.68 | 4.25 | 14.1 | 0.64× | 0.94 |

(Mann-Whitney U, one-sided OOD > in-domain.)

---

## 3. Interpretation

- **`epi_μ` behaves correctly as epistemic uncertainty.** It is near-zero
  in-domain (correct — the posterior is data-informed there) and rises
  significantly OOD: **mean ×2.77, median ×1.44, p < 0.001**. The mean rises
  more than the median, i.e. OOD grows a **heavier right tail** — the signature
  of some OOD inputs landing far from the training subspace and being flagged as
  high-epistemic. **The Setup-2 "collapse" was not a bug; it was the correct
  in-domain answer, and the signal is responsive to genuine domain shift.**
- **`epi_logit` is confirmed NOT epistemic.** It *decreases* OOD (mean ×0.64,
  p=0.94 in the wrong direction). A real epistemic signal must rise OOD; this one
  drops. This independently corroborates the Phase 9.2-1 verdict that `epi_logit`
  is a confidence / μ̂ proxy, not parameter uncertainty.

---

## 4. Caveats

- **Pilot scale**: 15 prompts, a single topic (computer-security), 292 sentences.
  Direction and significance are encouraging but a larger, multi-topic OOD set
  would strengthen the claim.
- **Annotation-free**: this validates the epistemic *magnitude shift*, not whether
  OOD `epi_μ` correlates with OOD *error* (that needs annotation). The
  magnitude-shift is the core "does epistemic behave correctly" test and it passes.
- **Absolute magnitude stays small even OOD — open concern.** `epi_μ` is the
  *variance* of μ̂ ∈ [0,1], so √epi_μ is the epistemic std on the factuality
  scale:

  | population | epi_μ (var) | epistemic std |
  |---|---|---|
  | in-domain mean | 8.1e-4 | ±0.028 |
  | OOD mean | 2.2e-3 | ±0.047 |
  | OOD p95 (tail) | 4.8e-3 | ±0.069 |

  So the epistemic std on a sentence's factuality probability is only ±0.03
  in-domain → ±0.05 OOD (±0.07 in the OOD tail). The ×2.77 rise is 2.77× of a
  *tiny* number. Against aleatoric (≈ μ̂(1−μ̂)/m, typically 0.05–0.06), epistemic
  is only a **few percent of total uncertainty**. This is structural, not
  accidental: the same sigmoid saturation that drives the in-domain collapse
  (Phase 9.1) also applies to OOD tokens, and averaging μ̂ over many tokens
  further shrinks its variance. **The OOD experiment validates the *direction*
  (relative rise), but the absolute magnitude is weak for a standalone rejection
  signal.**

---

## 5. Decision log (updated)

- **Posterior tempering**: DROPPED (Phase 9.1 — premise refuted).
- **Logit-space `epi_logit`**: NOT ADOPTED as epistemic (fails Phase 9.2-1 partial
  correlation AND drops OOD here). May only be described as a confidence-based
  rejection signal, never as epistemic.
- **`epi_μ`**: VALIDATED as a genuine epistemic signal — correctly ~0 in-domain,
  rises on OOD. The decomposition works as intended; the in-domain collapse is the
  correct answer, not a defect.
- **Open issue — small absolute magnitude.** Even OOD, epistemic is only a few %
  of total uncertainty (see §4). Three directions to resolve this:
  1. **Accept + relative framing**: publish "epistemic is (correctly) small and
     rises OOD". Risk: reviewer asks "so what, if it's ~3% of total?".
  2. **Attack the root cause (saturation)**: reduce token over-confidence (e.g.
     token-logit temperature, alternative feature/aggregation) to enlarge ĝ.
     Touches model/training. *Recommended to explore first* — saturation is the
     single largest suppressor.
  3. **Re-define where epistemic is measured**: token-level or a different
     summary (note: the simple logit-space variant `epi_logit` already failed in
     9.2-1 / 9.3, so this needs a genuinely epistemic alternative).
- **Other next steps (optional)**: scale the OOD set to multiple LongFact topics;
  annotate the OOD pilot to test OOD `epi_μ` vs OOD error correlation (API cost);
  draft the paper section.

---

## 7. Saturation attack — step 1: post-hoc temperature sweep

To test whether softening per-token confidence recovers `epi_μ` *before* retraining,
we swept a temperature on the fixed probe logits: `π̃ = σ(θ̂ᵀz / T)`. Artifacts:
`document/temperature_sweep.{json,png}`. Script: `scripts/09d_temperature_sweep.py`.

**Source of saturation**: ‖θ̂‖=9.75 (≈ the prior-implied 1.2·√66≈9.83, so θ̂ is *not*
abnormal) and ‖z‖ median≈5.3 — but their product+alignment gives **|θ̂ᵀz| median=21,
p90=52, max=132**. Per-token predictions are wildly over-confident; this is the
trained solution to matching K/m ratios via near-binary token votes.

| T | epi_grad mean (honest, ÷T²) | epi_nofac mean (ĝ-channel) | ECE | MAE | sat% |
|---|---|---|---|---|---|
| 1.0 | 8.07e-4 | 8.07e-4 | 0.067 | 0.218 | 0.92 |
| 2.0 | 5.64e-4 | 2.26e-3 | 0.067 | 0.219 | 0.85 |
| 3.0 | 4.68e-4 | 4.22e-3 | 0.069 | 0.220 | 0.77 |
| 5.0 | 3.78e-4 | 9.44e-3 | 0.062 | 0.225 | 0.64 |
| 8.0 | 3.21e-4 | **2.05e-2** | 0.071 | 0.233 | 0.46 |

**Findings**:
- **Post-hoc tempering fails**: `epi_grad` only *falls* with T (the 1/T² prefactor
  beats the ĝ growth). Confirms the math caveat — inference-time temperature cannot
  recover `epi_μ`.
- **But the ĝ-channel has ~25× headroom**: `epi_nofac` (the ĝ-only term, a proxy for
  what *retraining* with smaller logits unlocks — no 1/T penalty, and a less-saturated
  fit also *enlarges* Σ̂ via a smaller Fisher weight) rises from 8e-4 to **2.05e-2** at T=8.
- **Calibration is ~invariant to logit scale**: ECE (0.06–0.07) and MAE (~0.22) barely
  move, because μ̂ = mean(π̂) is robust to per-token sharpness. So we can shrink logits
  without harming sentence-level predictions.

**Conclusion**: the lever is real but only reachable by **retraining with a
logit-magnitude regulariser** (penalise `(θᵀz_ℓ)²` / encourage smoother π̂), not by
post-hoc tempering. Expected to raise `epi_μ` via two channels (bigger ĝ + bigger Σ̂)
while preserving μ̂ calibration. → next step.

## 6. Reproduce

```bash
python scripts/01_generate_data.py --setup 3 --config configs/default.yaml --limit 15 --device cuda
python scripts/01b_cache_scalars.py --config configs/default.yaml --dataset longfact
python scripts/09c_ood_epistemic.py --setup 2 --device cpu
```
