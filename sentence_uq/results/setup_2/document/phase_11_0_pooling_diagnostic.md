# Phase 11-0 — Pooling Diagnostic: Saturation by Token Type (read-only)

**Date**: 2026-06-16
**Setup**: 2 (FActScore-Bio, in-domain). Model: Llama-3-8B-Instruct probe, the
promoted λ=1e-2 production model (`results/setup_2/trained_model.pt`).
**Action**: gates Phase 9 `phase_9_summary.md` §3 item 2 ("epistemic magnitude
still modest; ĝ↔Σ̂ coupling caps de-saturation; larger gains need a different
parameterisation"). Tests whether **content-mask pooling** is that lever —
**before** implementing it. Verdict: **NO-GO**.

---

## 1. Question & hypothesis

The per-sentence epistemic term is

```
Epi_mu(j) = g_j^T Sigma_hat g_j,    g_j = (1/L_j) sum_l gate_l * z_l,
gate_l    = pi_l * (1 - pi_l),       pi_l = sigmoid(theta_hat^T z_l)
```

`Epi_mu` collapses toward a near-zero floor because **saturated tokens**
(`pi_l` near 0 or 1) have `gate_l ≈ 0` and dilute `g_j`. Content-mask pooling
(averaging `g_j` over content tokens only — NOUN/PROPN/NUM/ADJ) can rescue `g_j`
**iff content tokens are meaningfully less saturated than function tokens**.
This phase measures that hypothesis directly on the *existing* trained probe,
**no retraining**.

The test is an **exact counterfactual**: for every eval sentence we recompute
`g_j` and `Epi_mu` under uniform pooling vs content-mask pooling using the
*same* `theta_hat` / `Sigma_hat`. It answers "if I had pooled over content
tokens with this trained probe, how much larger would `Epi_mu` be?" It does
**not** capture how retraining under pooling would further move `theta_hat`
(that is the actual experiment, gated on a GO here).

## 2. Method (read-only)

- New: `src/data/pos_tags.py` — `compute_token_pos()` maps spaCy word UPOS onto
  the Llama **BPE** token axis using the *same* char-offset machinery as
  `src/data/sentence_split.py` (subwords inherit their word's POS; whitespace /
  unmatched → `SPACE`). `is_content(pos, content_set)`, default content set
  `{PROPN, NUM, NOUN, ADJ}` (parameterised).
- New: `src/analysis/saturation_diag.py` — `diagnose_saturation_by_pos()`
  (fp32, `no_grad`) + `format_verdict()`.
- New: `scripts/06_diagnose_saturation.py` — loads the trained artifacts,
  rebuilds the eval split via `SentenceUQTrainer.prepare_data` (the exact
  `03_train` path), POS-tags each source once, writes the artifacts below.
- Saturation flag: `pi_l < tau` or `pi_l > 1-tau`, `tau = 0.05`.
- **Nothing in training / inference / Fisher / extractor / prior was modified.**

## 3. Results — Setup-2, `tau=0.05`, content = {PROPN, NUM, NOUN, ADJ}

| | test (387 sent, 10 790 tok) | train (1 638 sent) |
|---|---|---|
| overall saturation frac | 0.42 | 0.40 |
| mean gate — content | 0.0965 | 0.0981 |
| mean gate — function | 0.0969 | 0.0977 |
| **gate ratio content/function** | **1.00×** | **1.00×** |
| sat frac content / function | 0.43 / 0.42 | 0.40 / 0.40 |
| **Epi_mask / Epi_unif (median)** | **1.30×** (IQR 0.99–1.83) | **1.28×** (IQR 0.94–1.75) |
| abs Epi_mu: unif → mask (median) | 1.54e-3 → 1.87e-3 | 1.64e-3 → 2.02e-3 |
| mu shift (unif − mask), median | −0.003 | +0.002 |
| sentences with \|C_j\|==0 / ≤1 | 0.3% / 0.8% | 0.1% / 0.6% |

**Mean gate by UPOS is flat** (test split; gate = `pi*(1-pi)`, higher = less
saturated). Content tags (★) are **not** less saturated than function tags —
PROPN is among the *most* saturated:

| UPOS | count | sat_frac | mean_gate | content |
|---|---|---|---|---|
| ADV | 170 | 0.36 | 0.111 | |
| PRON | 539 | 0.36 | 0.105 | |
| NOUN ★ | 1631 | 0.38 | 0.103 | ★ |
| AUX | 357 | 0.40 | 0.102 | |
| PUNCT | 1354 | 0.40 | 0.101 | |
| ADJ ★ | 648 | 0.42 | 0.100 | ★ |
| VERB | 847 | 0.43 | 0.098 | |
| NUM ★ | 545 | 0.42 | 0.097 | ★ |
| ADP | 1089 | 0.44 | 0.093 | |
| PROPN ★ | 2027 | 0.47 | 0.090 | ★ |
| DET | 690 | 0.48 | 0.087 | |
| PART | 154 | 0.47 | 0.084 | |

(Full table incl. SPACE/SCONJ/CCONJ/SYM/INTJ/X in
`results/setup_2/diagnostics/saturation_by_pos.csv`.)

## 4. Integrity note — the spec's "sanity ~0.92" anchor is STALE (not a bug)

The Phase-11-0 spec expected overall saturation ≈ 0.92, but we measure **0.42**.
This is **not a diagnostic bug** — it reflects the model, which changed:

- `results/setup_2/epistemic_diagnostics.json` (Phase 9.1, `frac_saturated=0.92`)
  was written **2026-05-28 04:54**, *before* `trained_model.pt` was re-saved
  **2026-05-28 10:01** with `logit_reg_lambda=1e-2` (the Phase 9.6 promotion).
- Re-running the established `09_diagnose_epistemic` code path on the **current**
  model gives `frac_saturated = 0.4316` over the **same 10 059-token** test set
  that produced the 0.92 — same tokens, different model. The λ=1e-2 logit
  penalty already de-saturated the probe (logit median 21 → 2.5; Phase 9.5/9.6).
- The new diagnostic's `z` (full-then-slice) matches the per-sentence path to
  `1.4e-6`. So the 0.42 is correct; the 0.92 anchor is a pre-logit-reg record.

**Implication**: the logit penalty already addressed the saturation that pooling
was meant to fix, and what remains is **uniform across token types**.

## 5. Verdict & decision — NO-GO

```
VERDICT: NO-GO. gate ratio 1.00x < 2x (content tokens are NOT clearly less
saturated); median Epi lift 1.30x < 2x. Pooling will not rescue g_j; redirect
effort to the loss/prior side (binomial focal, prior scale, or feature/layer
changes). A uniform-saturation result is itself reportable.
```

Verdict rule (in `saturation_diag.format_verdict`): GO requires (1) content
mean-gate ≥ 2× function, (2) median `Epi_mask/Epi_unif` ≥ 2×, and (3) absolute
`Epi_mask` median ≥ 2× the collapse floor (8e-4). Condition (3) passes (mask
1.9e-3), but (1) and (2) fail decisively. **Do not implement switchable pooling
for this model.** The saturation is uniform across POS, so masking cannot
differentially rescue `g_j` — and PROPN being the *most* saturated means a
content mask could even slightly hurt.

## 6. Connection to Phase 9

This closes the "is pooling the different parameterisation?" branch of
`phase_9_summary.md` §3 item 2 with a **no**. The ĝ↔Σ̂ coupling that caps
de-saturation (Phase 9.5 §4) is not relieved by re-weighting tokens by POS,
because the gate is already POS-agnostic on the λ=1e-2 model. Larger epistemic
magnitude still needs a genuinely different parameterisation or a loss/prior
change (binomial focal, fixed prior scale per §3 item 6, feature/layer changes)
— not pooling.

## 7. Reproduce

```bash
python scripts/06_diagnose_saturation.py --setup 2 --eval-split test --device cpu
python scripts/06_diagnose_saturation.py --setup 2 --eval-split train --device cpu  # confirm
# cross-check vs the established path: 09_diagnose_epistemic reports frac_saturated≈0.43 now
python -m pytest tests/test_pos_tags.py -q                                          # 7 passed
```

Artifacts (under `results/setup_2/diagnostics/`, default = test split):
`saturation_by_pos.csv`, `epi_counterfactual.csv`, `saturation_diag.png`
(mean-gate-by-UPOS bars + Epi-ratio histogram), `saturation_diag.json`,
`verdict.md`.

## 8. Out of scope (deliberate)

No change to training / inference / Fisher / feature extractor / prior. The
actual pooling implementation (`model.pooling: content_mask | saliency`) and
focal loss are **not** done — pooling was gated on a GO here and is now dropped;
focal loss is a separate, independent change.
