# Phase 11-B — Does epistemic beat baselines OOD? (make-or-break for epi)

**Status**: IN PROGRESS — annotation-free pre-check done; labelled error-detection
test pending a go/no-go.
**Question (user)**: confirm epi genuinely beats baselines OOD; if not, rethink
epi from scratch.

---

## 1. What Phase 9.3 did and did NOT show

Phase 9.3 (`document/phase_9_3_ood_findings.md`) showed `epi_mu` **rises** OOD
(mean ×2.77, p<0.001 on 292 LongFact computer-security sentences) — i.e. it
*behaves* like epistemic uncertainty. But it did **not**: (a) compare against
baselines, (b) test whether epi flags OOD *errors* (annotation-free). The user's
question targets exactly these gaps.

## 2. Pre-check (Option A, annotation-free) — `scripts/diag_ood_detection.py`

Pool in-domain Bio test (353) + OOD LongFact (292); AUROC of each per-sentence
signal at detecting "is_OOD" (no m_j needed):

| signal | OOD-detect AUROC | ID mean | OOD mean |
|---|---|---|---|
| **epi_mu (OURS)** | **0.589** | 0.0017 | 0.0027 |
| mean_entropy (= token_entropy baseline) | 0.456 | 0.475 | 0.426 |
| neg_mu (OURS confidence) | 0.462 | — | — |
| neg_top1 (confidence proxy) | 0.467 | — | — |

**Reading:**
- **epi_mu is the ONLY signal that rises OOD** (AUROC > 0.5). Entropy and
  confidence go the WRONG way (<0.5): the model is *more* confident / lower
  entropy on the OOD technical text. This is the textbook motivation for
  epistemic UQ — **confidence-based signals are fooled by confidently-fluent OOD
  generation; only parameter uncertainty catches it.** Baselines structurally
  cannot do what epi does here.
- **BUT the magnitude is weak (0.589)** and this is OOD *detection*, not OOD
  *error* detection (the real claim). Encouraging, not yet decisive.

## 3. Decisive test (Option B) — needs OOD labels (feasible)

The annotation pipeline supports LongFact (`retrieve_knowledge(..., "longfact")`,
`02_annotate_factuality.py --setup 3`, Wikipedia-search verification). Plan:

1. Annotate OOD sentences → per-sentence (K_j, m_j) ⇒ U_j / A_j.
2. Compute every method's uncertainty on the SAME OOD sentences (ours: epi_mu /
   total_U / confidence; baselines: token_entropy, and the probes re-scored OOD).
3. Head-to-head **error detection on OOD**: error-AUROC (rank A_j=0 / low U_j) and
   PRR. Hypothesis from §2: confidence baselines collapse OOD (fooled), epi/total
   should rank OOD errors better. If epi does NOT beat baselines here → rethink epi.

Scale options:
- **B-small**: annotate the existing 292 computer-security sentences (1 topic).
  Fast/cheap (~$1-3 GPT-4o-mini, ~30-60 min). Decisive first signal, weak
  generalisation.
- **B-full**: generate + annotate Setup-3 LongFact test (8 topics) for a robust,
  multi-topic OOD claim. More compute (LLM generation + annotation).

## 4. DECISIVE RESULT (Option B-small) — epi FAILS, verdict NEGATIVE

Annotated the 292 OOD sentences (`02_annotate --dataset longfact`): 222 with
m_j>0, 775 atoms, strict-correct rate 0.198. Head-to-head OOD error detection
(`scripts/diag_ood_error_detection.py`):

| signal | strict err-AUROC | ratio PRR | strict PRR | Spearman(U) |
|---|---|---|---|---|
| **epi_mu (OURS epistemic)** | **0.364** ❌ | 0.281 | 0.148 | **+0.186** (wrong sign) |
| **total_U (alea+epi)** | **0.367** ❌ | 0.296 | 0.141 | +0.100 |
| conf = −μ̂ (OURS confidence) | 0.625 | 0.449 | 0.270 | −0.238 ✓ |
| **mean_entropy (= token_entropy)** | **0.673** ✓ | 0.425 | 0.284 | −0.202 |
| logreg (probe, refit Bio→OOD) | 0.639 | 0.447 | 0.266 | −0.252 |

**epi_mu does NOT detect OOD errors** — err-AUROC 0.364 is *below chance*, and
Spearman(epi, U) = +0.19 (wrong direction: higher epi ⇒ *more* factual). total_U
inherits the failure (0.367). Every baseline — mean_entropy (0.673), logreg
(0.639), even **our own confidence** (0.625) — beats epi decisively.

**Root cause (structural, not tuning):** epi_mu = ĝᵀΣ̂ĝ with ĝ ∝ w·z,
w = π(1−π). The epistemic readout is *gated by the confidence term* w = μ(1−μ):
- pearson(epi_mu, μ(1−μ)) = **+0.791**
- pearson(epi_mu, μ) = +0.652

So epi_mu is effectively a **mid-confidence / aleatoric proxy**, not genuine
parameter uncertainty about correctness. This corroborates Phase 9.2 (epi_logit
is a confidence proxy) — BOTH natural readouts are confidence-entangled. The
Phase 9.3 "epi rises ×2.77 OOD" is a population-level mean shift that does NOT
translate into ranking *which* OOD predictions are wrong.

## 5. Verdict & options

**The epistemic/aleatoric decomposition, as currently formulated, does not earn
its place as an error-detection signal**: in-domain it collapses (correct but
useless), OOD it fails to rank errors (below chance) and is beaten by mean
entropy / confidence. This is the "rethink epi" trigger.

Paths (user decision):
- **Reframe** the contribution around what DOES work — binomial strict
  calibration (strict ECE 0.052 ≪ Han 0.17), single-pass efficiency, competitive
  AUROC (0.827) — and report the epistemic finding as an honest negative result.
- **Reformulate** the epistemic readout to decouple from w (e.g. residualise epi
  vs confidence; whitened/normalised functional) — but Phase 9.2 + this §4
  suggest Laplace-on-linear-probe epistemic is fundamentally confidence-entangled.
- **Confirm** with B-full (multi-topic OOD) before abandoning — though below-chance
  + a structural cause make reversal unlikely.
