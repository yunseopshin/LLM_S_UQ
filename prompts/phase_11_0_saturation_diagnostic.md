# Phase 11-0 — Pooling Diagnostic: Saturation by Token Type (read-only)

Before implementing any pooling, measure whether the data even supports it. The epistemic
collapse is $\text{Epi}_\mu = \hat{g}_*^\top \hat{\Sigma}\, \hat{g}_*$ vanishing because
$\hat{g}_j = \frac{1}{L_j}\sum_\ell \hat{\pi}_\ell(1-\hat{\pi}_\ell)\, z_\ell$ is diluted by
saturated tokens. Content-mask pooling can only rescue $\hat{g}$ **if content tokens
(NOUN/PROPN/NUM/ADJ) are meaningfully less saturated than function tokens**. This phase tests
that hypothesis directly, on the already-trained model, with **no retraining**.

**Prerequisite**: a trained model artifact exists (`results/{setup}/trained_model.pt` with
`theta_hat`, `Sigma_hat`, `feature_params_state_dict`, `cfg`) and the prepared per-sentence
eval data is reproducible via `SentenceUQTrainer.prepare_data`.

---

## 0. Hard Constraints

1. **Read-only.** Do NOT modify the training loop, inference, Fisher scoring, feature
   extractor, or any model code path. No retraining. This phase only *reads* the trained
   `theta_hat`, `Sigma_hat`, `feature_params`, and the cached per-token tensors.
2. Reuse existing machinery: build $z_\ell$ with `extract_token_features`; reuse the
   character-offset token mapping already in `src/data/sentence_split.py`.
3. The one new reusable component (`compute_token_pos`) is written so that the later pooling
   phase can import it unchanged — it is not throwaway.
4. English only, plain-ASCII math in code (`pi_l`, `g_j`, `mu_j`), no Unicode combining marks.

---

## 1. Quantities to compute

For every token $\ell$ in every eval-sentence span (using the trained model):

- Feature: $z_\ell = [\,W\,\textstyle\sum_q \mathrm{softmax}(\alpha)_q\, h_\ell^{(q)},\; H_\ell,\; p^{(1)}_\ell\,]$ via `extract_token_features` with the loaded `feature_params`.
- Logit: $\hat{\theta}^\top z_\ell$.
- Probability: $\hat{\pi}_\ell = \sigma(\hat{\theta}^\top z_\ell)$.
- Gate: $\mathrm{gate}_\ell = \hat{\pi}_\ell(1-\hat{\pi}_\ell)$  (this is the per-token weight in $\hat{g}_j$).
- Saturated flag: $\mathbf{1}[\hat{\pi}_\ell < \tau \text{ or } \hat{\pi}_\ell > 1-\tau]$, default $\tau = 0.05$.
- POS tag and content flag (Section 2).

Note: the gate is what actually matters for the collapse. A token can be "not saturated" by
the $\tau$ rule yet still contribute little. Report both the saturation fraction (interpretable)
and the mean gate (mechanistic).

---

## 2. NEW: `src/data/pos_tags.py` — POS aligned to the token axis

spaCy tags *word* tokens; our features live on Llama *BPE* tokens. Map word POS onto the BPE
token axis using the **same** char-offset machinery as sentence splitting.

```python
def compute_token_pos(text, token_ids, tokenizer, nlp) -> list[str]:
    """Return a per-token UPOS tag aligned to the token_ids axis (length T).

    Steps (mirror src/data/sentence_split.py):
      1. spaCy-parse `text`; collect (char_start, char_end, pos_) per spaCy token.
      2. Per-BPE-token char spans via the existing offset helpers
         (_offsets_via_reencoding, falling back to _offsets_via_incremental_decode).
      3. Assign each BPE token the UPOS of the spaCy token containing its first
         NON-whitespace character (mirror _assign_tokens_to_sentences).
      4. Whitespace-only / unmatched BPE tokens -> "SPACE" (treated as function).
    A BPE token split from one word (e.g. 'prof' + 'essor') inherits that word's POS.
    """
```

- Import and reuse the private offset/assignment helpers from `sentence_split.py` rather than
  reimplementing them (refactor them to module-level if needed, but do not change their
  behavior).
- `is_content(pos, content_set)` returns `pos in content_set`. Default
  `content_set = {"PROPN", "NUM", "NOUN", "ADJ"}`, but it is a parameter so the diagnostic can
  be re-run for narrower sets (e.g. `{"PROPN", "NUM"}`) and wider ones
  (`{"PROPN","NUM","NOUN","ADJ","VERB"}`).

---

## 3. NEW: `src/analysis/saturation_diag.py` — the diagnostic

A single function that takes the trained artifacts + eval data and returns a results dict.

```python
def diagnose_saturation_by_pos(
    eval_data,            # list of per-sentence records from prepare_data
    feature_params,       # SentenceUQParams with loaded state_dict
    theta_hat,            # (k,)
    Sigma_hat,            # (k, k)
    pos_by_source,        # dict: (dataset, source_id) -> per-token UPOS list (length T)
    *,
    sat_threshold=0.05,
    content_set=("PROPN","NUM","NOUN","ADJ"),
):
    """Read-only. Returns a dict of statistics; no model state is modified."""
```

Per-sentence loop (everything in fp32, `torch.no_grad`):

1. `z = extract_token_features(rec["hidden_states"], rec["entropy"], rec["top1"], feature_params)`
   then slice to `token_range` -> `z_span` of shape `(L_j, k)`.
2. `logit = z_span @ theta_hat`; `pi = sigmoid(logit)`; `gate = pi * (1 - pi)`.
3. POS slice for `token_range` from `pos_by_source[(dataset, source_id)]`; `content_mask` bool.

**Token-level aggregation (across all eval tokens):**
- Overall saturation fraction (sanity check: should land near the known ~92%).
- Saturation fraction and mean gate, split by `{content, function}`.
- Same, broken down by individual UPOS (PROPN, NOUN, NUM, ADJ, DET, ADP, AUX, PUNCT, ...).
- Mean-gate ratio: `mean_gate[content] / mean_gate[function]`.

**Sentence-level counterfactual (the decisive metric):** for each sentence with content subset
$C_j$, compute the two un-normalized gradients under the *existing* trained model:

$$\hat{g}_j^{\text{unif}} = \frac{1}{L_j}\sum_{\ell} \mathrm{gate}_\ell\, z_\ell, \qquad \hat{g}_j^{\text{mask}} = \frac{1}{|C_j|}\sum_{\ell \in C_j} \mathrm{gate}_\ell\, z_\ell,$$

then

$$\mathrm{Epi}^{\text{unif}}_j = \hat{g}_j^{\text{unif}\top}\hat{\Sigma}\,\hat{g}_j^{\text{unif}}, \qquad \mathrm{Epi}^{\text{mask}}_j = \hat{g}_j^{\text{mask}\top}\hat{\Sigma}\,\hat{g}_j^{\text{mask}}.$$

Report:
- Distribution (median, mean, IQR) of the ratio `Epi_mask / Epi_unif` over sentences.
- Absolute distributions of both `Epi_unif` and `Epi_mask` (the collapse is about the absolute
  scale ~8e-4, so the ratio alone is not enough — a 4x lift on a near-zero base may still be
  near zero).
- Point-estimate shift: distribution of `mu_unif - mu_mask` where
  `mu_unif = mean(pi)`, `mu_mask = mean(pi[content])`. (Tests the secondary benefit: do function
  tokens inflate `mu_j` toward 1?)
- Degeneracy guard: fraction of sentences with `|C_j| == 0` and `|C_j| <= 1`. If `|C_j| == 0`,
  exclude from the mask metrics and count separately (mask pooling would fall back to uniform
  there in the real implementation).

This counterfactual is exact for the *existing* model: it answers "if I had pooled over content
tokens with this same trained probe, how much larger would `Epi` be?" without any retraining.
(It does not capture how retraining under pooling would further change `theta_hat`; that is the
actual experiment, gated on this diagnostic being positive.)

---

## 4. NEW: `scripts/06_diagnose_saturation.py`

```
python scripts/06_diagnose_saturation.py --setup 2 \
    --eval-split test \
    --sat-threshold 0.05 \
    --content-set PROPN NUM NOUN ADJ
```

- Load `results/{setup}/trained_model.pt` -> rebuild `feature_params`, `theta_hat`, `Sigma_hat`.
- Rebuild eval-split per-sentence data via the same `prepare_data` path used in `03_train.py`.
- For each distinct `(dataset, source_id)` in the eval split, load the generation record
  (`data/generations/{ds}/...pt`) to get `text` + `token_ids`, run `compute_token_pos`, and
  cache the per-token POS list. (Load each source once.)
- Call `diagnose_saturation_by_pos(...)`.
- Write artifacts to `results/{setup}/diagnostics/`:
  - `saturation_by_pos.csv` (per-UPOS: count, sat_fraction, mean_gate).
  - `epi_counterfactual.csv` (per-sentence: L_j, n_content, mu_unif, mu_mask, Epi_unif, Epi_mask, ratio).
  - `saturation_diag.png`: two panels — (left) mean gate by UPOS as a bar chart with the
    content/function split colored; (right) histogram of `Epi_mask / Epi_unif` (log x) with the
    median marked.
  - `verdict.md`: the printed summary (Section 5) saved to disk.
- Also print the verdict to stdout.

Optionally accept `--eval-split train` to compare; the collapse was reported on the full run,
so default to `test` but allow both.

---

## 5. Go / No-Go verdict (print and save)

Emit a short, explicit verdict block, e.g.:

```
=== Pooling Diagnostic (setup 2, test split) ===
Overall saturation fraction (tau=0.05): 0.91   [sanity vs known ~0.92]
Mean gate  content=0.143  function=0.011   ratio=13.0x
Saturation fraction  content=0.38  function=0.97
Epi_mask / Epi_unif:  median=3.7x  mean=4.4x  IQR=[2.1, 6.0]
Absolute Epi_mu:  unif median=7.9e-4   mask median=2.9e-3
mu shift (unif - mask):  median=+0.12   (function tokens inflate mu toward 1)
Sentences with |C_j|==0: 1.4%   |C_j|<=1: 6.8%

VERDICT: GO. Content tokens are markedly less saturated (gate 13x higher) and content-mask
pooling lifts Epi_mu ~3-4x on this trained probe. Proceed to implement switchable pooling.
```

Verdict rule (state it in the script):
- **GO** if content mean-gate is clearly higher than function (say `>= 2x`) AND median
  `Epi_mask/Epi_unif >= ~2x` AND the resulting absolute `Epi_mask` is no longer near the
  collapse floor. Proceed to implement `model.pooling: content_mask | saliency`.
- **NO-GO** if content tokens are also saturated (gate ratio near 1) and the Epi ratio is near 1.
  Pooling will not rescue $\hat{g}$; redirect effort to the loss/prior side (binomial focal,
  fixing the prior scale, or the feature/layer changes).

Do not soften the NO-GO case: a negative result here saves a full pooling implementation +
retraining cycle and is itself reportable ("the saturation is uniform across token types").

---

## 6. Tests (`tests/test_pos_tags.py`, light)

1. `compute_token_pos` length equals `T`; on a short hand-checked string
   (e.g. "Vovin was a professor at UC Davis.") PROPN/NOUN/DET/ADP/PUNCT land on the right BPE
   tokens; subword pieces inherit their word's POS.
2. Whitespace/unmatched tokens -> "SPACE".
3. `diagnose_saturation_by_pos` on a tiny synthetic batch: gates equal `pi*(1-pi)`;
   `Epi_unif`/`Epi_mask` are non-negative; `|C_j|==0` sentences are excluded from mask metrics
   without raising.

---

## 7. Out of scope

- Any change to training, inference, Fisher scoring, the feature extractor, or the prior.
- Implementing the actual pooling (`mu_j`/`g_j` reweighting) — that is the next phase, gated on
  a GO verdict here.
- Focal loss — separate, independent change.
