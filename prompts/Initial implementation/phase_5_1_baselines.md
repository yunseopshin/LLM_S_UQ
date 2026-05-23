# Phase 5-1 — Baselines

Implement all files in `src/baselines/`.

## 1. `token_entropy.py`

Function `compute_token_entropy_baseline(entropy, token_range)`:
- Returns: mean entropy of tokens in the sentence (higher = more uncertain)

## 2. `semantic_entropy.py`

Function `compute_semantic_entropy(prompt, model, tokenizer, nli_model, num_samples=10)`:
- Generate m=10 samples (temperature > 0)
- Cluster via NLI (DeBERTa: microsoft/deberta-large-mnli)
- Compute entropy over semantic clusters
- Reference: https://github.com/lorenzkuhn/semantic_uncertainty

## 3. `luq.py`

Function `compute_luq(prompt, model, tokenizer, nli_model, num_samples=10)`:
- Generate m responses
- Per-sentence: compute support score via NLI entailment across other responses
- U(x) = 1 - mean(consistency)
- Reference: Zhang et al. (2024)

## 4. `logistic_regression.py`

Class `LogisticRegressionBaseline`:
- sklearn LogisticRegression on sentence-level aggregate features
  (mean hidden state + entropy + top1)
- Point estimate only (not Bayesian)

## 5. `factuality_probe.py` — Han et al. (2025) baseline

**This is the most important baseline — direct comparison target.**
Reference code: https://github.com/JThh/fact-probe

Class `FactualityProbeBaseline`:
- Reproduce Han et al.'s approach:
  * L1-regularized logistic regression (sklearn, penalty='l1', solver='liblinear')
  * Input: hidden state vector for each claim/sentence
  
- Two variants for controlled comparison:
  
  **(a) Original variant** (faithful to Han et al.):
  - Decompose sentence into atomic claims (using auxiliary LM or simple heuristic)
  - Re-encode each claim through the LLM
  - Extract last token's hidden state from single layer (layer 14 by default)
  - Train L1-logistic regression on (h_c, y_c) pairs
  - At sentence level: aggregate claim-level predictions
  
  **(b) Adapted variant** (for ablation — isolates re-encoding effect):
  - Use generation-time hidden states (same as our method)
  - Extract last token's hidden state from single layer (layer 14)
  - Train L1-logistic regression
  - This isolates the effect of re-encoding vs generation-time hidden states

- Both variants provide only point estimates — no uncertainty decomposition.
- Han et al. reported numbers: Llama-3.1-8B in-domain AUROC 0.7357

**Key comparison axes**:
- **Ratio-level** (primary): baselines produce μ̂_j estimates; compare MAE, Pearson r against U_j = K_j/m_j
- **Strict factuality** (secondary): AUROC, AUPRC on A_j = 1{K_j = m_j}
- ECE (calibration): our Bayesian method should beat point estimate probes — this is the core hypothesis
- Binomial NLL: only our method can be evaluated here (baselines don't model counts)
- Rejection curve: compare shape and AUC

**Note on baseline evaluation under binomial**:
Baselines output a single score per sentence (probability or uncertainty).
For ratio-level comparison, their predicted probability maps to μ̂_j.
For strict factuality, their probability is compared against A_j via standard binary metrics.
Only our method provides the additional binomial NLL comparison.

## Script `scripts/05_baselines.py`:
- Run each baseline on test set
- Cache results (semantic entropy and LUQ are expensive — ~10× generation each)
- Measure wall-clock time per method
- Save all results to dict

**Important**:
- NLI model: load once globally (singleton pattern)
- Semantic Entropy / LUQ require multiple generations — precompute offline
- For factuality probe: claim decomposition requires API calls to auxiliary LM.
  Budget ~$5 for GPT-4o-mini. Reference Han et al. repo for prompt templates.
