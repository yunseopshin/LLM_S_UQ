# Bayesian Sentence-Level Factuality Uncertainty Quantification

Estimate per-sentence factuality probability and decompose **epistemic / aleatoric uncertainty** from LLM hidden states in a **single forward pass**, using a Bayesian model with a binomial observation likelihood over atomic-fact verification counts.

## Project Goal

Given an LLM response, for every sentence `s_j` compute:

- a factuality probability `μ_j ∈ [0, 1]`
- an epistemic uncertainty estimate (posterior parameter uncertainty)
- an aleatoric uncertainty estimate (residual binomial noise at given `m_j`)

using only the hidden states produced during generation — no extra forward passes, no resampling.

## Core Model

```
π_ℓ(θ) = σ(θᵀ z_ℓ)                     # per-token latent factuality
μ_j(θ) = (1 / L_j) Σ_{ℓ∈s_j} π_ℓ(θ)    # sentence factuality (token average)
K_j | θ, m_j ~ Binomial(m_j, μ_j(θ))   # observation: m_j atomic facts, K_j supported
θ ~ N(μ_0, Σ_0)                         # Gaussian prior over the probe
p(θ | D) ≈ N(θ̂, Σ̂)                    # Laplace approximation (Fisher-type precision)
```

Feature `z_ℓ ∈ ℝ^66` combines a learnable layer-mixture of hidden states with cached per-token entropy and top-1 probability scalars. Inner loop is Fisher scoring on `θ`; outer loop is gradient descent on probe hyper-parameters `ψ = {W, α, μ_0, σ_0}`.

## Model-Agnostic Design

The codebase works with **any HuggingFace `AutoModelForCausalLM`**. The default checkpoint is `meta-llama/Meta-Llama-3-8B-Instruct`, but Gemma, Mistral, Qwen, etc. work without code changes — switch via `configs/*.yaml`:

```yaml
model:
  name: meta-llama/Meta-Llama-3-8B-Instruct   # change me
  selected_layers: null                        # null = auto-select evenly spaced layers
```

Model dimensions (`hidden_size`, `num_hidden_layers`, `vocab_size`) are read from `model.config` at runtime — **never** hardcoded. Generation `.pt` files store a `model_config` block for reproducibility, and tests parameterize over multiple `(hidden_dim, num_layers)` configurations.

## Installation

```bash
# 1. create environment (Python ≥ 3.10 recommended)
python -m venv .venv && source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. spaCy English model (for sentence splitting)
python -m spacy download en_core_web_sm
```

A CUDA-capable GPU with ≥ 24 GB is recommended for Llama-3-8B-Instruct generation; the trainer itself runs on a single GPU or CPU.

## Phase-by-Phase Execution

| Phase | Script | Purpose |
|-------|--------|---------|
| 0 | — | Project initialization (this phase) |
| 1-0 | `scripts/00_prepare_dataset.py` | Dataset download + train/test splits (`--setup N`) |
| 1-1 | `scripts/01_generate_data.py` | LLM generation + hidden-state extraction |
| 1-2 | (within 01) | Sentence splitting + token mapping |
| 1-3 | `scripts/01b_cache_scalars.py` | Per-token entropy / top-1 probability cache |
| 1-4 | `scripts/02_annotate_factuality.py` | Atom decomposition + Wikipedia verification → `(K_j, m_j)` |
| 2-1 | — (library code) | Feature extractor (`W · Σ α h^(l)`, entropy, top-1) |
| 3-1 | — | Fisher scoring inner loop |
| 3-2 | — | Main Bayesian model class |
| 3-3 | — | Predictive inference (4-level uncertainty decomposition) |
| 4-1 | `scripts/03_train.py` | Bilevel trainer (setup-aware) |
| 4-2 | — | Auxiliary Bayesian logistic regression |
| 5-1 | `scripts/05_baselines.py` | Token entropy, semantic entropy, LUQ, logistic regression, Han et al. factuality probe |
| 6-1 | — | Evaluation metrics (ratio-level + strict) |
| 6-2 | `scripts/04_evaluate.py` | End-to-end evaluation + ablations |
| 7 | `scripts/run_experiment.sh` | One-shot orchestration |

End-to-end run (default = Setup 2, FActScore-Bio in-domain):

```bash
bash scripts/run_experiment.sh 2
```

### Quick start

```bash
# Quick start (50 entities)
bash scripts/run_pilot.sh

# Full experiment (500 entities)
bash scripts/run_full.sh
```

Both wrappers (`scripts/run_pilot.sh`, `scripts/run_full.sh`) chain Phases 0 → 6 from a single YAML config (`configs/pilot.yaml` / `configs/default.yaml` by default). They write per-phase logs and a consolidated summary under `<results_dir>/logs/`, drop a stamp file in `<results_dir>/stamps/` after every successful phase, and resume from the first missing stamp on the next invocation — pass `--force` to redo everything. On failure, the failing phase, exit code, and log path are printed and no stamp is written, so re-running picks up where it left off. Phase 4 (auxiliary head) only runs when both the Phase 3 checkpoint and an offline `data/processed/u_star*.{json,pt}` file are present; otherwise it is reported as skipped. The pilot script also prints the full experiment checklist at the end.

## Experimental Setups

| Setup | Train | Test | Notes |
|-------|-------|------|-------|
| 1 | LongFact-Objects | FActScore-Bio (30 entities) | Cross-domain + Han et al. reproduction |
| 2 | FActScore-Bio (120 entities) | FActScore-Bio (30 entities) | **Default** |
| 3 | LongFact (26 topics) | LongFact (8 topics) | Multi-domain generalization |

## Evaluation

- **Primary (ratio-level)** on `U_j = K_j / m_j`: MAE, RMSE, Pearson r, binomial NLL (ours only), ECE, PRR.
- **Secondary (strict)** on `A_j = 1{K_j = m_j}`: AUROC, AUPRC, Brier, ECE with bootstrapped 95% CIs.
- **Core hypothesis**: Bayesian ECE < point-estimate ECE < Han et al. ECE.

## Related Work

| Paper | Code | Relation |
|-------|------|----------|
| **Han et al. (2025, EMNLP Findings)** — "Simple Factuality Probes" | <https://github.com/JThh/fact-probe> (local: `/home/ys971217/LLM_S_UQ/fact-probe-main`) | Direct comparison target. Trains lightweight probes on LLM hidden states for sentence-level factuality, but produces only point estimates. Our work adds principled Bayesian uncertainty on top: a binomial observation model over atomic-fact counts, Laplace posterior over the probe, and an epistemic / aleatoric decomposition at the ratio level. |
| **Kossen, Han et al. (2024)** — "Semantic Entropy Probes" | <https://github.com/OATML/semantic-entropy-probes> (local: `/home/ys971217/LLM_S_UQ/semantic-entropy-probes-main`) | Approximates semantic entropy from a single forward pass via a linear probe on hidden states. Methodologically adjacent (probe-on-hidden-states), used as a baseline for sentence-level uncertainty. |
| Min et al. (2023) — FActScore | <https://github.com/shmsw25/FActScore> | Provides the atom decomposition + Wikipedia verification pipeline used to obtain `(K_j, m_j)` labels. |
| Wei et al. (2024) — LongFact | <https://github.com/google-deepmind/long-form-factuality> | Multi-domain prompt set used in Setups 1 and 3. |

## Project Layout

```
sentence_uq/
├── README.md
├── requirements.txt
├── configs/        # default.yaml, pilot.yaml
├── src/
│   ├── data/       # generation, sentence_split, annotation
│   ├── features/   # extractor, cached_scalars
│   ├── models/     # bayesian_main, bayesian_aux, fisher_scoring
│   ├── train/      # trainer
│   ├── inference/  # predict
│   ├── baselines/  # token_entropy, semantic_entropy, luq, logistic_regression, factuality_probe
│   ├── evaluation/ # metrics
│   └── utils/      # io, logging
├── scripts/        # 01_generate_data → 05_baselines
├── tests/
└── data/
    ├── raw/        # FActScore-Bio, LongFact
    ├── generations/
    ├── processed/  # annotation results
    └── cache/      # entropy / top-1 caches
```
