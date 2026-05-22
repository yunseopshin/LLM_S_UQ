# Phase 0 — Project Initialization

## Context

I am conducting research on Bayesian sentence-level factuality uncertainty quantification for LLMs.
Set up the project structure first.

**Project goal**:
Given an LLM response, compute factuality probability and 
epistemic/aleatoric uncertainty for each sentence using only hidden states from a single forward pass.
Default model: Llama-3-8B-Instruct. Architecture is **model-agnostic** (any HuggingFace causal LM).

**Core idea**:
- Per-token latent factuality: π_ℓ(θ) = σ(θ^T z_ℓ)
- Sentence factuality: μ_j(θ) = (1/L_j) Σ_{ℓ∈s_j} π_ℓ(θ)
- Observation model: K_j | θ, m_j ~ Binomial(m_j, μ_j(θ)), where m_j = atomic fact count, K_j = supported count
- Prior: θ ~ N(μ_0, Σ_0)
- Posterior: Laplace approximation with Fisher-type precision (m_j-weighted)
- Inference: closed-form epistemic/aleatoric decomposition at ratio level (U_j = K_j/m_j)

**Prior art**: Han et al. (2025, EMNLP Findings) showed that LLM hidden states are 
highly predictive of factuality via lightweight probes (point estimates only).
Our work extends this by providing principled Bayesian uncertainty quantification.
Reference code: https://github.com/JThh/fact-probe

**Create the following directory structure**:

```
sentence_uq/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   └── pilot.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generation.py
│   │   ├── annotation.py
│   │   └── sentence_split.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   └── cached_scalars.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bayesian_main.py
│   │   ├── bayesian_aux.py
│   │   └── fisher_scoring.py
│   ├── train/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── token_entropy.py
│   │   ├── semantic_entropy.py
│   │   ├── luq.py
│   │   ├── logistic_regression.py
│   │   └── factuality_probe.py   # Han et al. (2025) baseline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       └── logging.py
├── scripts/
│   ├── 01_generate_data.py
│   ├── 01b_cache_scalars.py
│   ├── 02_annotate_factuality.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   └── 05_baselines.py
├── tests/
│   ├── test_features.py
│   ├── test_fisher_scoring.py
│   ├── test_bayesian_main.py
│   └── test_decomposition.py
└── data/
    ├── raw/
    ├── generations/
    ├── processed/
    └── cache/
```

**requirements.txt** should include:
- torch>=2.1
- transformers>=4.40
- spacy>=3.7
- scikit-learn
- numpy
- scipy
- pyyaml
- tqdm
- datasets (HuggingFace)

**README.md** should contain:
- Project goal and overview
- **Model-agnostic design**: default Llama-3-8B-Instruct, supports any HuggingFace causal LM via config
- Installation instructions
- Phase-by-phase execution guide
- Related Work section mentioning:
  - Han et al. (2025) "Simple Factuality Probes" — code: https://github.com/JThh/fact-probe (/home/ys971217/LLM_S_UQ/fact-probe-main)
  - Kossen, Han et al. (2024) "Semantic Entropy Probes" — code: https://github.com/OATML/semantic-entropy-probes (/home/ys971217/LLM_S_UQ/semantic-entropy-probes-main)

Create all Python files as empty stubs for now (except `__init__.py` which are blank).
Write actual content for README.md, requirements.txt.
