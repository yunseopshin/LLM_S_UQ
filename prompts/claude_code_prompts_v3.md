# Claude Code Implementation Prompts (v3)

This document is a consolidated reference for all implementation prompts.
Each Phase can be executed independently in Claude Code; the output of each Phase feeds the next.

**v3 changes from v2**:
- Observation model: Bernoulli(F_j) → Binomial(K_j | m_j, μ_j(θ))
- Dataset: single FActScore-Bio → multi-setup (FActScore-Bio + LongFact-Objects, 3 setups)
- Han et al. (2025) baseline fully integrated
- Evaluation: two-tiered (ratio-level primary + strict factuality secondary)
- 4-level uncertainty decomposition (latent / ratio / count / strict)

---

## Usage

1. Run Phase 0 (project initialization) first.
2. Execute each Phase sequentially. Multiple prompts within a Phase run in order.
3. Verify outputs between Phases before proceeding.
4. On error: re-run the failed Phase with the specific error message.
5. **Session management**: start a new Claude Code session per Phase to keep context clean.

---

# Phase 0 — Project Initialization

## Prompt 0-1: Initial Setup

```
I am conducting research on Bayesian sentence-level factuality uncertainty quantification for LLMs.
Set up the project structure first.

**Project goal**:
Given an LLM (Llama-3-8B-Instruct) response, compute factuality probability and 
epistemic/aleatoric uncertainty for each sentence using only hidden states from a single forward pass.

**Core idea**:
- Per-token latent factuality: π_ℓ(θ) = σ(θ^T z_ℓ)
- Sentence factuality: μ_j(θ) = (1/L_j) Σ_{ℓ∈s_j} π_ℓ(θ)
- Observation model: K_j | θ, m_j ~ Binomial(m_j, μ_j(θ))
  where m_j = atomic fact count, K_j = supported count
- Prior: θ ~ N(μ_0, Σ_0)
- Posterior: Laplace approximation with Fisher-type precision (m_j-weighted)
- Inference: closed-form epistemic/aleatoric decomposition at ratio level (U_j = K_j/m_j)

**Prior art**: Han et al. (2025, EMNLP Findings) showed that LLM hidden states are 
highly predictive of factuality via lightweight probes (point estimates only).
Our work extends this by providing principled Bayesian uncertainty quantification.
Reference code: https://github.com/JThh/fact-probe

**Create the following directory structure**:

sentence_uq/
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── pilot.yaml
│   ├── setup_1.yaml
│   ├── setup_2.yaml
│   └── setup_3.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # Dataset download + split generation
│   │   ├── generation.py      # LLM generation + hidden state extraction
│   │   ├── annotation.py      # Factuality annotation (FActScore → K_j, m_j)
│   │   └── sentence_split.py  # Sentence splitting + token mapping
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extractor.py       # Feature extractor (W, alpha, entropy, top-1)
│   │   └── cached_scalars.py  # Entropy/top-1 offline cache
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bayesian_main.py   # Main model: token-aggregated Bayesian (binomial)
│   │   ├── bayesian_aux.py    # Aux model: logit-Gaussian regression
│   │   └── fisher_scoring.py  # Damped Fisher scoring inner loop (binomial)
│   ├── train/
│   │   ├── __init__.py
│   │   └── trainer.py         # Bilevel training loop (setup-aware)
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py         # 4-level uncertainty decomposition
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── token_entropy.py
│   │   ├── semantic_entropy.py
│   │   ├── luq.py
│   │   ├── logistic_regression.py
│   │   └── factuality_probe.py   # Han et al. (2025) baseline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Two-tiered metrics (ratio + strict)
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── logging.py
│       └── debug.py           # Diagnostic utilities (binomial-aware)
├── scripts/
│   ├── 00_prepare_dataset.py     # Phase 1-0
│   ├── 01_generate_data.py       # Phase 1-1
│   ├── 01b_cache_scalars.py      # Phase 1-3
│   ├── 02_annotate_factuality.py # Phase 1-4
│   ├── 03_train.py               # Phase 4-1
│   ├── 04_evaluate.py            # Phase 6-2
│   ├── 04_train_aux.py           # Phase 4-2
│   ├── 05_baselines.py           # Phase 5-1
│   └── run_experiment.sh         # End-to-end per setup
├── tests/
│   ├── test_features.py
│   ├── test_fisher_scoring.py
│   ├── test_bayesian_main.py
│   └── test_decomposition.py
└── data/
    ├── raw/
    │   ├── factscore_bio/        # Entity list + Han et al. test set
    │   └── longfact/             # LongFact-Objects prompts
    ├── splits/                   # setup_{1,2,3}.json
    ├── generations/
    │   ├── factscore_bio/        # Per-entity .pt files
    │   └── longfact/             # Per-topic/prompt .pt files
    ├── cache/
    │   ├── factscore_bio/
    │   └── longfact/
    └── processed/
        ├── factscore_bio/        # Annotation results
        └── longfact/

requirements.txt:
- torch>=2.1
- transformers>=4.40
- spacy>=3.7
- scikit-learn
- numpy
- scipy
- pyyaml
- tqdm
- datasets (HuggingFace)

README.md should contain:
- Project goal and overview
- Installation instructions
- Phase-by-phase execution guide
- Related Work: Han et al. (2025) — https://github.com/JThh/fact-probe
- Related Work: Kossen, Han et al. (2024) SEP — https://github.com/OATML/semantic-entropy-probes

Create all Python files as empty stubs for now except __init__.py (blank).
Write actual content for README.md, requirements.txt, and all configs.
```

---

# Phase 1 — Data Pipeline

## Prompt 1-0: Dataset Preparation

```
Implement src/data/dataset.py.

**Purpose**:
Download/prepare prompt lists from two benchmarks (FActScore-Bio, LongFact-Objects)
and generate train/val/test splits according to the specified experimental setup.

**1. Dataset Download**:

Function prepare_factscore_bio(save_dir="data/raw/factscore_bio"):
- Download FActScore entity list (https://github.com/shmsw25/FActScore or HuggingFace)
- Save 183 entity names to entities.json: [{"entity": "Albert Einstein", "prompt": "Tell me a bio of Albert Einstein."}, ...]
- Save the 30 test entities used by Han et al. separately to test_entities_han.json
  (Check Han et al. repo https://github.com/JThh/fact-probe for exact entity list)

Function prepare_longfact_objects(save_dir="data/raw/longfact"):
- Download LongFact-Objects (https://github.com/google-deepmind/long-form-factuality)
- 38 topics × 30 prompts = 1,140 prompts
- Parse per-topic JSONL files into prompts.json:
  [{"topic": "chemistry", "prompt": "What are the key milestones in ...", "prompt_idx": 0}, ...]
- Save topic list to topics.json

**2. Split Generation**:

Function create_split(dataset, setup, seed=42):
- Input: dataset name ("factscore_bio" or "longfact"), setup (1, 2, or 3)
- Output: {"train": [...], "val": [...], "test": [...]}
- Save to data/splits/setup_{N}.json

Split logic:

  Setup 1 (Cross-domain — Han et al. reproduction):
    train = LongFact prompts (all or topic subset)
    test = FActScore entities (Han et al. 30 test entities)
    val = hold out a few LongFact topics

  Setup 2 (In-domain Biography — FActScore-Bio entity-level split):
    test = Han et al. 30 test entities (fixed)
    remaining = 183 - 30 = 153 entities
    train = 120 entities, val = 33 entities

  Setup 3 (In-domain Multi-domain — LongFact topic-level split):
    38 topics shuffled → train 26, val 4, test 8 topics
    All 30 prompts per topic go into the same split

Fix seed for reproducibility.

**3. Config integration**:
Add to configs/default.yaml:
  dataset:
    setup: 2
    factscore_bio_dir: data/raw/factscore_bio
    longfact_dir: data/raw/longfact
    split_file: null  # auto-generated if null
    seed: 42
  generation:
    factscore_bio_dir: data/generations/factscore_bio
    longfact_dir: data/generations/longfact

**Script scripts/00_prepare_dataset.py**:
  python scripts/00_prepare_dataset.py --setup 2 --seed 42
- Download datasets (skip if present), create split, print summary.
```

## Prompt 1-1: LLM Generation + Hidden State Extraction

```
Implement src/data/generation.py.

**Purpose**:
Generate responses from Llama-3-8B-Instruct and save per-token hidden states and logits.
Supports both datasets (FActScore-Bio and LongFact-Objects).

**Critical difference from Han et al. (2025)**: Han et al. re-encode extracted claims 
through the LLM and use only the last token's single-layer hidden state. 
We save generation-time hidden states for ALL tokens across multiple layers.
This captures the model's internal state at the moment of generation, not post-hoc re-encoding.

**Requirements**:

1. Function generate_with_hidden_states(model, tokenizer, prompt, max_new_tokens=512, selected_layers=None):
   - selected_layers default: [0, 8, 12, 16, 20, 24, 28, 32]
   - Greedy decoding (temperature=0), single response
   - Each step: save selected layers' hidden state (last position only) + full logit
   - Returns: dict {"text", "token_ids" (T,), "hidden_states" (T, num_layers, hidden_dim) fp16, "logits" (T, vocab_size) fp16}

2. Function load_model(model_name="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda", dtype=torch.float16):
   - output_hidden_states=True, device_map="auto"

3. Function save_generation(result, save_path): save as .pt (fp16)

4. Function batch_generate(model, tokenizer, prompts, save_dir, selected_layers=None):
   - Save {save_dir}/{idx:05d}.pt, tqdm progress, resume support (skip existing)

**Script scripts/01_generate_data.py**:
  python scripts/01_generate_data.py --setup 2 --config configs/default.yaml
- --setup selects which datasets to generate:
  Setup 1: LongFact (train) + FActScore (test)
  Setup 2: FActScore only
  Setup 3: LongFact only
- Output: data/generations/factscore_bio/{entity_name}.pt, data/generations/longfact/{topic}/{prompt_idx:03d}.pt
- Prompt construction:
  factscore_bio: "Tell me a bio of {entity}."
  longfact: use prompt as-is

**Important**:
- Manual generation loop with KV cache (past_key_values), NOT model.generate()
- output_hidden_states=True in each forward call
- GPU memory: move to CPU for long sequences if needed
- Setups share FActScore entities — skip duplicates
```

## Prompt 1-2: Sentence Splitting + Token Mapping

```
Implement src/data/sentence_split.py.

**Purpose**: Split generated text into sentences and map each sentence to token index range.

**Requirements**:

1. Function load_spacy_model(lang="en"): load en_core_web_sm, auto-install if missing

2. Function split_into_sentences(text, nlp):
   Returns: [{"text", "char_start", "char_end"}, ...]

3. Function map_sentences_to_tokens(sentences, token_ids, tokenizer):
   - Compute (tok_start, tok_end) range [tok_start, tok_end) per sentence
   - Use return_offsets_mapping=True for char→token mapping
   - Fallback: decode tokens one-by-one if re-encoding length differs
   Returns: list of (tok_start, tok_end) tuples

4. Function process_generation(generation_result, tokenizer, nlp):
   Returns: {"sentences": [{"text", "char_start", "char_end", "token_range"}, ...]}
   Filter out invalid sentences (empty token ranges)

**Important**:
- BPE token boundaries ≠ word boundaries
- If sentence boundary falls mid-subword, assign token to preceding sentence

**Tests tests/test_sentence_split.py**:
- "Hello world. This is a test." → 2 sentences
- Token ranges decode correctly
- Edge cases: empty text, single sentence
```

## Prompt 1-3: Entropy / Top-1 Offline Cache

```
Implement src/features/cached_scalars.py.

**Purpose**: Compute per-token predictive entropy and top-1 probability from logits.
These are ψ-independent — compute once and cache.

**Requirements**:

1. Function compute_token_entropy_and_top1(logits):
   - Input: (T, vocab_size) Tensor
   - probs = softmax(logits), entropy = -Σ p log p, top1 = max(probs)
   Returns: entropy (T,), top1_prob (T,)

2. Function cache_scalars_for_directory(generations_dir, cache_dir):
   - Process all .pt files, save entropy + top1 to cache_dir/{idx:05d}.pt

3. Function load_scalars(idx, cache_dir): load cached tensors

**Script scripts/01b_cache_scalars.py**: run with tqdm

**Important**: Numerical stability — log_softmax or subtract max; handle 0*log(0); cast fp16→fp32
```

## Prompt 1-4: Factuality Annotation (Binomial)

```
Implement src/data/annotation.py.

**Purpose**: For each sentence, extract atomic facts and judge supported/not-supported
to produce binomial counts (K_j, m_j) — NOT binary F_j.

**Pipeline**: Following Han et al. (2025) Stage 1.
π_aux = GPT-4o-mini (claim decomposition, revision, subjectivity filtering).
Retrieval-based scoring (Wikipedia or knowledge source).
Reference: https://github.com/JThh/fact-probe

**Requirements**:

1. Function decompose_to_atomic_facts(sentence, entity_or_topic, api_client):
   - Decompose sentence into atomic facts with claim revision + subjectivity filtering
   Returns: list of str

2. Function judge_atomic_fact(fact, knowledge_context, api_client):
   Returns: 1 (supported) or 0 (not supported)

3. Function retrieve_knowledge(entity_or_topic, dataset_type):
   - "factscore_bio": Wikipedia article for entity
   - "longfact": Wikipedia search or web search fallback
   Returns: knowledge context (str)

4. Function annotate_sentence(sentence, entity_or_topic, dataset_type, api_client):
   Returns: {"m_j": int, "K_j": int, "claims": [{"text", "label"}, ...]}

5. Function annotate_batch(processed_data, dataset_type, api_client):
   - Annotate all sentences with rate limiting + resume support

**Script scripts/02_annotate_factuality.py**:
  python scripts/02_annotate_factuality.py --setup 2 --config configs/default.yaml
- Setup determines which datasets to annotate
- Save to data/processed/{dataset}/annotated.json

**Key change from v2**: F_j ∈ {0,1} → (K_j, m_j) binomial counts.
Aligns with binomial observation model in research_document_v8 Part II §2.1.

**Cost estimate**: FActScore ~$16, LongFact ~$68, Both ~$84

**Important**: temperature=0 for deterministic judgments; filter very short sentences
```

---

# Phase 2 — Feature Extraction

## Prompt 2-1: Feature Extractor

```
Implement src/features/extractor.py.

**Mathematical definition** (research_document Part VI):
  z_ℓ = [W · h_ℓ^agg, entropy_ℓ, top1_ℓ] ∈ R^k
  h_ℓ^agg = Σ_l α_l · h_ℓ^(l),  α_l = softmax(α)_l

Learnable parameters ψ:
  W ∈ R^{p×d} (d=4096 → p=64), α ∈ R^{num_layers}, μ_0 ∈ R^k, log σ_0 ∈ R^k
  k = p + 2 = 66

**Requirements**:

1. Class SentenceUQParams(nn.Module):
   - W: nn.Linear(hidden_dim, projection_dim, bias=False)
   - alpha: nn.Parameter(zeros(num_layers))
   - mu_0: nn.Parameter(zeros(k))
   - log_sigma_0: nn.Parameter(zeros(k))
   - get_Sigma_0_inv(), get_Sigma_0(), feature_dim property

2. Function extract_token_features(hidden_states, entropy, top1_prob, params):
   Returns: (T, k) Tensor

3. Function extract_sentence_token_features(..., token_range, params):
   Returns: (L_j, k) Tensor (tokens in range only)

4. Function extract_sentence_aggregate_feature(z_tokens):
   Returns: (3k,) Tensor — concat [mean(z), std(z), z_last] for auxiliary model

**Tests tests/test_features.py**:
- Dim == k, gradients flow through W and alpha
- Smoke test with real dimensions (T=20, layers=8, hidden_dim=4096)
```

---

# Phase 3 — Bayesian Core Model

## Prompt 3-1: Fisher Scoring Inner Loop (Binomial)

```
Implement src/models/fisher_scoring.py.

**Mathematical definition** (research_document_v8 Part III, VII):

Clipped binomial objective:
  L̃(θ) = Σ_j [K_j log μ̃_j + (m_j - K_j) log(1 - μ̃_j)] - (1/2)(θ-μ_0)^T Σ_0^{-1}(θ-μ_0)
  where μ̃_j = clip(μ_j, ε, 1-ε). Sentences with m_j=0 are skipped.

Gradient:
  ∇_θ L̃ = -Σ_0^{-1}(θ-μ_0) + Σ_j R_j^bin g_j
  R_j^bin = (K_j - m_j μ̃_j) / (μ̃_j(1-μ̃_j))
  g_j = (1/L_j) Σ_ℓ π_ℓ(1-π_ℓ) z_ℓ

Fisher-type precision (m_j-weighted):
  H_fisher = Σ_0^{-1} + Σ_j m_j/(μ̃_j(1-μ̃_j)) · g_j g_j^T

Damped update: θ ← θ + (H_fisher + λI)^{-1} ∇_θ L̃

**Key change from v2**: F_j → (K_j, m_j). R_j → R_j^bin. Fisher weight gains m_j factor.
Bernoulli recovered when m_j=1, K_j∈{0,1}.

**Requirements**:

1. Function _compute_grad_and_fisher(theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps):
   - Skip m_j=0 sentences
   - MUST be differentiable (no detach) for outer loop backward
   Returns: grad (k,), H_fisher (k,k)

2. Function _compute_clipped_objective(theta, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps):
   Returns: scalar L̃(θ)

3. Function fisher_scoring_map(all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, num_iters=15, ...):
   - Damped Fisher scoring with adaptive λ
   - MUST be differentiable (unrolled optimization)
   Returns: theta_hat (k,), H_fisher_final (k,k)

4. Function fisher_scoring_map_detached(...): same with torch.no_grad() for inference

**Important**: torch.linalg.solve, no in-place ops, moderate num_iters (10-15)

**Tests tests/test_fisher_scoring.py**:
- Synthetic (k=5, N=20): convergence with random m_j∈{1..5}
- Bernoulli special case: m_j=1 matches old version
- m_j=0: skipped without error
- Gradient check: torch.autograd.gradcheck
- Fisher PD at convergence
```

## Prompt 3-2: Main Bayesian Model (Binomial)

```
Implement src/models/bayesian_main.py.

**Reference**: research_document_v8 Part II, III, VII.
Observation model: Binomial(K_j | m_j, μ_j(θ)), NOT Bernoulli.

**Requirements**:

1. Class BayesianSentenceUQ(nn.Module):
   - __init__(feature_params, num_fisher_iters=10, eps=1e-6)
   
   - compute_map(all_z_tokens, all_K, all_m, differentiable=True):
     Returns: theta_hat, H_fisher
   
   - compute_loss(all_z_tokens, all_K, all_m):
     Binomial NLL loss (skip m_j=0), differentiable through theta_hat → feature_params
     Returns: scalar loss
   
   - predict(z_tokens, m_j=None): post-training inference (implemented in Phase 3-3)

2. Function verify_local_pd(theta_hat, all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps):
   - Check Fisher-type AND true Hessian for PD
   Returns: {"fisher_min_eig", "true_min_eig", "fisher_pd", "true_pd", "laplace_valid_local"}

**Important**: Sum (not mean) over sentences. Skip m_j=0. Call verify_local_pd every 5 epochs.
```

## Prompt 3-3: Predictive Inference (4-Level Binomial)

```
Implement src/inference/predict.py.

**4-level uncertainty decomposition** (research_document_v8 Part IV, V):

Given trained (theta_hat, Sigma_hat, feature_params), for new sentence with m_* atomic facts:
  ĝ = (1/L) Σ_ℓ π̂_ℓ(1-π̂_ℓ) z_ℓ

  Latent level (μ):     Epi_μ = ĝ^T Σ̂ ĝ
  Ratio level (U=K/m):  Aleatoric_U = max(0, (μ̂(1-μ̂) - Epi_μ) / m_*)
                         Total_U = Aleatoric_U + Epi_μ
  Count level (K):       Epi_K = m_*² · Epi_μ
                         Aleatoric_K = m_* · max(0, μ̂(1-μ̂) - Epi_μ)
  Strict (A=1{K=m}):    p(A=1) = μ̂^{m_*}

Key: Aleatoric_U has 1/m_* factor — more atoms → less ratio noise.
When m_* unknown, report only latent-level Epi_μ.

Token-level (unchanged):
  Attr_ℓ = (1/L) g_ℓ^T Σ̂ ĝ          (signed, sums to Epi_μ)
  LocalEpi_ℓ = [π̂_ℓ(1-π̂_ℓ)]² · z_ℓ^T Σ̂ z_ℓ   (non-negative)

Probit-shrinkage: π̃_ℓ = σ(θ̂^T z_ℓ / √(1 + (π/8) z_ℓ^T Σ̂ z_ℓ))

**Requirements**:

1. Class Predictor:
   - __init__(theta_hat, Sigma_hat, feature_params, use_probit_shrinkage=False)
   
   - predict_sentence(z_tokens, m_j=None):
     Returns: {"mu_hat", "p_factual_probit", "epi_mu",
               "aleatoric_U" (None if no m_j), "total_U", "epi_K", "aleatoric_K",
               "p_strict_factual", "token_pi", "token_attr", "token_local_epi"}
   
   - predict_from_hidden_states(hidden_states, entropy, top1, token_range, m_j=None):
     High-level wrapper including feature extraction
   
   - predict_mc_epistemic(z_tokens, num_samples=100, m_j=None):
     Sample θ ~ N(θ̂, Σ̂), compute MC variance at all levels

2. Class BatchPredictor: vectorized over multiple sentences (each with own m_j)

3. save_trained_model / load_trained_model

**Tests tests/test_decomposition.py**:
- Invariants: epi_mu ≥ 0, aleatoric_U ≥ 0, sum(attr) ≈ epi_mu
- Bernoulli special case: m_j=1 matches v7
- Large m_j → aleatoric_U shrinks toward 0
- m_j=None: ratio/count/strict fields are None
```

---

# Phase 4 — Training

## Prompt 4-1: Trainer (Setup-Aware, Binomial)

```
Implement src/train/trainer.py.

**Reference**: research_document Part VII §7.6 (Outer Loop).
Key changes: setup-aware data loading + binomial (K_j, m_j) labels.

1. Class SentenceUQTrainer:
   - __init__, train_epoch, evaluate, fit — same interface
   
   - prepare_data(split_file, generations_dirs, cache_dirs):
     * split_file: data/splits/setup_{N}.json
     * generations_dirs: {"factscore_bio": ..., "longfact": ...}
     * Load split, match prompts to .pt files + annotations
     * Flatten to sentence-level: {"dataset", "source_id", "token_range", "K_j", "m_j"}

2. Script scripts/03_train.py:
   python scripts/03_train.py --setup 2 --config configs/default.yaml
   - Loads pre-generated split (does NOT compute splits itself)
   - Saves to results/setup_{N}/

3. Per-setup config overrides (setup_1.yaml, etc.):
   Only override dataset.setup and results_dir; inherit everything else.

4. Evaluation script changes (scripts/04_evaluate.py):
   python scripts/04_evaluate.py --setup 2 --config configs/default.yaml
   python scripts/04_evaluate.py --compare-all   # Cross-setup comparison

5. Full pipeline script (scripts/run_experiment.sh):
   #!/bin/bash
   set -e
   SETUP=${1:-2}
   python scripts/00_prepare_dataset.py --setup $SETUP
   python scripts/01_generate_data.py --setup $SETUP
   python scripts/01b_cache_scalars.py --setup $SETUP
   python scripts/02_annotate_factuality.py --setup $SETUP
   python scripts/03_train.py --setup $SETUP
   python scripts/04_evaluate.py --setup $SETUP

   Usage:
   bash scripts/run_experiment.sh 1   # Han et al. reproduction (cross-domain)
   bash scripts/run_experiment.sh 2   # FActScore-Bio in-domain
   bash scripts/run_experiment.sh 3   # LongFact multi-domain
```

## Prompt 4-2: Auxiliary Bayesian Regression

```
Implement src/models/bayesian_aux.py.

**Model** (research_document Part VIII):
Logit-transformed Bayesian Gaussian regression.
  V_j := logit(U_j*) ~ N(θ^T z_j, σ²)
  θ ~ N(μ_0, Σ_0)

Exact conjugate posterior:
  Σ_N^{-1} = Σ_0^{-1} + (1/σ²) Z^T Z
  θ_N = Σ_N (Σ_0^{-1} μ_0 + (1/σ²) Z^T V)

**Requirements**:

1. Function safe_logit(u, eps=1e-3)

2. Class BayesianLogitRegression:
   - fit(Z, U_star): closed-form posterior
   - predict(z_new): p_factual, epistemic_logit, aleatoric_logit
   - estimate_noise_variance(Z, U_star)

3. Script scripts/04_train_aux.py

**Tests**: Synthetic recovery check, sufficient statistics verification
```

---

# Phase 5 — Baselines

## Prompt 5-1: All Baselines

```
Implement all files in src/baselines/.

1. token_entropy.py: mean entropy of sentence tokens

2. semantic_entropy.py: m=10 samples, NLI clustering (DeBERTa), cluster entropy
   Reference: https://github.com/lorenzkuhn/semantic_uncertainty

3. luq.py: m responses, NLI support score, U = 1 - mean(consistency)
   Reference: Zhang et al. (2024)

4. logistic_regression.py: sklearn LogisticRegression on aggregate features. Point estimate only.

5. factuality_probe.py — Han et al. (2025) baseline [MOST IMPORTANT]:
   Reference: https://github.com/JThh/fact-probe
   
   Class FactualityProbeBaseline:
   
   (a) Original variant (faithful to Han et al.):
   - Decompose → re-encode claims → last token hidden state (layer 14) → L1-logistic regression
   - Sentence-level: aggregate claim predictions
   
   (b) Adapted variant (isolates re-encoding effect):
   - Generation-time hidden states → last token (layer 14) → L1-logistic regression
   
   Both: point estimates only, no uncertainty decomposition.
   Han et al. reported: Llama-3.1-8B AUROC 0.7357

**Key comparison axes**:
- Ratio-level (primary): MAE, Pearson r of μ̂_j vs U_j = K_j/m_j
- Strict factuality (secondary): AUROC, AUPRC on A_j = 1{K_j = m_j}
- ECE (calibration): Bayesian should beat point estimates — core hypothesis
- Binomial NLL: only our method (baselines don't model counts)
- Rejection curve

Script scripts/05_baselines.py: run all baselines, cache results, measure time.
NLI model loaded once (singleton). Semantic Entropy / LUQ are expensive (~10× generation).
Factuality probe claim decomposition: ~$5 for GPT-4o-mini.
```

---

# Phase 6 — Evaluation

## Prompt 6-1: Metrics (Two-Tiered, Binomial)

```
Implement src/evaluation/metrics.py.

**Two-tiered evaluation**:
- Primary: ratio-level (U_j = K_j/m_j) — continuous target
- Secondary: strict factuality (A_j = 1{K_j = m_j}) — binary target

**Requirements**:

1. compute_ratio_level_metrics(U_true, mu_hat, m_j=None):
   Returns: {"MAE", "RMSE", "Pearson_r", "binomial_NLL" (if m_j given)}

2. compute_strict_factuality_metrics(A_true, p_strict, uncertainty):
   Returns: {"AUROC", "AUPRC", "Brier", "ECE"}

3. compute_calibration_metrics(y_true, p_pred, n_bins=10):
   Returns: {"Brier", "ECE"} (general-purpose)

4. compute_prr(y_true, uncertainty, num_thresholds=100):
   Returns: {"rejection_rates", "remaining_quality", "prr_auc"}

5. compute_bootstrapped_ci(y_true, scores, metric_fn, n_bootstrap=1000, alpha=0.05):
   Returns: {"mean", "lower", "upper"}

6. plot_reliability_diagram(y_true, p_pred, n_bins, save_path, title)

7. compare_mc_vs_linear_epistemic(predictor, test_sentences, num_mc_samples=100)

8. full_evaluation(predictions, K_true, m_true, uncertainties):
   Returns: pandas DataFrame with all metrics

**Tests tests/test_metrics.py**:
- Perfect prediction → MAE=0, Pearson=1, AUROC=1, Brier=0
- Known binomial NLL closed-form case
```

## Prompt 6-2: Evaluation Script (Binomial)

```
Implement scripts/04_evaluate.py.

**Two-tiered evaluation pipeline**:

1. Compute predictions for all test sentences (our method + all baselines)

2. Ratio-level metrics (primary):
   MAE, RMSE, Pearson r (μ̂_j vs U_j), Binomial NLL (ours only), ECE, PRR

3. Strict factuality metrics (secondary):
   AUROC, AUPRC (A_j detection), Brier, ECE, bootstrapped 95% CI, inference time

4. Save results:
   results/final_metrics_ratio.csv
   results/final_metrics_strict.csv
   results/reliability_diagrams/
   results/prr_curves.png
   results/mc_vs_linear.png
   results/token_heatmaps/

5. Key ablations:
   - Bayesian vs Point estimate (Sigma on/off)
   - Uniform vs Attention weights
   - Linear vs MC approximation
   - Laplace-EB correction
   - Ours vs Factuality Probe (Han et al.): ECE + rejection curve
   - Binomial vs Bernoulli: m_j=1 ablation — does count awareness help?
   - Layer alpha distribution: learned alpha bar chart vs Han et al. layer 14
   - Generation-time vs re-encoded hidden state

**Expected output**:

=== Ratio-Level Metrics (Primary) ===
                    MAE      RMSE     Pearson   Binom NLL  ECE
Ours (Bayesian)    0.120    0.180    0.780     1.250      0.060
Ours (Point)       0.130    0.190    0.760     N/A        0.090
Fact Probe (Han)   0.145    0.210    0.720     N/A        0.110

=== Strict Factuality Metrics (Secondary, bootstrapped 95% CI) ===
                    AUROC    AUPRC    Brier    ECE      Time(ms)
Token Entropy      0.650    0.550    0.280    0.150    1
Fact Probe (Han)   0.735    0.640    0.245    0.130    15
Fact Probe (adapt) 0.730    0.635    0.250    0.135    10
LUQ (m=10)         0.740    0.640    0.230    0.110    5000
Semantic Entropy   0.760    0.660    0.220    0.100    5000
Log Reg            0.740    0.640    0.230    0.115    5
Ours (Main)        0.770    0.680    0.210    0.095    10
Ours (Aux)         0.765    0.670    0.215    0.100    5

=== Binomial vs Bernoulli Ablation ===
                    Binom NLL  Ratio MAE  Strict ECE
Ours (Binomial)    1.250      0.120      0.095
Ours (Bernoulli)   N/A        0.145      0.110
```

---

# Phase 7 — Integration & Debugging

## Prompt 7-1: Integration Scripts

```
Create end-to-end pipeline scripts.

scripts/run_pilot.sh:
#!/bin/bash
set -e
python scripts/01_generate_data.py --config configs/pilot.yaml
python scripts/01b_cache_scalars.py --config configs/pilot.yaml
python scripts/02_annotate_factuality.py --config configs/pilot.yaml
python scripts/03_train.py --config configs/pilot.yaml
python scripts/04_train_aux.py --config configs/pilot.yaml
python scripts/05_baselines.py --config configs/pilot.yaml
python scripts/04_evaluate.py --config configs/pilot.yaml

scripts/run_full.sh: same structure with default.yaml

**Experiment checklist** (print at end):
- [ ] 10 entity smoke test passed
- [ ] 50 entity pilot complete, all metrics computed
- [ ] Ratio-level: MAE and Pearson r reasonable (primary)
- [ ] Strict AUROC at least comparable to baselines
- [ ] Bayesian ECE < Point estimate ECE (core hypothesis)
- [ ] Our ECE < Factuality Probe (Han et al.) ECE (key comparison)
- [ ] Binomial NLL computed and reasonable
- [ ] Rejection curve: Ours ≥ Han et al.
- [ ] MC vs linear correlation > 0.9
- [ ] Learned alpha distribution visualized (vs Han et al. layer 14)
- [ ] m_j distribution checked (no excessive m_j=0 or extreme dominance)
- [ ] 500 entity full experiment
- [ ] Cross-setup comparison (Setup 1 vs 2 vs 3)
- [ ] All ablations complete
- [ ] Paper figures generated
```

## Prompt 7-2: Debugging Utilities (Binomial)

```
Implement src/utils/debug.py.

1. Function check_gradient_flow(loss, params):
   Print grad norm per parameter (W, alpha, mu_0, log_sigma_0). Warn if None.

2. Function visualize_feature_distribution(feature_params, sample_hidden_states, save_path):
   - Per-dimension histogram of projected features
   - Bar chart of softmax(alpha) with "Han et al. optimal: layer 14" annotation

3. Function diagnose_fisher_scoring(all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv, eps):
   - Verbose Fisher scoring: per-iteration objective, grad norm, H min eigenvalue
   - Report m_j=0 count and m_j distribution

4. Function sanity_check_boundary_fraction(all_z_tokens, all_K, all_m, theta_hat, eps):
   - % of sentences at μ_j clip boundary (>5% → tighten prior)
   - U_j = K_j/m_j vs μ̂_j scatter plot

5. Function check_m_j_distribution(all_m):
   - Summary stats (min, max, mean, median)
   - Count m_j=0 (should be rare)
   - Warn on highly skewed distribution (§XV.3 dominance concern)

All functions notebook-compatible (matplotlib inline).

**Common issues**:
- Tokenizer length mismatch → re-tokenize
- Hidden state > GPU memory → offline + lazy loading
- Fisher not converging → reduce prior_sigma or increase lambda_init
- Val metrics degrading → tighten prior or early stop
- verify_local_pd false → tighter prior
- Han et al. baseline: claim decomposition needs API calls (~$5 GPT-4o-mini)
- m_j=0 skipped in likelihood — check count
- Large m_j dominance → α-weighting ablation (§XV.3)
```

---

# Execution Summary

```
Phase 0:   Project initialization
Phase 1-0: Dataset download + split generation (--setup N)
Phase 1-1: LLM generation + hidden state extraction
Phase 1-2: Sentence splitting + token mapping
Phase 1-3: Entropy / top-1 caching
Phase 1-4: Factuality annotation → (K_j, m_j)
Phase 2-1: Feature extractor (W, α, entropy, top-1)
Phase 3-1: Fisher scoring inner loop (binomial)
Phase 3-2: Main Bayesian model (binomial)
Phase 3-3: Predictive inference (4-level decomposition)
Phase 4-1: Trainer (setup-aware, binomial)
Phase 4-2: Auxiliary Bayesian regression
Phase 5-1: Baselines (including Han et al. factuality probe)
Phase 6-1: Evaluation metrics (two-tiered: ratio + strict)
Phase 6-2: Evaluation script + ablations
Phase 7-1: Integration scripts
Phase 7-2: Debugging utilities

Run end-to-end:
  bash scripts/run_experiment.sh 2   # Default: FActScore-Bio in-domain

Run specific setup:
  bash scripts/run_experiment.sh 1   # Cross-domain (Han et al. reproduction)
  bash scripts/run_experiment.sh 3   # LongFact multi-domain
```

---

# Claude Code Usage Tips

- **Session per Phase**: start a new Claude Code session for each Phase to keep context clean.
- **Model selection**: Use Sonnet for Phase 1-2, 4-1, 5, 7. Opus for Phase 3 (math-heavy), 4-2, 6.
- **Verify between Phases**: "위 코드에 대한 단위 테스트를 실행해서 통과하는지 확인해줘"
- **Pilot first**: always run 10-entity smoke test → 50-entity pilot → full experiment.
- **Cost-sensitive phases**: Phase 1-4 (annotation) uses GPT-4o-mini API (~$84 for both datasets).
  Phase 5-1 (Semantic Entropy, LUQ) requires 10× generation per prompt.
