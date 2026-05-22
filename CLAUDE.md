# CLAUDE.md — Bayesian Sentence-Level Factuality UQ

## Project Summary

Estimate per-sentence factuality probability + epistemic/aleatoric uncertainty from LLM hidden states in a **single forward pass**, using a Bayesian model with binomial observation. Default model: Llama-3-8B-Instruct. Architecture is **model-agnostic** — supports any HuggingFace causal LM (Gemma, Mistral, Qwen, etc.).

## Core Math

```
π_ℓ(θ) = σ(θᵀ z_ℓ)                          # per-token latent factuality
μ_j(θ) = (1/L_j) Σ_{ℓ∈s_j} π_ℓ(θ)          # sentence factuality (token average)
K_j | θ, m_j ~ Binomial(m_j, μ_j(θ))        # observation (m_j=atom count, K_j=supported)
θ ~ N(μ_0, Σ_0)                              # prior
p(θ|D) ≈ N(θ̂, Σ̂)                           # Laplace approximation
```

- **Feature**: z_ℓ = [W · Σ_l α_l h_ℓ^(l), entropy_ℓ, top1_ℓ] ∈ ℝ^66
- **Training**: Bilevel — inner: Fisher scoring (θ MAP), outer: gradient descent (ψ = {W, α, μ_0, σ_0})
- **Inference**: 4-level uncertainty (latent / ratio / count / strict), token attribution, probit shrinkage

## Prior Art & Reference Code

| Paper | Relation | Code | Path |
|-------|----------|------|------| 
| Han et al. (2025, EMNLP) — Factuality Probes | **Direct comparison target**. Hidden state → point-estimate probe. We add Bayesian UQ on top | https://github.com/JThh/fact-probe | /home/ys971217/LLM_S_UQ/sentence_uq |
| Kossen, Han et al. (2024, ICML-W) — SEP | Approximates semantic entropy from hidden states via linear probes | https://github.com/OATML/semantic-entropy-probes  | /home/ys971217/LLM_S_UQ/semantic-entropy-probes-main |
| Min et al. (2023) — FActScore | Annotation pipeline (atom decomposition + Wikipedia verification) | https://github.com/shmsw25/FActScore | /home/ys971217/LLM_S_UQ/FActScore-main | 
| Wei et al. (2024) — LongFact | Multi-domain prompt set | https://github.com/google-deepmind/long-form-factuality | /home/ys971217/LLM_S_UQ/long-form-factuality-main |

**Key differences from Han et al.**: (1) Bayesian uncertainty decomposition, (2) generation-time hidden states (no re-encoding), (3) multi-layer aggregation (learnable α), (4) token-level attribution (Theorem 2), (5) binomial observation model.

---

## Code Rules

- **PyTorch 2.x**, type hints required on all functions
- Every function must have a **docstring** (math reference, tensor dimensions, return values)
- Unit tests required (`tests/`)
- **Numerical safety**: eps clipping, compute in fp32, store in fp16
- No in-place tensor ops (autograd compatibility)
- Skip `m_j = 0` sentences in all computations
- Code style: Black formatter, isort
- **Model-agnostic**: never hardcode model-specific dimensions (hidden_dim, num_layers, vocab_size). Always read from `model.config` at runtime or accept as constructor/config arguments. See Model Compatibility section below.

---

## Project Structure

```
sentence_uq/
├── configs/           # default.yaml, pilot.yaml, setup_{1,2,3}.yaml
├── src/
│   ├── data/          # dataset.py, generation.py, annotation.py, sentence_split.py
│   ├── features/      # extractor.py, cached_scalars.py
│   ├── models/        # bayesian_main.py, bayesian_aux.py, fisher_scoring.py
│   ├── train/         # trainer.py
│   ├── inference/     # predict.py
│   ├── baselines/     # token_entropy, semantic_entropy, luq, logistic_reg, factuality_probe
│   ├── evaluation/    # metrics.py
│   └── utils/         # io.py, logging.py, debug.py
├── scripts/           # 00~05 + run_experiment.sh
├── tests/
└── data/
    ├── raw/           # factscore_bio/, longfact/
    ├── splits/        # setup_{1,2,3}.json
    ├── generations/   # per-entity/prompt .pt files
    ├── cache/         # entropy, top1 offline cache
    └── processed/     # annotation results (K_j, m_j per sentence)
```

---

## Dataset & Experimental Setups

| Setup | Train | Test | Purpose |
|-------|-------|------|---------|
| 1 (Cross-domain) | LongFact-Objects | FActScore-Bio 30 entities (Han et al.) | Han et al. reproduction + cross-domain |
| 2 (In-domain Bio) | FActScore-Bio 120 entities | FActScore-Bio 30 entities | **Default setup** |
| 3 (Multi-domain) | LongFact 26 topics | LongFact 8 topics | Multi-domain generalization |

---

## Implementation Order

```
Phase 0:   Project initialization (dirs, configs, README)
Phase 1-0: Dataset download + split generation (--setup N)
Phase 1-1: LLM generation + hidden state extraction
Phase 1-2: Sentence splitting + token mapping
Phase 1-3: Entropy / top-1 caching
Phase 1-4: Factuality annotation → (K_j, m_j)
Phase 2-1: Feature extractor (W, α, entropy, top-1)
Phase 3-1: Fisher scoring inner loop (binomial, differentiable)
Phase 3-2: Main Bayesian model class
Phase 3-3: Predictive inference (4-level decomposition)
Phase 4-1: Trainer (setup-aware, bilevel)
Phase 4-2: Auxiliary Bayesian regression
Phase 5-1: Baselines (incl. Han et al. factuality probe)
Phase 6-1: Evaluation metrics (ratio-level + strict)
Phase 6-2: Evaluation script + ablations
Phase 7-1: Integration scripts (run_experiment.sh)
Phase 7-2: Debugging utilities
```

Detailed prompts: see `claude_code_prompts_v3.md` or individual `phase_*.md` files.

---

## Current Phase

<!-- Update at the start of every session -->
**Current: Phase 0 (project initialization) — not yet started**

---

## Evaluation Structure (Two-Tiered)

**Primary — Ratio-level** (U_j = K_j / m_j, continuous):
  MAE, RMSE, Pearson r, Binomial NLL (ours only), ECE, PRR

**Secondary — Strict factuality** (A_j = 1{K_j = m_j}, binary):
  AUROC, AUPRC, Brier, ECE, bootstrapped 95% CI

**Core hypothesis**: Bayesian ECE < Point estimate ECE < Han et al. ECE

---

## Experiment Checklist

- [ ] 10-entity smoke test passes
- [ ] 50-entity pilot complete, all metrics computed
- [ ] Ratio-level MAE and Pearson r in reasonable range
- [ ] Strict AUROC ≥ baselines
- [ ] Bayesian ECE < Point ECE (core hypothesis)
- [ ] Our ECE < Han et al. ECE (key comparison)
- [ ] Binomial NLL reasonable
- [ ] Rejection curve: Ours ≥ Han et al.
- [ ] MC vs linear epistemic correlation > 0.9
- [ ] Learned α distribution visualized (compare with Han et al. layer 14)
- [ ] m_j distribution checked (no excessive m_j=0 or extreme skew)
- [ ] Binomial vs Bernoulli ablation
- [ ] Cross-setup comparison (Setup 1 vs 2 vs 3)
- [ ] 500-entity full experiment
- [ ] Paper figures generated

---

## Critical Guidelines

1. **Verify all tests from the previous Phase pass before starting a new Phase.**
2. **Pilot expensive steps (LLM generation, annotation) with 5 entities first.**
3. Fisher scoring not converging → reduce prior_sigma, increase lambda_init, check iteration count.
4. OOM → reduce selected_layers, lower batch size, enable gradient checkpointing.
5. **Only do what is explicitly requested. Do not modify unrelated files or expand scope.**
6. Start a new Claude Code session per Phase to keep context clean.
7. Annotation API calls (GPT-4o-mini): temperature=0, respect rate limits.
8. `m_j = 0` sentences must be skipped in likelihood — warn if count is excessive.
9. `_compute_grad_and_fisher` must remain **differentiable** (no detach — needed for outer loop backward).
10. Store hidden states in fp16; always compute numerics in fp32.

---

## Model Compatibility

The codebase must work with any HuggingFace `AutoModelForCausalLM`. Config drives everything:

```yaml
# configs/default.yaml
model:
  name: meta-llama/Meta-Llama-3-8B-Instruct   # change this to switch models
  selected_layers: null   # null = auto-select evenly spaced layers
  # hidden_dim, num_layers, vocab_size: auto-detected from model.config
```

**Rules for model-agnostic code**:
1. `load_model()` reads `model.config.hidden_size`, `model.config.num_hidden_layers`, etc. and returns them alongside the model. Never assume 4096 or 33 layers.
2. `selected_layers=null` triggers `auto_select_layers(num_hidden_layers, target_count=8)` which picks ~8 evenly spaced layers including first and last.
3. `SentenceUQParams.__init__` requires explicit `hidden_dim` and `num_layers` — no defaults. These come from `model.config` at runtime.
4. Generation `.pt` files store a `model_config` dict (name, hidden_dim, num_layers, selected_layers) for reproducibility.
5. Tests parameterize over at least two configs (e.g., hidden_dim=4096/num_layers=8, hidden_dim=2048/num_layers=6) to catch hardcoded assumptions.

| Model | hidden_size | num_hidden_layers | Notes |
|-------|-------------|-------------------|-------|
| Llama-3-8B-Instruct | 4096 | 32 | Default |
| Gemma-2-9B | 3584 | 42 | Different dim + deeper |
| Gemma-7B | 3072 | 28 | Smaller dim |
| Mistral-7B-Instruct | 4096 | 32 | Same as Llama |
| Qwen2.5-7B | 3584 | 28 | Different dim |

---

## Pipeline Execution

```bash
# End-to-end (select setup)
bash scripts/run_experiment.sh 2   # default: FActScore-Bio in-domain

# Individual phases
python scripts/00_prepare_dataset.py --setup 2
python scripts/01_generate_data.py --setup 2
python scripts/01b_cache_scalars.py --setup 2
python scripts/02_annotate_factuality.py --setup 2
python scripts/03_train.py --setup 2
python scripts/04_evaluate.py --setup 2

# Cross-setup comparison
python scripts/04_evaluate.py --compare-all
```

---

## Reference Documents

| Document | Content |
|----------|---------|
| `research_document_v8.md` | Full mathematical theory (Parts I–XVI) |
| `claude_code_prompts_v3.md` | Consolidated per-Phase implementation prompts |
| `phase_*.md` (16 files) | Individual Phase detailed specs |
