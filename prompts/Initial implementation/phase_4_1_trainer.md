# Phase 4-1 — Trainer (Updated)

Implement `src/train/trainer.py`.

**Reference**: research_document Part VII §7.6 (Outer Loop).

**Key changes: setup-aware data loading + binomial (K_j, m_j) labels.**

## 1. Class `SentenceUQTrainer` (same interface as before)

`__init__`, `train_epoch`, `evaluate`, `fit` — unchanged.

Only change: `prepare_data` now loads `(K_j, m_j)` instead of binary `F_j`.

Method `prepare_data(split_file, generations_dirs, cache_dirs)`:
  - `split_file`: path to `data/splits/setup_{N}.json`
  - `generations_dirs`: dict `{"factscore_bio": "data/generations/factscore_bio", "longfact": "data/generations/longfact"}`
  - `cache_dirs`: dict (same structure)
  - Load train/val/test prompt lists from split_file
  - Match each prompt to its generation .pt file and annotation results
  - Flatten into sentence-level list. Each sentence contains:
    ```python
    {
        "dataset": "factscore_bio" | "longfact",
        "source_id": entity_name | f"{topic}/{prompt_idx}",
        "token_range": (start, end),
        "K_j": int,     # supported claim count
        "m_j": int,     # total claim count
    }
    ```

## 2. Script `scripts/03_train.py` (changed)

```
python scripts/03_train.py --setup 2 --config configs/default.yaml
```

**Behavior**:
1. `--setup` selects the experimental setup
2. Load pre-generated split from `data/splits/setup_{N}.json` (created in Phase 1-0)
3. Build train/val/test sentence sets from the split
4. Train + evaluate
5. Save results to `results/setup_{N}/`

**The script does NOT compute splits itself** — it loads the split file generated in Phase 1-0.
This ensures identical splits across generation, annotation, training, and evaluation.

## 3. Config Changes

**configs/default.yaml**:
```yaml
# === Dataset ===
dataset:
  setup: 2
  split_file: data/splits/setup_2.json
  factscore_bio_dir: data/generations/factscore_bio
  longfact_dir: data/generations/longfact
  cache_factscore_bio_dir: data/cache/factscore_bio
  cache_longfact_dir: data/cache/longfact

# === Model (model-agnostic: change name to switch models) ===
model:
  name: meta-llama/Meta-Llama-3-8B-Instruct
  selected_layers: null    # null = auto-select evenly spaced layers
  max_new_tokens: 512
  # hidden_dim, num_layers: auto-detected from model.config at runtime

# === Feature Extractor ===
projection_dim: 64

# === Training ===
num_epochs: 50
lr: 1e-3
num_fisher_iters: 10
prior_sigma_init: 1.0
batch_size: null
eval_every: 1
pd_check_every: 5

# === Output ===
results_dir: results/setup_2
```

**Per-setup config overrides** (`configs/setup_1.yaml`, `configs/setup_2.yaml`, `configs/setup_3.yaml`):
Each only overrides `dataset.setup` and `results_dir`, inheriting everything else from default.

```yaml
# configs/setup_1.yaml
dataset:
  setup: 1
  split_file: data/splits/setup_1.json
results_dir: results/setup_1
```

## 4. Evaluation Script Changes (`scripts/04_evaluate.py`)

```
python scripts/04_evaluate.py --setup 2 --config configs/default.yaml
```

- Load test set for the specified setup
- Save results to `results/setup_{N}/`
- Cross-setup comparison:

```
python scripts/04_evaluate.py --compare-all
```

Example output:
```
=== Cross-setup Comparison (Test AUROC / ECE) ===
                    Setup 1 (cross)  Setup 2 (bio)   Setup 3 (multi)
Fact Probe (Han)    0.735 / 0.130    0.750 / 0.120   0.710 / 0.140
Ours (Bayesian)     0.770 / 0.095    0.790 / 0.085   0.755 / 0.100
```

## 5. Full Pipeline Script (run_experiment.sh)

```bash
#!/bin/bash
set -e
SETUP=${1:-2}  # default: setup 2

echo "=== Running Setup ${SETUP} ==="

echo "Phase 0: Prepare dataset"
python scripts/00_prepare_dataset.py --setup $SETUP

echo "Phase 1: Generate"
python scripts/01_generate_data.py --setup $SETUP

echo "Phase 1b: Cache scalars"
python scripts/01b_cache_scalars.py --setup $SETUP

echo "Phase 2: Annotate"
python scripts/02_annotate_factuality.py --setup $SETUP

echo "Phase 3: Train"
python scripts/03_train.py --setup $SETUP

echo "Phase 4: Evaluate"
python scripts/04_evaluate.py --setup $SETUP

echo "=== Setup ${SETUP} Done ==="
```

Usage:
```
bash scripts/run_experiment.sh 1   # Han et al. reproduction (cross-domain)
bash scripts/run_experiment.sh 2   # FActScore-Bio in-domain
bash scripts/run_experiment.sh 3   # LongFact multi-domain
```
