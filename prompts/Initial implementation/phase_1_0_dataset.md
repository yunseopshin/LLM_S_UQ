# Phase 1-0 — Dataset Preparation

Implement `src/data/dataset.py`.

**Purpose**:
Download/prepare prompt lists from two benchmarks (FActScore-Bio, LongFact-Objects)
and generate train/val/test splits according to the specified experimental setup.

## 1. Dataset Download

Function `prepare_factscore_bio(save_dir="data/raw/factscore_bio")`:
- Download FActScore entity list (https://github.com/shmsw25/FActScore or HuggingFace)
- Save 183 entity names to `entities.json`: `[{"entity": "Albert Einstein", "prompt": "Tell me a bio of Albert Einstein."}, ...]`
- Save the 30 test entities used by Han et al. separately to `test_entities_han.json`
  (Check Han et al. repo https://github.com/JThh/fact-probe for exact entity list)

Function `prepare_longfact_objects(save_dir="data/raw/longfact")`:
- Download LongFact-Objects (https://github.com/google-deepmind/long-form-factuality)
- 38 topics × 30 prompts = 1,140 prompts
- Parse per-topic JSONL files into `prompts.json`:
  `[{"topic": "chemistry", "prompt": "What are the key milestones in ...", "prompt_idx": 0}, ...]`
- Save topic list to `topics.json`

## 2. Split Generation

Function `create_split(dataset, setup, seed=42)`:
- Input: dataset name ("factscore_bio" or "longfact"), setup (1, 2, or 3)
- Output: `{"train": [...], "val": [...], "test": [...]}` where each element is a prompt dict
- Save to `data/splits/setup_{N}.json`

**Split logic per setup**:

```python
if setup == 1:
    # Cross-domain: LongFact train -> FActScore test (Han et al. reproduction)
    train = load_longfact()  # all or topic subset
    test = load_factscore(entities=han_test_30)
    val = None  # or hold out a few LongFact topics

elif setup == 2:
    # In-domain Biography: FActScore-Bio entity-level split
    all_entities = load_factscore()  # 183 entities
    test_entities = han_test_30  # fixed
    remaining = [e for e in all_entities if e not in test_entities]  # 153
    shuffle(remaining)
    train = remaining[:120]
    val = remaining[120:]  # 33
    test = test_entities  # 30

elif setup == 3:
    # In-domain Multi-domain: LongFact topic-level split
    all_topics = load_topics()  # 38 topics
    shuffle(all_topics)
    train_topics = all_topics[:26]
    val_topics = all_topics[26:30]
    test_topics = all_topics[30:]  # 8 topics
    # Include all 30 prompts per topic in the corresponding split
```

**Important**: Fix seed for reproducibility. Save split as JSON for reuse in all subsequent phases.

## 3. Config Integration

Add to `configs/default.yaml`:

```yaml
# Dataset configuration
dataset:
  setup: 2                          # 1, 2, or 3
  factscore_bio_dir: data/raw/factscore_bio
  longfact_dir: data/raw/longfact
  split_file: null                  # auto-generated if null
  seed: 42

# Generation outputs separated by dataset
generation:
  factscore_bio_dir: data/generations/factscore_bio
  longfact_dir: data/generations/longfact
```

## 4. Script `scripts/00_prepare_dataset.py`

```
python scripts/00_prepare_dataset.py --setup 2 --seed 42
```

- Download datasets (skip if already present)
- Create split for the specified setup
- Output: `data/splits/setup_{N}.json` + summary print

```
=== Setup 2: In-domain Biography ===
Train: 120 entities (FActScore-Bio)
Val:    33 entities (FActScore-Bio)
Test:   30 entities (FActScore-Bio, Han et al. set)
Split saved to: data/splits/setup_2.json
```

## 5. Directory Structure (add to Phase 0)

```
data/
├── raw/
│   ├── factscore_bio/
│   │   ├── entities.json
│   │   └── test_entities_han.json
│   └── longfact/
│       ├── prompts.json
│       └── topics.json
├── splits/
│   ├── setup_1.json
│   ├── setup_2.json
│   └── setup_3.json
├── generations/
│   ├── factscore_bio/    # per-entity .pt files
│   └── longfact/         # per-topic/prompt .pt files
├── cache/
│   ├── factscore_bio/    # entropy, top1
│   └── longfact/
└── processed/
    ├── factscore_bio/    # sentence split + annotation results
    └── longfact/
```
