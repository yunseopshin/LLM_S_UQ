# Phase 8-1 — Notebook: Data Overview

Create `notebooks/01_data_overview.ipynb`.

**Purpose**: Visually inspect the Phase 1 pipeline outputs (generation, sentence split,
annotation) to confirm the data is healthy before model training.

---

## Cell Layout

### Cell 0: Configuration + Imports

```python
# === Configuration ===
import sys, os
PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

RESULTS_DIR = "results/pilot"   # ← change to inspect a different run
CONFIG_PATH = "configs/pilot.yaml"
SETUP = 2

import yaml, json, glob, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

%matplotlib inline
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
```

### Cell 1: Load Split Information

Load `data/splits/setup_{N}.json`. Print train / val / test entity counts.
Reference `src.data.dataset` constants (`SETUPS`, `HAN_TEST_SIZE`) and display a
split-statistics table.

```python
split_file = cfg["dataset"].get("split_file") or f"data/splits/setup_{SETUP}.json"
with open(split_file) as f:
    splits = json.load(f)

for split_name in ["train", "val", "test"]:
    entities = splits.get(split_name, [])
    print(f"  {split_name}: {len(entities)} entities")
```

### Cell 2: Generation File Inspection

List `.pt` files under `data/generations/factscore_bio/`.
For the first file, print the key structure: `text`, `hidden_states`, `logits`,
`token_offsets`, `model_config` (hidden_dim, num_hidden_layers, vocab_size).

```python
gen_dir = Path(cfg["generation"]["factscore_bio_dir"])
pt_files = sorted(gen_dir.glob("*.pt"))
print(f"Generation files: {len(pt_files)}")

sample = torch.load(pt_files[0], map_location="cpu", weights_only=False)
print(f"Keys: {list(sample.keys())}")
print(f"  text length: {len(sample['text'])} chars")
print(f"  hidden_states shape: {sample['hidden_states'].shape}")  # (T, num_layers, hidden_dim)
print(f"  model_config: {sample.get('model_config', 'N/A')}")
```

Plot a histogram of generated text lengths across all entities.

### Cell 3: Sentence Split Check

Load annotation files from `data/processed/factscore_bio/*.json`.
Each file contains a list of sentence records with fields:
`text`, `token_range`, `K_j`, `m_j`.

```python
proc_dir = Path(cfg["processed"]["factscore_bio_dir"])
json_files = sorted(proc_dir.glob("*.json"))
print(f"Annotation files: {len(json_files)}")

all_records = []
for jf in json_files:
    with open(jf) as f:
        records = json.load(f)
    all_records.extend(records)

print(f"Total sentences: {len(all_records)}")
```

Histogram of sentences per entity.

### Cell 4: m_j Distribution Analysis (Critical)

Visualise `m_j` (atomic-fact count per sentence). Directly connected to CLAUDE.md
rule 8 ("m_j=0 sentences must be skipped in likelihood computation").

```python
from src.utils.debug import check_m_j_distribution

all_m = torch.tensor([r["m_j"] for r in all_records])
m_stats = check_m_j_distribution(all_m)
print(f"m_j stats: {m_stats}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) m_j histogram
axes[0].hist(all_m.numpy(), bins=range(0, int(all_m.max()) + 2), edgecolor="black", alpha=0.7)
axes[0].axvline(x=0.5, color="red", linestyle="--",
                label=f"m_j=0: {m_stats['n_zero']} ({m_stats['frac_zero']:.1%})")
axes[0].set_xlabel("m_j (atomic facts per sentence)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of m_j")
axes[0].legend()

# (b) m_j=0 fraction (warn if >20%)
frac_zero = m_stats["frac_zero"]
color = "red" if frac_zero > 0.2 else "orange" if frac_zero > 0.1 else "green"
axes[1].bar(["m_j=0", "m_j>0"],
            [m_stats["n_zero"], len(all_m) - m_stats["n_zero"]],
            color=[color, "steelblue"])
axes[1].set_title(f"m_j=0 fraction: {frac_zero:.1%}")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/data_mj_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 5: K_j / m_j Distribution (Factuality Ratio)

Plot `U_j = K_j / m_j` — the ratio-level prediction target. Use only sentences
with `m_j > 0`.

```python
mask = all_m > 0
all_K = torch.tensor([r["K_j"] for r in all_records])[mask]
all_m_pos = all_m[mask]
U = (all_K.float() / all_m_pos.float()).numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) U_j histogram
axes[0].hist(U, bins=20, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("U_j = K_j / m_j")
axes[0].set_title(f"Factuality Ratio (n={len(U)})")

# (b) K_j vs m_j scatter
axes[1].scatter(all_m_pos.numpy(), all_K.numpy(), alpha=0.3, s=10)
axes[1].plot([0, all_m_pos.max()], [0, all_m_pos.max()], "r--", label="K=m (fully factual)")
axes[1].set_xlabel("m_j"); axes[1].set_ylabel("K_j")
axes[1].set_title("K_j vs m_j"); axes[1].legend()

# (c) Strict factuality fraction
A = (all_K == all_m_pos).float()
axes[2].bar(["Hallucinated\n(A=0)", "Fully Factual\n(A=1)"],
            [(1 - A.mean()).item(), A.mean().item()],
            color=["salmon", "mediumseagreen"])
axes[2].set_title(f"Strict Factuality: {A.mean():.1%} fully factual")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/data_factuality_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 6: Per-Entity Factuality Distribution

Boxplot of mean U_j per entity. Highlight entities with extreme factuality.

### Cell 7: Summary Table

Pandas DataFrame summarising: total entities, sentences, sentences with m_j>0,
m_j stats (min, max, mean, median, frac_zero), K_j stats, U_j stats, A_j fraction,
per-entity sentence count (min, max, mean).

---

## Dependencies

| Module | Functions / Classes Used |
|--------|-------------------------|
| `src.utils.debug` | `check_m_j_distribution(all_m)` |
| `src.data.dataset` | split-file loading logic |

## Input Paths (config-driven)

- `data/splits/setup_{N}.json`
- `data/generations/factscore_bio/*.pt`
- `data/processed/factscore_bio/*.json`
- (Setup 1/3) `data/generations/longfact/`, `data/processed/longfact/`

## Outputs

- `{RESULTS_DIR}/data_mj_distribution.png`
- `{RESULTS_DIR}/data_factuality_distribution.png`
- Inline summary table
