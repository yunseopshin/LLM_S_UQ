# Phase 8-2 — Notebook: Hidden State Inspection

Create `notebooks/02_hidden_state_inspection.ipynb`.

**Purpose**: Visually verify that LLM hidden states carry a factuality signal.
Reproduce Han et al.'s key finding (layer 14 vicinity is optimal) on our data.

---

## Cell Layout

### Cell 0: Configuration + Imports

```python
# === Configuration ===
import sys, os
PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

RESULTS_DIR = "results/pilot"
CONFIG_PATH = "configs/pilot.yaml"

import yaml, json, torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

%matplotlib inline
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
```

### Cell 1: Load Hidden States + Labels

Load generation `.pt` files (`hidden_states`: shape `(T, num_layers, hidden_dim)`)
and annotation `.json` files (`K_j`, `m_j`, `token_range`). Aggregate to
sentence-level mean hidden state and assign a strict label (factual if K_j == m_j).

```python
gen_dir = Path(cfg["generation"]["factscore_bio_dir"])
proc_dir = Path(cfg["processed"]["factscore_bio_dir"])

MAX_ENTITIES = 20  # cap to save memory
sentence_layers = []   # will hold (N, num_layers, hidden_dim)
sentence_labels = []   # 0=hallucinated, 1=factual

gen_files = sorted(gen_dir.glob("*.pt"))[:MAX_ENTITIES]
proc_files = sorted(proc_dir.glob("*.json"))[:MAX_ENTITIES]

for gf, pf in zip(gen_files, proc_files):
    gen = torch.load(gf, map_location="cpu", weights_only=False)
    with open(pf) as f:
        records = json.load(f)
    hs = gen["hidden_states"].float()  # fp16 → fp32
    for rec in records:
        if rec["m_j"] == 0:
            continue
        start, end = rec["token_range"]
        mean_hs = hs[start:end].mean(dim=0)  # (num_layers, hidden_dim)
        sentence_layers.append(mean_hs)
        sentence_labels.append(1 if rec["K_j"] == rec["m_j"] else 0)

sentence_layers = torch.stack(sentence_layers)  # (N, num_layers, hidden_dim)
sentence_labels = np.array(sentence_labels)
num_layers = sentence_layers.shape[1]
print(f"Loaded {len(sentence_labels)} sentences, {num_layers} layers")
print(f"  Factual: {sentence_labels.sum()}, Hallucinated: {(1-sentence_labels).sum()}")
```

### Cell 2: Per-Layer t-SNE / PCA (Key Visualisation)

Select ~4 representative layers (early, mid, layer-14 vicinity, late).
2D scatter with factual (green) vs hallucinated (red) colour coding.

```python
representative_layers = [0, num_layers // 4,
                         14 if num_layers > 14 else num_layers // 2,
                         num_layers - 1]
representative_layers = [l for l in representative_layers if l < num_layers]

fig, axes = plt.subplots(1, len(representative_layers),
                         figsize=(5 * len(representative_layers), 5))
if len(representative_layers) == 1:
    axes = [axes]

for ax, layer_idx in zip(axes, representative_layers):
    X = sentence_layers[:, layer_idx, :].numpy()
    if X.shape[0] > 50:
        X_pca = PCA(n_components=min(50, X.shape[1])).fit_transform(X)
        X_2d = TSNE(n_components=2, random_state=42,
                     perplexity=min(30, len(X) - 1)).fit_transform(X_pca)
    else:
        X_2d = PCA(n_components=2).fit_transform(X)
    colors = ["#e74c3c" if l == 0 else "#2ecc71" for l in sentence_labels]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.5, s=15)
    ax.set_title(f"Layer {layer_idx}")
    ax.set_xticks([]); ax.set_yticks([])

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color="#2ecc71", label="Factual"),
                    Patch(color="#e74c3c", label="Hallucinated")],
           loc="lower center", ncol=2, fontsize=11)
plt.suptitle("Hidden State t-SNE by Layer (sentence mean)", fontsize=14)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(f"{RESULTS_DIR}/hidden_state_tsne.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 3: Layer-wise Linear Separability (Probe Accuracy)

Train a simple L1-LogisticRegression per layer and report 3-fold CV AUROC.
Reproduces Han et al. Figure 6 (Left): layer index vs AUROC.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

layer_aurocs = []
for layer_idx in range(num_layers):
    X = sentence_layers[:, layer_idx, :].numpy()
    y = sentence_labels
    if len(np.unique(y)) < 2:
        layer_aurocs.append(0.5); continue
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
    layer_aurocs.append(scores.mean())

plt.figure(figsize=(12, 5))
plt.plot(range(num_layers), layer_aurocs, "b-o", markersize=4)
plt.axvline(x=14, color="red", linestyle="--", alpha=0.7, label="Han et al. optimal (layer 14)")
best_layer = np.argmax(layer_aurocs)
plt.axvline(x=best_layer, color="green", linestyle="--", alpha=0.7,
            label=f"Our best (layer {best_layer})")
plt.xlabel("Layer Index"); plt.ylabel("3-fold CV AUROC")
plt.title("Layer-wise Linear Separability (L1 Logistic Regression)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(f"{RESULTS_DIR}/layer_separability.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Best layer: {best_layer} (AUROC={layer_aurocs[best_layer]:.4f})")
```

### Cell 4: Cross-Layer Cosine Similarity Heatmap

Cosine similarity between mean hidden states across layers. Shows whether
factuality information is concentrated in a specific layer band.

```python
from torch.nn.functional import cosine_similarity

mean_per_layer = sentence_layers.mean(dim=0)  # (num_layers, hidden_dim)
cos_sim = torch.zeros(num_layers, num_layers)
for i in range(num_layers):
    for j in range(num_layers):
        cos_sim[i, j] = cosine_similarity(
            mean_per_layer[i].unsqueeze(0), mean_per_layer[j].unsqueeze(0)).item()

plt.figure(figsize=(8, 7))
sns.heatmap(cos_sim.numpy(), cmap="viridis", xticklabels=4, yticklabels=4)
plt.xlabel("Layer"); plt.ylabel("Layer")
plt.title("Cross-Layer Cosine Similarity (sentence-mean hidden states)")
plt.savefig(f"{RESULTS_DIR}/layer_cosine_similarity.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 5: Factual vs Hallucinated — Per-Layer Norm Difference

Per-layer mean L2 norm for factual vs hallucinated sentences. A clear separation
at certain layers indicates a strong factuality signal.

```python
factual_mask = sentence_labels == 1
halluc_mask = sentence_labels == 0

norms_factual = sentence_layers[factual_mask].norm(dim=-1).mean(dim=0).numpy()
norms_halluc = sentence_layers[halluc_mask].norm(dim=-1).mean(dim=0).numpy()

plt.figure(figsize=(12, 5))
plt.plot(range(num_layers), norms_factual, "g-o", markersize=3, label="Factual", alpha=0.8)
plt.plot(range(num_layers), norms_halluc, "r-o", markersize=3, label="Hallucinated", alpha=0.8)
plt.fill_between(range(num_layers), norms_factual, norms_halluc, alpha=0.15, color="gray")
plt.xlabel("Layer Index"); plt.ylabel("Mean L2 Norm")
plt.title("Hidden State Norm: Factual vs Hallucinated")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(f"{RESULTS_DIR}/hidden_norm_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 6: Single-Entity Deep Dive

Pick one entity and visualise all its sentences in representation space. Show
sentence text + label + PCA position for a chosen layer.

---

## Dependencies

| Module | Functions / Classes Used |
|--------|-------------------------|
| (none — pure torch + sklearn) | `torch.load`, `PCA`, `TSNE`, `LogisticRegression` |

## Input Paths

- `data/generations/factscore_bio/*.pt` (hidden_states)
- `data/processed/factscore_bio/*.json` (K_j, m_j, token_range)

## Outputs

- `{RESULTS_DIR}/hidden_state_tsne.png`
- `{RESULTS_DIR}/layer_separability.png`
- `{RESULTS_DIR}/layer_cosine_similarity.png`
- `{RESULTS_DIR}/hidden_norm_comparison.png`
