# Phase 8-4 — Notebook: Model Internals

Create `notebooks/04_model_internals.ipynb`.

**Purpose**: Dissect the trained model to understand *what* it learned.
The key analysis is the `softmax(α)` layer-weight distribution compared with
Han et al.'s finding that layer 14 is optimal.

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
from pathlib import Path

%matplotlib inline
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
```

### Cell 1: Load Trained Model

```python
from src.inference.predict import load_trained_model
from src.features.extractor import SentenceUQParams

trained = load_trained_model(Path(RESULTS_DIR) / "trained_model.pt")
feature_params = trained["feature_params"]  # SentenceUQParams instance
theta_hat = trained["theta_hat"]             # (k,) where k = projection_dim + 2
Sigma_hat = trained["Sigma_hat"]             # (k, k)

# SentenceUQParams attributes:
#   .W             : (projection_dim, hidden_dim)
#   .alpha         : (num_layers,) raw layer weights before softmax
#   .mu_0          : (k,) prior mean
#   .log_sigma_0   : (k,) log prior std (diagonal)

print(f"theta_hat shape: {theta_hat.shape}")
print(f"Sigma_hat shape: {Sigma_hat.shape}")
print(f"W shape: {feature_params.W.shape}")
print(f"alpha shape: {feature_params.alpha.shape}")
print(f"projection_dim: {feature_params.projection_dim}")
print(f"num_layers: {feature_params.num_layers}")
print(f"hidden_dim: {feature_params.hidden_dim}")
```

### Cell 2: Layer α Weight Distribution (Key — Han et al. Comparison)

Bar chart of `softmax(α)`. Compare with Han et al.'s layer-14-optimal finding.
If Phase 6-2 saved `alpha_distribution.csv`, use it for `selected_layers`.

```python
with torch.no_grad():
    alpha_raw = feature_params.alpha.detach().float()
    alpha_softmax = torch.softmax(alpha_raw, dim=0).numpy()

alpha_csv = Path(RESULTS_DIR) / "alpha_distribution.csv"
if alpha_csv.exists():
    import pandas as pd
    alpha_df = pd.read_csv(alpha_csv)
    selected_layers = alpha_df["selected_layer"].tolist()
else:
    selected_layers = list(range(len(alpha_softmax)))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) softmax(α) bar chart
bars = axes[0].bar(range(len(alpha_softmax)), alpha_softmax,
                   color="steelblue", edgecolor="navy", alpha=0.8)
if 14 in selected_layers:
    idx_14 = selected_layers.index(14)
    bars[idx_14].set_color("red"); bars[idx_14].set_edgecolor("darkred")
axes[0].set_xticks(range(len(selected_layers)))
axes[0].set_xticklabels([str(l) for l in selected_layers], rotation=45, fontsize=9)
axes[0].set_xlabel("Layer Index"); axes[0].set_ylabel("softmax(α)")
axes[0].set_title("Learned Layer Weights (softmax(α))")
axes[0].axhline(y=1.0/len(alpha_softmax), color="gray", linestyle="--", alpha=0.5, label="Uniform")
axes[0].legend()

# (b) raw α (logits)
axes[1].bar(range(len(alpha_raw)), alpha_raw.numpy(),
            color="darkorange", edgecolor="brown", alpha=0.8)
axes[1].set_xticks(range(len(selected_layers)))
axes[1].set_xticklabels([str(l) for l in selected_layers], rotation=45, fontsize=9)
axes[1].set_xlabel("Layer Index"); axes[1].set_ylabel("α (raw logit)")
axes[1].set_title("Raw Layer Logits (before softmax)")
axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/alpha_distribution_detailed.png", dpi=150, bbox_inches="tight")
plt.show()

top3_idx = np.argsort(alpha_softmax)[::-1][:3]
for i, idx in enumerate(top3_idx):
    print(f"  #{i+1}: Layer {selected_layers[idx]}, weight={alpha_softmax[idx]:.4f}")
```

### Cell 3: Projection Matrix W Analysis

SVD of `W` (projection_dim × hidden_dim). Effective rank, singular value spectrum.

```python
W = feature_params.W.detach().float()
U, S, Vh = torch.linalg.svd(W, full_matrices=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].semilogy(range(len(S)), S.numpy(), "b-o", markersize=4)
axes[0].set_xlabel("Index"); axes[0].set_ylabel("Singular Value")
axes[0].set_title(f"W Singular Values (shape {tuple(W.shape)})"); axes[0].grid(alpha=0.3)

if W.shape[0] <= 128:
    sns.heatmap(W.numpy(), cmap="RdBu_r", center=0, ax=axes[1],
                xticklabels=False, yticklabels=False)
    axes[1].set_xlabel("hidden_dim"); axes[1].set_ylabel("projection_dim")
    axes[1].set_title("Projection Matrix W")
else:
    axes[1].text(0.5, 0.5, f"W too large to visualize\n{tuple(W.shape)}",
                 ha="center", va="center", fontsize=14)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/W_matrix_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

effective_rank = (S.sum() / S.max()).item()
print(f"W effective rank: {effective_rank:.1f} / {min(W.shape)}")
```

### Cell 4: Prior Parameters (μ₀, σ₀) — Prior vs Posterior

Prior mean and std compared with posterior std. Shrinkage ratio shows how much
the data informed the posterior relative to the prior.

```python
mu_0 = feature_params.mu_0.detach().float().numpy()
sigma_0 = torch.exp(feature_params.log_sigma_0.detach().float()).numpy()
posterior_std = torch.sqrt(torch.diag(Sigma_hat)).numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(range(len(mu_0)), mu_0, color="steelblue")
axes[0].set_title("Prior Mean μ₀"); axes[0].set_xlabel("Dim")

x = np.arange(len(sigma_0)); width = 0.35
axes[1].bar(x - width/2, sigma_0, width, label="Prior σ₀", color="lightcoral")
axes[1].bar(x + width/2, posterior_std, width, label="Posterior √Σ̂_ii", color="steelblue")
axes[1].set_title("Prior vs Posterior Std"); axes[1].set_xlabel("Dim"); axes[1].legend()

shrinkage = posterior_std / np.maximum(sigma_0, 1e-8)
axes[2].bar(range(len(shrinkage)), shrinkage, color="mediumpurple")
axes[2].axhline(y=1, color="red", linestyle="--", alpha=0.5, label="No shrinkage")
axes[2].set_title("Shrinkage Ratio (posterior/prior std)"); axes[2].set_xlabel("Dim"); axes[2].legend()

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/prior_posterior_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 5: Feature Distribution After Projection

Use `src.utils.debug.visualize_feature_distribution()` to inspect the
`z_ℓ = [W · Σ αₗ h^(l), H_ℓ, p^(1)_ℓ]` distribution after projection.

### Cell 6: θ̂ Dimension Interpretation

The last two dimensions of θ̂ correspond to entropy and top-1 probability.
Sign and magnitude directly reveal: "Does higher entropy imply more hallucination?"

```python
k = len(theta_hat)
projection_dim = k - 2
print(f"θ̂ dimensions: {k} (projection: {projection_dim}, entropy: 1, top-1: 1)")
print(f"  θ̂[{k-2}] (entropy coefficient): {theta_hat[k-2].item():.4f}")
print(f"  θ̂[{k-1}] (top-1 prob coefficient): {theta_hat[k-1].item():.4f}")
if theta_hat[k-2] < 0:
    print("  → Higher entropy → lower π̂ (less factual) ✓ (expected)")
else:
    print("  → Higher entropy → higher π̂ ⚠️ (unexpected)")
if theta_hat[k-1] > 0:
    print("  → Higher top-1 prob → higher π̂ (more factual) ✓ (expected)")
else:
    print("  → Higher top-1 prob → lower π̂ ⚠️ (unexpected)")
```

---

## Dependencies

| Module | Functions / Classes Used |
|--------|-------------------------|
| `src.inference.predict` | `load_trained_model(path)` → `{"theta_hat", "Sigma_hat", "feature_params"}` |
| `src.features.extractor` | `SentenceUQParams` (`.W`, `.alpha`, `.mu_0`, `.log_sigma_0`) |
| `src.utils.debug` | `visualize_feature_distribution(feature_params, hidden_states, save_path)` |

## Inputs

- `{RESULTS_DIR}/trained_model.pt`
- `{RESULTS_DIR}/alpha_distribution.csv` (Phase 6-2 output, optional)

## Outputs

- `{RESULTS_DIR}/alpha_distribution_detailed.png`
- `{RESULTS_DIR}/W_matrix_analysis.png`
- `{RESULTS_DIR}/prior_posterior_comparison.png`
