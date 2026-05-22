# Phase 8-3 — Notebook: Training Diagnostics

Create `notebooks/03_training_diagnostics.ipynb`.

**Purpose**: Diagnose whether Phase 4-1 (`scripts/03_train.py`) training converged
properly. Visualise the Fisher scoring inner loop, bilevel outer loop, and PD checks.

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
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

%matplotlib inline
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
```

### Cell 1: Load Training History

`SentenceUQTrainer.fit()` returns a history dict, expected to be saved at
`{RESULTS_DIR}/training_history.pt` (or `.json`).

History structure (from Phase 4-1 `trainer.py` `fit()` return value):
```python
{
    "train_loss": [float, ...],        # per-epoch training loss
    "val_metrics": [dict, ...],        # per-epoch val MAE/RMSE/Pearson_r/binomial_NLL
    "pd_checks": [dict, ...],          # every 5 epochs verify_local_pd result
    "theta_hat": Tensor,               # final theta
    "Sigma_hat": Tensor,               # final Sigma
    "test_metrics": dict               # final test-set result
}
```

```python
history_path = Path(RESULTS_DIR) / "training_history.pt"
if history_path.exists():
    history = torch.load(history_path, map_location="cpu", weights_only=False)
else:
    with open(Path(RESULTS_DIR) / "training_history.json") as f:
        history = json.load(f)
print(f"Epochs trained: {len(history['train_loss'])}")
```

### Cell 2: Training Loss Curve

```python
fig, ax = plt.subplots(figsize=(10, 5))
epochs = range(1, len(history["train_loss"]) + 1)
ax.plot(epochs, history["train_loss"], "b-", linewidth=1.5, label="Train Loss (Binomial NLL)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Training Loss (Bilevel Outer Loop)")
ax.legend(); ax.grid(alpha=0.3)

ax2 = ax.twinx()
ax2.plot(epochs, history["train_loss"], "b--", alpha=0.3)
ax2.set_yscale("log"); ax2.set_ylabel("Loss (log scale)", alpha=0.5)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 3: Validation Metrics Over Epochs

Extract MAE, Pearson r, binomial NLL from `val_metrics` and plot per-epoch trends.

```python
if history.get("val_metrics"):
    val_df_rows = []
    for i, vm in enumerate(history["val_metrics"]):
        if vm is not None:
            vm["epoch"] = i + 1
            val_df_rows.append(vm)
    val_df = pd.DataFrame(val_df_rows)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, color, title in [
        (axes[0], "MAE", "tab:blue", "Validation MAE"),
        (axes[1], "Pearson_r", "tab:green", "Validation Pearson r"),
        (axes[2], "binomial_NLL", "tab:orange", "Validation Binomial NLL"),
    ]:
        if col in val_df.columns:
            ax.plot(val_df["epoch"], val_df[col], "-o", color=color, markersize=3)
            ax.set_xlabel("Epoch"); ax.set_title(title); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/val_metrics_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### Cell 4: PD (Positive Definiteness) Check History

`verify_local_pd` runs every 5 epochs. Tracks Fisher-type and true Hessian PD status
plus minimum eigenvalues. If PD breaks, Laplace approximation is invalid.

```python
if history.get("pd_checks"):
    pd_df = pd.DataFrame(history["pd_checks"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # (a) Minimum eigenvalue trend
    if "fisher_min_eig" in pd_df.columns:
        axes[0].plot(pd_df.index * 5, pd_df["fisher_min_eig"], "b-o", markersize=4, label="Fisher min eig")
    if "true_min_eig" in pd_df.columns:
        axes[0].plot(pd_df.index * 5, pd_df["true_min_eig"], "r-s", markersize=4, label="True Hessian min eig")
    axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Min Eigenvalue")
    axes[0].set_title("Positive Definiteness Check"); axes[0].legend(); axes[0].grid(alpha=0.3)
    
    # (b) PD pass/fail timeline
    if "laplace_valid_local" in pd_df.columns:
        colors = ["green" if v else "red" for v in pd_df["laplace_valid_local"]]
        axes[1].bar(pd_df.index * 5, [1]*len(pd_df), color=colors, width=4)
        axes[1].set_xlabel("Epoch"); axes[1].set_yticks([])
        axes[1].set_title("Laplace Valid (green=OK, red=FAIL)")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/pd_check_history.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### Cell 5: Fisher Scoring Inner-Loop Diagnostics

Use `src.utils.debug.diagnose_fisher_scoring()` to visualise per-iteration
objective, gradient norm, and Hessian min eigenvalue inside a single Fisher solve.

```python
from src.utils.debug import diagnose_fisher_scoring
from src.inference.predict import load_trained_model
from src.features.extractor import SentenceUQParams, extract_sentence_token_features

trained = load_trained_model(Path(RESULTS_DIR) / "trained_model.pt")
feature_params = trained["feature_params"]

# Load a few test sentences and run Fisher scoring diagnosis
# diag = diagnose_fisher_scoring(all_z_tokens, all_K, all_m, mu_0, Sigma_0_inv)
# Plot: (a) per-iter objective, (b) per-iter grad norm, (c) per-iter H min eigenvalue
```

### Cell 6: Gradient Flow Check

Use `src.utils.debug.check_gradient_flow()` to verify that gradients reach
every learnable parameter (W, alpha, mu_0, log_sigma_0). `None` means detached.

```python
from src.utils.debug import check_gradient_flow
# grad_info = check_gradient_flow(loss, feature_params)
# Bar chart of grad norms per parameter
```

### Cell 7: Final θ̂ and Σ̂ Summary

Distribution of learned θ̂, diagonal of Σ̂ (per-parameter posterior variance),
and Σ̂ eigenspectrum with condition number.

```python
theta = trained["theta_hat"]
Sigma = trained["Sigma_hat"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(range(len(theta)), theta.numpy())
axes[0].set_xlabel("Dimension"); axes[0].set_ylabel("θ̂"); axes[0].set_title("Learned θ̂ (MAP)")

diag_sigma = torch.diag(Sigma).numpy()
axes[1].bar(range(len(diag_sigma)), diag_sigma)
axes[1].set_xlabel("Dimension"); axes[1].set_ylabel("Σ̂_ii")
axes[1].set_title("Posterior Variance (diagonal)")

eigvals = torch.linalg.eigvalsh(Sigma).numpy()
axes[2].semilogy(range(len(eigvals)), np.sort(eigvals)[::-1], "b-o", markersize=3)
axes[2].set_xlabel("Index (sorted)"); axes[2].set_ylabel("Eigenvalue")
axes[2].set_title(f"Σ̂ Eigenspectrum (cond={eigvals.max()/max(eigvals.min(), 1e-10):.0f})")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/theta_sigma_summary.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Dependencies

| Module | Functions / Classes Used |
|--------|-------------------------|
| `src.inference.predict` | `load_trained_model(path)` |
| `src.models.bayesian_main` | `verify_local_pd(...)` |
| `src.utils.debug` | `diagnose_fisher_scoring(...)`, `check_gradient_flow(...)` |
| `src.features.extractor` | `SentenceUQParams`, `extract_sentence_token_features(...)` |

## Inputs

- `{RESULTS_DIR}/trained_model.pt`
- `{RESULTS_DIR}/training_history.pt` (or `.json`)

## Outputs

- `{RESULTS_DIR}/training_loss_curve.png`
- `{RESULTS_DIR}/val_metrics_curve.png`
- `{RESULTS_DIR}/pd_check_history.png`
- `{RESULTS_DIR}/theta_sigma_summary.png`
