# Phase 8-6 — Notebook: Calibration & Baseline Comparison

Create `notebooks/06_calibration_comparison.ipynb`.

**Purpose**: Directly test the core hypothesis **"Bayesian ECE < Point ECE < Han et al. ECE"**.
Reliability diagrams, PRR curves, ECE bar charts, AUROC comparison across all methods.

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
results_dir = Path(RESULTS_DIR)
```

### Cell 1: Load Phase 6-2 Evaluation Results

```python
ratio_df = pd.read_csv(results_dir / "final_metrics_ratio.csv")
strict_df = pd.read_csv(results_dir / "final_metrics_strict.csv")
print("=== Ratio-Level Metrics (Primary) ===")
display(ratio_df)
print("\n=== Strict Factuality Metrics (Secondary) ===")
display(strict_df)
```

### Cell 2: ECE Bar Chart — Core Hypothesis Test

Horizontal bar chart of ECE for all methods. Colour-code: Bayesian (green),
Point (orange), Han et al. (red), others (blue).

```python
methods = ratio_df["method"].tolist() if "method" in ratio_df.columns else ratio_df.index.tolist()
ece_values = ratio_df["ECE"].values if "ECE" in ratio_df.columns else []

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = []
for m in methods:
    if "Bayesian" in str(m): colors.append("#2ecc71")
    elif "Point" in str(m): colors.append("#f39c12")
    elif "Han" in str(m) or "Fact Probe" in str(m): colors.append("#e74c3c")
    else: colors.append("#3498db")

axes[0].barh(methods, ece_values, color=colors, edgecolor="black", alpha=0.8)
axes[0].set_xlabel("ECE (lower = better)")
axes[0].set_title("Ratio-Level ECE: Core Hypothesis Test")
axes[0].invert_yaxis()

# Strict ECE subplot
if "ECE" in strict_df.columns:
    s_methods = strict_df["method"].tolist() if "method" in strict_df.columns else strict_df.index.tolist()
    s_ece = strict_df["ECE"].values
    s_colors = []
    for m in s_methods:
        if "Bayesian" in str(m) or "Ours (Main)" in str(m): s_colors.append("#2ecc71")
        elif "Point" in str(m): s_colors.append("#f39c12")
        elif "Han" in str(m) or "Fact Probe" in str(m): s_colors.append("#e74c3c")
        else: s_colors.append("#3498db")
    axes[1].barh(s_methods, s_ece, color=s_colors, edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("ECE (lower = better)")
    axes[1].set_title("Strict ECE Comparison"); axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/ece_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 3: Reliability Diagrams (Ratio-Level)

Load per-method reliability diagram PNGs from Phase 6-2 output, or regenerate
using `src.evaluation.metrics.plot_reliability_diagram()`.

```python
from src.evaluation.metrics import plot_reliability_diagram

rel_dir = results_dir / "reliability_diagrams"
if rel_dir.exists():
    from IPython.display import Image, display as ipy_display
    for png in sorted(rel_dir.glob("*.png")):
        print(f"\n--- {png.name} ---")
        ipy_display(Image(filename=str(png), width=500))
else:
    print("reliability_diagrams/ not found. Run 04_evaluate.py first.")
```

### Cell 4: PRR (Prediction Rejection Ratio) Curves

All methods' PRR curves overlaid. Epistemic uncertainty as rejection signal.

```python
prr_png = results_dir / "prr_curves.png"
if prr_png.exists():
    from IPython.display import Image, display as ipy_display
    ipy_display(Image(filename=str(prr_png), width=700))
else:
    from src.evaluation.metrics import compute_prr
    # Re-compute PRR for each method from baselines.json + our predictions
    pass
```

### Cell 5: AUROC & AUPRC Comparison (Strict)

```python
if "AUROC" in strict_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    methods_s = strict_df["method"].tolist() if "method" in strict_df.columns else strict_df.index.tolist()
    axes[0].barh(methods_s, strict_df["AUROC"].values, color="steelblue", edgecolor="navy", alpha=0.8)
    axes[0].set_xlabel("AUROC"); axes[0].set_title("Strict AUROC (higher = better)")
    axes[0].invert_yaxis()
    if "AUPRC" in strict_df.columns:
        axes[1].barh(methods_s, strict_df["AUPRC"].values, color="darkorange", edgecolor="brown", alpha=0.8)
        axes[1].set_xlabel("AUPRC"); axes[1].set_title("Strict AUPRC (higher = better)")
        axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/auroc_auprc_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### Cell 6: Bootstrapped Confidence Intervals

Strict AUROC with bootstrapped 95% CI as error bars.

```python
if "AUROC_lower" in strict_df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    means = strict_df["AUROC"].values
    lowers = strict_df["AUROC_lower"].values
    uppers = strict_df["AUROC_upper"].values
    xerr = np.array([means - lowers, uppers - means])
    ax.barh(range(len(methods_s)), means, xerr=xerr, capsize=4,
            color="steelblue", alpha=0.7, ecolor="black")
    ax.set_yticks(range(len(methods_s))); ax.set_yticklabels(methods_s)
    ax.set_xlabel("AUROC (with 95% CI)")
    ax.set_title("AUROC with Bootstrapped Confidence Intervals"); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/auroc_with_ci.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### Cell 7: Binomial NLL — Binomial vs Bernoulli Ablation

Load `ablation_binomial_vs_bernoulli.csv` and visualise: does count-awareness help?

```python
binom_csv = results_dir / "ablation_binomial_vs_bernoulli.csv"
if binom_csv.exists():
    binom_df = pd.read_csv(binom_csv)
    display(binom_df)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    variants = binom_df["variant"].tolist()
    for ax, col, title in [
        (axes[0], "ratio_MAE", "Ratio MAE"),
        (axes[1], "strict_ECE", "Strict ECE"),
        (axes[2], "binomial_NLL", "Binomial NLL"),
    ]:
        if col in binom_df.columns:
            ax.bar(variants, binom_df[col].values, color=["#2ecc71", "#e74c3c"],
                   edgecolor="black", alpha=0.8)
            ax.set_title(title)
    plt.suptitle("Binomial vs Bernoulli Ablation", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/binomial_vs_bernoulli.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### Cell 8: Inference Time Comparison

Wall-clock inference time per method. Shows that our single-pass approach is
orders of magnitude faster than Semantic Entropy / LUQ (10× generation).

---

## Dependencies

| Module | Functions / Classes Used |
|--------|-------------------------|
| `src.evaluation.metrics` | `plot_reliability_diagram(...)`, `compute_prr(...)` |

## Inputs (Phase 6-2 outputs)

- `{RESULTS_DIR}/final_metrics_ratio.csv`
- `{RESULTS_DIR}/final_metrics_strict.csv`
- `{RESULTS_DIR}/ablation_binomial_vs_bernoulli.csv`
- `{RESULTS_DIR}/reliability_diagrams/*.png`
- `{RESULTS_DIR}/prr_curves.png`

## Outputs

- `{RESULTS_DIR}/ece_comparison.png`
- `{RESULTS_DIR}/auroc_auprc_comparison.png`
- `{RESULTS_DIR}/auroc_with_ci.png`
- `{RESULTS_DIR}/binomial_vs_bernoulli.png`
