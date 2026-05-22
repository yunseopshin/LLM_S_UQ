# Phase 8-8 — Notebook: Paper Figures

Create `notebooks/08_paper_figures.ipynb`.

**Purpose**: Generate publication-quality figures for the paper. Every figure is saved
to `figures/` as both PDF and PNG. Consistent `rcParams` across all cells.

---

## Cell Layout

### Cell 0: Configuration + Style Setup

```python
# === Configuration ===
import sys, os
PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

RESULTS_DIR = "results/pilot"   # ← switch to "results/setup_2" for the final run
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

import yaml, json, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# === Publication-quality style ===
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 10, "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.figsize": (3.5, 2.8),
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "lines.markersize": 4,
})

# Colour-blind-friendly palette
COLORS = {
    "ours_bayesian": "#1b9e77",
    "ours_point":    "#d95f02",
    "han_original":  "#7570b3",
    "han_adapted":   "#e7298a",
    "token_entropy": "#66a61e",
    "semantic_ent":  "#e6ab02",
    "luq":           "#a6761d",
    "log_reg":       "#666666",
}
DOUBLE_COL = (7.0, 3.0)
SINGLE_COL = (3.5, 2.8)

def save_fig(fig, name):
    """Save as PDF + PNG simultaneously."""
    fig.savefig(f"{FIGURES_DIR}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{FIGURES_DIR}/{name}.png", bbox_inches="tight", pad_inches=0.02)
    print(f"  Saved: {FIGURES_DIR}/{name}.{{pdf,png}}")

results_dir = Path(RESULTS_DIR)
```

### Cell 1: Figure 1 — Method Overview / Architecture Diagram

Usually created externally (draw.io, TikZ). Placeholder cell.

```python
print("Figure 1 (method overview) — create with TikZ or draw.io.")
print("Content: prompt → LLM → hidden states → feature extractor (W, α)")
print("         → Fisher scoring → Bayesian posterior → uncertainty decomposition")
```

### Cell 2: Figure 2 — Reliability Diagrams (Ours vs Han et al.)

Three subplots: Ours (Bayesian), Ours (Point), Han et al. ECE annotated.

```python
from src.evaluation.metrics import compute_calibration_metrics

fig, axes = plt.subplots(1, 3, figsize=DOUBLE_COL, sharey=True)
methods_to_plot = [
    ("Ours (Bayesian)", COLORS["ours_bayesian"]),
    ("Ours (Point)", COLORS["ours_point"]),
    ("Fact Probe (Han)", COLORS["han_original"]),
]
for ax, (method_name, color) in zip(axes, methods_to_plot):
    # Load y_true, p_pred for each method and draw reliability diagram
    # ax.bar(bin_centers, bin_acc, ...) + ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("Predicted μ̂"); ax.set_title(method_name, fontsize=10)
axes[0].set_ylabel("Observed U_j")
# save_fig(fig, "fig2_reliability_diagrams")
print("Figure 2: complete after loading per-method predictions")
```

### Cell 3: Figure 3 — PRR Curves (All Methods)

All methods overlaid on one plot. Legend on the right.

```python
fig, ax = plt.subplots(figsize=(4.5, 3.0))
# for method_name, color in COLORS.items():
#     ax.plot(rejection_rates, remaining_quality, color=color, label=method_name)
ax.set_xlabel("Rejection Rate"); ax.set_ylabel("Remaining Quality (1 − MAE)")
ax.set_title("Prediction Rejection Ratio")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
# save_fig(fig, "fig3_prr_curves")
print("Figure 3: complete after loading PRR data")
```

### Cell 4: Figure 4 — Layer α Distribution

Learned `softmax(α)` bar chart with Han et al. layer-14 annotation.

```python
alpha_csv = results_dir / "alpha_distribution.csv"
if alpha_csv.exists():
    alpha_df = pd.read_csv(alpha_csv)
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    layers = alpha_df["selected_layer"].values
    weights = alpha_df["softmax_alpha"].values
    bar_colors = [COLORS["han_original"] if l == 14 else COLORS["ours_bayesian"] for l in layers]
    ax.bar(range(len(layers)), weights, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], rotation=45, fontsize=7)
    ax.set_xlabel("Layer Index"); ax.set_ylabel("softmax(α)")
    ax.axhline(y=1.0/len(layers), color="gray", linestyle=":", alpha=0.6, label="Uniform")
    if 14 in list(layers):
        idx_14 = list(layers).index(14)
        ax.annotate("Han et al.\noptimal", xy=(idx_14, weights[idx_14]),
                    xytext=(idx_14 + 2, weights[idx_14] + 0.02),
                    arrowprops=dict(arrowstyle="->", color=COLORS["han_original"]),
                    fontsize=8, color=COLORS["han_original"])
    ax.legend(fontsize=8)
    save_fig(fig, "fig4_alpha_distribution"); plt.show()
```

### Cell 5: Figure 5 — μ̂ vs U Scatter (Compact)

Paper-ready scatter. Marker size ∝ m_j.

### Cell 6: Figure 6 — Token-Level Heatmap Example

Han et al. Figure 1 style: token-level π̂_ℓ colour bar + attribution.

### Cell 7: Figure 7 — ECE Comparison Bar Chart (Compact)

Three methods only: Ours Bayesian, Ours Point, Han et al. Core-hypothesis arrow.

### Cell 8: Table 1 — Main Results (LaTeX Output)

```python
ratio_csv = results_dir / "final_metrics_ratio.csv"
strict_csv = results_dir / "final_metrics_strict.csv"
if ratio_csv.exists():
    print("=== Table 1: Ratio-Level (LaTeX) ===")
    print(pd.read_csv(ratio_csv).to_latex(index=False, float_format="%.3f"))
if strict_csv.exists():
    print("\n=== Table 2: Strict (LaTeX) ===")
    print(pd.read_csv(strict_csv).to_latex(index=False, float_format="%.3f"))
```

### Cell 9: Table 2 — Ablation Summary (LaTeX Output)

### Cell 10: Figure Inventory

List all generated figures with file sizes.

```python
fig_dir = Path(FIGURES_DIR)
if fig_dir.exists():
    pdfs = sorted(fig_dir.glob("*.pdf"))
    print(f"Generated {len(pdfs)} PDF figures:")
    for p in pdfs:
        print(f"  {p.name:40s} ({p.stat().st_size / 1024:.0f} KB)")
```

---

## Dependencies

| Module | Functions / Classes Used |
|--------|-------------------------|
| `src.evaluation.metrics` | `compute_calibration_metrics(...)`, `plot_reliability_diagram(...)` |
| `src.inference.predict` | `load_trained_model(...)`, `Predictor` |

## Inputs

All Phase 6-2 outputs: `final_metrics_ratio.csv`, `final_metrics_strict.csv`,
`alpha_distribution.csv`, `ablation_*.csv`, `reliability_diagrams/*.png`,
`prr_curves.png`, `trained_model.pt`

## Outputs

- `figures/fig2_reliability_diagrams.{pdf,png}`
- `figures/fig3_prr_curves.{pdf,png}`
- `figures/fig4_alpha_distribution.{pdf,png}`
- `figures/fig5_prediction_scatter.{pdf,png}`
- `figures/fig6_token_heatmap.{pdf,png}`
- `figures/fig7_ece_comparison.{pdf,png}`
- LaTeX table output (copy-paste ready)

Note: Figure numbering is provisional — adjust to match the final paper structure.
