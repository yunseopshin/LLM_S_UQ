# Phase 8-7 — Notebook: Ablation Results

Create `notebooks/07_ablation_results.ipynb`.

**Purpose**: Aggregate and visualise all ablation results from Phase 6-2 built-in
ablations and the `experiments/run_ablation_suite.sh` outputs.

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
ABLATION_DIR = "results/ablations"   # ← run_ablation_suite.sh output

import yaml, json, torch, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

%matplotlib inline
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})

results_dir = Path(RESULTS_DIR)
ablation_dir = Path(ABLATION_DIR)
```

### Cell 1: Load All Ablation CSVs

Phase 6-2 built-in ablations:
- `ablation_bayesian_vs_point.csv`
- `ablation_binomial_vs_bernoulli.csv`
- `ablation_mc_vs_linear.csv`

Ablation suite results:
- `results/ablations/ablation_{name}/final_metrics_ratio.csv`

```python
ab_bp = pd.read_csv(results_dir / "ablation_bayesian_vs_point.csv") \
    if (results_dir / "ablation_bayesian_vs_point.csv").exists() else None
ab_bb = pd.read_csv(results_dir / "ablation_binomial_vs_bernoulli.csv") \
    if (results_dir / "ablation_binomial_vs_bernoulli.csv").exists() else None
ab_mc = pd.read_csv(results_dir / "ablation_mc_vs_linear.csv") \
    if (results_dir / "ablation_mc_vs_linear.csv").exists() else None

suite_results = {}
if ablation_dir.exists():
    for sub in sorted(ablation_dir.iterdir()):
        if sub.is_dir():
            ratio_csv = sub / "final_metrics_ratio.csv"
            if ratio_csv.exists():
                suite_results[sub.name] = pd.read_csv(ratio_csv)

print(f"Phase 6-2 ablations: bp={ab_bp is not None}, bb={ab_bb is not None}, mc={ab_mc is not None}")
print(f"Suite ablation dirs: {list(suite_results.keys())}")
```

### Cell 2: Bayesian vs Point Estimate

Core ablation — Bayesian (with Σ̂) vs Point (Σ̂ = 0). Side-by-side AUROC, Brier, ECE.

```python
if ab_bp is not None:
    display(ab_bp)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    variants = ab_bp["variant"].tolist()
    colors = ["#2ecc71", "#f39c12"]
    for ax, col, title in [
        (axes[0], "AUROC", "AUROC (higher ↑)"),
        (axes[1], "Brier", "Brier Score (lower ↓)"),
        (axes[2], "ECE", "ECE (lower ↓)"),
    ]:
        if col in ab_bp.columns:
            ax.bar(variants, ab_bp[col].values, color=colors, edgecolor="black", alpha=0.85)
            ax.set_title(title)
            vals = ab_bp[col].values
            if len(vals) == 2:
                diff = vals[0] - vals[1]
                ax.annotate(f"Δ={diff:+.4f}", xy=(0.5, max(vals)),
                           ha="center", fontsize=10, color="navy")
    plt.suptitle("Bayesian vs Point Estimate (Core Hypothesis)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ablation_bayesian_vs_point.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### Cell 3: Binomial vs Bernoulli

Does modelling the count `m_j` help? Phase 6-2 spec "m_j=1 ablation".

```python
if ab_bb is not None:
    display(ab_bb)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    variants = ab_bb["variant"].tolist()
    colors = ["#2ecc71", "#e74c3c"]
    for ax, col, title in [
        (axes[0], "ratio_MAE", "Ratio MAE (lower ↓)"),
        (axes[1], "strict_ECE", "Strict ECE (lower ↓)"),
        (axes[2], "strict_AUROC", "Strict AUROC (higher ↑)"),
    ]:
        if col in ab_bb.columns:
            ax.bar(variants, ab_bb[col].values, color=colors, edgecolor="black", alpha=0.85)
            ax.set_title(title)
    plt.suptitle("Binomial vs Bernoulli (Does Count Awareness Help?)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ablation_binomial_vs_bernoulli_detail.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### Cell 4: MC vs Linear Epistemic

Delta-method (closed-form) vs Monte Carlo epistemic scatter + correlation.
If correlation > 0.9, the delta method is sufficiently accurate.

```python
mc_png = results_dir / "mc_vs_linear.png"
if mc_png.exists():
    from IPython.display import Image, display as ipy_display
    ipy_display(Image(filename=str(mc_png), width=600))
if ab_mc is not None:
    display(ab_mc)
    if "correlation" in ab_mc.columns:
        corr = ab_mc["correlation"].values[0]
        print(f"Correlation: {corr:.4f} {'✓' if corr > 0.9 else '⚠️'}")
```

### Cell 5: Uniform vs Learned α

Compare uniform α (all layers equally weighted) vs learned α.

```python
uniform_key = [k for k in suite_results if "uniform" in k.lower()]
if uniform_key:
    uniform_df = suite_results[uniform_key[0]]
    learned_df = pd.read_csv(results_dir / "final_metrics_ratio.csv")
    ours_learned = learned_df[learned_df["method"].str.contains("Ours|Bayesian", case=False, na=False)]
    ours_uniform = uniform_df[uniform_df["method"].str.contains("Ours|Bayesian", case=False, na=False)]
    comparison = pd.DataFrame({
        "Metric": ["MAE", "Pearson_r", "ECE"],
        "Learned α": [ours_learned["MAE"].values[0], ours_learned["Pearson_r"].values[0], ours_learned["ECE"].values[0]],
        "Uniform α": [ours_uniform["MAE"].values[0], ours_uniform["Pearson_r"].values[0], ours_uniform["ECE"].values[0]],
    })
    display(comparison)
else:
    print("Uniform α ablation not found in suite results.")
```

### Cell 6: Prior Sigma Sweep

Heatmap or line plot of metrics across different `prior_sigma_init` values.

```python
sigma_keys = sorted([k for k in suite_results if "sigma" in k.lower() or "prior" in k.lower()])
if sigma_keys:
    rows = []
    for key in sigma_keys:
        df = suite_results[key]
        ours = df[df["method"].str.contains("Ours|Bayesian", case=False, na=False)]
        if len(ours) > 0:
            rows.append({"config": key, "MAE": ours["MAE"].values[0],
                         "ECE": ours["ECE"].values[0], "Pearson_r": ours["Pearson_r"].values[0]})
    if rows:
        sweep_df = pd.DataFrame(rows); display(sweep_df)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(sweep_df)), sweep_df["ECE"], "r-o", label="ECE")
        ax.plot(range(len(sweep_df)), sweep_df["MAE"], "b-s", label="MAE")
        ax.set_xticks(range(len(sweep_df)))
        ax.set_xticklabels(sweep_df["config"], rotation=45, ha="right")
        ax.set_title("Prior Sigma Sweep"); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/prior_sigma_sweep.png", dpi=150, bbox_inches="tight")
        plt.show()
else:
    print("No prior sigma sweep results found.")
```

### Cell 7: Generation-time vs Re-encoded Hidden States

Han et al. original (re-encode claims) vs adapted (generation-time) comparison.
Maps to the two `FactualityProbeBaseline` variants in `src/baselines/factuality_probe.py`.

```python
strict_path = results_dir / "final_metrics_strict.csv"
if strict_path.exists():
    strict_df = pd.read_csv(strict_path)
    han_rows = strict_df[strict_df["method"].str.contains("Fact Probe|Han", case=False, na=False)]
    if len(han_rows) >= 2:
        display(han_rows[["method", "AUROC", "ECE", "Brier"]])
```

### Cell 8: Cross-Setup Comparison (Setup 1 vs 2 vs 3)

Load per-setup results from `results/setup_{1,2,3}/`.

```python
setup_results = {}
for i in [1, 2, 3]:
    ratio_csv = Path(f"results/setup_{i}") / "final_metrics_ratio.csv"
    if ratio_csv.exists():
        setup_results[f"setup_{i}"] = pd.read_csv(ratio_csv)
if len(setup_results) >= 2:
    for name, df in setup_results.items():
        ours = df[df["method"].str.contains("Ours|Bayesian", case=False, na=False)]
        if len(ours) > 0:
            print(f"  {name}: MAE={ours['MAE'].values[0]:.3f}, "
                  f"Pearson_r={ours['Pearson_r'].values[0]:.3f}, ECE={ours['ECE'].values[0]:.3f}")
else:
    print("Cross-setup results not yet available. Run: bash experiments/run_cross_setup.sh")
```

### Cell 9: Ablation Summary Table

Aggregate all ablations into one summary table and save.

---

## Dependencies

| Module | Functions / Classes Used |
|--------|-------------------------|
| (none — CSV/PNG loading only) | pandas, matplotlib |

## Inputs

Phase 6-2 outputs: `ablation_bayesian_vs_point.csv`, `ablation_binomial_vs_bernoulli.csv`,
`ablation_mc_vs_linear.csv`, `mc_vs_linear.png`, `final_metrics_*.csv`

Suite outputs (optional): `results/ablations/ablation_{name}/final_metrics_ratio.csv`

Cross-setup (optional): `results/setup_{1,2,3}/final_metrics_ratio.csv`

## Outputs

- `{RESULTS_DIR}/ablation_bayesian_vs_point.png`
- `{RESULTS_DIR}/ablation_binomial_vs_bernoulli_detail.png`
- `{RESULTS_DIR}/prior_sigma_sweep.png`
- `{RESULTS_DIR}/ablation_summary.csv`
