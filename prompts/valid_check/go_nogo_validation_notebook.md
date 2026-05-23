# Go/No-Go Validation Notebook

Create `validation/go_nogo_validation.ipynb`.

**Purpose**: A single condensed notebook that answers one question:
**"Does the Bayesian model beat baselines enough to justify continuing this project?"**

This is NOT a deep-dive analysis notebook. It is a fast, visual checklist that
a researcher can run after `scripts/run_pilot.sh` (or any experiment) completes,
skim the outputs top-to-bottom, and reach a go/no-go decision in under 5 minutes.

**Important**: This notebook lives in `validation/`, not `notebooks/`.
It assumes it is run from the project root (`sentence_uq/`), so paths are
relative to the project root (no `..` parent directory hacks).

**Prerequisite**: Phase 6-2 evaluation (`scripts/04_evaluate.py`) must have run
successfully, producing `results/{run}/final_metrics_ratio.csv`,
`final_metrics_strict.csv`, and related artefacts.

---

## Design Principles

1. **Every cell ends with a PASS / WARN / FAIL verdict** printed in bold.
   Use a helper function `verdict(condition, metric_name, detail_str)` that prints
   a colour-coded (green/yellow/red via ANSI codes) one-liner.
2. **Minimal code, maximal signal.** Load pre-computed Phase 6-2 outputs; do NOT
   re-run inference or training inside this notebook.
3. **Self-contained.** A single run cell at the top sets `RESULTS_DIR`. Everything
   else flows from that path.
4. **Final summary cell** aggregates all verdicts into a single GO / CONDITIONAL / NO-GO.

---

## Cell Layout

### Cell 0: Configuration + Verdict Helper

```python
# === Configuration — edit this one line ===
RESULTS_DIR = "results/pilot"      # ← change per experiment
CONFIG_PATH = "configs/pilot.yaml"
# ==========================================

import sys, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure project root is importable (notebook may be run from validation/ or root)
PROJECT_ROOT = os.path.abspath(".")
if os.path.basename(PROJECT_ROOT) == "validation":
    PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import yaml
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

%matplotlib inline
plt.rcParams.update({"figure.figsize": (12, 5), "font.size": 11})

results_dir = Path(RESULTS_DIR)

# --- verdict helper ---
_verdicts = []  # accumulate for final summary

def verdict(passed, metric_name, detail="", warn=False):
    """Print and record a PASS / WARN / FAIL line."""
    if passed and not warn:
        tag = "\033[92m✓ PASS\033[0m"
        _verdicts.append(("PASS", metric_name))
    elif warn:
        tag = "\033[93m⚠ WARN\033[0m"
        _verdicts.append(("WARN", metric_name))
    else:
        tag = "\033[91m✗ FAIL\033[0m"
        _verdicts.append(("FAIL", metric_name))
    print(f"  {tag}  {metric_name}: {detail}")

print(f"Results dir : {results_dir}")
print(f"Config      : {CONFIG_PATH}")
assert results_dir.exists(), f"Results directory not found: {results_dir}"
```

### Cell 1: Data Health Quick-Check

Load annotation files and verify the dataset is large enough and balanced enough
to trust downstream metrics. This cell does NOT reproduce the full Phase 8-1
data overview — it checks only the bare minimum for go/no-go.

**Checks**:
- Total sentence count (with m_j > 0) — FAIL if < 200 for pilot, WARN if < 500
- m_j = 0 fraction — WARN if > 20%
- K_j/m_j distribution — WARN if > 80% of sentences have U_j = 0 or U_j = 1
  (extreme imbalance makes continuous modelling uninformative)
- Strict factual fraction (A_j = 1) — WARN if < 10% or > 90%

```python
import glob

# Determine processed dir from config
setup = cfg.get("evaluation", {}).get("setup", 2)
proc_dir = Path(cfg.get("processed", {}).get("factscore_bio_dir",
                f"data/processed/factscore_bio"))

# Load split to identify test entities
split_file = cfg.get("dataset", {}).get("split_file",
             f"data/splits/setup_{setup}.json")
with open(split_file) as f:
    splits = json.load(f)

test_entities = set(splits.get("test", []))

all_records = []
for jf in sorted(proc_dir.glob("*.json")):
    entity_name = jf.stem
    with open(jf) as f:
        records = json.load(f)
    for rec in records:
        rec["entity"] = entity_name
        rec["split"] = "test" if entity_name in test_entities else "train_or_val"
    all_records.extend(records)

df = pd.DataFrame(all_records)
df["U_j"] = df["K_j"] / df["m_j"].replace(0, np.nan)
df["A_j"] = (df["K_j"] == df["m_j"]).astype(int)
df_valid = df[df["m_j"] > 0].copy()
df_test = df_valid[df_valid["split"] == "test"].copy()

print(f"Total sentences: {len(df)}, valid (m_j>0): {len(df_valid)}, test split: {len(df_test)}")
print(f"m_j=0 fraction: {(df['m_j']==0).mean():.2%}")

# Distribution summary
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df_valid["m_j"], bins=range(0, df_valid["m_j"].max()+2), edgecolor="k", alpha=0.7)
axes[0].set_title("m_j distribution (m_j > 0)"); axes[0].set_xlabel("m_j")
axes[1].hist(df_valid["U_j"].dropna(), bins=20, edgecolor="k", alpha=0.7, color="tab:orange")
axes[1].set_title("U_j = K_j/m_j distribution"); axes[1].set_xlabel("U_j")
axes[2].bar(["Factual (A=1)", "Hallucinated (A=0)"],
            [df_test["A_j"].mean(), 1 - df_test["A_j"].mean()], color=["tab:green", "tab:red"])
axes[2].set_title("Test split class balance"); axes[2].set_ylabel("Fraction")
plt.tight_layout(); plt.show()

# Verdicts
n_valid = len(df_test)
verdict(n_valid >= 200, "Test set size",
        f"{n_valid} sentences (need ≥200)", warn=(200 <= n_valid < 500))
verdict((df["m_j"]==0).mean() <= 0.20, "m_j=0 fraction",
        f"{(df['m_j']==0).mean():.1%}",
        warn=(0.10 < (df["m_j"]==0).mean() <= 0.20))

boundary_frac = ((df_valid["U_j"] == 0) | (df_valid["U_j"] == 1)).mean()
verdict(boundary_frac <= 0.80, "U_j boundary fraction",
        f"{boundary_frac:.1%} at 0 or 1", warn=(0.60 < boundary_frac <= 0.80))

strict_frac = df_test["A_j"].mean()
verdict(0.10 <= strict_frac <= 0.90, "Class balance (test)",
        f"strict factual = {strict_frac:.1%}",
        warn=(strict_frac < 0.15 or strict_frac > 0.85))
```

### Cell 2: Hidden-State Signal Sanity (Per-Layer AUROC)

A fast check that the hidden states carry factuality information at all.
Compute per-layer logistic regression AUROC on a random subsample of test
sentences. If peak AUROC < 0.55, probing approaches are fundamentally broken.

**This cell re-uses generation .pt and annotation .json files directly.**
Subsample to at most `MAX_ENTITIES` entities for speed.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch

gen_dir = Path(cfg["generation"]["factscore_bio_dir"])
MAX_ENTITIES = 20  # keep it fast

test_entity_files = sorted([
    gf for gf in gen_dir.glob("*.pt")
    if gf.stem in test_entities
])[:MAX_ENTITIES]

sentence_layers = []  # (N, num_layers, hidden_dim)
labels = []

for gf in test_entity_files:
    pf = proc_dir / f"{gf.stem}.json"
    if not pf.exists():
        continue
    gen = torch.load(gf, map_location="cpu", weights_only=False)
    with open(pf) as f:
        records = json.load(f)
    hs = gen["hidden_states"].float()
    for rec in records:
        if rec["m_j"] == 0:
            continue
        start, end = rec["token_range"]
        if end <= start or end > hs.shape[0]:
            continue
        mean_hs = hs[start:end].mean(dim=0)  # (num_layers, hidden_dim)
        sentence_layers.append(mean_hs)
        labels.append(1 if rec["K_j"] == rec["m_j"] else 0)

sentence_layers = torch.stack(sentence_layers)  # (N, L, D)
labels = np.array(labels)
num_layers = sentence_layers.shape[1]
print(f"Loaded {len(labels)} test sentences, {num_layers} layers")

# Per-layer AUROC (fast LogReg)
layer_aurocs = []
for l in range(num_layers):
    X = sentence_layers[:, l, :].numpy()
    if np.unique(labels).size < 2:
        layer_aurocs.append(0.5)
        continue
    try:
        clf = LogisticRegression(max_iter=300, C=0.1, solver="liblinear",
                                 penalty="l1", random_state=42)
        clf.fit(X, labels)
        probs = clf.predict_proba(X)[:, 1]
        layer_aurocs.append(roc_auc_score(labels, probs))
    except Exception:
        layer_aurocs.append(0.5)

layer_aurocs = np.array(layer_aurocs)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(num_layers), layer_aurocs, "-o", markersize=4)
best_layer = np.argmax(layer_aurocs)
ax.axvline(x=best_layer, color="red", linestyle="--", alpha=0.5,
           label=f"Best: layer {best_layer} (AUROC={layer_aurocs[best_layer]:.4f})")
ax.axvline(x=14, color="gray", linestyle=":", alpha=0.5, label="Han et al. layer 14")
ax.axhline(y=0.55, color="orange", linestyle="--", alpha=0.3, label="Minimum bar (0.55)")
ax.set_xlabel("Layer"); ax.set_ylabel("AUROC"); ax.set_title("Per-Layer Probing AUROC (test set)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

peak = layer_aurocs.max()
verdict(peak >= 0.60, "Hidden-state signal",
        f"peak AUROC = {peak:.4f} at layer {best_layer}",
        warn=(0.55 <= peak < 0.60))
verdict(peak >= 0.55, "Minimum viability",
        f"peak AUROC = {peak:.4f} (hard floor 0.55)")
```

### Cell 3: Training Convergence

Load `trained_model.pt` training history. Verify loss converged and
Laplace PD checks passed.

```python
model_path = results_dir / "trained_model.pt"
assert model_path.exists(), f"Trained model not found: {model_path}"

ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
history = ckpt.get("history", {})

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# (a) Loss curve
if "train_loss" in history:
    axes[0].plot(history["train_loss"], label="train")
if "val_loss" in history:
    axes[0].plot(history["val_loss"], label="val", linestyle="--")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Loss Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

# (b) Validation metrics if available
if history.get("val_metrics"):
    val_df_rows = [vm | {"epoch": i+1} for i, vm in enumerate(history["val_metrics"]) if vm]
    if val_df_rows:
        vdf = pd.DataFrame(val_df_rows)
        for col, color in [("MAE", "tab:blue"), ("Pearson_r", "tab:green")]:
            if col in vdf.columns:
                axes[1].plot(vdf["epoch"], vdf[col], "-o", color=color,
                             markersize=3, label=col)
        axes[1].set_xlabel("Epoch"); axes[1].set_title("Validation Metrics")
        axes[1].legend(); axes[1].grid(alpha=0.3)

# (c) PD checks
if history.get("pd_checks"):
    pd_df = pd.DataFrame(history["pd_checks"])
    if "fisher_min_eig" in pd_df.columns:
        axes[2].plot(pd_df["fisher_min_eig"], "b-o", markersize=3, label="Fisher min eig")
    axes[2].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[2].set_title("PD Check (min eigenvalue)"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout(); plt.show()

# Verdicts
if "train_loss" in history and len(history["train_loss"]) >= 5:
    last5 = history["train_loss"][-5:]
    converged = (max(last5) - min(last5)) / (abs(last5[0]) + 1e-8) < 0.10
    verdict(converged, "Training convergence",
            f"last 5 epoch loss range: {min(last5):.4f}–{max(last5):.4f}")
else:
    verdict(False, "Training convergence", "Not enough epochs in history")

if history.get("pd_checks"):
    all_pd_ok = all(c.get("laplace_valid_local", False) for c in history["pd_checks"])
    last_pd_ok = history["pd_checks"][-1].get("laplace_valid_local", False)
    verdict(last_pd_ok, "Laplace PD (final)",
            "positive definite" if last_pd_ok else "NOT positive definite")
    if not all_pd_ok:
        verdict(False, "Laplace PD (all epochs)", "PD failed at some epochs", warn=True)
else:
    verdict(True, "Laplace PD", "no PD check history (may be OK for short runs)", warn=True)
```

### Cell 4: Core Hypothesis — ECE Comparison

**THE most important cell.** Load Phase 6-2 metric tables and check whether
Bayesian ECE < Point ECE and Bayesian ECE < Han et al. ECE.

```python
ratio_csv = results_dir / "final_metrics_ratio.csv"
strict_csv = results_dir / "final_metrics_strict.csv"
assert ratio_csv.exists(), f"Not found: {ratio_csv} — run scripts/04_evaluate.py first"

ratio_df = pd.read_csv(ratio_csv)
strict_df = pd.read_csv(strict_csv) if strict_csv.exists() else None

print("=" * 60)
print("RATIO-LEVEL METRICS (Primary Evaluation)")
print("=" * 60)
display(ratio_df.round(4))

if strict_df is not None:
    print("\n" + "=" * 60)
    print("STRICT FACTUALITY METRICS (Secondary Evaluation)")
    print("=" * 60)
    display(strict_df.round(4))

# --- Extract key methods ---
def get_metric(df, method_substr, col):
    """Get a metric value for a method (fuzzy match on name)."""
    mask = df["method"].str.lower().str.contains(method_substr.lower())
    if mask.any():
        return df.loc[mask, col].values[0]
    return np.nan

ece_bayesian = get_metric(ratio_df, "bayesian", "ECE")
ece_point = get_metric(ratio_df, "point", "ECE")
ece_han_orig = get_metric(ratio_df, "han.*original", "ECE")
ece_han_adapt = get_metric(ratio_df, "han.*adapted", "ECE")
ece_han = np.nanmin([ece_han_orig, ece_han_adapt])  # best Han variant

# ECE bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Ratio-level ECE
methods_r = ratio_df["method"].values
ece_r = ratio_df["ECE"].values
colors_r = []
for m in methods_r:
    ml = m.lower()
    if "bayesian" in ml: colors_r.append("tab:green")
    elif "point" in ml: colors_r.append("tab:orange")
    elif "han" in ml: colors_r.append("tab:red")
    else: colors_r.append("tab:blue")
axes[0].barh(methods_r, ece_r, color=colors_r, edgecolor="k", alpha=0.8)
axes[0].set_xlabel("ECE (lower is better)"); axes[0].set_title("Ratio-Level ECE")
axes[0].invert_yaxis()

# Strict ECE if available
if strict_df is not None and "ECE" in strict_df.columns:
    methods_s = strict_df["method"].values
    ece_s = strict_df["ECE"].values
    colors_s = []
    for m in methods_s:
        ml = m.lower()
        if "bayesian" in ml: colors_s.append("tab:green")
        elif "point" in ml: colors_s.append("tab:orange")
        elif "han" in ml: colors_s.append("tab:red")
        else: colors_s.append("tab:blue")
    axes[1].barh(methods_s, ece_s, color=colors_s, edgecolor="k", alpha=0.8)
    axes[1].set_xlabel("ECE (lower is better)"); axes[1].set_title("Strict Factuality ECE")
    axes[1].invert_yaxis()

plt.tight_layout(); plt.show()

# Verdicts — core hypothesis
verdict(ece_bayesian < ece_point, "ECE: Bayesian < Point",
        f"Bayesian={ece_bayesian:.4f} vs Point={ece_point:.4f}")
verdict(ece_bayesian < ece_han, "ECE: Bayesian < Han et al.",
        f"Bayesian={ece_bayesian:.4f} vs Han={ece_han:.4f}")
if not np.isnan(ece_point) and not np.isnan(ece_han):
    verdict(ece_point < ece_han, "ECE: Point < Han (sanity)",
            f"Point={ece_point:.4f} vs Han={ece_han:.4f}", warn=True)
```

### Cell 5: AUROC — Competitive with Baselines?

Check strict factuality AUROC. The model does not need to dominate on AUROC
(that is not the core claim), but it must be competitive.
Reference: Han et al. Llama-3.1-8B in-domain AUROC ≈ 0.7357.

```python
if strict_df is not None and "AUROC" in strict_df.columns:
    auroc_ours = get_metric(strict_df, "bayesian", "AUROC")
    auroc_han = np.nanmin([
        get_metric(strict_df, "han.*original", "AUROC"),
        get_metric(strict_df, "han.*adapted", "AUROC"),
    ])  # note: take the BETTER Han variant for conservative comparison

    # Actually, for AUROC we want to compare against the best Han variant
    auroc_han_best = np.nanmax([
        get_metric(strict_df, "han.*original", "AUROC"),
        get_metric(strict_df, "han.*adapted", "AUROC"),
    ])

    print(f"Ours (Bayesian) AUROC : {auroc_ours:.4f}")
    print(f"Han et al. best AUROC : {auroc_han_best:.4f}")
    print(f"Reference (Han paper) : 0.7357")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    methods_s = strict_df["method"].values
    auroc_s = strict_df["AUROC"].values
    colors_a = []
    for m in methods_s:
        ml = m.lower()
        if "bayesian" in ml: colors_a.append("tab:green")
        elif "point" in ml: colors_a.append("tab:orange")
        elif "han" in ml: colors_a.append("tab:red")
        else: colors_a.append("tab:blue")
    ax.barh(methods_s, auroc_s, color=colors_a, edgecolor="k", alpha=0.8)
    ax.axvline(x=0.7357, color="gray", linestyle=":", alpha=0.7,
               label="Han et al. paper (0.7357)")
    ax.set_xlabel("AUROC (higher is better)"); ax.set_title("Strict Factuality AUROC")
    ax.legend(); ax.invert_yaxis()
    plt.tight_layout(); plt.show()

    # With 95% CI if available
    if "AUROC_CI_lo" in strict_df.columns:
        ci_lo = get_metric(strict_df, "bayesian", "AUROC_CI_lo")
        ci_hi = get_metric(strict_df, "bayesian", "AUROC_CI_hi")
        print(f"Ours 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    delta = auroc_ours - auroc_han_best
    verdict(delta >= -0.03, "AUROC competitive",
            f"Δ vs Han = {delta:+.4f}",
            warn=(-0.03 <= delta < 0.0))
else:
    verdict(False, "AUROC competitive", "strict metrics CSV not found")
```

### Cell 6: Rejection Curve (PRR) — Does Uncertainty Help?

Plot PRR curves for all methods. The Bayesian model's epistemic uncertainty
should provide a better rejection signal than baselines.

```python
# Try to load pre-computed PRR data, or compute from CSVs
prr_png = results_dir / "prr_curves.png"

if prr_png.exists():
    from IPython.display import Image, display as ipy_display
    print("Pre-computed PRR curves (from Phase 6-2):")
    ipy_display(Image(filename=str(prr_png), width=800))
else:
    print("PRR curves image not found — check Phase 6-2 output.")

# Verdict based on PRR_AUC from CSV
prr_ours = get_metric(ratio_df, "bayesian", "PRR_AUC")
prr_han = np.nanmax([
    get_metric(ratio_df, "han.*original", "PRR_AUC"),
    get_metric(ratio_df, "han.*adapted", "PRR_AUC"),
])

print(f"\nPRR AUC — Ours: {prr_ours:.4f}, Han best: {prr_han:.4f}")

verdict(prr_ours >= prr_han, "PRR: Ours ≥ Han",
        f"Ours={prr_ours:.4f} vs Han={prr_han:.4f}",
        warn=(prr_han - prr_ours < 0.02))
```

### Cell 7: Reliability Diagrams — Visual Calibration

Show reliability diagrams for the three key methods side-by-side.
If pre-generated plots exist, display them; otherwise build from scratch.

```python
rel_dir = results_dir / "reliability_diagrams"

key_methods = ["Ours_Bayesian", "Ours_Point", "Han_original", "Han_adapted"]

if rel_dir.exists():
    from IPython.display import Image, display as ipy_display
    pngs = sorted(rel_dir.glob("*.png"))
    if pngs:
        # Show up to 4 key plots
        fig, axes = plt.subplots(1, min(len(pngs), 4), figsize=(5 * min(len(pngs), 4), 4))
        if not hasattr(axes, '__len__'):
            axes = [axes]
        for ax, png in zip(axes, pngs[:4]):
            img = plt.imread(str(png))
            ax.imshow(img); ax.axis("off"); ax.set_title(png.stem, fontsize=9)
        plt.tight_layout(); plt.show()
    else:
        print("No reliability diagram PNGs found.")
else:
    print("Reliability diagrams directory not found — run Phase 6-2 with plots enabled.")

# No separate verdict — ECE numbers already checked in Cell 4
print("(Visual confirmation of Cell 4 ECE numbers — no separate verdict)")
```

### Cell 8: Bayesian-Specific Advantages

Quick checks on features unique to the Bayesian model.

```python
# (a) MC vs Linear approximation correlation
mc_lin_png = results_dir / "mc_vs_linear.png"
if mc_lin_png.exists():
    from IPython.display import Image, display as ipy_display
    print("MC vs Linear epistemic approximation:")
    ipy_display(Image(filename=str(mc_lin_png), width=500))

# (b) Binomial NLL (only our method can produce this)
binom_nll = get_metric(ratio_df, "bayesian", "binomial_NLL")
print(f"\nBinomial NLL (Ours): {binom_nll:.4f}")

# (c) Learned layer-alpha distribution
alpha_csv = results_dir / "alpha_distribution.csv"
alpha_png = results_dir / "alpha_distribution.png"

if alpha_png.exists():
    from IPython.display import Image, display as ipy_display
    ipy_display(Image(filename=str(alpha_png), width=600))
elif alpha_csv.exists():
    alpha_df = pd.read_csv(alpha_csv)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(alpha_df.iloc[:, 0], alpha_df.iloc[:, 1], edgecolor="k", alpha=0.7)
    ax.axvline(x=14, color="red", linestyle="--", alpha=0.5, label="Han et al. layer 14")
    ax.set_xlabel("Layer"); ax.set_ylabel("α weight"); ax.set_title("Learned Layer Weights")
    ax.legend(); plt.tight_layout(); plt.show()

# (d) Ablation: Binomial vs Bernoulli
abl_binom = results_dir / "ablation_binomial_vs_bernoulli.csv"
if abl_binom.exists():
    abl_df = pd.read_csv(abl_binom)
    print("\nBinomial vs Bernoulli ablation:")
    display(abl_df.round(4))
    binom_better = (abl_df.loc[abl_df["method"].str.contains("binomial", case=False), "MAE"].values[0]
                    < abl_df.loc[abl_df["method"].str.contains("bernoulli", case=False), "MAE"].values[0])
    verdict(binom_better, "Binomial > Bernoulli (MAE)",
            "count-awareness helps" if binom_better else "Bernoulli is better or equal",
            warn=not binom_better)
else:
    print("Binomial vs Bernoulli ablation not found (optional)")

# Verdict: is binomial NLL finite and reasonable?
verdict(np.isfinite(binom_nll) and binom_nll < 5.0, "Binomial NLL reasonable",
        f"NLL = {binom_nll:.4f}")
```

### Cell 9: Full Metric Summary Table

Print a single consolidated table with all methods and all key metrics,
highlighted for easy scanning.

```python
# Merge ratio + strict into one table
if strict_df is not None:
    # Join on method
    merged = ratio_df.merge(strict_df, on="method", suffixes=("_ratio", "_strict"),
                            how="outer")
else:
    merged = ratio_df.copy()

# Select key columns
key_cols = ["method"]
for c in ["MAE", "Pearson_r", "ECE", "PRR_AUC", "binomial_NLL",
           "AUROC", "AUPRC", "Brier_strict", "ECE_strict"]:
    # Handle suffixed columns from merge
    candidates = [c, f"{c}_ratio", f"{c}_strict"]
    for cand in candidates:
        if cand in merged.columns and cand not in key_cols:
            key_cols.append(cand)
            break

summary = merged[[c for c in key_cols if c in merged.columns]].copy()
print("=" * 80)
print("CONSOLIDATED METRIC SUMMARY")
print("=" * 80)
display(summary.round(4))
```

### Cell 10: GO / NO-GO Final Verdict

Aggregate all verdicts and produce a single decision.

Decision logic:
- **GO**: zero FAIL and ≤ 1 WARN
- **CONDITIONAL**: zero FAIL but ≥ 2 WARN, or exactly 1 FAIL on a non-core metric
- **NO-GO**: ≥ 1 FAIL on a core metric (ECE hypothesis, minimum viability, convergence)

Core metrics: "ECE: Bayesian < Point", "ECE: Bayesian < Han et al.",
"Minimum viability", "Training convergence", "Laplace PD (final)".

```python
CORE_METRICS = {
    "ECE: Bayesian < Point",
    "ECE: Bayesian < Han et al.",
    "Minimum viability",
    "Training convergence",
    "Laplace PD (final)",
}

n_pass = sum(1 for s, _ in _verdicts if s == "PASS")
n_warn = sum(1 for s, _ in _verdicts if s == "WARN")
n_fail = sum(1 for s, _ in _verdicts if s == "FAIL")
core_fails = [(s, m) for s, m in _verdicts if s == "FAIL" and m in CORE_METRICS]

print("=" * 80)
print(f"  PASS: {n_pass}   WARN: {n_warn}   FAIL: {n_fail}")
print("=" * 80)

if n_fail > 0:
    print("\nFailed checks:")
    for s, m in _verdicts:
        if s == "FAIL":
            core_tag = " ★ CORE" if m in CORE_METRICS else ""
            print(f"  ✗ {m}{core_tag}")

if n_warn > 0:
    print("\nWarnings:")
    for s, m in _verdicts:
        if s == "WARN":
            print(f"  ⚠ {m}")

print("\n" + "=" * 80)
if len(core_fails) > 0:
    print("\033[91m" + "  ██  NO-GO  ██" + "\033[0m")
    print("  Core hypothesis failed. Review the failed metrics above.")
    print("  Possible actions: re-tune hyperparameters, check data pipeline,")
    print("  or reconsider the modelling approach.")
elif n_fail == 0 and n_warn <= 1:
    print("\033[92m" + "  ██  GO  ██" + "\033[0m")
    print("  All core checks passed. Proceed to full-scale experiments.")
else:
    print("\033[93m" + "  ██  CONDITIONAL  ██" + "\033[0m")
    print("  Core hypothesis holds but there are warnings/minor failures.")
    print("  Address warnings before scaling up, or proceed with caution.")
print("=" * 80)
```

---

## File Structure After Creation

```
sentence_uq/
├── validation/
│   └── go_nogo_validation.ipynb    ← THIS FILE
├── notebooks/                       # Phase 8 detailed analysis (separate)
│   ├── 01_data_overview.ipynb
│   ├── ...
```

## Tests

No separate test file — this notebook IS the test. Verify by running:
```bash
cd sentence_uq
jupyter nbconvert --execute validation/go_nogo_validation.ipynb --to html
```
The output HTML should show a GO / CONDITIONAL / NO-GO verdict at the bottom.
