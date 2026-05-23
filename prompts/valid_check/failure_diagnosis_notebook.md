# Failure Diagnosis Notebook

Create `validation/failure_diagnosis.ipynb`.

**Purpose**: When go/no-go validation returns **CONDITIONAL** or **NO-GO**, this
notebook systematically isolates the root cause by testing each pipeline layer
independently. It answers: **"Is the problem in the backbone, annotation,
data, or model?"**

**Important**: This notebook lives in `validation/`, alongside `go_nogo_validation.ipynb`.
It assumes it is run from the project root (`sentence_uq/`).

**Prerequisite**: The full pipeline (through Phase 6-2) must have run at least once.
Some cells (label corruption, learning curve) launch lightweight re-training and
may take 10–30 minutes on GPU.

---

## Design Principles

1. **Diagnostic tree, top-to-bottom.** Each cell tests one layer of the pipeline.
   If a cell FAILs, the cells below it are tainted — the root cause is at or
   above the failing layer.
2. **Every cell produces a DIAGNOSIS** (not just PASS/FAIL) that names the
   suspected component and suggests a concrete remediation.
3. **Cells are ordered by "cheapest to check first"**: annotation spot-check
   (seconds), backbone signal (minutes), data sufficiency (minutes),
   model ablation chain (minutes–tens of minutes).
4. **Reuse existing artefacts** where possible. Only re-train when absolutely
   necessary (learning curve, label corruption).

---

## Diagnostic Levels (Summary)

```
Level 0  Annotation quality     — Are the labels trustworthy?
Level 1  Backbone signal        — Does the hidden state encode factuality at all?
Level 2  Data sufficiency       — Enough data? Distribution shift?
Level 3  Model ablation chain   — Which model component breaks?
         (a) sklearn LogReg on layer-14 mean HS     → Han et al. reproduction
         (b) sklearn LogReg on OUR features (Wα)    → feature extractor check
         (c) Our model, point estimate (Σ=0)        → Fisher scoring check
         (d) Our model, full Bayesian                → Laplace / posterior check
Level 4  Label noise sensitivity — Is annotation noise the bottleneck?
```

---

## Cell Layout

### Cell 0: Configuration + Helpers

```python
# === Configuration — edit these lines ===
RESULTS_DIR = "results/pilot"
CONFIG_PATH = "configs/pilot.yaml"
SETUP = 2
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
# =========================================

import sys, os, json, warnings, time, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from collections import Counter

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

# --- Diagnosis helper ---
_diagnoses = []

def diagnose(level, component, status, detail, remediation=""):
    """Record and print a diagnosis entry.

    status: 'OK', 'SUSPECT', 'ROOT_CAUSE'
    """
    icon = {"OK": "\033[92m✓\033[0m",
            "SUSPECT": "\033[93m⚠\033[0m",
            "ROOT_CAUSE": "\033[91m✗\033[0m"}[status]
    print(f"  {icon}  [L{level}] {component}: {detail}")
    if remediation:
        print(f"        → Remediation: {remediation}")
    _diagnoses.append({
        "level": level, "component": component,
        "status": status, "detail": detail,
        "remediation": remediation,
    })

print(f"Results dir : {results_dir}")
print(f"Device      : {DEVICE}")
```

### Cell 1: Load All Data (Shared Across Cells)

Load generation files, annotation records, and split information once.
All subsequent cells reference these variables.

```python
from src.data.dataset import SETUPS

gen_dir = Path(cfg["generation"]["factscore_bio_dir"])
proc_dir = Path(cfg.get("processed", {}).get("factscore_bio_dir",
                "data/processed/factscore_bio"))
split_file = cfg.get("dataset", {}).get("split_file",
             f"data/splits/setup_{SETUP}.json")

with open(split_file) as f:
    splits = json.load(f)

train_entities = set(splits.get("train", []))
val_entities = set(splits.get("val", []))
test_entities = set(splits.get("test", []))

# Load all annotation records
all_records = []
for jf in sorted(proc_dir.glob("*.json")):
    entity = jf.stem
    with open(jf) as f:
        recs = json.load(f)
    for r in recs:
        r["entity"] = entity
        if entity in test_entities:
            r["split"] = "test"
        elif entity in val_entities:
            r["split"] = "val"
        else:
            r["split"] = "train"
    all_records.extend(recs)

df = pd.DataFrame(all_records)
df["U_j"] = df["K_j"] / df["m_j"].replace(0, np.nan)
df["A_j"] = (df["K_j"] == df["m_j"]).astype(int)
df_valid = df[df["m_j"] > 0].copy()

print(f"Total: {len(df)} sentences, valid (m_j>0): {len(df_valid)}")
for sp in ["train", "val", "test"]:
    n = len(df_valid[df_valid["split"] == sp])
    print(f"  {sp}: {n} sentences")
```

---

### Cell 2: Level 0 — Annotation Quality (Spot-Check)

Display a random sample of sentences with their K_j, m_j labels alongside
the original generated text. The researcher visually inspects whether the
FActScore annotations make sense.

This is a MANUAL check — the notebook shows the data, the human decides.

```python
# Sample 20 test sentences, stratified: 10 with U_j < 0.5, 10 with U_j >= 0.5
df_test = df_valid[df_valid["split"] == "test"].copy()
low_u = df_test[df_test["U_j"] < 0.5].sample(n=min(10, len(df_test[df_test["U_j"] < 0.5])),
                                               random_state=42)
high_u = df_test[df_test["U_j"] >= 0.5].sample(n=min(10, len(df_test[df_test["U_j"] >= 0.5])),
                                                 random_state=42)
spot_check = pd.concat([low_u, high_u]).sample(frac=1, random_state=0)

print("=" * 80)
print("MANUAL ANNOTATION SPOT-CHECK")
print("Review the sentences below. For each, ask: does K_j/m_j feel right?")
print("=" * 80)

for i, (_, row) in enumerate(spot_check.iterrows()):
    print(f"\n--- Sample {i+1} / {len(spot_check)} [{row['entity']}] ---")
    print(f"  Text: {row['text'][:200]}{'...' if len(row['text']) > 200 else ''}")
    print(f"  m_j={row['m_j']}  K_j={row['K_j']}  U_j={row['U_j']:.2f}  "
          f"label={'FACTUAL' if row['A_j'] else 'HALLUCINATED'}")

print("\n" + "=" * 80)
print("After reviewing, set your subjective accuracy estimate below.")
print("=" * 80)

# The researcher fills this in after reviewing:
SPOT_CHECK_ACCURACY = 0.80  # ← EDIT THIS after manual review (0.0 to 1.0)

diagnose(0, "Annotation (spot-check)", 
         "OK" if SPOT_CHECK_ACCURACY >= 0.75 else
         ("SUSPECT" if SPOT_CHECK_ACCURACY >= 0.60 else "ROOT_CAUSE"),
         f"Subjective accuracy = {SPOT_CHECK_ACCURACY:.0%}",
         remediation="" if SPOT_CHECK_ACCURACY >= 0.75 else
         ("Debug FActScore pipeline: check Wikipedia retrieval, "
          "atom decomposition prompt, and supported/not-supported threshold."
          if SPOT_CHECK_ACCURACY < 0.60 else
          "Annotation is marginal — consider human annotation on test set."))
```

### Cell 3: Level 0b — Annotation Consistency (Inter-Atom Agreement)

For sentences with m_j > 1, check whether atom-level labels within a sentence
are internally consistent. If atoms from the same sentence have wildly
mixed labels AND the sentence text is clearly all-factual or all-hallucinated,
the annotation pipeline may be unreliable.

```python
# Distribution of K_j/m_j — if heavily bimodal (all 0 or all 1), annotation
# might be too coarse or too aggressive
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# (a) U_j histogram
axes[0].hist(df_test["U_j"].dropna(), bins=20, edgecolor="k", alpha=0.7)
axes[0].set_title("U_j distribution (test set)")
axes[0].set_xlabel("U_j = K_j / m_j")
# Mark fully-factual and fully-hallucinated fractions
frac_0 = (df_test["U_j"] == 0).mean()
frac_1 = (df_test["U_j"] == 1).mean()
frac_mid = 1 - frac_0 - frac_1
axes[0].axvline(x=0.5, color="red", linestyle="--", alpha=0.3)
axes[0].text(0.02, 0.95, f"U=0: {frac_0:.1%}\nU=1: {frac_1:.1%}\nmixed: {frac_mid:.1%}",
             transform=axes[0].transAxes, va="top", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

# (b) m_j vs U_j scatter — are high-m_j sentences noisier?
axes[1].scatter(df_test["m_j"], df_test["U_j"], alpha=0.3, s=10)
axes[1].set_xlabel("m_j (atom count)"); axes[1].set_ylabel("U_j")
axes[1].set_title("m_j vs U_j — higher m_j should have more spread")

plt.tight_layout(); plt.show()

# Diagnosis: if nearly everything is 0 or 1, the continuous model adds little
boundary_frac = frac_0 + frac_1
diagnose(0, "Annotation granularity",
         "OK" if boundary_frac < 0.70 else
         ("SUSPECT" if boundary_frac < 0.85 else "ROOT_CAUSE"),
         f"{boundary_frac:.0%} of test sentences at U_j=0 or U_j=1",
         remediation="" if boundary_frac < 0.70 else
         "Most sentences have binary U_j — the Binomial model's advantage "
         "over Bernoulli may be marginal. Consider m_j=1 (Bernoulli) as "
         "the primary evaluation target.")
```

---

### Cell 4: Level 1 — Backbone Signal (Per-Layer AUROC)

Test whether the Llama-3-8B-Instruct hidden states carry ANY factuality
information. This is independent of our model — uses plain sklearn.

If this fails, the backbone is not expressive enough for this task, or
the annotation is too noisy for any probe to learn from.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MAX_ENTITIES_DIAG = 30  # use more than go_nogo for robustness

# Collect hidden states and labels from test entities
test_files = sorted([gf for gf in gen_dir.glob("*.pt") if gf.stem in test_entities])
train_files = sorted([gf for gf in gen_dir.glob("*.pt") if gf.stem in train_entities])

def collect_layer_data(entity_files, max_n=None):
    """Load mean hidden states per sentence + strict labels."""
    layers_list, labels_list = [], []
    for gf in entity_files[:max_n or len(entity_files)]:
        pf = proc_dir / f"{gf.stem}.json"
        if not pf.exists():
            continue
        gen = torch.load(gf, map_location="cpu", weights_only=False)
        with open(pf) as f:
            recs = json.load(f)
        hs = gen["hidden_states"].float()
        for rec in recs:
            if rec["m_j"] == 0:
                continue
            s, e = rec["token_range"]
            if e <= s or e > hs.shape[0]:
                continue
            layers_list.append(hs[s:e].mean(dim=0))  # (num_layers, D)
            labels_list.append(1 if rec["K_j"] == rec["m_j"] else 0)
    if not layers_list:
        return None, None
    return torch.stack(layers_list), np.array(labels_list)

train_layers, train_labels = collect_layer_data(train_files, MAX_ENTITIES_DIAG)
test_layers, test_labels = collect_layer_data(test_files, MAX_ENTITIES_DIAG)
num_layers = train_layers.shape[1]

print(f"Train: {len(train_labels)} sentences, Test: {len(test_labels)} sentences")
print(f"Layers: {num_layers}")

# Per-layer AUROC — train on train, evaluate on test (proper split)
layer_aurocs_train = []  # in-sample (for comparison)
layer_aurocs_test = []   # out-of-sample (what matters)

for l in range(num_layers):
    X_tr = train_layers[:, l, :].numpy()
    X_te = test_layers[:, l, :].numpy()
    if np.unique(train_labels).size < 2 or np.unique(test_labels).size < 2:
        layer_aurocs_train.append(0.5)
        layer_aurocs_test.append(0.5)
        continue
    try:
        clf = LogisticRegression(max_iter=500, C=0.1, solver="liblinear",
                                 penalty="l1", random_state=42)
        clf.fit(X_tr, train_labels)
        layer_aurocs_train.append(roc_auc_score(train_labels, clf.predict_proba(X_tr)[:, 1]))
        layer_aurocs_test.append(roc_auc_score(test_labels, clf.predict_proba(X_te)[:, 1]))
    except Exception as exc:
        print(f"  Layer {l}: {exc}")
        layer_aurocs_train.append(0.5)
        layer_aurocs_test.append(0.5)

layer_aurocs_test = np.array(layer_aurocs_test)
layer_aurocs_train = np.array(layer_aurocs_train)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(num_layers), layer_aurocs_test, "-o", markersize=5, label="Test AUROC", color="tab:blue")
ax.plot(range(num_layers), layer_aurocs_train, "--s", markersize=3, label="Train AUROC", color="tab:blue", alpha=0.4)
best_layer = np.argmax(layer_aurocs_test)
best_auroc = layer_aurocs_test[best_layer]
ax.axvline(x=best_layer, color="red", linestyle="--", alpha=0.5,
           label=f"Best: layer {best_layer} (AUROC={best_auroc:.4f})")
ax.axvline(x=14, color="gray", linestyle=":", alpha=0.5, label="Han et al. layer 14")
ax.axhline(y=0.55, color="orange", linestyle="--", alpha=0.3, label="Hard floor (0.55)")
ax.axhline(y=0.7357, color="green", linestyle=":", alpha=0.3, label="Han et al. reported (0.7357)")
ax.set_xlabel("Layer"); ax.set_ylabel("AUROC")
ax.set_title("Per-Layer Probing AUROC (sklearn LogReg, L1, C=0.1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# Also check train-test gap (overfitting signal)
train_test_gap = layer_aurocs_train[best_layer] - layer_aurocs_test[best_layer]
print(f"\nBest layer: {best_layer}, Test AUROC: {best_auroc:.4f}")
print(f"Train AUROC at best layer: {layer_aurocs_train[best_layer]:.4f}")
print(f"Train-Test gap: {train_test_gap:.4f}")

diagnose(1, "Backbone signal (peak)",
         "OK" if best_auroc >= 0.65 else
         ("SUSPECT" if best_auroc >= 0.55 else "ROOT_CAUSE"),
         f"Peak test AUROC = {best_auroc:.4f} at layer {best_layer}",
         remediation="" if best_auroc >= 0.65 else
         ("Hidden state carries weak signal. Could be backbone limitation "
          "(try re-encoding variant) or annotation noise (check Level 0). "
          "If Level 0 is OK, consider using a larger model or the "
          "re-encoding approach from Han et al."
          if best_auroc >= 0.55 else
          "No usable signal in hidden states. Either annotation is fundamentally "
          "broken or this backbone cannot encode factuality for this data."))

diagnose(1, "Backbone overfitting",
         "OK" if train_test_gap < 0.10 else "SUSPECT",
         f"Train-Test AUROC gap = {train_test_gap:.4f}",
         remediation="" if train_test_gap < 0.10 else
         "Large gap suggests overfitting — data may be too small or "
         "entity-level distribution shift exists.")
```

---

### Cell 5: Level 2 — Data Sufficiency (Learning Curve)

Train a simple sklearn LogReg at the best layer with increasing fractions
of training data. If performance is still rising at 100%, more data would help.

```python
from sklearn.model_selection import StratifiedShuffleSplit

best_l = best_layer  # from Cell 4
X_tr_best = train_layers[:, best_l, :].numpy()
X_te_best = test_layers[:, best_l, :].numpy()

fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
lc_aurocs = []
lc_ns = []

for frac in fractions:
    n_use = max(10, int(len(X_tr_best) * frac))
    # Stratified subsample
    if frac < 1.0 and n_use < len(X_tr_best):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_use, random_state=42)
        idx, _ = next(sss.split(X_tr_best, train_labels))
    else:
        idx = np.arange(len(X_tr_best))
    X_sub, y_sub = X_tr_best[idx], train_labels[idx]
    if np.unique(y_sub).size < 2:
        lc_aurocs.append(0.5)
        lc_ns.append(n_use)
        continue
    clf = LogisticRegression(max_iter=500, C=0.1, solver="liblinear",
                             penalty="l1", random_state=42)
    clf.fit(X_sub, y_sub)
    auc = roc_auc_score(test_labels, clf.predict_proba(X_te_best)[:, 1])
    lc_aurocs.append(auc)
    lc_ns.append(n_use)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lc_ns, lc_aurocs, "-o", markersize=8, linewidth=2)
ax.set_xlabel("Training samples"); ax.set_ylabel("Test AUROC")
ax.set_title(f"Learning Curve (sklearn LogReg, layer {best_l})")
ax.grid(alpha=0.3)
for n, a in zip(lc_ns, lc_aurocs):
    ax.annotate(f"{a:.3f}", (n, a), textcoords="offset points",
                xytext=(0, 10), fontsize=9, ha="center")
plt.tight_layout(); plt.show()

# Is the curve still rising?
if len(lc_aurocs) >= 3:
    slope_last = lc_aurocs[-1] - lc_aurocs[-2]
    still_rising = slope_last > 0.005
else:
    still_rising = False

diagnose(2, "Data sufficiency",
         "SUSPECT" if still_rising else "OK",
         f"Last segment slope = {slope_last:.4f}" if len(lc_aurocs) >= 3 else "Not enough points",
         remediation="Learning curve still rising — more entities would likely help. "
         "Consider expanding to full 150 entities or adding LongFact data."
         if still_rising else "")
```

### Cell 6: Level 2b — Distribution Shift (Train vs Test)

Compare the U_j distribution between train and test. A large shift
can explain poor generalisation even when backbone signal is strong.

```python
df_tr = df_valid[df_valid["split"] == "train"]
df_te = df_valid[df_valid["split"] == "test"]

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# (a) U_j distributions overlaid
axes[0].hist(df_tr["U_j"].dropna(), bins=20, alpha=0.5, label="Train", density=True, edgecolor="k")
axes[0].hist(df_te["U_j"].dropna(), bins=20, alpha=0.5, label="Test", density=True, edgecolor="k")
axes[0].set_title("U_j distribution: Train vs Test"); axes[0].legend()

# (b) m_j distributions
axes[1].hist(df_tr["m_j"], bins=range(0, max(df_tr["m_j"].max(), df_te["m_j"].max())+2),
             alpha=0.5, label="Train", density=True, edgecolor="k")
axes[1].hist(df_te["m_j"], bins=range(0, max(df_tr["m_j"].max(), df_te["m_j"].max())+2),
             alpha=0.5, label="Test", density=True, edgecolor="k")
axes[1].set_title("m_j distribution: Train vs Test"); axes[1].legend()

# (c) Strict label fraction per entity
entity_stats = df_valid.groupby(["entity", "split"]).agg(
    strict_frac=("A_j", "mean"), n_sentences=("A_j", "count")
).reset_index()
for sp, color in [("train", "tab:blue"), ("test", "tab:red")]:
    sub = entity_stats[entity_stats["split"] == sp]
    axes[2].scatter(sub["n_sentences"], sub["strict_frac"], alpha=0.5,
                    color=color, label=sp, s=30)
axes[2].set_xlabel("Sentences per entity"); axes[2].set_ylabel("Strict factual fraction")
axes[2].set_title("Per-Entity Factuality Rate"); axes[2].legend()

plt.tight_layout(); plt.show()

# KS test for distribution shift
from scipy.stats import ks_2samp
ks_stat, ks_p = ks_2samp(df_tr["U_j"].dropna(), df_te["U_j"].dropna())
print(f"KS test (U_j): statistic={ks_stat:.4f}, p-value={ks_p:.4f}")

diagnose(2, "Distribution shift (U_j)",
         "OK" if ks_p > 0.05 else "SUSPECT",
         f"KS statistic = {ks_stat:.4f}, p = {ks_p:.4f}",
         remediation="" if ks_p > 0.05 else
         "Significant train-test U_j shift detected. "
         "This is expected for Setup 1 (cross-domain) but problematic for Setup 2. "
         "Consider stratified splitting or data augmentation.")
```

---

### Cell 7: Level 3 — Model Ablation Chain

The critical diagnostic. Test four models on the SAME data with increasing
complexity. Performance should monotonically improve (or at least not degrade).

```
(a) sklearn LogReg, layer 14, mean HS         → Han et al. reproduction
(b) sklearn LogReg, our features (Wα + ent + top1)  → feature extractor value
(c) Our Fisher-scoring MAP, Σ zeroed out       → Fisher scoring correctness
(d) Our full Bayesian (Laplace posterior)       → Bayesian value-add
```

Any drop between adjacent steps localises the problem.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# ---- (a) sklearn LogReg on layer-14 mean hidden state ----
# Already computed in Cell 4; grab the best-layer result
auroc_a = layer_aurocs_test[best_layer]

# Also compute ECE for the sklearn model
clf_a = LogisticRegression(max_iter=500, C=0.1, solver="liblinear",
                           penalty="l1", random_state=42)
clf_a.fit(train_layers[:, best_layer, :].numpy(), train_labels)
probs_a = clf_a.predict_proba(test_layers[:, best_layer, :].numpy())[:, 1]
from src.evaluation.metrics import compute_calibration_metrics
cal_a = compute_calibration_metrics(test_labels, probs_a)
ece_a = cal_a["ECE"]

print(f"(a) sklearn LogReg (layer {best_layer}):  AUROC={auroc_a:.4f}  ECE={ece_a:.4f}")

# ---- (b) sklearn LogReg on our extracted features ----
# Load trained model to get feature extractor parameters
model_path = results_dir / "trained_model.pt"
ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

from src.features.extractor import SentenceUQParams, extract_token_features
from src.features.cached_scalars import load_cached_scalars

params = ckpt.get("params") or ckpt.get("model_state")
# Reconstruct feature extractor — adapt to however the checkpoint stores it
# This section may need adjustment based on actual checkpoint format

# Build features using the trained W and α
# For each sentence: z_tokens → mean → feature vector
our_features_train = []
our_features_test = []
our_labels_train_b = []
our_labels_test_b = []

# Helper: extract features for a set of entities
def extract_our_features(entity_files, entity_set, ckpt):
    """Extract mean sentence features using the trained W and α."""
    from src.features.extractor import SentenceUQParams, extract_sentence_token_features
    
    # Reconstruct params from checkpoint
    feat_params = SentenceUQParams(
        hidden_dim=ckpt["config"]["hidden_dim"],
        num_layers=ckpt["config"]["num_layers"],
        projection_dim=ckpt["config"].get("projection_dim", 64),
    )
    feat_params.load_state_dict(ckpt["params_state_dict"])
    feat_params.eval()
    
    features, labels = [], []
    for gf in entity_files:
        if gf.stem not in entity_set:
            continue
        pf = proc_dir / f"{gf.stem}.json"
        if not pf.exists():
            continue
        gen = torch.load(gf, map_location="cpu", weights_only=False)
        with open(pf) as f:
            recs = json.load(f)
        hs = gen["hidden_states"].float()
        
        # Load cached entropy and top1
        cache_dir = Path(cfg.get("cache", {}).get("dir", "data/cache"))
        ent_path = cache_dir / f"{gf.stem}_entropy.pt"
        top1_path = cache_dir / f"{gf.stem}_top1.pt"
        entropy = torch.load(ent_path, map_location="cpu") if ent_path.exists() else torch.zeros(hs.shape[0])
        top1 = torch.load(top1_path, map_location="cpu") if top1_path.exists() else torch.zeros(hs.shape[0])
        
        with torch.no_grad():
            for rec in recs:
                if rec["m_j"] == 0:
                    continue
                s, e = rec["token_range"]
                if e <= s or e > hs.shape[0]:
                    continue
                z_tok = extract_sentence_token_features(
                    hs, entropy, top1, (s, e), feat_params
                )  # (L_j, k)
                features.append(z_tok.mean(dim=0).numpy())
                labels.append(1 if rec["K_j"] == rec["m_j"] else 0)
    
    return np.array(features), np.array(labels)

all_pt_files = sorted(gen_dir.glob("*.pt"))

try:
    X_tr_b, y_tr_b = extract_our_features(all_pt_files, train_entities, ckpt)
    X_te_b, y_te_b = extract_our_features(all_pt_files, test_entities, ckpt)
    
    clf_b = LogisticRegression(max_iter=500, C=0.1, solver="liblinear",
                               penalty="l1", random_state=42)
    clf_b.fit(X_tr_b, y_tr_b)
    probs_b = clf_b.predict_proba(X_te_b)[:, 1]
    auroc_b = roc_auc_score(y_te_b, probs_b)
    cal_b = compute_calibration_metrics(y_te_b, probs_b)
    ece_b = cal_b["ECE"]
    print(f"(b) sklearn LogReg (our features):     AUROC={auroc_b:.4f}  ECE={ece_b:.4f}")
except Exception as exc:
    print(f"(b) FAILED to extract features: {exc}")
    print("    Falling back — check checkpoint format and feature extractor compatibility.")
    auroc_b, ece_b = np.nan, np.nan

# ---- (c) and (d) from Phase 6-2 CSV ----
ratio_df = pd.read_csv(results_dir / "final_metrics_ratio.csv")
strict_df = pd.read_csv(results_dir / "final_metrics_strict.csv") if (results_dir / "final_metrics_strict.csv").exists() else None

def get_m(df, substr, col):
    mask = df["method"].str.lower().str.contains(substr.lower())
    return df.loc[mask, col].values[0] if mask.any() else np.nan

# (c) Point estimate
auroc_c = get_m(strict_df, "point", "AUROC") if strict_df is not None else np.nan
ece_c = get_m(ratio_df, "point", "ECE")
print(f"(c) Our model (point, Σ=0):             AUROC={auroc_c:.4f}  ECE={ece_c:.4f}")

# (d) Full Bayesian
auroc_d = get_m(strict_df, "bayesian", "AUROC") if strict_df is not None else np.nan
ece_d = get_m(ratio_df, "bayesian", "ECE")
print(f"(d) Our model (full Bayesian):           AUROC={auroc_d:.4f}  ECE={ece_d:.4f}")

# ---- Visualise the chain ----
chain_labels = [
    f"(a) LogReg L{best_layer}",
    "(b) LogReg Wα-feat",
    "(c) Ours (Point)",
    "(d) Ours (Bayesian)",
]
chain_auroc = [auroc_a, auroc_b, auroc_c, auroc_d]
chain_ece = [ece_a, ece_b, ece_c, ece_d]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors_chain = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]

# AUROC chain
axes[0].bar(chain_labels, chain_auroc, color=colors_chain, edgecolor="k", alpha=0.8)
axes[0].set_ylabel("AUROC"); axes[0].set_title("Ablation Chain — AUROC (higher is better)")
axes[0].tick_params(axis="x", rotation=15)
for i, v in enumerate(chain_auroc):
    if np.isfinite(v):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)

# ECE chain
axes[1].bar(chain_labels, chain_ece, color=colors_chain, edgecolor="k", alpha=0.8)
axes[1].set_ylabel("ECE"); axes[1].set_title("Ablation Chain — ECE (lower is better)")
axes[1].tick_params(axis="x", rotation=15)
for i, v in enumerate(chain_ece):
    if np.isfinite(v):
        axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)

plt.tight_layout(); plt.show()

# ---- Diagnose each transition ----
# a → b: feature extractor
if np.isfinite(auroc_b):
    delta_ab = auroc_b - auroc_a
    diagnose(3, "Feature extractor (a→b)",
             "OK" if delta_ab >= -0.02 else "SUSPECT",
             f"AUROC change: {delta_ab:+.4f}",
             remediation="" if delta_ab >= -0.02 else
             "Our Wα features hurt AUROC vs raw layer-14. "
             "Check W initialisation, α learning rate, or projection_dim.")

# b → c: Fisher scoring
if np.isfinite(auroc_c) and np.isfinite(auroc_b):
    delta_bc = auroc_c - auroc_b
    diagnose(3, "Fisher scoring (b→c)",
             "OK" if delta_bc >= -0.03 else
             ("SUSPECT" if delta_bc >= -0.05 else "ROOT_CAUSE"),
             f"AUROC change: {delta_bc:+.4f}",
             remediation="" if delta_bc >= -0.03 else
             "Fisher scoring MAP is worse than sklearn LogReg on same features. "
             "Check: convergence (grad norm), prior_sigma (too loose?), "
             "lambda_init, number of inner iterations.")

# c → d: Bayesian (the value proposition)
if np.isfinite(ece_d) and np.isfinite(ece_c):
    ece_improvement = ece_c - ece_d  # positive = Bayesian better
    diagnose(3, "Bayesian value-add (c→d) ECE",
             "OK" if ece_improvement > 0 else
             ("SUSPECT" if ece_improvement > -0.01 else "ROOT_CAUSE"),
             f"ECE change: {ece_improvement:+.4f} (positive = Bayesian better)",
             remediation="" if ece_improvement > 0 else
             "Bayesian posterior is NOT improving calibration over point estimate. "
             "Check: PD status (Laplace validity), posterior covariance scale, "
             "probit shrinkage. The Laplace approximation may be too crude.")
```

---

### Cell 8: Level 4 — Label Noise Sensitivity

Corrupt 20% of training labels and retrain a simple probe. If performance
barely changes, the original labels are too noisy to learn from.

**This cell re-trains a sklearn model — fast (seconds), not the full pipeline.**

```python
# Corrupt 20% of training labels
rng = np.random.RandomState(123)
n_corrupt = int(0.20 * len(train_labels))
corrupt_idx = rng.choice(len(train_labels), n_corrupt, replace=False)
train_labels_corrupt = train_labels.copy()
train_labels_corrupt[corrupt_idx] = 1 - train_labels_corrupt[corrupt_idx]

# Retrain best-layer LogReg with corrupted labels
clf_corrupt = LogisticRegression(max_iter=500, C=0.1, solver="liblinear",
                                 penalty="l1", random_state=42)
clf_corrupt.fit(train_layers[:, best_layer, :].numpy(), train_labels_corrupt)
probs_corrupt = clf_corrupt.predict_proba(test_layers[:, best_layer, :].numpy())[:, 1]
auroc_corrupt = roc_auc_score(test_labels, probs_corrupt)

# Also test with 40% corruption
n_corrupt_40 = int(0.40 * len(train_labels))
corrupt_idx_40 = rng.choice(len(train_labels), n_corrupt_40, replace=False)
train_labels_corrupt_40 = train_labels.copy()
train_labels_corrupt_40[corrupt_idx_40] = 1 - train_labels_corrupt_40[corrupt_idx_40]
clf_corrupt_40 = LogisticRegression(max_iter=500, C=0.1, solver="liblinear",
                                     penalty="l1", random_state=42)
clf_corrupt_40.fit(train_layers[:, best_layer, :].numpy(), train_labels_corrupt_40)
probs_corrupt_40 = clf_corrupt_40.predict_proba(test_layers[:, best_layer, :].numpy())[:, 1]
auroc_corrupt_40 = roc_auc_score(test_labels, probs_corrupt_40)

print(f"Clean labels AUROC:           {auroc_a:.4f}")
print(f"20% corrupted labels AUROC:   {auroc_corrupt:.4f}  (Δ = {auroc_corrupt - auroc_a:+.4f})")
print(f"40% corrupted labels AUROC:   {auroc_corrupt_40:.4f}  (Δ = {auroc_corrupt_40 - auroc_a:+.4f})")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(["Clean", "20% corrupt", "40% corrupt"],
       [auroc_a, auroc_corrupt, auroc_corrupt_40],
       color=["tab:green", "tab:orange", "tab:red"], edgecolor="k", alpha=0.8)
ax.set_ylabel("Test AUROC"); ax.set_title("Label Noise Sensitivity")
for i, v in enumerate([auroc_a, auroc_corrupt, auroc_corrupt_40]):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11)
plt.tight_layout(); plt.show()

delta_20 = auroc_a - auroc_corrupt
diagnose(4, "Label noise sensitivity",
         "OK" if delta_20 > 0.02 else
         ("SUSPECT" if delta_20 > 0.005 else "ROOT_CAUSE"),
         f"AUROC drop with 20% corruption = {delta_20:.4f}",
         remediation="" if delta_20 > 0.02 else
         "Model is insensitive to label corruption — this suggests the CLEAN "
         "labels are already at noise level. The annotation pipeline may need "
         "human verification or a more reliable LLM judge.")
```

### Cell 9: Level 4b — Oracle Ceiling (m_j = 1 subset)

Test on sentences with exactly 1 atomic fact (m_j = 1). These have no
aggregation ambiguity — U_j is either 0 or 1, same as A_j. If performance
is good here but bad overall, the issue is sentence-level aggregation.

```python
# Filter to m_j = 1 sentences
df_m1_test = df_test[df_test["m_j"] == 1].copy()
df_m1_train = df_valid[(df_valid["split"] == "train") & (df_valid["m_j"] == 1)].copy()

print(f"m_j=1 sentences — train: {len(df_m1_train)}, test: {len(df_m1_test)}")
print(f"(Total test: {len(df_test)}, m_j=1 fraction: {len(df_m1_test)/len(df_test):.1%})")

if len(df_m1_test) >= 20 and len(df_m1_train) >= 20:
    # Collect hidden states for m_j=1 subset
    def collect_m1_data(entity_files, entity_set, proc_dir):
        layers, labels = [], []
        for gf in entity_files:
            if gf.stem not in entity_set:
                continue
            pf = proc_dir / f"{gf.stem}.json"
            if not pf.exists():
                continue
            gen = torch.load(gf, map_location="cpu", weights_only=False)
            with open(pf) as f:
                recs = json.load(f)
            hs = gen["hidden_states"].float()
            for rec in recs:
                if rec["m_j"] != 1:
                    continue
                s, e = rec["token_range"]
                if e <= s or e > hs.shape[0]:
                    continue
                layers.append(hs[s:e].mean(dim=0))
                labels.append(1 if rec["K_j"] == rec["m_j"] else 0)
        if not layers:
            return None, None
        return torch.stack(layers), np.array(labels)
    
    m1_tr_layers, m1_tr_labels = collect_m1_data(all_pt_files, train_entities, proc_dir)
    m1_te_layers, m1_te_labels = collect_m1_data(all_pt_files, test_entities, proc_dir)
    
    if m1_tr_layers is not None and m1_te_layers is not None:
        clf_m1 = LogisticRegression(max_iter=500, C=0.1, solver="liblinear",
                                     penalty="l1", random_state=42)
        clf_m1.fit(m1_tr_layers[:, best_layer, :].numpy(), m1_tr_labels)
        probs_m1 = clf_m1.predict_proba(m1_te_layers[:, best_layer, :].numpy())[:, 1]
        auroc_m1 = roc_auc_score(m1_te_labels, probs_m1) if np.unique(m1_te_labels).size > 1 else 0.5
        
        print(f"\nm_j=1 subset AUROC: {auroc_m1:.4f}  (vs full test: {auroc_a:.4f})")
        delta_m1 = auroc_m1 - auroc_a
        
        diagnose(4, "Oracle ceiling (m_j=1)",
                 "SUSPECT" if delta_m1 > 0.05 else "OK",
                 f"m_j=1 AUROC = {auroc_m1:.4f}, full AUROC = {auroc_a:.4f}, Δ = {delta_m1:+.4f}",
                 remediation="" if delta_m1 <= 0.05 else
                 "Much better on m_j=1 subset — sentence-level aggregation "
                 "(K_j/m_j conversion) may be losing signal. Consider "
                 "claim-level evaluation or alternative aggregation strategies.")
    else:
        print("Not enough m_j=1 data to evaluate.")
else:
    print(f"Too few m_j=1 sentences for meaningful evaluation (need ≥20, have {len(df_m1_test)} test).")
    print("Skipping this check.")
```

---

### Cell 10: Diagnosis Summary & Decision Tree

Aggregate all diagnoses and present the root-cause diagnosis with remediation.

```python
print("=" * 80)
print("FAILURE DIAGNOSIS SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame(_diagnoses)

# Print by level
for level in sorted(summary_df["level"].unique()):
    level_rows = summary_df[summary_df["level"] == level]
    level_names = {0: "Annotation", 1: "Backbone", 2: "Data", 3: "Model", 4: "Label Noise"}
    print(f"\n  Level {level} — {level_names.get(level, '?')}:")
    for _, row in level_rows.iterrows():
        icon = {"OK": "✓", "SUSPECT": "⚠", "ROOT_CAUSE": "✗"}[row["status"]]
        print(f"    {icon} {row['component']}: {row['detail']}")

# Find root cause
root_causes = summary_df[summary_df["status"] == "ROOT_CAUSE"]
suspects = summary_df[summary_df["status"] == "SUSPECT"]

print("\n" + "=" * 80)
if len(root_causes) > 0:
    print("\033[91mROOT CAUSE IDENTIFIED:\033[0m")
    for _, rc in root_causes.iterrows():
        print(f"  Level {rc['level']} — {rc['component']}")
        print(f"  Detail: {rc['detail']}")
        if rc["remediation"]:
            print(f"  → {rc['remediation']}")
        print()
elif len(suspects) > 0:
    print("\033[93mNO CLEAR ROOT CAUSE — SUSPECTS:\033[0m")
    for _, s in suspects.iterrows():
        print(f"  Level {s['level']} — {s['component']}: {s['detail']}")
        if s["remediation"]:
            print(f"    → {s['remediation']}")
else:
    print("\033[92mALL CHECKS PASSED — no obvious failure point.\033[0m")
    print("If go_nogo still shows CONDITIONAL/NO-GO, consider:")
    print("  - Hyperparameter tuning (prior_sigma, learning rate, inner iterations)")
    print("  - Longer training (more epochs)")
    print("  - Different setup (1 vs 2 vs 3)")

print("=" * 80)

# Decision tree diagram
print("""
DIAGNOSTIC DECISION TREE:

    Level 0 FAIL → Fix annotation pipeline first
        ↓ OK
    Level 1 FAIL → Backbone cannot encode factuality
                   → Try re-encoding (Han et al. original) or larger model
        ↓ OK
    Level 2 FAIL → Insufficient data or distribution shift
                   → Add entities, check split strategy
        ↓ OK
    Level 3 (a→b) FAIL → Feature extractor (W, α) is harmful
                         → Check init, LR, projection_dim
    Level 3 (b→c) FAIL → Fisher scoring diverges or under-performs
                         → Check prior_sigma, lambda_init, convergence
    Level 3 (c→d) FAIL → Bayesian posterior hurts calibration
                         → Check PD, posterior scale, probit shrinkage
        ↓ OK
    Level 4 FAIL → Annotation noise is the bottleneck
                   → Human annotation or better LLM judge needed
""")
```

---

## File Structure After Creation

```
sentence_uq/
├── validation/
│   ├── go_nogo_validation.ipynb     # Quick verdict (run first)
│   └── failure_diagnosis.ipynb      # Root-cause analysis (run if not GO)
├── notebooks/                        # Phase 8 detailed analysis
│   ├── 01_data_overview.ipynb
│   ├── ...
```

## Tests

No separate test file. Verify by running:
```bash
cd sentence_uq
jupyter nbconvert --execute validation/failure_diagnosis.ipynb --to html
```
The output HTML should show a clear root-cause diagnosis at the bottom.
If all checks pass, the notebook says so and suggests hyperparameter tuning.
