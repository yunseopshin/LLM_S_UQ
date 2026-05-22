# Phase 8-5 — Notebook: Prediction Analysis

Create `notebooks/05_prediction_analysis.ipynb`.

**Purpose**: Deep analysis of the trained model's predictions. μ̂_j vs U_j scatter,
4-level uncertainty decomposition, residual analysis, best/worst case studies.

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
import seaborn as sns
from pathlib import Path

%matplotlib inline
plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
```

### Cell 1: Load Model + Test Data + Run Predictions

Load the trained model and run `Predictor.predict_sentence()` on the test split.
If Phase 6-2 CSVs already exist, load them directly.

```python
from src.inference.predict import load_trained_model, Predictor, BatchPredictor
from src.features.extractor import SentenceUQParams, extract_sentence_token_features
from src.train.trainer import SentenceUQTrainer

results_dir = Path(RESULTS_DIR)

# Option A: use pre-computed Phase 6-2 outputs
ratio_csv = results_dir / "final_metrics_ratio.csv"
strict_csv = results_dir / "final_metrics_strict.csv"
if ratio_csv.exists():
    ratio_df = pd.read_csv(ratio_csv)
    strict_df = pd.read_csv(strict_csv)
    print("Loaded pre-computed evaluation results")
    display(ratio_df)

# Option B: run predictions directly
trained = load_trained_model(results_dir / "trained_model.pt")
predictor = Predictor(
    theta_hat=trained["theta_hat"],
    Sigma_hat=trained["Sigma_hat"],
    feature_params=trained["feature_params"],
)

trainer = SentenceUQTrainer(feature_params=trained["feature_params"], cfg=cfg)
data_splits = trainer.prepare_data(
    split_file=cfg["dataset"].get("split_file") or f"data/splits/setup_{cfg['dataset']['setup']}.json",
    generations_dirs={"factscore_bio": cfg["generation"]["factscore_bio_dir"]},
    cache_dirs={"factscore_bio": cfg["cache"]["factscore_bio_dir"]},
    processed_dirs={"factscore_bio": cfg["processed"]["factscore_bio_dir"]},
)
test_data = data_splits["test"]
```

### Cell 2: Collect Predictions

```python
all_mu_hat, all_epi_mu, all_aleatoric_U, all_total_U = [], [], [], []
all_p_strict, all_U_true, all_A_true, all_K, all_m, all_texts = [], [], [], [], [], []

for rec in test_data:
    if rec["m_j"] == 0:
        continue
    out = predictor.predict_sentence(rec["z_tokens"], m_j=rec["m_j"])
    all_mu_hat.append(out["mu_hat"])
    all_epi_mu.append(out["epi_mu"])
    all_aleatoric_U.append(out["aleatoric_U"])
    all_total_U.append(out["total_U"])
    all_p_strict.append(out["p_strict_factual"])
    all_U_true.append(rec["K_j"] / rec["m_j"])
    all_A_true.append(1.0 if rec["K_j"] == rec["m_j"] else 0.0)
    all_K.append(rec["K_j"]); all_m.append(rec["m_j"])
    all_texts.append(rec.get("text", ""))

mu_hat = np.array(all_mu_hat); epi_mu = np.array(all_epi_mu)
aleatoric_U = np.array(all_aleatoric_U); total_U = np.array(all_total_U)
p_strict = np.array(all_p_strict); U_true = np.array(all_U_true)
A_true = np.array(all_A_true); m_arr = np.array(all_m)
print(f"Test sentences: {len(mu_hat)} (m_j>0)")
```

### Cell 3: μ̂_j vs U_j Scatter Plot (Key Figure)

Core ratio-level prediction quality. Closer to the diagonal = better.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) basic scatter
axes[0].scatter(U_true, mu_hat, alpha=0.4, s=15, c="steelblue")
axes[0].plot([0, 1], [0, 1], "r--", linewidth=2, label="Perfect prediction")
axes[0].set_xlabel("U_j = K_j / m_j (true)"); axes[0].set_ylabel("μ̂_j (predicted)")
r = np.corrcoef(U_true, mu_hat)[0, 1]
axes[0].set_title(f"Ratio-Level Prediction (Pearson r = {r:.3f})")
axes[0].legend(); axes[0].set_xlim(-0.05, 1.05); axes[0].set_ylim(-0.05, 1.05)

# (b) coloured by m_j
sc = axes[1].scatter(U_true, mu_hat, c=m_arr, cmap="viridis", alpha=0.5, s=20)
axes[1].plot([0, 1], [0, 1], "r--", linewidth=2)
axes[1].set_xlabel("U_j (true)"); axes[1].set_ylabel("μ̂_j (predicted)")
axes[1].set_title("Coloured by m_j"); plt.colorbar(sc, ax=axes[1], label="m_j")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/mu_hat_vs_U_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 4: Residual Analysis

```python
residual = mu_hat - U_true
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(residual, bins=30, edgecolor="black", alpha=0.7)
axes[0].axvline(x=0, color="red", linestyle="--")
axes[0].set_xlabel("Residual (μ̂ − U)"); axes[0].set_title(f"MAE = {np.abs(residual).mean():.3f}")

axes[1].scatter(m_arr, np.abs(residual), alpha=0.3, s=10)
axes[1].set_xlabel("m_j"); axes[1].set_ylabel("|Residual|")
axes[1].set_title("Absolute Error vs m_j")

# binned MAE by m_j range
m_bins = [(1,2), (3,5), (6,10), (11,100)]
bin_labels, bin_maes = [], []
for lo, hi in m_bins:
    mask = (m_arr >= lo) & (m_arr <= hi)
    if mask.sum() > 0:
        bin_labels.append(f"{lo}-{hi}"); bin_maes.append(np.abs(residual[mask]).mean())
axes[2].bar(bin_labels, bin_maes, color="mediumpurple")
axes[2].set_xlabel("m_j range"); axes[2].set_ylabel("MAE")
axes[2].set_title("MAE by m_j Group (more atoms → better?)")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/residual_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 5: 4-Level Uncertainty Decomposition

Distributions and correlations of `epi_mu`, `aleatoric_U`, `total_U`.
Verify: "Do sentences with high epistemic uncertainty actually have high prediction error?"

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0,0].hist(epi_mu, bins=30, edgecolor="black", alpha=0.7, color="coral")
axes[0,0].set_xlabel("Epistemic (Epi_μ)"); axes[0,0].set_title("Epistemic Distribution")

axes[0,1].hist(aleatoric_U, bins=30, edgecolor="black", alpha=0.7, color="skyblue")
axes[0,1].set_xlabel("Aleatoric (Aleatoric_U)"); axes[0,1].set_title("Aleatoric Distribution")

corr_epi = np.corrcoef(epi_mu, np.abs(residual))[0, 1]
axes[1,0].scatter(epi_mu, np.abs(residual), alpha=0.3, s=10, color="coral")
axes[1,0].set_xlabel("Epi_μ"); axes[1,0].set_ylabel("|Residual|")
axes[1,0].set_title(f"Epistemic vs Error (r={corr_epi:.3f})")

corr_total = np.corrcoef(total_U, np.abs(residual))[0, 1]
axes[1,1].scatter(total_U, np.abs(residual), alpha=0.3, s=10, color="mediumpurple")
axes[1,1].set_xlabel("Total_U"); axes[1,1].set_ylabel("|Residual|")
axes[1,1].set_title(f"Total_U vs Error (r={corr_total:.3f})")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/uncertainty_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Cell 6: Best & Worst Predictions (Case Study)

Print the 5 sentences with smallest |residual| and the 5 with largest.
For each: text, (K_j, m_j), U_j, μ̂_j, epi_mu.

```python
abs_res = np.abs(residual)
best_idx = np.argsort(abs_res)[:5]
worst_idx = np.argsort(abs_res)[-5:][::-1]

print("=== Best Predictions (lowest |residual|) ===")
for i in best_idx:
    print(f"  [{i}] U={U_true[i]:.3f} → μ̂={mu_hat[i]:.3f} "
          f"(K={all_K[i]}, m={all_m[i]}, epi={epi_mu[i]:.4f})")
    print(f"       {all_texts[i][:100]}...")

print("\n=== Worst Predictions (highest |residual|) ===")
for i in worst_idx:
    print(f"  [{i}] U={U_true[i]:.3f} → μ̂={mu_hat[i]:.3f} "
          f"(K={all_K[i]}, m={all_m[i]}, epi={epi_mu[i]:.4f})")
    print(f"       {all_texts[i][:100]}...")
```

### Cell 7: Token-Level Heatmap (1–2 Sentences)

Use `token_pi`, `token_attr`, `token_local_epi` from `predict_sentence()` to build
a Han et al. Figure 1-style token-level factuality heatmap.

---

## Dependencies

| Module | Functions / Classes Used |
|--------|-------------------------|
| `src.inference.predict` | `load_trained_model`, `Predictor`, `BatchPredictor` |
| `src.features.extractor` | `SentenceUQParams`, `extract_sentence_token_features` |
| `src.train.trainer` | `SentenceUQTrainer.prepare_data(...)` |

## Inputs

- `{RESULTS_DIR}/trained_model.pt`
- `{RESULTS_DIR}/final_metrics_ratio.csv` (optional, Phase 6-2)
- Config → data paths (generation, cache, processed dirs)
- `data/splits/setup_{N}.json`

## Outputs

- `{RESULTS_DIR}/mu_hat_vs_U_scatter.png`
- `{RESULTS_DIR}/residual_analysis.png`
- `{RESULTS_DIR}/uncertainty_decomposition.png`
