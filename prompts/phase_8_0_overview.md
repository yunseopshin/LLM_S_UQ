# Phase 8-0 — Experiment Notebooks & Bash Launchers (Overview)

Phase 8 provides the tooling to **run experiments and analyse results** on top of the
Phase 1–7 codebase. Two kinds of artefacts:

1. **Bash scripts** (`experiments/`): editable config block + pipeline execution
2. **Jupyter notebooks** (`notebooks/`): result visualisation + model introspection

---

## Artefact List

### Notebooks (Phase 8-1 – 8-8)

| Phase | File | Purpose |
|-------|------|---------|
| 8-1 | `notebooks/01_data_overview.ipynb` | Data health check: m_j distribution, K_j/m_j, per-entity stats |
| 8-2 | `notebooks/02_hidden_state_inspection.ipynb` | Does the hidden state carry a factuality signal? (t-SNE, PCA, per-layer AUROC) |
| 8-3 | `notebooks/03_training_diagnostics.ipynb` | Convergence check: loss curve, Fisher diagnostics, PD checks |
| 8-4 | `notebooks/04_model_internals.ipynb` | Dissect the trained model: α weights, W matrix, prior parameters |
| 8-5 | `notebooks/05_prediction_analysis.ipynb` | μ̂ vs U scatter, uncertainty decomposition, case studies |
| 8-6 | `notebooks/06_calibration_comparison.ipynb` | Reliability diagrams, ECE comparison, PRR curves |
| 8-7 | `notebooks/07_ablation_results.ipynb` | Aggregate ablation-suite visualisation |
| 8-8 | `notebooks/08_paper_figures.ipynb` | Publication-quality figures for the paper |

### Bash scripts (Phase 8-9)

| File | Purpose |
|------|---------|
| `experiments/run_smoke.sh` | 10-entity quick sanity check |
| `experiments/run_ablation_suite.sh` | Core ablation loop |
| `experiments/run_cross_setup.sh` | Setup 1 / 2 / 3 comparison |
| `experiments/run_baselines_only.sh` | Re-run baselines only |
| `experiments/run_hyperparam_sweep.sh` | Grid search over prior_sigma, lr, etc. |

---

## Recommended Execution Order

```
1. bash experiments/run_smoke.sh          # 10-entity — verify full pipeline
2. notebooks/01 → 02                      # inspect data + hidden states
3. bash scripts/run_pilot.sh              # 50-entity pilot
4. notebooks/03 → 04                      # training diagnostics + model internals
5. notebooks/05 → 06                      # prediction analysis + calibration
6. bash experiments/run_ablation_suite.sh  # run all ablations
7. notebooks/07                           # visualise ablation results
8. bash scripts/run_full.sh               # 500-entity full experiment
9. bash experiments/run_cross_setup.sh     # 3-setup comparison
10. notebooks/08                           # generate paper figures
```

---

## Shared Conventions

Every notebook starts with:

```python
# === Configuration ===
import sys, os
PROJECT_ROOT = os.path.abspath("..")
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

RESULTS_DIR = "results/pilot"   # ← change this one line to analyse a different run
CONFIG_PATH = "configs/pilot.yaml"
```

Every bash script starts with:

```bash
# ============================================================
# ★ EDIT THIS BLOCK ★
# ============================================================
CONFIG="configs/pilot.yaml"
SETUP=2
N_ENTITIES=10
SELECTED_LAYERS=""  # empty = auto
PRIOR_SIGMA=1.0
LR=1e-3
# ============================================================
```
