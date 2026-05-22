# Phase 8-9 — Bash Experiment Launchers

Create 5 bash scripts under `experiments/`.

**Purpose**: Easy-to-edit experiment launchers. Every script has a clearly marked
`★ EDIT THIS BLOCK ★` section at the top for configuration.

**Project path reference**:
```
sentence_uq/
├── scripts/
│   ├── 00_prepare_dataset.py     # --config CFG [--setup N]
│   ├── 01_generate_data.py       # --config CFG
│   ├── 01b_cache_scalars.py      # --config CFG
│   ├── 02_annotate_factuality.py # --config CFG
│   ├── 03_train.py               # --config CFG
│   ├── 04_evaluate.py            # --config CFG [--setup N] [--no-plots] [--compare-all]
│   ├── 04_train_aux.py           # --config CFG
│   ├── 05_baselines.py           # --config CFG
│   ├── run_pilot.sh              # --config --label --force
│   └── run_full.sh               # --config --label --force
├── configs/
│   ├── default.yaml
│   ├── pilot.yaml
│   └── setup_{1,2,3}.yaml
└── experiments/                   # ← created by Phase 8-9
    ├── run_smoke.sh
    ├── run_ablation_suite.sh
    ├── run_cross_setup.sh
    ├── run_baselines_only.sh
    └── run_hyperparam_sweep.sh
```

---

## 1. `experiments/run_smoke.sh`

Quick 10-entity end-to-end sanity check. Creates a temp config that overrides
`pilot.yaml` with a smaller entity count and shorter generation length.

```bash
#!/usr/bin/env bash
# experiments/run_smoke.sh — 10-entity smoke test
# Runs the full pipeline at reduced scale to catch errors early.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================
# ★ EDIT THIS BLOCK ★
# ============================================================
BASE_CONFIG="configs/pilot.yaml"
N_ENTITIES=10
MAX_NEW_TOKENS=128         # short generation for speed
LABEL="smoke"
# ============================================================

RESULTS_DIR="results/${LABEL}"
mkdir -p "${RESULTS_DIR}"

echo "=== Smoke Test (${N_ENTITIES} entities) ==="
echo "Config: ${BASE_CONFIG}, Results: ${RESULTS_DIR}"

# Generate temp config with overrides
TEMP_CONFIG="${RESULTS_DIR}/smoke_config.yaml"
python -c "
import yaml
with open('${BASE_CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['dataset']['pilot_size'] = ${N_ENTITIES}
cfg['generation']['max_new_tokens'] = ${MAX_NEW_TOKENS}
cfg['results_dir'] = '${RESULTS_DIR}'
with open('${TEMP_CONFIG}', 'w') as f:
    yaml.dump(cfg, f)
print('Temp config written:', '${TEMP_CONFIG}')
"

# Re-use run_pilot.sh stamp/log infrastructure
bash scripts/run_pilot.sh --config "${TEMP_CONFIG}" --label "${LABEL}" --force

echo ""
echo "=== Smoke Test Complete ==="
echo "Results: ${RESULTS_DIR}"
echo "Next: open notebooks/01_data_overview.ipynb with RESULTS_DIR='${RESULTS_DIR}'"
```

**Key points**:
- Copies `pilot.yaml` and overrides `pilot_size` via a temp config
- `--force` ensures a clean run (important for smoke tests)
- Re-uses `scripts/run_pilot.sh` stamp/log infrastructure

---

## 2. `experiments/run_ablation_suite.sh`

Loops over core ablation conditions. Each condition gets its own results directory.

```bash
#!/usr/bin/env bash
# experiments/run_ablation_suite.sh — core ablation loop
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================
# ★ EDIT THIS BLOCK ★
# ============================================================
BASE_CONFIG="configs/pilot.yaml"
ABLATION_ROOT="results/ablations"

# Comment out lines to skip specific ablations
ABLATIONS=(
    "bayesian_vs_point"
    "uniform_alpha"
    "prior_sigma_0.1"
    "prior_sigma_0.5"
    "prior_sigma_2.0"
    # "layer_subset_14_only"
    # "layer_subset_10_18"
)
# ============================================================

mkdir -p "${ABLATION_ROOT}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${ABLATION_ROOT}/ablation_summary_${TIMESTAMP}.txt"

echo "=== Ablation Suite (${#ABLATIONS[@]} conditions) ===" | tee "${SUMMARY}"
echo "Base config: ${BASE_CONFIG}" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

for abl in "${ABLATIONS[@]}"; do
    echo "--- Running: ${abl} ---" | tee -a "${SUMMARY}"
    ABL_DIR="${ABLATION_ROOT}/ablation_${abl}"
    ABL_CONFIG="${ABL_DIR}/config.yaml"
    mkdir -p "${ABL_DIR}"
    
    # Generate per-ablation config override
    python -c "
import yaml
with open('${BASE_CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['results_dir'] = '${ABL_DIR}'

abl = '${abl}'
if abl == 'bayesian_vs_point':
    cfg.setdefault('training', {})['disable_posterior'] = True   # Sigma=0
elif abl == 'uniform_alpha':
    cfg.setdefault('training', {})['freeze_alpha'] = True        # freeze α (uniform)
elif abl.startswith('prior_sigma_'):
    sigma = float(abl.split('_')[-1])
    cfg['prior_sigma_init'] = sigma
elif abl.startswith('layer_subset_'):
    layers_str = abl.replace('layer_subset_', '')
    if '_' in layers_str:
        lo, hi = layers_str.split('_')
        cfg.setdefault('model', {})['selected_layers'] = list(range(int(lo), int(hi)+1))
    else:
        cfg.setdefault('model', {})['selected_layers'] = [int(layers_str)]

with open('${ABL_CONFIG}', 'w') as f:
    yaml.dump(cfg, f)
"
    
    # Only re-run train + evaluate (generation/annotation shared with base)
    echo "  Training..."
    python scripts/03_train.py --config "${ABL_CONFIG}" 2>&1 | tail -5
    
    echo "  Evaluating..."
    python scripts/04_evaluate.py --config "${ABL_CONFIG}" 2>&1 | tail -5
    
    # Record key metric
    if [[ -f "${ABL_DIR}/final_metrics_ratio.csv" ]]; then
        echo "  Results:" | tee -a "${SUMMARY}"
        head -2 "${ABL_DIR}/final_metrics_ratio.csv" | tee -a "${SUMMARY}"
    fi
    echo "" | tee -a "${SUMMARY}"
done

echo "=== Ablation Suite Complete ===" | tee -a "${SUMMARY}"
echo "Summary: ${SUMMARY}"
echo "Visualise: notebooks/07_ablation_results.ipynb with ABLATION_DIR='${ABLATION_ROOT}'"
```

**Key points**:
- `ABLATIONS` array — comment/uncomment to toggle conditions
- Generation and annotation are shared — only train + evaluate are re-run
- Config overrides are generated via Python (preserves YAML structure)
- Summary file auto-records the key metric from each condition

---

## 3. `experiments/run_cross_setup.sh`

Run the full pipeline for Setup 1 (cross-domain), 2 (in-domain), 3 (multi-domain)
and produce a cross-setup comparison.

```bash
#!/usr/bin/env bash
# experiments/run_cross_setup.sh — Setup 1/2/3 comparison
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================
# ★ EDIT THIS BLOCK ★
# ============================================================
SETUPS=(1 2 3)
FORCE=0                    # set to 1 to pass --force
# ============================================================

for setup in "${SETUPS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Setup ${setup}"
    echo "=========================================="
    CONFIG="configs/setup_${setup}.yaml"
    if [[ ! -f "${CONFIG}" ]]; then
        echo "Config not found: ${CONFIG}, skipping."; continue
    fi
    FORCE_FLAG=""
    [[ "${FORCE}" -eq 1 ]] && FORCE_FLAG="--force"
    bash scripts/run_full.sh --config "${CONFIG}" --label "setup_${setup}" ${FORCE_FLAG}
done

echo ""
echo "=== Cross-Setup Comparison ==="
python scripts/04_evaluate.py --compare-all 2>&1 || echo "(--compare-all not yet implemented)"

echo ""
echo "Visualise: notebooks/07_ablation_results.ipynb Cell 8 (Cross-Setup Comparison)"
```

---

## 4. `experiments/run_baselines_only.sh`

Re-run baselines only (generation + annotation already complete). Expensive
baselines (Semantic Entropy, LUQ) are toggled via flags.

```bash
#!/usr/bin/env bash
# experiments/run_baselines_only.sh — re-run baselines selectively
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================
# ★ EDIT THIS BLOCK ★
# ============================================================
CONFIG="configs/pilot.yaml"

# 1=run, 0=skip
RUN_TOKEN_ENTROPY=1
RUN_LOGISTIC_REG=1
RUN_FACTUALITY_PROBE=1
RUN_SEMANTIC_ENTROPY=0    # ⚠️ expensive: 10× generation per prompt
RUN_LUQ=0                 # ⚠️ expensive: 10× generation + NLI
# ============================================================

echo "=== Baselines Only ==="
echo "Config: ${CONFIG}"
echo "Token Entropy: ${RUN_TOKEN_ENTROPY}, LogReg: ${RUN_LOGISTIC_REG}"
echo "Fact Probe: ${RUN_FACTUALITY_PROBE}"
echo "Semantic Entropy: ${RUN_SEMANTIC_ENTROPY}, LUQ: ${RUN_LUQ}"

BASELINE_FLAGS=""
[[ "${RUN_TOKEN_ENTROPY}" -eq 0 ]] && BASELINE_FLAGS="${BASELINE_FLAGS} --skip-token-entropy"
[[ "${RUN_LOGISTIC_REG}" -eq 0 ]] && BASELINE_FLAGS="${BASELINE_FLAGS} --skip-logistic-regression"
[[ "${RUN_FACTUALITY_PROBE}" -eq 0 ]] && BASELINE_FLAGS="${BASELINE_FLAGS} --skip-factuality-probe"
[[ "${RUN_SEMANTIC_ENTROPY}" -eq 0 ]] && BASELINE_FLAGS="${BASELINE_FLAGS} --skip-semantic-entropy"
[[ "${RUN_LUQ}" -eq 0 ]] && BASELINE_FLAGS="${BASELINE_FLAGS} --skip-luq"

python scripts/05_baselines.py --config "${CONFIG}" ${BASELINE_FLAGS}

echo ""
echo "Baselines complete. To evaluate:"
echo "  python scripts/04_evaluate.py --config ${CONFIG}"
```

**Note**: `scripts/05_baselines.py` may not yet support `--skip-*` flags. If so,
implement them as environment-variable overrides or add a `baselines.skip` config field.

---

## 5. `experiments/run_hyperparam_sweep.sh`

Grid search over `prior_sigma` × `learning_rate`. Results aggregated into a
summary CSV with automatic best-config output.

```bash
#!/usr/bin/env bash
# experiments/run_hyperparam_sweep.sh — grid search
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ============================================================
# ★ EDIT THIS BLOCK ★
# ============================================================
BASE_CONFIG="configs/pilot.yaml"
SWEEP_ROOT="results/sweep"

PRIOR_SIGMAS=(0.1 0.5 1.0 2.0 5.0)
LEARNING_RATES=(1e-4 5e-4 1e-3 5e-3)
# FISHER_ITERS=(5 10 15 20)          # uncomment to add a third axis
# ============================================================

mkdir -p "${SWEEP_ROOT}"
SUMMARY="${SWEEP_ROOT}/sweep_summary.csv"
echo "prior_sigma,lr,MAE,Pearson_r,ECE,binomial_NLL" > "${SUMMARY}"

TOTAL=$((${#PRIOR_SIGMAS[@]} * ${#LEARNING_RATES[@]}))
COUNT=0

for sigma in "${PRIOR_SIGMAS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        COUNT=$((COUNT + 1))
        RUN_NAME="sigma_${sigma}_lr_${lr}"
        RUN_DIR="${SWEEP_ROOT}/${RUN_NAME}"
        RUN_CONFIG="${RUN_DIR}/config.yaml"
        mkdir -p "${RUN_DIR}"
        
        echo "[${COUNT}/${TOTAL}] ${RUN_NAME}"
        
        python -c "
import yaml
with open('${BASE_CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['results_dir'] = '${RUN_DIR}'
cfg['prior_sigma_init'] = ${sigma}
cfg['lr'] = ${lr}
with open('${RUN_CONFIG}', 'w') as f:
    yaml.dump(cfg, f)
"
        
        python scripts/03_train.py --config "${RUN_CONFIG}" 2>&1 | tail -1
        python scripts/04_evaluate.py --config "${RUN_CONFIG}" --no-plots 2>&1 | tail -1
        
        # Extract metrics into summary CSV
        python -c "
import pandas as pd
try:
    df = pd.read_csv('${RUN_DIR}/final_metrics_ratio.csv')
    ours = df[df['method'].str.contains('Ours|Bayesian', case=False, na=False)]
    if len(ours) > 0:
        r = ours.iloc[0]
        print(f'${sigma},${lr},{r.get(\"MAE\",\"\")},{r.get(\"Pearson_r\",\"\")},{r.get(\"ECE\",\"\")},{r.get(\"binomial_NLL\",\"\")}')
    else:
        print(f'${sigma},${lr},,,,,')
except Exception:
    print(f'${sigma},${lr},,,,,')
" >> "${SUMMARY}"
    done
done

echo ""
echo "=== Sweep Complete (${TOTAL} runs) ==="
echo "Summary: ${SUMMARY}"

# Auto-print best config
python -c "
import pandas as pd
df = pd.read_csv('${SUMMARY}')
df = df.dropna(subset=['ECE'])
if len(df) > 0:
    best = df.loc[df['ECE'].idxmin()]
    print(f'Best config (lowest ECE): sigma={best[\"prior_sigma\"]}, lr={best[\"lr\"]}')
    print(f'  MAE={best[\"MAE\"]:.4f}, Pearson_r={best[\"Pearson_r\"]:.4f}, ECE={best[\"ECE\"]:.4f}')
"

echo ""
echo "Visualise: notebooks/07_ablation_results.ipynb Cell 6 (Prior Sigma Sweep)"
```

**Key points**:
- Double loop (sigma × lr); extendable to triple (+ fisher_iters)
- Each run's metrics are auto-appended to a summary CSV
- Best config printed automatically at the end
- `--no-plots` on evaluate for speed

---

## Shared Requirements

1. All scripts use `#!/usr/bin/env bash` + `set -euo pipefail`
2. Auto-detect `SCRIPT_DIR` / `PROJECT_ROOT` and `cd "${PROJECT_ROOT}"`
3. The `★ EDIT THIS BLOCK ★` section is delimited by `# ====` lines
4. Each script ends with a "Next step" hint (which notebook to open)
5. `experiments/` directory itself is tracked in git; generated result dirs
   (`results/ablations/`, `results/sweep/`) should be in `.gitignore`

## Important Notes

- `run_ablation_suite.sh` and `run_hyperparam_sweep.sh` **only re-run train + evaluate**.
  They assume generation and annotation data already exist from a prior
  `run_pilot.sh` or `run_full.sh` invocation.
- Config overrides are written via `yaml.dump`, which loses comments from the
  base config. This is intentional — override configs are disposable.
- `run_baselines_only.sh` requires `--skip-*` flag support in `scripts/05_baselines.py`.
  If not yet implemented, fall back to environment-variable overrides
  (e.g. `SKIP_SEMANTIC_ENTROPY=1`).
