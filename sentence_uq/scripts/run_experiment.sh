#!/bin/bash
# Full Bayesian sentence-level UQ pipeline for one experimental setup.
#
# Usage
# -----
#   bash scripts/run_experiment.sh 1   # Han et al. reproduction (cross-domain)
#   bash scripts/run_experiment.sh 2   # FActScore-Bio in-domain (default)
#   bash scripts/run_experiment.sh 3   # LongFact multi-domain
set -e

SETUP=${1:-2}  # default: setup 2 (FActScore-Bio in-domain)

echo "=== Running Setup ${SETUP} ==="

echo "Phase 0: Prepare dataset"
python scripts/00_prepare_dataset.py --setup "$SETUP"

echo "Phase 1: Generate"
python scripts/01_generate_data.py --setup "$SETUP"

echo "Phase 1b: Cache scalars"
python scripts/01b_cache_scalars.py --config "configs/setup_${SETUP}.yaml"

echo "Phase 2: Annotate"
python scripts/02_annotate_factuality.py --setup "$SETUP"

echo "Phase 3: Train"
python scripts/03_train.py --setup "$SETUP" --config "configs/setup_${SETUP}.yaml"

echo "Phase 4: Evaluate"
python scripts/04_evaluate.py --setup "$SETUP" --config "configs/setup_${SETUP}.yaml"

echo "=== Setup ${SETUP} Done ==="
