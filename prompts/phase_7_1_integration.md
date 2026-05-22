# Phase 7-1 — Integration Scripts

Create end-to-end pipeline scripts.

**scripts/run_pilot.sh**:
```bash
#!/bin/bash
set -e

echo "=== Phase 1: Generate ==="
python scripts/01_generate_data.py --config configs/pilot.yaml

echo "=== Phase 1b: Cache entropy/top-1 ==="
python scripts/01b_cache_scalars.py --config configs/pilot.yaml

echo "=== Phase 2: Annotate factuality ==="
python scripts/02_annotate_factuality.py --config configs/pilot.yaml

echo "=== Phase 3: Train main model ==="
python scripts/03_train.py --config configs/pilot.yaml

echo "=== Phase 4: Train auxiliary model ==="
python scripts/04_train_aux.py --config configs/pilot.yaml

echo "=== Phase 5: Baselines ==="
python scripts/05_baselines.py --config configs/pilot.yaml

echo "=== Phase 6: Evaluate ==="
python scripts/04_evaluate.py --config configs/pilot.yaml

echo "=== Done. Results in results/ ==="
```

**Requirements**:
- Clear error messages on failure at each step
- Resume support (skip completed steps)
- Logs saved to results/logs/
- Summary output at the end

**scripts/run_full.sh**: same structure, uses default.yaml (500 entities)

**Update README.md** with:
```
# Quick start (50 entities)
bash scripts/run_pilot.sh

# Full experiment (500 entities)
bash scripts/run_full.sh
```

**Experiment checklist** (print at end of run_pilot.sh):
- [ ] 10 entity smoke test passed
- [ ] 50 entity pilot complete, all metrics computed
- [ ] Ratio-level: MAE and Pearson r reasonable (primary evaluation)
- [ ] Strict AUROC at least comparable to baselines
- [ ] Bayesian ECE < Point estimate ECE (core hypothesis)
- [ ] Our ECE < Factuality Probe (Han et al.) ECE (key comparison)
- [ ] Binomial NLL computed and reasonable (count-aware model advantage)
- [ ] Rejection curve: Ours comparable or better than Han et al.
- [ ] MC vs linear correlation > 0.9
- [ ] Learned alpha distribution visualized (compare with Han et al. layer 14)
- [ ] m_j distribution checked (no excessive m_j=0, no extreme dominance)
