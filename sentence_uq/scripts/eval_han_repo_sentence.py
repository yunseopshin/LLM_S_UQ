"""Repo-faithful Han baseline at the SENTENCE level.

Takes Han's *stock* CV-selected probe (trained by fact-probe-main/train.py — no
retraining here), re-encodes each test atom EXACTLY as their eval.py does
(last-token hidden state at the probe's layer group, fp16), then adds the one
step their eval.py omits: aggregate per-atom probabilities to a sentence score
(mean). Scores it with OUR metric functions on the SAME m_j>0 test pool the
pipeline uses, and injects the result into results/setup_2/baselines.json as
``factuality_probe_original_repo`` so 04_evaluate folds it into the tables.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJ = Path("/home/ys971217/LLM_S_UQ/sentence_uq")
sys.path.insert(0, str(PROJ))
os.chdir(PROJ)

from src.train.trainer import SentenceUQTrainer  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.features.extractor import SentenceUQParams  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_ratio_level_metrics,
    compute_calibration_metrics,
    compute_strict_factuality_metrics,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PROBE_PKL = Path(
    "/home/ys971217/LLM_S_UQ/fact-probe-main/fact-probe-main/_han_on_ours/"
    "probes/Llama3-8B_train_logistic_regression_C0.5_group5.pkl"
)
RESULTS = PROJ / "results/setup_2"
cfg = yaml.safe_load(open(PROJ / "configs/default.yaml"))

# --- 1. test pool in pipeline order, m_j > 0 (same as 04_evaluate) -----------
dummy = SentenceUQParams(hidden_dim=8, num_layers=2, projection_dim=4)
trainer = SentenceUQTrainer(model=BayesianSentenceUQ(feature_params=dummy), device="cpu")
split_file = (cfg.get("dataset") or {}).get("split_file") or "data/splits/setup_2.json"
gen_dirs = {"factscore_bio": "data/generations/factscore_bio", "longfact": "data/generations/longfact"}
cache_dirs = {"factscore_bio": "data/cache/factscore_bio", "longfact": "data/cache/longfact"}
proc_dirs = {"factscore_bio": "data/processed/factscore_bio", "longfact": "data/processed/longfact"}

data = trainer.prepare_data(
    split_file=split_file, generations_dirs=gen_dirs,
    cache_dirs=cache_dirs, processed_dirs=proc_dirs,
)
test = [r for r in (data.get("test") or []) if int(r.get("m_j", 0) or 0) > 0]

# --- 2. attach claim texts via (dataset, source_id, token_range) -------------
ann_map = {}
for ds, pdir in proc_dirs.items():
    anns = SentenceUQTrainer._load_annotations(ds, Path(pdir))
    for sid, rec in anns.items():
        for s in rec.get("sentences") or []:
            tr = s.get("token_range")
            if tr and len(tr) == 2:
                texts = [str(c.get("text", "")).strip()
                         for c in (s.get("claims") or [])
                         if str(c.get("text", "")).strip()]
                ann_map[(ds, str(sid), int(tr[0]), int(tr[1]))] = texts

for r in test:
    key = (r["dataset"], str(r["source_id"]), int(r["token_range"][0]), int(r["token_range"][1]))
    r["_claims"] = ann_map.get(key, [])

n_sent = len(test)
n_atoms = sum(len(r["_claims"]) for r in test)
n_missing = sum(1 for r in test if not r["_claims"])
print(f"test sentences (m>0): {n_sent} | atoms: {n_atoms} | sentences w/o claims: {n_missing}")

# --- 3. load Han's stock probe + model exactly as their eval.py does ---------
pb = pickle.load(open(PROBE_PKL, "rb"))
probe, lg = pb["probe"], pb["layer_group"]
layers = list(range(lg[0], lg[1] + 1))
print(f"stock probe: {PROBE_PKL.name} | layer_group {lg} -> layers {layers} | C={pb.get('C')}")

hf_name = cfg["model"]["name"]
tok = AutoTokenizer.from_pretrained(hf_name)
model = AutoModelForCausalLM.from_pretrained(hf_name, device_map="auto", torch_dtype=torch.float16)
model.eval()


def atom_prob(text: str) -> float:
    """Han eval.py extraction: last-token hidden at layer group -> probe prob."""
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states
    vec = np.concatenate([hs[L][0, -1].float().cpu().numpy() for L in layers])
    return float(probe.predict_proba(vec.reshape(1, -1))[0, 1])


# --- 4. per-atom prob -> sentence mean ---------------------------------------
t0 = time.perf_counter()
mu_hat = np.empty(n_sent, dtype=np.float64)
K = np.empty(n_sent, dtype=np.float64)
m = np.empty(n_sent, dtype=np.float64)
for i, r in enumerate(test):
    claims = r["_claims"]
    mu_hat[i] = float(np.mean([atom_prob(c) for c in claims])) if claims else 0.5
    K[i] = int(r.get("K_j", 0) or 0)
    m[i] = int(r.get("m_j", 0) or 0)
elapsed = time.perf_counter() - t0

U = K / np.maximum(m, 1.0)
A = (K == m).astype(np.float64)

# --- 5. metrics with OUR functions (same as 04_evaluate baseline rows) -------
ratio = compute_ratio_level_metrics(U, mu_hat, m_j=m)
calib = compute_calibration_metrics(U, mu_hat, n_bins=10)
strict = compute_strict_factuality_metrics(A, mu_hat, -mu_hat)

print("\n=== Repo-faithful Han (sentence-level, mean-aggregated) ===")
print(f"RATIO : MAE={ratio['MAE']:.4f}  RMSE={ratio['RMSE']:.4f}  "
      f"Pearson={ratio['Pearson_r']:.4f}  ratio_ECE={calib['ECE']:.4f}  "
      f"binom_NLL={ratio.get('binomial_NLL', float('nan')):.4f}")
print(f"STRICT: AUROC={strict['AUROC']:.4f}  AUPRC={strict['AUPRC']:.4f}  "
      f"Brier={strict['Brier']:.4f}  ECE={strict['ECE']:.4f}")
print(f"(encoded {n_atoms} atoms in {elapsed:.1f}s)")

# --- 6. inject into baselines.json as repo-faithful row ----------------------
bj = RESULTS / "baselines.json"
bl = json.load(open(bj))
bl["baselines"]["factuality_probe_original_repo"] = {
    "name": "factuality_probe_original_repo",
    "mu_hat": mu_hat.tolist(),
    "wall_clock_seconds": float(elapsed),
    "config": {
        "source": "fact-probe-main stock probe (no retraining)",
        "probe_file": PROBE_PKL.name,
        "layer_group": list(lg),
        "C": pb.get("C"),
        "aggregation": "mean",
    },
}
json.dump(bl, open(bj, "w"), indent=2)
print(f"\ninjected 'factuality_probe_original_repo' (n={n_sent}) into {bj}")
