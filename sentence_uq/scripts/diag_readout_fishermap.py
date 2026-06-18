"""Why did end-to-end readout=last regress? Isolate features vs optimisation.

The sklearn diagnostic (diag_readout_pool.py) got strict AUROC 0.797 with
last-token pooling on the PRODUCTION token_mean-trained features (W/α fixed) +
a standardised LR. End-to-end joint training of W/α under readout=last collapsed
to 0.667. This script tests the middle ground: keep the PRODUCTION W/α FIXED,
swap to last-token pooling, and refit ONLY θ through OUR Fisher-MAP + binomial
(the real inference path), no sklearn.

- ~0.79  → signal transfers through our pipeline; the failure is the joint W/α
           retraining → warm-start / standardisation is the fix.
- ~0.67  → even with good features, Fisher-MAP+binomial last-readout underperforms
           the standardised LR → deeper (scaling / prior) issue.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJ = Path("/home/ys971217/LLM_S_UQ/sentence_uq")
sys.path.insert(0, str(PROJ))
os.chdir(PROJ)

from src.train.trainer import SentenceUQTrainer  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.models.fisher_scoring import fisher_scoring_map_detached  # noqa: E402
from src.features.extractor import (  # noqa: E402
    extract_sentence_token_features,
    pool_token_features,
)
from src.inference.predict import load_trained_model  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_calibration_metrics,
    compute_strict_factuality_metrics,
    compute_bootstrapped_ci,
)


def _auroc(A, p):
    return compute_strict_factuality_metrics(A, p, 1 - np.asarray(p))["AUROC"]

cfg = yaml.safe_load(open(PROJ / "configs/default.yaml"))
DEV = torch.device("cpu")

# Production model: W/α trained under token_mean.
loaded = load_trained_model(PROJ / "results/setup_2/trained_model.pt", map_location="cpu")
fp = loaded["feature_params"].to(DEV).eval()

trainer = SentenceUQTrainer(model=BayesianSentenceUQ(feature_params=fp), device="cpu")
data = trainer.prepare_data(
    split_file="data/splits/setup_2.json",
    generations_dirs={"factscore_bio": "data/generations/factscore_bio", "longfact": "data/generations/longfact"},
    cache_dirs={"factscore_bio": "data/cache/factscore_bio", "longfact": "data/cache/longfact"},
    processed_dirs={"factscore_bio": "data/processed/factscore_bio", "longfact": "data/processed/longfact"},
)
train = [r for r in data["train"] if int(r.get("m_j", 0) or 0) > 0]
test = [r for r in data["test"] if int(r.get("m_j", 0) or 0) > 0]


def pooled_z(records, readout):
    fp.readout = readout  # monkeypatch the readout on the FIXED-W/α params
    zs, K, m = [], [], []
    with torch.no_grad():
        for r in records:
            z = extract_sentence_token_features(
                hidden_states=r["hidden_states"].to(DEV), entropy=r["entropy"].to(DEV),
                top1_prob=r["top1"].to(DEV),
                token_range=(int(r["token_range"][0]), int(r["token_range"][1])), params=fp,
            ).to(torch.float32)
            zs.append(pool_token_features(z, fp))
            K.append(int(r.get("K_j", 0) or 0)); m.append(int(r.get("m_j", 0) or 0))
    return zs, torch.tensor(K), torch.tensor(m)


def auroc_of(score, A):
    return compute_strict_factuality_metrics(A, score, 1 - np.asarray(score))["AUROC"]


def run(readout, standardize):
    ztr, Ktr, mtr = pooled_z(train, readout)
    zte, Kte, mte = pooled_z(test, readout)
    if standardize:
        # Fit per-feature mean/std on train pooled-z; apply to both (no leakage).
        Ztr = torch.cat(ztr, 0)
        mean, std = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True).clamp_min(1e-6)
        ztr = [(z - mean) / std for z in ztr]
        zte = [(z - mean) / std for z in zte]
    mu0 = fp.mu_0.detach()
    Sig0_inv = fp.get_Sigma_0_inv().detach()
    theta, _ = fisher_scoring_map_detached(
        all_z_tokens=ztr, all_K=Ktr, all_m=mtr, mu_0=mu0, Sigma_0_inv=Sig0_inv,
        num_iters=int(cfg.get("num_fisher_iters", 10)), eps=1e-6, lambda_init=1e-4,
    )
    mte_np = mte.numpy().astype(float)
    A = (Kte.numpy() == mte.numpy()).astype(float)
    mu = np.array([float(torch.sigmoid(z.to(torch.float32) @ theta).mean()) for z in zte])
    auroc_mpow = auroc_of(np.power(mu, mte_np), A)   # rank by p_strict = μ^m
    auroc_mu = auroc_of(mu, A)                        # rank by raw μ
    ece = compute_calibration_metrics(A, np.power(mu, mte_np), n_bins=10)["ECE"]
    return auroc_mpow, auroc_mu, ece


print(f"train={len(train)} test={len(test)}  (production W/α FIXED, Fisher-MAP refits θ only)")
print(f"{'readout':12s} {'std?':>5s} {'AUROC(μ^m)':>11s} {'AUROC(μ)':>10s} {'strict_ECE':>10s}")
for std in (False, True):
    for ro in ["token_mean", "mean", "last"]:
        a_mp, a_mu, e = run(ro, std)
        print(f"{ro:12s} {str(std):>5s} {a_mp:11.4f} {a_mu:10.4f} {e:10.4f}")
print("REF: production token_mean 0.780 | sklearn-diag last 0.797 | end2end-trained last 0.667")

# Headline: production token_mean model, fair strict AUROC (rank by μ, like baselines),
# with bootstrap 95% CI; ECE stays on μ^m (calibration).
fp.readout = "token_mean"
zte, Kte, mte = pooled_z(test, "token_mean")
ztr, Ktr, mtr = pooled_z(train, "token_mean")
theta, _ = fisher_scoring_map_detached(
    all_z_tokens=ztr, all_K=Ktr, all_m=mtr, mu_0=fp.mu_0.detach(),
    Sigma_0_inv=fp.get_Sigma_0_inv().detach(), num_iters=int(cfg.get("num_fisher_iters", 10)),
    eps=1e-6, lambda_init=1e-4,
)
A = (Kte.numpy() == mte.numpy()).astype(float)
mte_np = mte.numpy().astype(float)
mu = np.array([float(torch.sigmoid(z.to(torch.float32) @ theta).mean()) for z in zte])
ci_mu = compute_bootstrapped_ci(A, mu, _auroc, n_bootstrap=2000, seed=0)
ci_mpow = compute_bootstrapped_ci(A, np.power(mu, mte_np), _auroc, n_bootstrap=2000, seed=0)
ece_mu = compute_calibration_metrics(A, mu, n_bins=10)
ece_mpow = compute_calibration_metrics(A, np.power(mu, mte_np), n_bins=10)
print("\n=== HEADLINE: production token_mean model, strict AUROC scoring fix ===")
print(f"  rank by μ^m (current, self-handicapped): AUROC {_auroc(A, np.power(mu, mte_np)):.4f} "
      f"CI[{ci_mpow['lower']:.4f},{ci_mpow['upper']:.4f}]")
print(f"  rank by μ   (fair, == baselines)        : AUROC {_auroc(A, mu):.4f} "
      f"CI[{ci_mu['lower']:.4f},{ci_mu['upper']:.4f}]")
print("  baselines (rank by μ): logreg 0.803 | adapted 0.811 | Han repo 0.858")
print("\n=== CALIBRATION: should ECE/Brier use μ or μ^m? ===")
print(f"  ECE(A, μ^m)  [P(all-correct), correct object] : {ece_mpow['ECE']:.4f}  Brier {ece_mpow['Brier']:.4f}")
print(f"  ECE(A, μ)    [per-atom prob, WRONG object]     : {ece_mu['ECE']:.4f}  Brier {ece_mu['Brier']:.4f}")
print("  (μ is E[U_j]=P(one atom ok); A_j asks P(ALL m ok)=μ^m. Calibrating μ vs A = overconfident.)")
