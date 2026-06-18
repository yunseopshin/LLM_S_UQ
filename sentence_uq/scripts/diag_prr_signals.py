"""Is the in-domain PRR salvageable by changing the rejection signal?

The Bayesian PRR ranks rejections by epi_mu (epistemic), which COLLAPSES in-domain
(Phase 9.2/9.3) — epi_mu ~ 1.7e-3 — so PRR is poor (ratio 0.103, strict 0.025).
The strict-AUROC scoring fix (μ vs μ^m) does NOT touch this. Question: does the
information exist in a DIFFERENT uncertainty signal (total / aleatoric /
confidence)? Compute ratio- and strict-PRR for the SAME production model under
each rejection signal.
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
from src.features.extractor import extract_sentence_token_features, pool_token_features  # noqa: E402
from src.inference.predict import Predictor, load_trained_model  # noqa: E402
from src.evaluation.metrics import compute_prr  # noqa: E402

cfg = yaml.safe_load(open(PROJ / "configs/default.yaml"))
DEV = torch.device("cpu")
loaded = load_trained_model(PROJ / "results/setup_2/trained_model.pt", map_location="cpu")
fp = loaded["feature_params"].to(DEV).eval()
predictor = Predictor(
    theta_hat=loaded["theta_hat"], Sigma_hat=loaded["Sigma_hat"],
    feature_params=fp, likelihood=loaded.get("likelihood"),
)

trainer = SentenceUQTrainer(model=BayesianSentenceUQ(feature_params=fp), device="cpu")
data = trainer.prepare_data(
    split_file="data/splits/setup_2.json",
    generations_dirs={"factscore_bio": "data/generations/factscore_bio", "longfact": "data/generations/longfact"},
    cache_dirs={"factscore_bio": "data/cache/factscore_bio", "longfact": "data/cache/longfact"},
    processed_dirs={"factscore_bio": "data/processed/factscore_bio", "longfact": "data/processed/longfact"},
)
test = [r for r in data["test"] if int(r.get("m_j", 0) or 0) > 0]

mu, epi, alea, tot, K, m = [], [], [], [], [], []
with torch.no_grad():
    for r in test:
        z = extract_sentence_token_features(
            hidden_states=r["hidden_states"].to(DEV), entropy=r["entropy"].to(DEV),
            top1_prob=r["top1"].to(DEV),
            token_range=(int(r["token_range"][0]), int(r["token_range"][1])), params=fp,
        )
        z = pool_token_features(z, fp)
        mj = int(r["m_j"])
        out = predictor.predict_sentence(z, m_j=mj)
        mu.append(out["mu_hat"]); epi.append(out["epi_mu"])
        alea.append(out["aleatoric_U"] or 0.0); tot.append(out["total_U"] or 0.0)
        K.append(int(r["K_j"])); m.append(mj)

mu = np.array(mu); epi = np.array(epi); alea = np.array(alea); tot = np.array(tot)
K = np.array(K, float); m = np.array(m, float)
U = K / np.maximum(m, 1.0); A = (K == m).astype(float)

# Rejection signals (higher = reject first). For quality=U/A (higher=better),
# a good signal puts LOW-quality items at HIGH uncertainty.
signals = {
    "epi_mu (current Bayes)": epi,
    "total_U (alea+epi)": tot,
    "aleatoric_U": alea,
    "confidence |mu-0.5|^-1": -np.abs(mu - 0.5),
    "low-mu  (-mu)": -mu,
}
print(f"n_test={len(U)}  epi_mu mean={epi.mean():.2e}  aleatoric mean={alea.mean():.3f}  total mean={tot.mean():.3f}")
print(f"{'rejection signal':26s} {'ratio PRR_AUC':>14s} {'strict PRR_AUC':>15s}")
for name, sig in signals.items():
    pr_ratio = compute_prr(U, sig, num_thresholds=100)["prr_auc"]
    pr_strict = compute_prr(A, sig, num_thresholds=100)["prr_auc"]
    print(f"{name:26s} {pr_ratio:14.4f} {pr_strict:15.4f}")
print("REF baselines ratio PRR: token_entropy .283 logreg .114 adapted .126 Han .096-.102")
print("REF baselines strict PRR: .144 / .157 / .158 / .177-.181")
