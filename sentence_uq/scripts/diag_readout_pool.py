"""Route-A viability diagnostic: project-then-pool vs pool-then-project.

Question
--------
Our model's discrimination (strict AUROC 0.780) trails simple probes on the
*same* generation-time hidden states (adapted last-token 0.811, logreg
token-mean 0.803). Is the bottleneck (a) our 66-d feature z_ℓ, or (b) the
``μ_j = mean_ℓ σ(θᵀ z_ℓ)`` *project-then-pool* readout that collapses each
token to a scalar before averaging?

Method
------
Hold the feature transform FIXED (the trained model's learned W, α → z_ℓ ∈ R^66)
and swap only the readout, fit on the setup_2 train split, evaluate on the
identical m_j>0 test pool (353 sentences) that 04_evaluate uses:

  A. ours (reproduce)      μ_j = mean_ℓ σ(θ̂ᵀ z_ℓ),  p_strict = μ_j^{m_j}
  B. pool-then-project mean ζ_j = mean_ℓ z_ℓ (66-d)        → L1-LR / Ridge
  C. pool-then-project last ζ_j = z_last     (66-d)        → L1-LR / Ridge   [↔ adapted]
  D. mean+std+last          ζ_j = [mean,std,last] (198-d)  → L1-LR / Ridge

Strict head: L1-LR on A_j = 1{K_j=m_j}; rank by P(A_j).  Ratio head: Ridge on
U_j = K_j/m_j.  Metrics use OUR functions on the SAME pool.

Interpretation
--------------
If C (last-token on our 66-d z) ≈ 0.81 → our features retain the signal; the
project-then-pool readout is the bottleneck → Route A is viable (relocate the
pooling, keep binomial for the strict head). If C ≪ 0.81 → the W compression to
66-d is the ceiling → richer features needed (see results/setup_2_projdim) or
pivot to Route B.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

PROJ = Path("/home/ys971217/LLM_S_UQ/sentence_uq")
sys.path.insert(0, str(PROJ))
os.chdir(PROJ)

from src.train.trainer import SentenceUQTrainer  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.features.extractor import (  # noqa: E402
    extract_sentence_token_features,
    extract_sentence_aggregate_feature,
)
from src.inference.predict import load_trained_model  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_ratio_level_metrics,
    compute_calibration_metrics,
    compute_strict_factuality_metrics,
)

RESULTS = PROJ / "results/setup_2"
cfg = yaml.safe_load(open(PROJ / "configs/default.yaml"))
DEVICE = torch.device("cpu")

# --- 1. trained model: learned W, α (feature transform) + θ̂ (ours readout) ---
loaded = load_trained_model(RESULTS / "trained_model.pt", map_location="cpu")
feature_params = loaded["feature_params"].to(DEVICE).eval()
theta_hat = loaded["theta_hat"].to(torch.float32)
k = theta_hat.shape[0]
print(f"feature dim k = {k}  (projection_dim+2)")

# --- 2. same split / pool as 04_evaluate ------------------------------------
trainer = SentenceUQTrainer(
    model=BayesianSentenceUQ(feature_params=feature_params), device="cpu"
)
split_file = (cfg.get("dataset") or {}).get("split_file") or "data/splits/setup_2.json"
gen_dirs = {
    "factscore_bio": "data/generations/factscore_bio",
    "longfact": "data/generations/longfact",
}
cache_dirs = {
    "factscore_bio": "data/cache/factscore_bio",
    "longfact": "data/cache/longfact",
}
proc_dirs = {
    "factscore_bio": "data/processed/factscore_bio",
    "longfact": "data/processed/longfact",
}
data = trainer.prepare_data(
    split_file=split_file, generations_dirs=gen_dirs,
    cache_dirs=cache_dirs, processed_dirs=proc_dirs,
)


def pos(records):
    return [r for r in (records or []) if int(r.get("m_j", 0) or 0) > 0]


train = pos(data.get("train"))
test = pos(data.get("test"))
print(f"train sentences (m>0): {len(train)} | test: {len(test)}")


# --- 3. per-sentence pooled features on the SHARED z_ℓ -----------------------
def build(records):
    feats = {"mean": [], "last": [], "msl": []}
    ours_mu, K, m = [], [], []
    with torch.no_grad():
        for r in records:
            z = extract_sentence_token_features(
                hidden_states=r["hidden_states"].to(DEVICE),
                entropy=r["entropy"].to(DEVICE),
                top1_prob=r["top1"].to(DEVICE),
                token_range=(int(r["token_range"][0]), int(r["token_range"][1])),
                params=feature_params,
            ).to(torch.float32)                                # (L_j, k)
            feats["mean"].append(z.mean(0).numpy())
            feats["last"].append(z[-1].numpy())
            feats["msl"].append(extract_sentence_aggregate_feature(z).numpy())
            # ours readout: project each token to a scalar prob, then average
            p_tok = torch.sigmoid(z @ theta_hat)               # (L_j,)
            ours_mu.append(float(p_tok.mean()))
            K.append(int(r.get("K_j", 0) or 0))
            m.append(int(r.get("m_j", 0) or 0))
    feats = {kk: np.asarray(v, dtype=np.float64) for kk, v in feats.items()}
    return feats, np.asarray(ours_mu), np.asarray(K, float), np.asarray(m, float)


ftr, mu_tr, Ktr, mtr = build(train)
fte, mu_te, Kte, mte = build(test)
Utr, Ute = Ktr / np.maximum(mtr, 1), Kte / np.maximum(mte, 1)
Atr, Ate = (Ktr == mtr).astype(float), (Kte == mte).astype(float)
print(f"strict-positive rate  train={Atr.mean():.3f}  test={Ate.mean():.3f}")


# --- 4. scoring helpers ------------------------------------------------------
def strict_row(name, p_strict, score_for_pearson_U):
    s = compute_strict_factuality_metrics(Ate, p_strict, 1.0 - np.asarray(p_strict))
    cal = compute_calibration_metrics(Ate, p_strict, n_bins=10)
    r = compute_ratio_level_metrics(Ute, np.clip(score_for_pearson_U, 0, 1), m_j=mte)
    return {
        "method": name,
        "strict_AUROC": s["AUROC"], "strict_AUPRC": s["AUPRC"],
        "strict_ECE": cal["ECE"],
        "ratio_MAE": r["MAE"], "ratio_Pearson": r["Pearson_r"],
    }


def fit_readout(name, Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    # strict head: L1-LR on A_j (matches adapted/Han probe recipe)
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000)
    lr.fit(Xtr_s, Atr)
    p_strict = lr.predict_proba(Xte_s)[:, 1]
    # ratio head: Ridge on U_j, clipped to [0,1] for ratio metrics
    rg = Ridge(alpha=1.0).fit(Xtr_s, Utr)
    u_hat = np.clip(rg.predict(Xte_s), 0, 1)
    row = strict_row(name, p_strict, u_hat)
    # ratio metrics use the dedicated ratio head, not the strict proba
    rr = compute_ratio_level_metrics(Ute, u_hat, m_j=mte)
    row["ratio_MAE"], row["ratio_Pearson"] = rr["MAE"], rr["Pearson_r"]
    return row


rows = []
# A. reproduce ours (project-then-pool), strict = μ^m
rows.append(strict_row("A_ours_proj_then_pool", np.power(mu_te, mte), mu_te))
# B/C/D. pool-then-project on the SAME z_ℓ
rows.append(fit_readout("B_pool_mean_66d", ftr["mean"], fte["mean"]))
rows.append(fit_readout("C_pool_last_66d", ftr["last"], fte["last"]))
rows.append(fit_readout("D_mean+std+last_198d", ftr["msl"], fte["msl"]))

# --- 5. report ---------------------------------------------------------------
print("\n" + "=" * 92)
print("Route-A diagnostic — readout swap on the SAME features z_ℓ (setup_2 test, n=%d)" % len(test))
print("=" * 92)
hdr = f"{'readout':28s} {'strict_AUROC':>12s} {'strict_AUPRC':>12s} {'strict_ECE':>10s} {'ratio_MAE':>10s} {'ratio_r':>8s}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['method']:28s} {r['strict_AUROC']:12.4f} {r['strict_AUPRC']:12.4f} "
          f"{r['strict_ECE']:10.4f} {r['ratio_MAE']:10.4f} {r['ratio_Pearson']:8.4f}")
print("-" * len(hdr))
print(f"{'REF ours (CSV, μ^m)':28s} {0.7797:12.4f} {0.2388:12.4f} {0.0518:10.4f} {0.2237:10.4f} {0.4526:8.4f}")
print(f"{'REF adapted (last 4096d)':28s} {0.8113:12.4f} {0.2659:12.4f} {0.0330:10.4f} {0.2072:10.4f} {0.3425:8.4f}")
print(f"{'REF logreg (mean 4096d)':28s} {0.8031:12.4f} {0.2796:12.4f} {0.0526:10.4f} {0.1922:10.4f} {0.3170:8.4f}")
print(f"{'REF Han repo (re-encode)':28s} {0.8576:12.4f} {0.4217:12.4f} {0.1725:10.4f} {0.2063:10.4f} {0.5214:8.4f}")
print("=" * 92)
print("\nVERDICT GUIDE:")
print("  C_pool_last_66d AUROC ≈ 0.81  -> features OK, project-then-pool is the bottleneck -> Route A VIABLE")
print("  C_pool_last_66d AUROC ≈ 0.78  -> 66-d W compression is the ceiling -> richer features or Route B")
