"""DECISIVE test: on LABELLED OOD, does epi beat baselines at flagging errors?

Runs after `02_annotate --dataset longfact` produces data/processed/longfact/
annotated.json (K_j, m_j per OOD sentence). For every OOD sentence (m_j>0) we
compute each method's uncertainty signal and score how well it ranks ERRORS:

  strict error-detection AUROC = roc_auc( 1{A_j=0}, uncertainty )   [A_j=1{K=m}]
  ratio  PRR_AUC                = reject high-uncertainty, remaining U rises
  Spearman(uncertainty, U)      = should be NEGATIVE (more uncertain ⇒ lower U)

Signals (all computable on OOD; ours need the Setup-2 model, no OOD training):
  ours: epi_mu (epistemic) | total_U (aleatoric+epi) | conf = -μ̂ (confidence)
  baselines: mean_entropy (= token_entropy) | logreg_unc (= -P_logreg(A),
             refit on Bio TRAIN, applied OOD — the deploy-in-domain-test-OOD setup)

Hypothesis from the pre-check (confidence is FOOLED OOD): epi/total should beat
mean_entropy & conf at OOD error detection. If not → rethink epi.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

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
predictor = Predictor(theta_hat=loaded["theta_hat"], Sigma_hat=loaded["Sigma_hat"],
                      feature_params=fp, likelihood=loaded.get("likelihood"))

ANN = PROJ / "data/processed/longfact/annotated.json"
if not ANN.exists():
    sys.exit(f"OOD annotations not found at {ANN} — run 02_annotate --dataset longfact first.")
ood_ann = json.load(open(ANN))
# map (topic, prompt_idx) -> annotation record
ann_map = {}
for r in (ood_ann if isinstance(ood_ann, list) else ood_ann.values()):
    key = (str(r.get("topic") or (r.get("meta") or {}).get("topic")), int(r.get("prompt_idx") or (r.get("meta") or {}).get("prompt_idx") or 0))
    ann_map[key] = r

gen_dir, cache_dir = Path("data/generations/longfact"), Path("data/cache/longfact")
index = SentenceUQTrainer._build_generation_index(gen_dir)


def collect_ood():
    """Per OOD sentence (m_j>0): signals + U/A."""
    rows = []
    with torch.no_grad():
        for rel_path, cidx in sorted(index.items(), key=lambda kv: kv[1]):
            hs, ent, t1 = SentenceUQTrainer._load_prompt_tensors(
                gen_dir=gen_dir, cache_dir=cache_dir, rel_path=rel_path, cache_idx=cidx)
            topic = Path(rel_path).parent.name
            try:
                pidx = int(Path(rel_path).stem)
            except ValueError:
                pidx = 0
            rec = ann_map.get((topic, pidx))
            if rec is None:
                continue
            for s in (rec.get("sentences") or []):
                m = int(s.get("m_j", 0) or 0)
                if m <= 0:
                    continue
                a, b = int(s["token_range"][0]), int(s["token_range"][1])
                if b <= a:
                    continue
                z = extract_sentence_token_features(
                    hidden_states=hs, entropy=ent, top1_prob=t1, token_range=(a, b), params=fp)
                zp = pool_token_features(z, fp)
                out = predictor.predict_sentence(zp, m_j=m)
                rows.append({
                    "epi_mu": out["epi_mu"], "total_U": out["total_U"] or 0.0,
                    "conf": -out["mu_hat"], "mean_entropy": float(ent[a:b].mean()),
                    "z_mean": z.mean(0).numpy(), "z_last": z[-1].numpy(),
                    "K": int(s.get("K_j", 0) or 0), "m": m,
                })
    return rows


# --- baseline probe: logistic_regression refit on Bio TRAIN, applied OOD ------
def fit_logreg_on_bio():
    trainer = SentenceUQTrainer(model=BayesianSentenceUQ(feature_params=fp), device="cpu")
    data = trainer.prepare_data(
        split_file="data/splits/setup_2.json",
        generations_dirs={"factscore_bio": "data/generations/factscore_bio", "longfact": "data/generations/longfact"},
        cache_dirs={"factscore_bio": "data/cache/factscore_bio", "longfact": "data/cache/longfact"},
        processed_dirs={"factscore_bio": "data/processed/factscore_bio", "longfact": "data/processed/longfact"})
    Xtr, ytr = [], []
    with torch.no_grad():
        for r in data["train"]:
            if int(r.get("m_j", 0) or 0) <= 0:
                continue
            a, b = int(r["token_range"][0]), int(r["token_range"][1])
            z = extract_sentence_token_features(
                hidden_states=r["hidden_states"], entropy=r["entropy"], top1_prob=r["top1"],
                token_range=(a, b), params=fp)
            Xtr.append(z.mean(0).numpy()); ytr.append(int(int(r["K_j"]) == int(r["m_j"])))
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return sc, lr


rows = collect_ood()
print(f"OOD labelled sentences (m_j>0): {len(rows)}")
if len(rows) < 10:
    sys.exit("too few OOD sentences to evaluate.")
m = np.array([r["m"] for r in rows], float); K = np.array([r["K"] for r in rows], float)
U = K / np.maximum(m, 1.0); A = (K == m).astype(float); err = 1.0 - A
print(f"OOD strict-positive rate: {A.mean():.3f}  mean U: {U.mean():.3f}")

sc, lr = fit_logreg_on_bio()
Zmean = np.vstack([r["z_mean"] for r in rows])
logreg_unc = -lr.predict_proba(sc.transform(Zmean))[:, 1]   # low P(A) => uncertain

signals = {
    "epi_mu (OURS epistemic)": np.array([r["epi_mu"] for r in rows]),
    "total_U (OURS alea+epi)": np.array([r["total_U"] for r in rows]),
    "conf=-mu (OURS confid.)": np.array([r["conf"] for r in rows]),
    "mean_entropy (token_ent)": np.array([r["mean_entropy"] for r in rows]),
    "logreg_unc (probe)": logreg_unc,
}
print(f"\n{'signal':28s} {'strict err-AUROC':>16s} {'ratio PRR':>10s} {'strict PRR':>11s} {'Spearman(U)':>12s}")
for nm, sig in signals.items():
    err_auroc = roc_auc_score(err, sig) if err.sum() not in (0, len(err)) else float("nan")
    pr_ratio = compute_prr(U, sig)["prr_auc"]
    pr_strict = compute_prr(A, sig)["prr_auc"]
    rho = spearmanr(sig, U).correlation
    print(f"{nm:28s} {err_auroc:16.4f} {pr_ratio:10.4f} {pr_strict:11.4f} {rho:12.4f}")
print("\nGood error signal: high err-AUROC, high PRR, NEGATIVE Spearman(U).")
print("Verdict hinges on whether epi_mu/total_U beat mean_entropy & conf.")

# --- WHY does epi fail? Is it entangled with the w=mu(1-mu) confidence term? --
epi = np.array([r["epi_mu"] for r in rows])
mu = -np.array([r["conf"] for r in rows])            # conf = -mu
w_term = mu * (1.0 - mu)                              # the gate w=mu(1-mu)
from scipy.stats import pearsonr
print("\n=== epi entanglement diagnostic (OOD) ===")
print(f"  pearson(epi_mu, mu(1-mu)) = {pearsonr(epi, w_term)[0]:+.3f}   (high ⇒ epi is a mid-confidence/aleatoric proxy)")
print(f"  pearson(epi_mu, mu)       = {pearsonr(epi, mu)[0]:+.3f}")
print(f"  pearson(epi_mu, mean_ent) = {pearsonr(epi, np.array([r['mean_entropy'] for r in rows]))[0]:+.3f}")
