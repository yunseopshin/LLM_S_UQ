"""OOD-detection head-to-head: does epi_mu beat baseline signals at flagging OOD?

Annotation-free Option-A check (a fast filter, NOT the definitive error-detection
test). Pool the in-domain Setup-2 Bio test sentences (353) with the OOD LongFact
sentences (computer-security, ~292) and ask: which per-sentence signal best
separates OOD from in-domain (AUROC of "is_OOD")? Signals that need no m_j:

  epi_mu      = ĝᵀΣ̂ĝ                 (ours, epistemic)
  mean_entropy= (1/L)Σ H_ℓ            (== token_entropy baseline)
  neg_mu      = -μ̂                    (ours, confidence: low factuality)
  neg_top1    = -(1/L)Σ p^(1)_ℓ       (mean top-1 prob, a confidence proxy)

If epi_mu does NOT beat mean_entropy / confidence here, the epistemic signal adds
little even under domain shift → reconsider epi. If it does, proceed to the
labelled OOD error-detection test (needs annotation).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

PROJ = Path("/home/ys971217/LLM_S_UQ/sentence_uq")
sys.path.insert(0, str(PROJ))
os.chdir(PROJ)

from src.train.trainer import SentenceUQTrainer  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.features.extractor import extract_sentence_token_features  # noqa: E402
from src.inference.predict import load_trained_model  # noqa: E402
from src.data.sentence_split import load_spacy_model, process_generation  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

cfg = yaml.safe_load(open(PROJ / "configs/default.yaml"))
DEV = torch.device("cpu")
loaded = load_trained_model(PROJ / "results/setup_2/trained_model.pt", map_location="cpu")
fp = loaded["feature_params"].to(DEV).eval()
theta, Sig = loaded["theta_hat"].to(torch.float32), loaded["Sigma_hat"].to(torch.float32)


def signals_for_sentence(z, ent_slice, top1_slice):
    """z:(L,k); return (epi_mu, mean_entropy, neg_mu, neg_top1)."""
    logits = z @ theta
    pi = torch.sigmoid(logits)
    w = pi * (1.0 - pi)
    g = (w.unsqueeze(1) * z).mean(0)
    epi_mu = float((g @ (Sig @ g)).clamp_min(0.0))
    mu = float(pi.mean())
    return epi_mu, float(ent_slice.mean()), -mu, -float(top1_slice.mean())


# --- in-domain: Setup-2 Bio test (has token_range from annotation pipeline) ---
trainer = SentenceUQTrainer(model=BayesianSentenceUQ(feature_params=fp), device="cpu")
data = trainer.prepare_data(
    split_file="data/splits/setup_2.json",
    generations_dirs={"factscore_bio": "data/generations/factscore_bio", "longfact": "data/generations/longfact"},
    cache_dirs={"factscore_bio": "data/cache/factscore_bio", "longfact": "data/cache/longfact"},
    processed_dirs={"factscore_bio": "data/processed/factscore_bio", "longfact": "data/processed/longfact"},
)
test = [r for r in data["test"] if int(r.get("m_j", 0) or 0) > 0]
ID = []
with torch.no_grad():
    for r in test:
        a, b = int(r["token_range"][0]), int(r["token_range"][1])
        z = extract_sentence_token_features(
            hidden_states=r["hidden_states"], entropy=r["entropy"], top1_prob=r["top1"],
            token_range=(a, b), params=fp,
        ).to(torch.float32)
        ID.append(signals_for_sentence(z, r["entropy"][a:b], r["top1"][a:b]))

# --- OOD: LongFact generations, annotation-free sentence split (09c pattern) ---
gen_dir, cache_dir = Path("data/generations/longfact"), Path("data/cache/longfact")
tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
nlp = load_spacy_model()
index = SentenceUQTrainer._build_generation_index(gen_dir)
OOD = []
with torch.no_grad():
    for rel_path, cidx in sorted(index.items(), key=lambda kv: kv[1]):
        hs, ent, t1 = SentenceUQTrainer._load_prompt_tensors(
            gen_dir=gen_dir, cache_dir=cache_dir, rel_path=rel_path, cache_idx=cidx)
        payload = torch.load(gen_dir / rel_path, map_location="cpu", weights_only=False)
        for s in (process_generation(payload, tokenizer=tok, nlp=nlp).get("sentences") or []):
            a, b = int(s["token_range"][0]), int(s["token_range"][1])
            if b <= a:
                continue
            z = extract_sentence_token_features(
                hidden_states=hs, entropy=ent, top1_prob=t1, token_range=(a, b), params=fp,
            ).to(torch.float32)
            OOD.append(signals_for_sentence(z, ent[a:b], t1[a:b]))

ID, OOD = np.array(ID), np.array(OOD)
y = np.concatenate([np.zeros(len(ID)), np.ones(len(OOD))])  # 1 = OOD
names = ["epi_mu (OURS)", "mean_entropy (token_entropy)", "neg_mu (OURS conf)", "neg_top1 (conf proxy)"]
allsig = np.vstack([ID, OOD])
print(f"in-domain={len(ID)}  OOD={len(OOD)}")
print(f"{'signal':32s} {'OOD-detect AUROC':>16s}  {'ID mean':>10s} {'OOD mean':>10s}")
for j, nm in enumerate(names):
    s = allsig[:, j]
    auc = roc_auc_score(y, s)
    print(f"{nm:32s} {auc:16.4f}  {ID[:,j].mean():10.4f} {OOD[:,j].mean():10.4f}")
print("\nAUROC>0.5 ⇒ signal is higher on OOD. Compare epi_mu vs mean_entropy/confidence.")
