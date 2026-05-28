"""Convert our annotation + setup-2 split into Han fact-probe .pkl format.

Output: list of [atom_text, is_supported(int)] pairs per split section,
which Han's flatten_scores() consumes directly. Labels are coerced to int
*here* because our annotations store them as strings ('0'/'1') and Han's
`int(bool(is_supported))` would turn the string '0' into 1.
"""
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/home/ys971217/LLM_S_UQ/sentence_uq")
WORK = Path("/home/ys971217/LLM_S_UQ/fact-probe-main/fact-probe-main/_han_on_ours")
(WORK / "train_data").mkdir(parents=True, exist_ok=True)
(WORK / "test_data").mkdir(parents=True, exist_ok=True)

split = json.load(open(ROOT / "data/splits/setup_2.json"))
ann = json.load(open(ROOT / "data/processed/factscore_bio/annotated.json"))

by_ent = {}
for r in ann:
    e = r.get("entity") or (r.get("meta") or {}).get("entity")
    if e:
        by_ent[e] = r


def to_int_label(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().lower()
    return 1 if s in ("1", "true", "yes", "supported", "s") else 0


def build(section):
    pairs, n_ent = [], 0
    for it in split.get(section) or []:
        r = by_ent.get(it.get("entity"))
        if not r:
            continue
        n_ent += 1
        for s in r.get("sentences") or []:
            for c in (s.get("claims") or []):
                t = str(c.get("text", "") or "").strip()
                if t:
                    pairs.append([t, to_int_label(c.get("label", 0))])
    return pairs, n_ent


tr, ntr = build("train")
te, nte = build("test")
pickle.dump(tr, open(WORK / "train_data/train.pkl", "wb"))
pickle.dump(te, open(WORK / "test_data/test.pkl", "wb"))

print(f"train: {len(tr)} atoms / {ntr} entities | support rate {np.mean([p[1] for p in tr]):.3f}")
print(f"test : {len(te)} atoms / {nte} entities | support rate {np.mean([p[1] for p in te]):.3f}")
print("sample:", tr[0])
