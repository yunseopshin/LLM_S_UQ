"""CLI for the OOD epistemic check (Phase 9, Plan B — annotation-free).

Usage
-----
    python scripts/09c_ood_epistemic.py --device cpu

Question
~~~~~~~~
Phase 9.1/9.2 established that in-domain (Setup 2) there is essentially no
epistemic signal to extract — near-zero ``epi_μ`` is the *correct* Bayesian
answer when the posterior is data-informed. The decomposition can only be
validated where parameter uncertainty actually exists: **out of distribution**.

This script feeds the **Setup-2-trained model** (trained on FActScore-Bio) two
sentence populations and compares their epistemic readouts:

* **in-domain** — the Setup-2 Bio test split (same loader as Phase 9.1), and
* **OOD** — LongFact generations (`data/generations/longfact`), an unseen
  domain.

For both we compute, per sentence:

    epi_μ      = ĝᵀ Σ̂ ĝ          (probability space; ĝ = mean π(1-π) z)
    epi_logit  = z̄ᵀ Σ̂ z̄         (logit space; z̄ = mean token feature)

**Hypothesis**: if the posterior is genuinely tight only in the in-domain
data subspace, OOD inputs should land partly in prior-scale directions →
**higher epi on OOD than in-domain**. A clear upward shift validates that the
decomposition behaves correctly; a flat result would mean the readout is inert.

No annotation (no API) is needed: sentence boundaries come from
:func:`src.data.sentence_split.process_generation`, the exact splitter the
annotation pipeline uses, so OOD sentences are defined identically to in-domain.

Prerequisites
~~~~~~~~~~~~~
* ``results/setup_2/trained_model.pt``
* LongFact generations under ``data/generations/longfact`` (run
  ``01_generate_data.py --setup 3 --limit N``) and their scalar cache (run
  ``01b_cache_scalars.py --setup 3 --limit N``).

Outputs
~~~~~~~
* ``results/setup_2/document/ood_epistemic.json``
* ``results/setup_2/document/ood_epistemic.png``
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.sentence_split import load_spacy_model, process_generation  # noqa: E402
from src.features.extractor import extract_sentence_token_features  # noqa: E402
from src.inference.predict import load_trained_model  # noqa: E402
from src.train.trainer import SentenceUQTrainer  # noqa: E402


def _load_diag_module() -> Any:
    """Import the digit-prefixed Phase 9.1 module to reuse its in-domain loaders."""
    path = _THIS_DIR / "09_diagnose_epistemic.py"
    spec = importlib.util.spec_from_file_location("diag09", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Epistemic readouts
# ---------------------------------------------------------------------------


def _epi_for_z(
    z: torch.Tensor, theta: torch.Tensor, Sig: torch.Tensor
) -> Tuple[float, float, float]:
    """Return ``(epi_mu, epi_logit, mu_hat)`` for one sentence's token features."""
    zf = z.to(torch.float64)
    pi = torch.sigmoid(zf @ theta)
    w = pi * (1.0 - pi)
    g_hat = (w.unsqueeze(1) * zf).mean(dim=0)
    epi_mu = float((g_hat @ (Sig @ g_hat)).clamp_min(0.0).item())
    z_bar = zf.mean(dim=0)
    epi_logit = float((z_bar @ (Sig @ z_bar)).clamp_min(0.0).item())
    return epi_mu, epi_logit, float(pi.mean().item())


# ---------------------------------------------------------------------------
# In-domain (Setup 2 Bio test) — reuse Phase 9.1 loaders
# ---------------------------------------------------------------------------


def _in_domain_signals(
    diag: Any,
    setup: int,
    device: torch.device,
    feature_params: Any,
    theta: torch.Tensor,
    Sig: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """epi_μ / epi_logit for the in-domain test split."""
    records = [
        r for r in diag._prepare_test_records({}, setup, device, feature_params)
        if int(r.get("m_j", 0) or 0) > 0
    ]
    epi_mu, epi_logit, mu = [], [], []
    for r in records:
        z = diag._extract_z_tokens(r, feature_params, device)
        a, b, c = _epi_for_z(z, theta, Sig)
        epi_mu.append(a); epi_logit.append(b); mu.append(c)
    return {
        "epi_mu": np.asarray(epi_mu, dtype=np.float64),
        "epi_logit": np.asarray(epi_logit, dtype=np.float64),
        "mu_hat": np.asarray(mu, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# OOD (LongFact) — sentence-split without annotation
# ---------------------------------------------------------------------------


def _ood_signals(
    gen_dir: Path,
    cache_dir: Path,
    feature_params: Any,
    theta: torch.Tensor,
    Sig: torch.Tensor,
    device: torch.device,
    tokenizer: Any,
    nlp: Any,
) -> Dict[str, np.ndarray]:
    """epi_μ / epi_logit for every LongFact sentence (annotation-free split)."""
    index = SentenceUQTrainer._build_generation_index(gen_dir)
    if not index:
        raise FileNotFoundError(f"no generations under {gen_dir}")

    epi_mu, epi_logit, mu = [], [], []
    n_prompts = 0
    for rel_path, cache_idx in sorted(index.items(), key=lambda kv: kv[1]):
        hidden_states, entropy, top1 = SentenceUQTrainer._load_prompt_tensors(
            gen_dir=gen_dir, cache_dir=cache_dir, rel_path=rel_path, cache_idx=cache_idx
        )
        payload = torch.load(gen_dir / rel_path, map_location="cpu", weights_only=False)
        split = process_generation(payload, tokenizer=tokenizer, nlp=nlp)
        sentences = split.get("sentences") or []
        if not sentences:
            continue
        n_prompts += 1
        hs = hidden_states.to(device)
        ent = entropy.to(device)
        t1 = top1.to(device)
        for s in sentences:
            a, b = int(s["token_range"][0]), int(s["token_range"][1])
            if b <= a:
                continue
            with torch.no_grad():
                z = extract_sentence_token_features(
                    hidden_states=hs, entropy=ent, top1_prob=t1,
                    token_range=(a, b), params=feature_params,
                ).detach().cpu().to(torch.float32)
            em, el, mh = _epi_for_z(z, theta, Sig)
            epi_mu.append(em); epi_logit.append(el); mu.append(mh)

    return {
        "epi_mu": np.asarray(epi_mu, dtype=np.float64),
        "epi_logit": np.asarray(epi_logit, dtype=np.float64),
        "mu_hat": np.asarray(mu, dtype=np.float64),
        "n_prompts": n_prompts,
    }


# ---------------------------------------------------------------------------
# Comparison + plot
# ---------------------------------------------------------------------------


def _describe(x: np.ndarray) -> Dict[str, float]:
    """Summary stats of a 1-D array."""
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
    }


def _compare(name: str, indom: np.ndarray, ood: np.ndarray) -> Dict[str, Any]:
    """In-domain vs OOD summary + Mann-Whitney U (one-sided OOD > in-domain)."""
    from scipy.stats import mannwhitneyu

    try:
        u, p = mannwhitneyu(ood, indom, alternative="greater")
    except ValueError:
        u, p = float("nan"), float("nan")
    di, do = _describe(indom), _describe(ood)
    return {
        "signal": name,
        "in_domain": di,
        "ood": do,
        "median_ratio_ood_over_indom": (
            do["median"] / di["median"] if di["median"] > 0 else float("inf")
        ),
        "mannwhitney_u": float(u),
        "p_value_ood_greater": float(p),
    }


def _plot(
    indom: Dict[str, np.ndarray], ood: Dict[str, np.ndarray], save_path: Path
) -> None:
    """Log-scale box/strip comparison for epi_μ and epi_logit."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, key, title in (
        (axes[0], "epi_mu", "epi_μ = ĝᵀΣ̂ĝ  (probability space)"),
        (axes[1], "epi_logit", "epi_logit = z̄ᵀΣ̂z̄  (logit space)"),
    ):
        data = [np.maximum(indom[key], 1e-12), np.maximum(ood[key], 1e-12)]
        ax.boxplot(data, labels=["in-domain\n(Bio test)", "OOD\n(LongFact)"],
                   showfliers=False)
        for i, d in enumerate(data, start=1):
            jit = (np.random.default_rng(0).random(d.size) - 0.5) * 0.12
            ax.scatter(np.full(d.size, i) + jit, d, s=6, alpha=0.3, color="C0")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_ylabel("value (log)")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("OOD epistemic check — Setup-2 model on Bio (in-domain) vs LongFact (OOD)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """argparse parser for ``scripts/09c_ood_epistemic.py``."""
    p = argparse.ArgumentParser(description="OOD epistemic check (annotation-free).")
    p.add_argument("--setup", type=int, default=2, help="In-domain setup for the model + test split.")
    p.add_argument("--device", type=str, default="cpu", help="Feature-extractor device.")
    p.add_argument("--trained-model", type=str, default=None, help="Override model path.")
    p.add_argument("--results-dir", type=str, default=None, help="Override output dir.")
    p.add_argument("--ood-gen-dir", type=str, default="data/generations/longfact")
    p.add_argument("--ood-cache-dir", type=str, default="data/cache/longfact")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point — see module docstring."""
    args = _build_parser().parse_args(argv)
    setup = int(args.setup)
    diag = _load_diag_module()
    device = torch.device(diag._resolve_device(args.device))

    results_dir = Path(args.results_dir or f"results/setup_{setup}")
    doc_dir = results_dir / "document"
    trained_path = Path(args.trained_model or results_dir / "trained_model.pt")

    print(f"=== OOD epistemic check — model from setup {setup} ===")
    if not trained_path.exists():
        print(f"error: trained model not found at {trained_path}", file=sys.stderr)
        return 2

    loaded = load_trained_model(trained_path, map_location="cpu")
    feature_params = loaded["feature_params"].to(device)
    feature_params.eval()
    theta = loaded["theta_hat"].to(torch.float64)
    Sig = 0.5 * (loaded["Sigma_hat"] + loaded["Sigma_hat"].T).to(torch.float64)

    # --- in-domain ----------------------------------------------------------
    print("Computing in-domain (Bio test) epistemic readouts ...")
    indom = _in_domain_signals(diag, setup, device, feature_params, theta, Sig)
    print(f"  in-domain sentences: {indom['epi_mu'].size}")

    # --- OOD ----------------------------------------------------------------
    gen_dir = Path(args.ood_gen_dir)
    cache_dir = Path(args.ood_cache_dir)
    if not gen_dir.exists() or not SentenceUQTrainer._build_generation_index(gen_dir):
        print(f"error: no OOD generations under {gen_dir}. Run "
              f"01_generate_data.py --setup 3 --limit N first.", file=sys.stderr)
        return 2
    if not cache_dir.exists():
        print(f"error: no OOD scalar cache under {cache_dir}. Run "
              f"01b_cache_scalars.py --setup 3 --limit N first.", file=sys.stderr)
        return 2

    print("Loading tokenizer + spaCy and splitting OOD generations ...")
    from transformers import AutoTokenizer
    model_name = (loaded.get("extra", {}) or {}).get("model_name") \
        or "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    nlp = load_spacy_model("en")
    ood = _ood_signals(gen_dir, cache_dir, feature_params, theta, Sig,
                       device, tokenizer, nlp)
    print(f"  OOD prompts: {ood['n_prompts']}, OOD sentences: {ood['epi_mu'].size}")
    if ood["epi_mu"].size == 0:
        print("error: no OOD sentences produced.", file=sys.stderr)
        return 2

    # --- compare ------------------------------------------------------------
    cmp_mu = _compare("epi_mu", indom["epi_mu"], ood["epi_mu"])
    cmp_logit = _compare("epi_logit", indom["epi_logit"], ood["epi_logit"])

    print("\n--- epistemic shift (in-domain vs OOD) ---")
    for c in (cmp_mu, cmp_logit):
        print(f"  {c['signal']}: in-domain median={c['in_domain']['median']:.3e}  "
              f"OOD median={c['ood']['median']:.3e}  "
              f"(ratio={c['median_ratio_ood_over_indom']:.2f}x)  "
              f"p(OOD>in)={c['p_value_ood_greater']:.2e}")

    summary = {
        "setup_model": setup,
        "ood_dataset": "longfact",
        "n_in_domain": int(indom["epi_mu"].size),
        "n_ood_prompts": int(ood["n_prompts"]),
        "n_ood_sentences": int(ood["epi_mu"].size),
        "epi_mu": cmp_mu,
        "epi_logit": cmp_logit,
    }
    doc_dir.mkdir(parents=True, exist_ok=True)
    json_path = doc_dir / "ood_epistemic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary -> {json_path}")

    if not args.no_plots:
        plot_path = doc_dir / "ood_epistemic.png"
        _plot(indom, ood, plot_path)
        print(f"Saved figure  -> {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
