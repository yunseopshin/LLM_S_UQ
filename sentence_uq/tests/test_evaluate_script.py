"""Tests for ``scripts/04_evaluate.py`` — Phase 6-2.

Drives the full evaluation pipeline end-to-end on a synthetic FActScore-Bio
fixture (one entity, two sentences) plus a fabricated trained model and
baseline cache. The tests assert that the script's output artefacts are
written with the expected schema; the metric *values* themselves are
covered by :mod:`tests.test_metrics`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import pandas as pd
import pytest
import torch

matplotlib.use("Agg")  # noqa: E402  -- before pyplot import inside the script

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.extractor import SentenceUQParams  # noqa: E402
from src.inference.predict import save_trained_model  # noqa: E402


# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------


def _load_evaluate_module():
    """Import ``scripts/04_evaluate.py`` as a module (numeric filename)."""
    spec = importlib.util.spec_from_file_location(
        "evaluate_module",
        _PROJECT_ROOT / "scripts" / "04_evaluate.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not locate scripts/04_evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------


def _write_fixture(
    tmp_path: Path,
    hidden_dim: int = 6,
    num_layers: int = 3,
    projection_dim: int = 4,
    T: int = 8,
) -> Tuple[Dict[str, Path], Path]:
    """Lay out a complete Phase 1-1 / 1-3 / 1-4 / 4-1 directory tree.

    Returns
    -------
    paths : dict
        Keys ``"gen", "cache", "proc", "results", "trained", "split",
        "baselines"`` mapping to the relevant file/dir.
    project_dir : Path
        Top-level fixture root.
    """
    project_dir = tmp_path / "proj"
    gen_dir = project_dir / "data" / "generations" / "factscore_bio"
    cache_dir = project_dir / "data" / "cache" / "factscore_bio"
    proc_dir = project_dir / "data" / "processed" / "factscore_bio"
    lf_gen = project_dir / "data" / "generations" / "longfact"
    lf_cache = project_dir / "data" / "cache" / "longfact"
    lf_proc = project_dir / "data" / "processed" / "longfact"
    results_dir = project_dir / "results" / "setup_2"
    for d in (gen_dir, cache_dir, proc_dir, lf_gen, lf_cache, lf_proc, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Generation .pt (FActScore-Bio entity "Alice") ---------------------
    torch.manual_seed(0)
    hidden = torch.randn(T, num_layers, hidden_dim, dtype=torch.float16)
    gen_payload = {
        "text": "First sentence. Second sentence. Third sentence.",
        "prompt": "Tell me a bio of Alice.",
        "prompt_text": "Tell me a bio of Alice.",
        "prompt_ids": torch.tensor([1, 2], dtype=torch.long),
        "token_ids": torch.arange(T, dtype=torch.long),
        "hidden_states": hidden,
        "logits": torch.zeros(T, 16, dtype=torch.float16),
        "selected_layers": list(range(num_layers)),
        "model_config": {
            "name": "test",
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_layers,
            "vocab_size": 16,
            "selected_layers": list(range(num_layers)),
        },
        "dataset": "factscore_bio",
        "meta": {"entity": "Alice"},
        "finished": True,
    }
    torch.save(gen_payload, gen_dir / "Alice.pt")

    # --- Cache scalars -----------------------------------------------------
    cache_payload = {
        "entropy": torch.full((T,), 1.5, dtype=torch.float32),
        "top1_prob": torch.full((T,), 0.4, dtype=torch.float32),
        "token_ids": torch.arange(T, dtype=torch.long),
        "source_path": "Alice.pt",
    }
    torch.save(cache_payload, cache_dir / "00000.pt")

    # --- Annotations with several sentences (varied K/m so metrics are not degenerate) ----
    ann_record = {
        "dataset": "factscore_bio",
        "entity": "Alice",
        "text": "First sentence. Second sentence. Third sentence.",
        "sentences": [
            {
                "text": "First sentence.",
                "token_range": [0, 3],
                "m_j": 4,
                "K_j": 4,
                "claims": [],
            },
            {
                "text": "Second sentence.",
                "token_range": [3, 5],
                "m_j": 3,
                "K_j": 1,
                "claims": [],
            },
            {
                "text": "Third sentence.",
                "token_range": [5, T],
                "m_j": 2,
                "K_j": 0,
                "claims": [],
            },
        ],
    }
    with open(proc_dir / "Alice.json", "w", encoding="utf-8") as f:
        json.dump(ann_record, f)

    # --- Split file --------------------------------------------------------
    split_file = project_dir / "data" / "splits" / "setup_2.json"
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split = {
        "setup": 2,
        "seed": 42,
        "train": [
            {"dataset": "factscore_bio", "entity": "Alice", "prompt_idx": 0}
        ],
        "val": [],
        "test": [
            {"dataset": "factscore_bio", "entity": "Alice", "prompt_idx": 0}
        ],
    }
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split, f)

    # --- Trained model -----------------------------------------------------
    fp = SentenceUQParams(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        projection_dim=projection_dim,
    )
    k = fp.feature_dim
    torch.manual_seed(1)
    theta_hat = torch.randn(k, dtype=torch.float32) * 0.1
    Sigma_hat = torch.eye(k, dtype=torch.float32) * 0.05
    trained_path = results_dir / "trained_model.pt"
    save_trained_model(
        trained_path,
        theta_hat=theta_hat,
        Sigma_hat=Sigma_hat,
        feature_params=fp,
        extra={
            "setup": 2,
            "model_dims": {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "projection_dim": projection_dim,
                "selected_layers": list(range(num_layers)),
            },
        },
    )

    # --- Baselines.json (single mock baseline aligned with the 3 sentences) ----
    baselines_path = results_dir / "baselines.json"
    baselines = {
        "setup": 2,
        "selected_layers": list(range(num_layers)),
        "baselines": {
            "token_entropy": {
                "name": "token_entropy",
                "n_test": 3,
                "wall_clock_seconds": 0.001,
                "mu_hat": [0.9, 0.4, 0.1],
                "scores_raw": [0.0, 0.0, 0.0],
                "ratio_metrics": {},
                "strict_metrics": {},
            },
            "logistic_regression": {
                "name": "logistic_regression",
                "n_train": 0,
                "n_test": 3,
                "wall_clock_seconds": 0.002,
                "mu_hat": [0.8, 0.3, 0.2],
                "ratio_metrics": {},
                "strict_metrics": {},
            },
            "skipped_one": {
                "name": "skipped_one",
                "skipped": True,
                "reason": "fixture skip path",
            },
        },
    }
    with open(baselines_path, "w", encoding="utf-8") as f:
        json.dump(baselines, f)

    return (
        {
            "gen": gen_dir,
            "cache": cache_dir,
            "proc": proc_dir,
            "lf_gen": lf_gen,
            "lf_cache": lf_cache,
            "lf_proc": lf_proc,
            "results": results_dir,
            "trained": trained_path,
            "split": split_file,
            "baselines": baselines_path,
        },
        project_dir,
    )


def _write_config_yaml(paths: Dict[str, Path], project_dir: Path) -> Path:
    """Tiny config YAML matching the fixture directory layout."""
    config_path = project_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  setup: 2",
                f"  splits_dir: {paths['split'].parent}",
                f"  split_file: {paths['split']}",
                "generation:",
                f"  factscore_bio_dir: {paths['gen']}",
                f"  longfact_dir: {paths['lf_gen']}",
                "cache:",
                f"  factscore_bio_dir: {paths['cache']}",
                f"  longfact_dir: {paths['lf_cache']}",
                "processed:",
                f"  factscore_bio_dir: {paths['proc']}",
                f"  longfact_dir: {paths['lf_proc']}",
                f"results_dir: {paths['results']}",
            ]
        )
    )
    return config_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_safe_filename_basics() -> None:
    """``_safe_filename`` strips disallowed characters but keeps separators."""
    module = _load_evaluate_module()
    assert module._safe_filename("Ours (Bayesian)") == "Ours__Bayesian"
    assert module._safe_filename("a/b\\c.txt") == "a_b_c_txt"
    assert module._safe_filename("---") == "unnamed" or module._safe_filename("---") == "---"


def test_ablation_binom_vs_bernoulli_schema() -> None:
    """The Binomial / Bernoulli ablation always returns two rows with the expected columns."""
    import numpy as np

    module = _load_evaluate_module()
    rng = np.random.default_rng(0)
    n = 20
    K = rng.integers(0, 5, size=n).astype(np.float64)
    m = np.maximum(K, 1).astype(np.float64) + rng.integers(0, 3, size=n)
    mu_hat = rng.uniform(0.1, 0.9, size=n)
    p_strict = mu_hat ** m

    df = module._ablation_binom_vs_bernoulli(mu_hat, p_strict, K, m)
    assert set(df.columns) == {
        "variant",
        "binomial_NLL",
        "ratio_MAE",
        "strict_ECE",
        "strict_AUROC",
    }
    assert df["variant"].tolist() == ["Binomial", "Bernoulli (m=1)"]


def test_evaluate_main_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test: every required artefact is produced."""
    module = _load_evaluate_module()
    paths, project_dir = _write_fixture(tmp_path)
    config_path = _write_config_yaml(paths, project_dir)

    rc = module.main(
        [
            "--setup", "2",
            "--config", str(config_path),
            "--device", "cpu",
            "--bootstrap-iters", "20",
            "--mc-samples", "12",
            "--num-heatmaps", "1",
            "--no-plots",
        ]
    )
    assert rc == 0

    results = paths["results"]
    for fname in (
        "final_metrics_ratio.csv",
        "final_metrics_strict.csv",
        "ablation_bayesian_vs_point.csv",
        "ablation_binomial_vs_bernoulli.csv",
        "ablation_mc_vs_linear.csv",
        "alpha_distribution.csv",
        "eval_summary.json",
    ):
        assert (results / fname).exists(), f"missing {fname}"

    ratio_df = pd.read_csv(results / "final_metrics_ratio.csv")
    strict_df = pd.read_csv(results / "final_metrics_strict.csv")
    assert "Ours (Bayesian)" in ratio_df["method"].tolist()
    assert "Ours (Point)" in ratio_df["method"].tolist()
    # token_entropy + logistic_regression baselines were aligned with the test pool
    methods = set(ratio_df["method"].tolist())
    assert {"token_entropy", "logistic_regression"}.issubset(methods)
    # Bootstrap CIs must respect the lo ≤ hi ordering when finite.
    for _, row in strict_df.iterrows():
        if pd.notna(row["AUROC_CI_lo"]) and pd.notna(row["AUROC_CI_hi"]):
            assert row["AUROC_CI_lo"] <= row["AUROC_CI_hi"] + 1e-9

    summary = json.loads((results / "eval_summary.json").read_text())
    assert summary["setup"] == 2
    assert summary["n_test_positive"] >= 1


def test_evaluate_main_with_plots(tmp_path: Path) -> None:
    """Running without ``--no-plots`` produces the PRR / α / heatmap PNGs."""
    module = _load_evaluate_module()
    paths, project_dir = _write_fixture(tmp_path)
    config_path = _write_config_yaml(paths, project_dir)

    rc = module.main(
        [
            "--setup", "2",
            "--config", str(config_path),
            "--device", "cpu",
            "--bootstrap-iters", "8",
            "--mc-samples", "8",
            "--num-heatmaps", "1",
        ]
    )
    assert rc == 0

    results = paths["results"]
    assert (results / "prr_curves.png").exists()
    assert (results / "mc_vs_linear.png").exists()
    assert (results / "alpha_distribution.png").exists()
    rel_dir = results / "reliability_diagrams"
    assert rel_dir.exists() and any(rel_dir.iterdir())
    heat_dir = results / "token_heatmaps"
    assert heat_dir.exists() and any(heat_dir.iterdir())


def test_evaluate_missing_trained_model_returns_2(tmp_path: Path) -> None:
    """Absent ``trained_model.pt`` produces a non-zero exit code, not a crash."""
    module = _load_evaluate_module()
    paths, project_dir = _write_fixture(tmp_path)
    paths["trained"].unlink()
    config_path = _write_config_yaml(paths, project_dir)

    rc = module.main(
        ["--setup", "2", "--config", str(config_path), "--no-plots"]
    )
    assert rc == 2
