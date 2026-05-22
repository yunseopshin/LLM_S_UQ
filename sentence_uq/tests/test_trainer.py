"""Tests for ``src.train.trainer`` — Phase 4-1.

Covers:
- ``SentenceUQTrainer.__init__`` validation,
- ``prepare_data`` flattening with on-disk synthetic generation /
  cache / annotation files,
- ``train_epoch`` forward + backward + Adam step on synthetic data,
- ``evaluate`` ratio-level metric schema and ``m_j = 0`` skipping,
- ``fit`` end-to-end with val + test data, including the final
  ``(θ̂, Σ̂)`` returned in the history.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.extractor import SentenceUQParams  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.train.trainer import SentenceUQTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(
    hidden_dim: int = 8,
    num_layers: int = 3,
    projection_dim: int = 4,
    num_fisher_iters: int = 4,
) -> BayesianSentenceUQ:
    params = SentenceUQParams(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        projection_dim=projection_dim,
    )
    return BayesianSentenceUQ(params, num_fisher_iters=num_fisher_iters)


def _make_sentence(
    T: int = 6,
    start: int = 0,
    end: int | None = None,
    hidden_dim: int = 8,
    num_layers: int = 3,
    m: int | None = None,
    K: int | None = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Synthetic per-sentence record.

    When ``m`` / ``K`` are not given, they are drawn so that ``K/m`` varies
    across sentences — the trivial ``μ = 0.5`` solution would otherwise sit
    at θ = 0 and zero out the W / α gradients.
    """
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(T, num_layers, hidden_dim, generator=g, dtype=torch.float32)
    entropy = torch.randn(T, generator=g).abs()
    top1 = torch.rand(T, generator=g)
    if end is None:
        end = T
    if m is None:
        m = int(torch.randint(2, 5, (1,), generator=g).item())
    if K is None:
        # Bias K toward either 0 or m so K/m ≠ 0.5; perturb by seed parity.
        K = m if (seed % 2 == 0) else 0
    return {
        "dataset": "factscore_bio",
        "source_id": f"test_{seed}",
        "token_range": (int(start), int(end)),
        "K_j": int(K),
        "m_j": int(m),
        "hidden_states": hidden,
        "entropy": entropy,
        "top1": top1,
    }


def _make_dataset(
    n: int = 5, seed: int = 0, **kwargs: Any
) -> List[Dict[str, Any]]:
    return [
        _make_sentence(seed=seed + i, **kwargs) for i in range(n)
    ]


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_stores_attributes_and_validates() -> None:
    model = _make_model()
    trainer = SentenceUQTrainer(
        model,
        lr=2e-3,
        num_epochs=3,
        eval_every=1,
        pd_check_every=0,
        device="cpu",
    )
    assert trainer.model is model
    assert trainer.lr == 2e-3
    assert trainer.num_epochs == 3
    assert trainer.eval_every == 1
    assert trainer.pd_check_every == 0
    assert trainer.device.type == "cpu"
    assert isinstance(trainer.optimizer, torch.optim.Adam)

    with pytest.raises(TypeError):
        SentenceUQTrainer("not a model")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SentenceUQTrainer(model, lr=0)
    with pytest.raises(ValueError):
        SentenceUQTrainer(model, num_epochs=0)
    with pytest.raises(ValueError):
        SentenceUQTrainer(model, eval_every=-1)
    with pytest.raises(ValueError):
        SentenceUQTrainer(model, pd_check_every=-1)


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------


def test_train_epoch_returns_stats() -> None:
    model = _make_model()
    trainer = SentenceUQTrainer(model, lr=1e-2, num_epochs=1, pd_check_every=0)
    data = _make_dataset(n=5, seed=10)

    stats = trainer.train_epoch(data)
    assert set(stats.keys()) == {"loss", "num_sentences", "num_positive"}
    assert isinstance(stats["loss"], float)
    assert stats["num_sentences"] == 5
    assert stats["num_positive"] == 5  # all m_j > 0 by construction


def test_train_epoch_updates_psi() -> None:
    """Adam step must move at least W, alpha, log_sigma_0 (gradients reach all)."""
    model = _make_model()
    trainer = SentenceUQTrainer(model, lr=1e-1, num_epochs=1, pd_check_every=0)
    data = _make_dataset(n=5, seed=20)

    W0 = model.feature_params.W.weight.detach().clone()
    alpha0 = model.feature_params.alpha.detach().clone()
    log_s0 = model.feature_params.log_sigma_0.detach().clone()

    trainer.train_epoch(data)

    assert not torch.allclose(W0, model.feature_params.W.weight, atol=1e-7)
    assert not torch.allclose(alpha0, model.feature_params.alpha, atol=1e-7)
    assert not torch.allclose(log_s0, model.feature_params.log_sigma_0, atol=1e-7)


def test_train_epoch_empty_data_raises() -> None:
    model = _make_model()
    trainer = SentenceUQTrainer(model, num_epochs=1, pd_check_every=0)
    with pytest.raises(ValueError):
        trainer.train_epoch([])


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_returns_ratio_metric_schema() -> None:
    model = _make_model()
    trainer = SentenceUQTrainer(model, num_epochs=1, pd_check_every=0)
    train = _make_dataset(n=5, seed=30)
    val = _make_dataset(n=3, seed=130)

    metrics = trainer.evaluate(train, val)
    assert set(metrics.keys()) == {"MAE", "RMSE", "Pearson_r", "binomial_NLL", "n"}
    assert metrics["n"] == 3
    assert metrics["MAE"] >= 0.0
    assert metrics["RMSE"] >= 0.0
    assert metrics["binomial_NLL"] >= 0.0


def test_evaluate_skips_m_zero_sentences() -> None:
    """Sentences with m_j = 0 must not contribute to ratio-level metrics."""
    model = _make_model()
    trainer = SentenceUQTrainer(model, num_epochs=1, pd_check_every=0)
    train = _make_dataset(n=5, seed=40)

    val = [
        _make_sentence(seed=200, m=0, K=0),
        _make_sentence(seed=201, m=2, K=1),
        _make_sentence(seed=202, m=0, K=0),
    ]
    metrics = trainer.evaluate(train, val)
    assert metrics["n"] == 1


def test_evaluate_empty_eval_returns_nans() -> None:
    model = _make_model()
    trainer = SentenceUQTrainer(model, num_epochs=1, pd_check_every=0)
    train = _make_dataset(n=4, seed=50)

    # Two m_j = 0 sentences only.
    val = [_make_sentence(seed=300, m=0, K=0), _make_sentence(seed=301, m=0, K=0)]
    metrics = trainer.evaluate(train, val)
    assert metrics["n"] == 0
    for key in ("MAE", "RMSE", "binomial_NLL"):
        assert metrics[key] != metrics[key]  # NaN check


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_fit_runs_end_to_end_and_returns_posterior() -> None:
    model = _make_model()
    trainer = SentenceUQTrainer(
        model,
        lr=1e-2,
        num_epochs=3,
        eval_every=1,
        pd_check_every=2,
        log_fn=lambda msg: None,
    )
    train = _make_dataset(n=5, seed=60)
    val = _make_dataset(n=3, seed=160)
    test = _make_dataset(n=4, seed=260)

    history = trainer.fit(train, val, test)
    k = model.feature_params.feature_dim

    assert len(history["train_loss"]) == 3
    assert len(history["val_metrics"]) == 3
    assert len(history["pd_checks"]) == 1  # epoch 2 only (eval at every / pd at 2)
    assert "test_metrics" in history
    assert history["theta_hat"].shape == (k,)
    assert history["Sigma_hat"].shape == (k, k)
    # Sigma_hat must be symmetric.
    assert torch.allclose(history["Sigma_hat"], history["Sigma_hat"].T, atol=1e-5)


def test_fit_without_val_and_test() -> None:
    model = _make_model()
    trainer = SentenceUQTrainer(
        model, lr=1e-2, num_epochs=2, eval_every=1, pd_check_every=0,
        log_fn=lambda msg: None,
    )
    train = _make_dataset(n=4, seed=70)

    history = trainer.fit(train)
    assert len(history["train_loss"]) == 2
    assert history["val_metrics"] == []
    assert "test_metrics" not in history


# ---------------------------------------------------------------------------
# prepare_data — synthetic on-disk fixture
# ---------------------------------------------------------------------------


def _write_factscore_fixture(
    tmp_path: Path,
    entity: str,
    hidden_dim: int,
    num_layers: int,
    T: int,
) -> tuple[Path, Path, Path]:
    """Write one synthetic FActScore-Bio generation + cache + annotation set."""
    gen_dir = tmp_path / "gen" / "factscore_bio"
    cache_dir = tmp_path / "cache" / "factscore_bio"
    proc_dir = tmp_path / "proc" / "factscore_bio"
    for d in (gen_dir, cache_dir, proc_dir):
        d.mkdir(parents=True, exist_ok=True)

    hidden = torch.randn(T, num_layers, hidden_dim, dtype=torch.float16)
    token_ids = torch.arange(T, dtype=torch.long)
    logits = torch.zeros(T, 16, dtype=torch.float16)
    gen_payload = {
        "text": "First sentence. Second sentence.",
        "prompt": f"Tell me a bio of {entity}.",
        "prompt_text": f"Tell me a bio of {entity}.",
        "prompt_ids": torch.tensor([1, 2], dtype=torch.long),
        "token_ids": token_ids,
        "hidden_states": hidden,
        "logits": logits,
        "selected_layers": list(range(num_layers)),
        "model_config": {
            "name": "test",
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_layers,
            "vocab_size": 16,
            "selected_layers": list(range(num_layers)),
        },
        "dataset": "factscore_bio",
        "meta": {"entity": entity},
        "finished": True,
    }
    torch.save(gen_payload, gen_dir / f"{entity}.pt")

    cache_payload = {
        "entropy": torch.full((T,), 1.5, dtype=torch.float32),
        "top1_prob": torch.full((T,), 0.4, dtype=torch.float32),
        "token_ids": token_ids,
        "source_path": f"{entity}.pt",
    }
    torch.save(cache_payload, cache_dir / "00000.pt")

    ann_record = {
        "dataset": "factscore_bio",
        "entity": entity,
        "text": "First sentence. Second sentence.",
        "sentences": [
            {
                "text": "First sentence.",
                "char_start": 0,
                "char_end": 15,
                "token_range": [0, 3],
                "m_j": 2,
                "K_j": 1,
                "claims": [],
            },
            {
                "text": "Second sentence.",
                "char_start": 16,
                "char_end": 32,
                "token_range": [3, T],
                "m_j": 0,
                "K_j": 0,
                "claims": [],
            },
        ],
    }
    with open(proc_dir / f"{entity}.json", "w", encoding="utf-8") as f:
        json.dump(ann_record, f)
    return gen_dir, cache_dir, proc_dir


def test_prepare_data_flattens_setup_2_split(tmp_path: Path) -> None:
    hidden_dim = 8
    num_layers = 3
    T = 6
    entity = "Alice"

    gen_dir, cache_dir, proc_dir = _write_factscore_fixture(
        tmp_path, entity=entity, hidden_dim=hidden_dim, num_layers=num_layers, T=T
    )
    lf_gen = tmp_path / "gen" / "longfact"
    lf_cache = tmp_path / "cache" / "longfact"
    lf_proc = tmp_path / "proc" / "longfact"
    for d in (lf_gen, lf_cache, lf_proc):
        d.mkdir(parents=True, exist_ok=True)

    split_file = tmp_path / "split.json"
    split = {
        "setup": 2,
        "seed": 42,
        "train": [
            {
                "dataset": "factscore_bio",
                "entity": entity,
                "prompt": f"Tell me a bio of {entity}.",
                "prompt_idx": 0,
            }
        ],
        "val": [],
        "test": [],
    }
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split, f)

    model = _make_model(hidden_dim=hidden_dim, num_layers=num_layers)
    trainer = SentenceUQTrainer(model, num_epochs=1, pd_check_every=0)

    data = trainer.prepare_data(
        split_file=split_file,
        generations_dirs={"factscore_bio": gen_dir, "longfact": lf_gen},
        cache_dirs={"factscore_bio": cache_dir, "longfact": lf_cache},
        processed_dirs={"factscore_bio": proc_dir, "longfact": lf_proc},
    )

    assert set(data.keys()) == {"train", "val", "test"}
    assert data["val"] == []
    assert data["test"] == []
    assert len(data["train"]) == 2

    s0, s1 = data["train"]
    assert s0["dataset"] == "factscore_bio"
    assert s0["source_id"] == entity
    assert s0["token_range"] == (0, 3)
    assert s0["K_j"] == 1
    assert s0["m_j"] == 2
    assert s0["hidden_states"].shape == (T, num_layers, hidden_dim)
    assert s0["entropy"].shape == (T,)
    assert s0["top1"].shape == (T,)
    # Both sentences should share the same hidden_states tensor reference.
    assert s0["hidden_states"].data_ptr() == s1["hidden_states"].data_ptr()
    # Second sentence is m_j = 0 — kept in the records but contributes nothing.
    assert s1["m_j"] == 0


def test_prepare_data_skips_unannotated_or_missing(tmp_path: Path) -> None:
    """Prompts without generation or annotation files are silently skipped."""
    hidden_dim = 8
    num_layers = 3
    T = 5
    gen_dir, cache_dir, proc_dir = _write_factscore_fixture(
        tmp_path, entity="Alice", hidden_dim=hidden_dim, num_layers=num_layers, T=T
    )
    lf_gen = tmp_path / "gen" / "longfact"
    lf_cache = tmp_path / "cache" / "longfact"
    lf_proc = tmp_path / "proc" / "longfact"
    for d in (lf_gen, lf_cache, lf_proc):
        d.mkdir(parents=True, exist_ok=True)

    split = {
        "setup": 2,
        "seed": 42,
        "train": [
            {"dataset": "factscore_bio", "entity": "Alice", "prompt": "", "prompt_idx": 0},
            {"dataset": "factscore_bio", "entity": "Bob", "prompt": "", "prompt_idx": 1},
        ],
        "val": [],
        "test": [],
    }
    split_file = tmp_path / "split.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split, f)

    model = _make_model(hidden_dim=hidden_dim, num_layers=num_layers)
    trainer = SentenceUQTrainer(model, num_epochs=1, pd_check_every=0)
    data = trainer.prepare_data(
        split_file=split_file,
        generations_dirs={"factscore_bio": gen_dir, "longfact": lf_gen},
        cache_dirs={"factscore_bio": cache_dir, "longfact": lf_cache},
        processed_dirs={"factscore_bio": proc_dir, "longfact": lf_proc},
    )
    # Only Alice has fixture files; Bob is skipped silently.
    sources = {s["source_id"] for s in data["train"]}
    assert sources == {"Alice"}
