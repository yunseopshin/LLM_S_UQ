"""CLI for Phase 4-1: train the Bayesian sentence-level UQ model.

Usage
-----
    python scripts/03_train.py --setup 2 --config configs/default.yaml

The script does **not** compute splits — it loads the JSON saved by
Phase 1-0 (``data/splits/setup_{N}.json``) so that generation,
annotation, training, and evaluation all see exactly the same prompts.

Output layout
~~~~~~~~~~~~~
::

    results/setup_{N}/
        trained_model.pt        # (θ̂, Σ̂, ψ.state_dict, config)
        train_history.json      # per-epoch losses + val metrics + PD checks
        train_summary.json      # final test metrics + sentence counts

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Tuple

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import SETUPS, split_save_filename  # noqa: E402
from src.features.extractor import SentenceUQParams  # noqa: E402
from src.inference.predict import save_trained_model  # noqa: E402
from src.models.bayesian_main import BayesianSentenceUQ  # noqa: E402
from src.train.trainer import SentenceUQTrainer  # noqa: E402


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cfg_get(cfg: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    sec = cfg.get(section) or {}
    val = sec.get(key)
    return val if val is not None else default


def _top_get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    val = cfg.get(key)
    return val if val is not None else default


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train the Bayesian sentence-level UQ model on the chosen "
            "experimental setup (Phase 4-1)."
        )
    )
    p.add_argument(
        "--setup", type=int, choices=list(SETUPS), required=False,
        help="Experimental setup (1, 2, or 3). Falls back to config value.",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="YAML config (e.g. configs/default.yaml).",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="cuda | cpu (default: cuda, falls back to cpu when unavailable).",
    )
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-fisher-iters", type=int, default=None)
    p.add_argument("--projection-dim", type=int, default=None)
    p.add_argument("--prior-sigma-init", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--pd-check-every", type=int, default=None)
    p.add_argument(
        "--results-dir", type=str, default=None,
        help="Override the output directory (default: results/setup_{N}).",
    )
    return p


def _detect_model_dims(
    generations_dirs: dict[str, str | Path],
) -> Tuple[int | None, int | None]:
    """Read ``hidden_dim`` / ``num_layers`` from the first generation .pt found."""
    for gen_dir in generations_dirs.values():
        gen_dir_p = Path(gen_dir)
        if not gen_dir_p.exists():
            continue
        files = sorted(
            (p for p in gen_dir_p.rglob("*.pt") if p.is_file()),
            key=lambda p: p.relative_to(gen_dir_p).as_posix(),
        )
        for p in files:
            try:
                payload = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            hidden = payload.get("hidden_states")
            mc = payload.get("model_config") or {}
            if hidden is None or hidden.dim() != 3 or hidden.shape[0] == 0:
                continue
            hidden_dim = int(mc.get("hidden_dim") or hidden.shape[-1])
            selected = mc.get("selected_layers") or payload.get("selected_layers")
            if selected:
                num_layers = int(len(selected))
            else:
                num_layers = int(hidden.shape[1])
            return hidden_dim, num_layers
    return None, None


def _resolve_device(name: str) -> str:
    if name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return name


def _resolve_setup(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    setup = args.setup
    if setup is None:
        setup = (cfg.get("dataset") or {}).get("setup")
    if setup is None:
        raise SystemExit(
            "error: --setup is required (or set dataset.setup in config)"
        )
    setup = int(setup)
    if setup not in SETUPS:
        raise SystemExit(f"error: unknown setup {setup}; valid: {list(SETUPS)}")
    return setup


def _serializable_history(history: dict[str, Any]) -> dict[str, Any]:
    """Strip non-JSON entries (theta_hat / Sigma_hat tensors)."""
    return {k: v for k, v in history.items() if k not in ("theta_hat", "Sigma_hat")}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg: dict[str, Any] = {}
    if args.config:
        cfg = _load_yaml(args.config)

    setup = _resolve_setup(args, cfg)

    splits_dir = _cfg_get(cfg, "dataset", "splits_dir", "data/splits")
    split_file = _cfg_get(cfg, "dataset", "split_file") or str(
        Path(splits_dir) / split_save_filename(setup)
    )

    generations_dirs = {
        "factscore_bio": _cfg_get(
            cfg, "generation", "factscore_bio_dir",
            "data/generations/factscore_bio",
        ),
        "longfact": _cfg_get(
            cfg, "generation", "longfact_dir", "data/generations/longfact"
        ),
    }
    cache_dirs = {
        "factscore_bio": _cfg_get(
            cfg, "cache", "factscore_bio_dir", "data/cache/factscore_bio"
        ),
        "longfact": _cfg_get(
            cfg, "cache", "longfact_dir", "data/cache/longfact"
        ),
    }
    processed_dirs = {
        "factscore_bio": _cfg_get(
            cfg, "processed", "factscore_bio_dir",
            "data/processed/factscore_bio",
        ),
        "longfact": _cfg_get(
            cfg, "processed", "longfact_dir", "data/processed/longfact"
        ),
    }

    num_epochs = (
        args.num_epochs
        if args.num_epochs is not None
        else int(_top_get(cfg, "num_epochs", 50))
    )
    lr = (
        args.lr
        if args.lr is not None
        else float(_top_get(cfg, "lr", 1e-3))
    )
    num_fisher_iters = (
        args.num_fisher_iters
        if args.num_fisher_iters is not None
        else int(_top_get(cfg, "num_fisher_iters", 10))
    )
    projection_dim = (
        args.projection_dim
        if args.projection_dim is not None
        else int(_top_get(cfg, "projection_dim", 64))
    )
    prior_sigma_init = (
        args.prior_sigma_init
        if args.prior_sigma_init is not None
        else float(_top_get(cfg, "prior_sigma_init", 1.0))
    )
    eval_every = (
        args.eval_every
        if args.eval_every is not None
        else int(_top_get(cfg, "eval_every", 1))
    )
    pd_check_every = (
        args.pd_check_every
        if args.pd_check_every is not None
        else int(_top_get(cfg, "pd_check_every", 5))
    )

    results_dir = Path(
        args.results_dir
        or _top_get(cfg, "results_dir", f"results/setup_{setup}")
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 4-1 training — setup {setup} ===")
    print(f"Split file:  {split_file}")
    print(f"Generations: {generations_dirs}")
    print(f"Cache:       {cache_dirs}")
    print(f"Processed:   {processed_dirs}")
    print(f"Results:     {results_dir}")

    hidden_dim, num_layers = _detect_model_dims(generations_dirs)
    if hidden_dim is None or num_layers is None:
        print(
            "error: could not detect model dims; no generation .pt files "
            "found under the configured generations_dirs. Run Phase 1-1 first.",
            file=sys.stderr,
        )
        return 2
    print(
        f"Detected model dims: hidden_dim={hidden_dim}, "
        f"num_layers={num_layers}, projection_dim={projection_dim}"
    )

    feature_params = SentenceUQParams(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        projection_dim=projection_dim,
    )
    if prior_sigma_init != 1.0:
        with torch.no_grad():
            feature_params.log_sigma_0.fill_(
                float(torch.log(torch.tensor(prior_sigma_init)).item())
            )

    bayes = BayesianSentenceUQ(
        feature_params=feature_params,
        num_fisher_iters=num_fisher_iters,
    )

    device = _resolve_device(args.device)
    trainer = SentenceUQTrainer(
        model=bayes,
        lr=lr,
        num_epochs=num_epochs,
        eval_every=eval_every,
        pd_check_every=pd_check_every,
        device=device,
    )
    print(
        f"Training:    epochs={num_epochs}, lr={lr}, fisher_iters={num_fisher_iters}, "
        f"eval_every={eval_every}, pd_check_every={pd_check_every}, device={device}"
    )

    data = trainer.prepare_data(
        split_file=split_file,
        generations_dirs=generations_dirs,
        cache_dirs=cache_dirs,
        processed_dirs=processed_dirs,
    )
    n_train = len(data.get("train", []))
    n_val = len(data.get("val", []))
    n_test = len(data.get("test", []))
    print(
        f"Sentences:   train={n_train}, val={n_val}, test={n_test} "
        f"(skipped m_j=0 sentences are kept in records but ignored by NLL)"
    )

    if n_train == 0:
        print(
            "error: no training sentences after joining splits with "
            "generations / cache / annotations. Check that Phases 1-1/1-3/1-4 "
            "have been run for this setup.",
            file=sys.stderr,
        )
        return 2

    history = trainer.fit(
        train_data=data["train"],
        val_data=data.get("val") or None,
        test_data=data.get("test") or None,
    )

    save_trained_model(
        results_dir / "trained_model.pt",
        theta_hat=history["theta_hat"],
        Sigma_hat=history["Sigma_hat"],
        feature_params=feature_params,
        extra={
            "setup": setup,
            "model_dims": {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "projection_dim": projection_dim,
            },
            "training": {
                "num_epochs": num_epochs,
                "lr": lr,
                "num_fisher_iters": num_fisher_iters,
                "prior_sigma_init": prior_sigma_init,
                "eval_every": eval_every,
                "pd_check_every": pd_check_every,
            },
            "split_file": str(split_file),
        },
    )

    with open(results_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(_serializable_history(history), f, indent=2)

    summary = {
        "setup": setup,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "val_metrics_last": (history["val_metrics"][-1] if history["val_metrics"] else None),
        "test_metrics": history.get("test_metrics"),
    }
    with open(results_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved trained model + history to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
