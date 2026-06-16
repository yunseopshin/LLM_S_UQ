"""Tests for the inheritance-aware config loader (``src.utils.io``).

Covers the reproducibility fix: ``setup_{1,2,3}.yaml`` must inherit the
``default.yaml`` training block + ``logit_reg_lambda`` so that
``run_experiment.sh N`` reproduces the promoted production model rather than
silently falling back to argparse defaults (num_epochs=50, logit_reg_lambda=0).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.io import deep_merge, load_config

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS = _PROJECT_ROOT / "configs"


# --------------------------------------------------------------------------- #
# deep_merge                                                                   #
# --------------------------------------------------------------------------- #
def test_deep_merge_child_wins_scalar() -> None:
    """Top-level scalars: override replaces base."""
    base = {"a": 1, "b": 2}
    override = {"b": 3, "c": 4}
    assert deep_merge(base, override) == {"a": 1, "b": 3, "c": 4}


def test_deep_merge_nested_dicts_merge_key_by_key() -> None:
    """Nested mappings merge recursively rather than being replaced wholesale."""
    base = {"dataset": {"setup": 2, "seed": 42, "split_file": None}}
    override = {"dataset": {"setup": 1, "split_file": "x.json"}}
    merged = deep_merge(base, override)
    # seed survives from base; setup + split_file come from override.
    assert merged == {"dataset": {"setup": 1, "seed": 42, "split_file": "x.json"}}


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"d": {"x": 1}}
    override = {"d": {"y": 2}}
    deep_merge(base, override)
    assert base == {"d": {"x": 1}}
    assert override == {"d": {"y": 2}}


# --------------------------------------------------------------------------- #
# load_config — plain (no inheritance)                                         #
# --------------------------------------------------------------------------- #
def test_load_plain_config_equivalent_to_safe_load(tmp_path: Path) -> None:
    cfg_path = tmp_path / "plain.yaml"
    cfg_path.write_text("num_epochs: 7\nlr: 0.5\n", encoding="utf-8")
    assert load_config(cfg_path) == {"num_epochs": 7, "lr": 0.5}


def test_load_empty_config_returns_empty_dict(tmp_path: Path) -> None:
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("", encoding="utf-8")
    assert load_config(cfg_path) == {}


# --------------------------------------------------------------------------- #
# load_config — inheritance                                                    #
# --------------------------------------------------------------------------- #
def test_inherits_pulls_parent_keys_and_child_overrides(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        "num_epochs: 300\nlr: 0.001\nlogit_reg_lambda: 0.01\n"
        "dataset:\n  setup: 2\n  seed: 42\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "inherits: parent.yaml\ndataset:\n  setup: 1\n",
        encoding="utf-8",
    )
    merged = load_config(child)
    assert "inherits" not in merged                 # stripped
    assert merged["num_epochs"] == 300              # inherited
    assert merged["logit_reg_lambda"] == 0.01       # inherited
    assert merged["dataset"]["setup"] == 1          # child override
    assert merged["dataset"]["seed"] == 42          # inherited nested key


def test_inherits_resolves_relative_to_child_dir(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "base.yaml").write_text("x: 1\n", encoding="utf-8")
    child = sub / "c.yaml"
    child.write_text("inherits: ../base.yaml\ny: 2\n", encoding="utf-8")
    assert load_config(child) == {"x": 1, "y": 2}


def test_multilevel_inheritance_chain(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("v: 1\nw: 1\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("inherits: a.yaml\nw: 2\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("inherits: b.yaml\nz: 3\n", encoding="utf-8")
    assert load_config(tmp_path / "c.yaml") == {"v": 1, "w": 2, "z": 3}


def test_missing_parent_raises(tmp_path: Path) -> None:
    child = tmp_path / "c.yaml"
    child.write_text("inherits: nope.yaml\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config(child)


def test_circular_inheritance_raises(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("inherits: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("inherits: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


# --------------------------------------------------------------------------- #
# Integration: the real project configs reproduce production hyperparameters   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("setup", [1, 2, 3])
def test_real_setup_configs_inherit_training_block(setup: int) -> None:
    """setup_{1,2,3}.yaml must inherit the full training block + lambda."""
    cfg = load_config(_CONFIGS / f"setup_{setup}.yaml")
    assert cfg["num_epochs"] == 300
    assert cfg["lr"] == pytest.approx(1.0e-3)
    assert cfg["num_fisher_iters"] == 10
    assert cfg["projection_dim"] == 64
    assert cfg["logit_reg_lambda"] == pytest.approx(1.0e-2)  # production value
    # per-setup overrides still apply
    assert cfg["dataset"]["setup"] == setup
    assert cfg["results_dir"] == f"results/setup_{setup}"
    assert cfg["dataset"]["split_file"] == f"data/splits/setup_{setup}.json"


def test_default_config_carries_production_lambda() -> None:
    cfg = load_config(_CONFIGS / "default.yaml")
    assert cfg["logit_reg_lambda"] == pytest.approx(1.0e-2)
    assert cfg["num_epochs"] == 300
