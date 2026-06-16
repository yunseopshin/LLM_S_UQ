"""Configuration / file I/O helpers for the Bayesian sentence-level UQ project.

The central utility is :func:`load_config`, an inheritance-aware YAML loader.
A config file may declare a single parent via a top-level ``inherits:`` key::

    # configs/setup_2.yaml
    inherits: default.yaml
    dataset:
      setup: 2

The parent is loaded first (recursively) and the child's keys are deep-merged
on top (child wins, nested dicts merged key-by-key). This lets the thin
``setup_{1,2,3}.yaml`` overrides truly "inherit everything else from
default.yaml" — including the Phase 4-1 training block and the Phase 9.5
``logit_reg_lambda`` regulariser — so that ``run_experiment.sh N`` reproduces
the same model the ``--config configs/default.yaml`` path produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Top-level key naming the parent config to inherit from.
INHERIT_KEY = "inherits"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (``override`` wins).

    Nested mappings are merged key-by-key; any non-dict value (or a key present
    only in one side) is taken from ``override`` when present, else ``base``.
    Neither input is mutated; a new dict is returned.

    Parameters
    ----------
    base : dict
        The lower-priority mapping (e.g. the inherited parent config).
    override : dict
        The higher-priority mapping (e.g. the child config).

    Returns
    -------
    dict
        A new merged mapping.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``inherits:`` parent chain.

    Behaviour for a plain config (no ``inherits`` key) is identical to
    ``yaml.safe_load`` followed by ``or {}`` — i.e. a drop-in replacement for
    the per-script ``_load_yaml`` helpers. When an ``inherits`` key is present,
    its value is treated as a path to a parent config (resolved relative to the
    *child* file's directory unless absolute); the parent is loaded recursively
    and this file's keys are deep-merged on top via :func:`deep_merge`. The
    ``inherits`` key is stripped from the returned mapping.

    Parameters
    ----------
    path : str | Path
        Path to the YAML config file.
    _seen : set[Path] | None
        Internal cycle-detection accumulator (resolved paths visited so far).
        Callers should not pass this.

    Returns
    -------
    dict
        The fully-resolved config mapping. Empty dict when the file is empty.

    Raises
    ------
    FileNotFoundError
        If an ``inherits`` parent path does not exist.
    ValueError
        If the YAML does not parse to a mapping, or an inheritance cycle is
        detected.
    """
    import yaml

    path = Path(path)
    resolved = path.resolve()
    seen = set() if _seen is None else _seen
    if resolved in seen:
        raise ValueError(
            f"circular config inheritance detected involving {path}"
        )
    seen.add(resolved)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"config {path} did not parse to a mapping (got {type(cfg).__name__})"
        )

    parent_ref = cfg.pop(INHERIT_KEY, None)
    if parent_ref is None:
        return cfg

    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    if not parent_path.exists():
        raise FileNotFoundError(
            f"config {path} declares '{INHERIT_KEY}: {parent_ref}' but the "
            f"resolved parent {parent_path} does not exist"
        )

    parent_cfg = load_config(parent_path, _seen=seen)
    return deep_merge(parent_cfg, cfg)
