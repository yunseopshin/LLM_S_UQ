"""Dataset preparation and split generation for Bayesian sentence-level UQ.

Two benchmarks are supported:

- **FActScore-Bio** (Min et al., 2023): 183 biography entities. The canonical
  prompt template is ``"Tell me a bio of {entity}."``.
- **LongFact-Objects** (Wei et al., 2024): 38 topics × 30 prompts = 1140 prompts.

Three experimental setups produce reproducible train/val/test splits saved as
JSON under ``data/splits/``.

Element schema (each item in a split list)::

    # FActScore-Bio
    {"dataset": "factscore_bio", "entity": str, "prompt": str, "prompt_idx": int}

    # LongFact-Objects
    {"dataset": "longfact",      "topic":  str, "prompt": str, "prompt_idx": int}
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Prompt template applied to every FActScore-Bio entity.
FACTSCORE_PROMPT_TEMPLATE: str = "Tell me a bio of {entity}."

#: Size of the fixed FActScore-Bio test set used in Setups 1 and 2.
HAN_TEST_SIZE: int = 30

#: Local fallback paths (the in-repo mirrors of the upstream data).
_LOCAL_FACTSCORE_ENTITIES: tuple[str, ...] = (
    "/home/ys971217/LLM_S_UQ/long-form-factuality-main/third_party/factscore/"
    "labeled_data/prompt_entities.txt",
    "/home/ys971217/LLM_S_UQ/fact-probe-main/fact-probe-main/"
    "factuality_benchmarks/FActScore/data/labeled/prompt_entities.txt",
)
_LOCAL_LONGFACT_DIRS: tuple[str, ...] = (
    "/home/ys971217/LLM_S_UQ/long-form-factuality-main/longfact/"
    "longfact-objects_gpt4_01-12-2024_noduplicates",
    "/home/ys971217/LLM_S_UQ/fact-probe-main/fact-probe-main/"
    "factuality_benchmarks/long-form-factuality/longfact/"
    "longfact-objects_gpt4_01-12-2024_noduplicates",
)

_LONGFACT_FILE_PREFIX: str = "longfact-objects_"
_LONGFACT_FILE_SUFFIX: str = ".jsonl"

#: Supported setup identifiers.
SETUPS: tuple[int, ...] = (1, 2, 3)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_existing(paths: Iterable[str]) -> Optional[str]:
    """Return the first path in ``paths`` that exists on disk, else ``None``."""
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _ensure_dir(path: str | os.PathLike) -> Path:
    """Create directory if it doesn't exist and return as ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: str | os.PathLike, obj: Any) -> None:
    """Write ``obj`` to ``path`` as pretty JSON (utf-8, 2-space indent)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_json(path: str | os.PathLike) -> Any:
    """Read JSON from ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# FActScore-Bio
# ---------------------------------------------------------------------------


def prepare_factscore_bio(
    save_dir: str | os.PathLike = "data/raw/factscore_bio",
    source_path: Optional[str | os.PathLike] = None,
) -> Path:
    """Materialise the 183 FActScore-Bio entities under ``save_dir``.

    Outputs
    -------
    - ``entities.json``: list of ``{"entity", "prompt"}`` dicts, in source order.
    - ``test_entities_han.json``: the first :data:`HAN_TEST_SIZE` entities of
      ``entities.json`` (deterministic, file-order). Han et al. (2025) use
      5-fold stratified CV rather than a fixed test set, so we adopt a
      deterministic fixed 30-entity split for reproducibility.

    Parameters
    ----------
    save_dir : path-like
        Destination directory; created if it doesn't exist.
    source_path : path-like, optional
        Override path to a ``prompt_entities.txt`` with one entity per line.
        If ``None``, falls back to known local mirrors.

    Returns
    -------
    Path
        The ``save_dir`` as a ``Path``.

    Raises
    ------
    FileNotFoundError
        If no local mirror can be located and ``source_path`` is not provided.
    """
    save_path = _ensure_dir(save_dir)

    src = (
        str(source_path)
        if source_path is not None
        else _find_existing(_LOCAL_FACTSCORE_ENTITIES)
    )
    if src is None or not os.path.exists(src):
        raise FileNotFoundError(
            "Could not locate FActScore prompt_entities.txt. Tried: "
            + ", ".join(_LOCAL_FACTSCORE_ENTITIES)
            + ". Pass source_path explicitly or download from "
            "https://github.com/shmsw25/FActScore."
        )

    with open(src, "r", encoding="utf-8") as f:
        raw_entities = [ln.strip() for ln in f if ln.strip()]

    entities = [
        {"entity": e, "prompt": FACTSCORE_PROMPT_TEMPLATE.format(entity=e)}
        for e in raw_entities
    ]
    _write_json(save_path / "entities.json", entities)
    _write_json(save_path / "test_entities_han.json", entities[:HAN_TEST_SIZE])
    return save_path


def load_factscore_bio_entities(
    save_dir: str | os.PathLike = "data/raw/factscore_bio",
) -> list[dict[str, str]]:
    """Load the prepared FActScore-Bio entity list."""
    return _read_json(Path(save_dir) / "entities.json")


def load_han_test_entities(
    save_dir: str | os.PathLike = "data/raw/factscore_bio",
) -> list[dict[str, str]]:
    """Load the fixed Han-style 30-entity test set."""
    return _read_json(Path(save_dir) / "test_entities_han.json")


# ---------------------------------------------------------------------------
# LongFact-Objects
# ---------------------------------------------------------------------------


def prepare_longfact_objects(
    save_dir: str | os.PathLike = "data/raw/longfact",
    source_dir: Optional[str | os.PathLike] = None,
) -> Path:
    """Materialise LongFact-Objects (38 topics × 30 prompts) under ``save_dir``.

    Outputs
    -------
    - ``prompts.json``: ``[{"topic", "prompt", "prompt_idx"}, ...]`` sorted by
      ``(topic, prompt_idx)``.
    - ``topics.json``: sorted list of the 38 topic strings.

    The topic name is derived from the JSONL filename: e.g.
    ``longfact-objects_chemistry.jsonl`` → ``"chemistry"``.

    Parameters
    ----------
    save_dir : path-like
        Destination directory.
    source_dir : path-like, optional
        Directory containing ``longfact-objects_<topic>.jsonl`` files. Falls
        back to known local mirrors when ``None``.

    Returns
    -------
    Path
        The ``save_dir`` as a ``Path``.
    """
    save_path = _ensure_dir(save_dir)

    src = (
        str(source_dir)
        if source_dir is not None
        else _find_existing(_LOCAL_LONGFACT_DIRS)
    )
    if src is None or not os.path.isdir(src):
        raise FileNotFoundError(
            "Could not locate longfact-objects JSONL directory. Tried: "
            + ", ".join(_LOCAL_LONGFACT_DIRS)
            + ". Pass source_dir explicitly or download from "
            "https://github.com/google-deepmind/long-form-factuality."
        )

    jsonl_files = sorted(
        f
        for f in os.listdir(src)
        if f.startswith(_LONGFACT_FILE_PREFIX) and f.endswith(_LONGFACT_FILE_SUFFIX)
    )

    topics: list[str] = []
    prompts: list[dict[str, Any]] = []
    for fname in jsonl_files:
        topic = fname[len(_LONGFACT_FILE_PREFIX) : -len(_LONGFACT_FILE_SUFFIX)]
        topics.append(topic)
        rows = _read_jsonl(os.path.join(src, fname))
        for idx, row in enumerate(rows):
            prompt_text = row.get("prompt")
            if not isinstance(prompt_text, str) or not prompt_text:
                continue
            prompts.append(
                {"topic": topic, "prompt": prompt_text, "prompt_idx": idx}
            )

    _write_json(save_path / "topics.json", topics)
    _write_json(save_path / "prompts.json", prompts)
    return save_path


def load_longfact_topics(
    save_dir: str | os.PathLike = "data/raw/longfact",
) -> list[str]:
    """Load the LongFact-Objects topic list."""
    return _read_json(Path(save_dir) / "topics.json")


def load_longfact_prompts(
    save_dir: str | os.PathLike = "data/raw/longfact",
) -> list[dict[str, Any]]:
    """Load the LongFact-Objects prompt list."""
    return _read_json(Path(save_dir) / "prompts.json")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _factscore_record(entity_dict: dict[str, str], prompt_idx: int) -> dict[str, Any]:
    return {
        "dataset": "factscore_bio",
        "entity": entity_dict["entity"],
        "prompt": entity_dict["prompt"],
        "prompt_idx": prompt_idx,
    }


def _longfact_record(prompt_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "longfact",
        "topic": prompt_dict["topic"],
        "prompt": prompt_dict["prompt"],
        "prompt_idx": prompt_dict["prompt_idx"],
    }


def _group_longfact_by_topic(
    prompts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for p in prompts:
        by_topic.setdefault(p["topic"], []).append(p)
    return by_topic


def create_split(
    dataset: str,
    setup: int,
    factscore_dir: str | os.PathLike = "data/raw/factscore_bio",
    longfact_dir: str | os.PathLike = "data/raw/longfact",
    save_path: Optional[str | os.PathLike] = None,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Build a train/val/test split for the given experimental setup.

    Parameters
    ----------
    dataset : {"factscore_bio", "longfact"}
        Primary dataset name. Used as a sanity hint; the actual datasets
        consumed are determined by ``setup`` (see the table below).
    setup : {1, 2, 3}
        - **Setup 1** — Cross-domain: train on LongFact (34 topics × 30 = 1020),
          val on 4 held-out LongFact topics (120 prompts), test on the
          Han 30-entity FActScore-Bio set.
        - **Setup 2** — In-domain Bio: 30 fixed Han test entities, then a
          seeded shuffle of the remaining 153 → 120 train / 33 val.
        - **Setup 3** — Multi-domain LongFact: seeded shuffle of 38 topics
          → 26 train / 4 val / 8 test, all 30 prompts per topic kept together.
    factscore_dir, longfact_dir : path-like
        Locations of the prepared raw datasets.
    save_path : path-like, optional
        If provided, the split is also written as JSON to this path.
    seed : int
        Seed for any random shuffling. Defaults to 42 (project-wide default).

    Returns
    -------
    dict
        ``{"setup": int, "seed": int, "train": [...], "val": [...], "test": [...]}``.

    Raises
    ------
    ValueError
        If ``setup`` is not in :data:`SETUPS` or ``dataset`` is unknown.
    """
    if setup not in SETUPS:
        raise ValueError(f"setup must be one of {SETUPS}, got {setup}")
    if dataset not in ("factscore_bio", "longfact"):
        raise ValueError(
            f"dataset must be 'factscore_bio' or 'longfact', got {dataset!r}"
        )

    rng = random.Random(seed)

    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]

    if setup == 1:
        longfact_prompts = load_longfact_prompts(longfact_dir)
        topics = load_longfact_topics(longfact_dir)
        han_entities = load_han_test_entities(factscore_dir)

        topics_sorted = sorted(topics)
        shuffled = topics_sorted[:]
        rng.shuffle(shuffled)
        val_topics = set(shuffled[:4])

        by_topic = _group_longfact_by_topic(longfact_prompts)
        train = [
            _longfact_record(p)
            for t in topics_sorted
            if t not in val_topics
            for p in by_topic.get(t, [])
        ]
        val = [
            _longfact_record(p)
            for t in topics_sorted
            if t in val_topics
            for p in by_topic.get(t, [])
        ]
        test = [_factscore_record(e, i) for i, e in enumerate(han_entities)]

    elif setup == 2:
        entities = load_factscore_bio_entities(factscore_dir)
        han_entities = load_han_test_entities(factscore_dir)
        han_names = {e["entity"] for e in han_entities}

        remaining = [e for e in entities if e["entity"] not in han_names]
        rng.shuffle(remaining)
        train_entities = remaining[:120]
        val_entities = remaining[120:]

        train = [_factscore_record(e, i) for i, e in enumerate(train_entities)]
        val = [_factscore_record(e, i) for i, e in enumerate(val_entities)]
        test = [_factscore_record(e, i) for i, e in enumerate(han_entities)]

    else:  # setup == 3
        longfact_prompts = load_longfact_prompts(longfact_dir)
        topics = load_longfact_topics(longfact_dir)

        topics_sorted = sorted(topics)
        shuffled = topics_sorted[:]
        rng.shuffle(shuffled)
        train_topics = shuffled[:26]
        val_topics = shuffled[26:30]
        test_topics = shuffled[30:]

        by_topic = _group_longfact_by_topic(longfact_prompts)

        def _collect(ts: list[str]) -> list[dict[str, Any]]:
            return [_longfact_record(p) for t in ts for p in by_topic.get(t, [])]

        train = _collect(train_topics)
        val = _collect(val_topics)
        test = _collect(test_topics)

    split = {
        "setup": setup,
        "seed": seed,
        "train": train,
        "val": val,
        "test": test,
    }

    if save_path is not None:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        _write_json(save_p, split)

    return split


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


_SETUP_TITLES: dict[int, str] = {
    1: "Cross-domain (LongFact → FActScore-Bio)",
    2: "In-domain Biography (FActScore-Bio)",
    3: "Multi-domain (LongFact topic-level)",
}


def summarise_split(split: dict[str, Any]) -> str:
    """Return a human-readable multi-line summary of a split dict."""
    setup = split.get("setup")
    title = _SETUP_TITLES.get(setup, f"setup {setup}")
    lines = [f"=== Setup {setup}: {title} ==="]
    for key in ("train", "val", "test"):
        items = split.get(key) or []
        lines.append(f"{key.capitalize():5s}: {len(items):4d} prompts")
    lines.append(f"Seed:  {split.get('seed')}")
    return "\n".join(lines)


def split_save_filename(setup: int) -> str:
    """Return the canonical filename for a setup's split JSON."""
    return f"setup_{setup}.json"


# ---------------------------------------------------------------------------
# Convenience for the CLI script
# ---------------------------------------------------------------------------


def prepare_all_and_split(
    setup: int,
    factscore_dir: str | os.PathLike = "data/raw/factscore_bio",
    longfact_dir: str | os.PathLike = "data/raw/longfact",
    splits_dir: str | os.PathLike = "data/splits",
    seed: int = 42,
    force_redownload: bool = False,
) -> dict[str, Any]:
    """One-shot: ensure raw datasets exist, then build & save the chosen split.

    Skips dataset preparation if the canonical output files already exist
    (unless ``force_redownload`` is True).
    """
    fs_dir = Path(factscore_dir)
    lf_dir = Path(longfact_dir)

    fs_done = (fs_dir / "entities.json").exists() and (
        fs_dir / "test_entities_han.json"
    ).exists()
    if force_redownload or not fs_done:
        prepare_factscore_bio(fs_dir)

    lf_done = (lf_dir / "prompts.json").exists() and (lf_dir / "topics.json").exists()
    if force_redownload or not lf_done:
        prepare_longfact_objects(lf_dir)

    dataset_name = "factscore_bio" if setup == 2 else "longfact"
    save_path = Path(splits_dir) / split_save_filename(setup)
    split = create_split(
        dataset=dataset_name,
        setup=setup,
        factscore_dir=fs_dir,
        longfact_dir=lf_dir,
        save_path=save_path,
        seed=seed,
    )
    return split


__all__ = [
    "FACTSCORE_PROMPT_TEMPLATE",
    "HAN_TEST_SIZE",
    "SETUPS",
    "prepare_factscore_bio",
    "prepare_longfact_objects",
    "load_factscore_bio_entities",
    "load_han_test_entities",
    "load_longfact_prompts",
    "load_longfact_topics",
    "create_split",
    "prepare_all_and_split",
    "split_save_filename",
    "summarise_split",
]
