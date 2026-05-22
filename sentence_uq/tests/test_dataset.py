"""Tests for ``src.data.dataset`` — Phase 1-0 dataset preparation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    FACTSCORE_PROMPT_TEMPLATE,
    HAN_TEST_SIZE,
    SETUPS,
    create_split,
    load_factscore_bio_entities,
    load_han_test_entities,
    load_longfact_prompts,
    load_longfact_topics,
    prepare_all_and_split,
    prepare_factscore_bio,
    prepare_longfact_objects,
    split_save_filename,
    summarise_split,
)


# ---------------------------------------------------------------------------
# Synthetic upstream sources (so tests don't depend on the user's environment)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_factscore_source(tmp_path: Path) -> Path:
    """Create a fake prompt_entities.txt with 183 deterministic entities."""
    src = tmp_path / "prompt_entities.txt"
    lines = [f"Entity_{i:03d}" for i in range(183)]
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return src


@pytest.fixture
def synthetic_longfact_source(tmp_path: Path) -> Path:
    """Create a fake longfact-objects directory: 38 topics × 30 prompts."""
    src = tmp_path / "longfact_objects"
    src.mkdir()
    topics = [f"topic_{i:02d}" for i in range(38)]
    for topic in topics:
        path = src / f"longfact-objects_{topic}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for j in range(30):
                f.write(
                    json.dumps({"prompt": f"prompt {j} for {topic}", "canary": "x"})
                    + "\n"
                )
    return src


@pytest.fixture
def prepared_raw(
    tmp_path: Path,
    synthetic_factscore_source: Path,
    synthetic_longfact_source: Path,
) -> tuple[Path, Path]:
    """Prepare both raw datasets under tmp_path and return their directories."""
    fs_dir = tmp_path / "raw" / "factscore_bio"
    lf_dir = tmp_path / "raw" / "longfact"
    prepare_factscore_bio(fs_dir, source_path=synthetic_factscore_source)
    prepare_longfact_objects(lf_dir, source_dir=synthetic_longfact_source)
    return fs_dir, lf_dir


# ---------------------------------------------------------------------------
# FActScore-Bio preparation
# ---------------------------------------------------------------------------


def test_prepare_factscore_bio_writes_expected_files(
    tmp_path: Path, synthetic_factscore_source: Path
) -> None:
    out = tmp_path / "factscore_bio"
    prepare_factscore_bio(out, source_path=synthetic_factscore_source)

    assert (out / "entities.json").exists()
    assert (out / "test_entities_han.json").exists()


def test_prepare_factscore_bio_counts_and_format(
    tmp_path: Path, synthetic_factscore_source: Path
) -> None:
    out = tmp_path / "factscore_bio"
    prepare_factscore_bio(out, source_path=synthetic_factscore_source)

    entities = load_factscore_bio_entities(out)
    han = load_han_test_entities(out)

    assert len(entities) == 183
    assert len(han) == HAN_TEST_SIZE
    # Every record has the expected shape
    for e in entities:
        assert set(e.keys()) == {"entity", "prompt"}
        assert e["prompt"] == FACTSCORE_PROMPT_TEMPLATE.format(entity=e["entity"])
    # Han set is a prefix of entities
    assert han == entities[:HAN_TEST_SIZE]


def test_prepare_factscore_bio_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_factscore_bio(
            tmp_path / "out", source_path=tmp_path / "does_not_exist.txt"
        )


# ---------------------------------------------------------------------------
# LongFact-Objects preparation
# ---------------------------------------------------------------------------


def test_prepare_longfact_writes_expected_files(
    tmp_path: Path, synthetic_longfact_source: Path
) -> None:
    out = tmp_path / "longfact"
    prepare_longfact_objects(out, source_dir=synthetic_longfact_source)
    assert (out / "prompts.json").exists()
    assert (out / "topics.json").exists()


def test_prepare_longfact_counts(
    tmp_path: Path, synthetic_longfact_source: Path
) -> None:
    out = tmp_path / "longfact"
    prepare_longfact_objects(out, source_dir=synthetic_longfact_source)

    topics = load_longfact_topics(out)
    prompts = load_longfact_prompts(out)

    assert len(topics) == 38
    assert len(prompts) == 38 * 30

    # Each prompt has the expected schema and prompt_idx is in [0, 30)
    for p in prompts:
        assert set(p.keys()) == {"topic", "prompt", "prompt_idx"}
        assert 0 <= p["prompt_idx"] < 30
        assert p["topic"] in set(topics)


def test_prepare_longfact_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_longfact_objects(
            tmp_path / "out", source_dir=tmp_path / "missing_dir"
        )


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


def test_setup2_sizes_and_disjointness(prepared_raw: tuple[Path, Path]) -> None:
    fs_dir, lf_dir = prepared_raw
    split = create_split(
        dataset="factscore_bio",
        setup=2,
        factscore_dir=fs_dir,
        longfact_dir=lf_dir,
        seed=42,
    )
    assert len(split["train"]) == 120
    assert len(split["val"]) == 33
    assert len(split["test"]) == HAN_TEST_SIZE
    # All FActScore records
    for part in ("train", "val", "test"):
        for r in split[part]:
            assert r["dataset"] == "factscore_bio"
            assert set(r.keys()) == {"dataset", "entity", "prompt", "prompt_idx"}
    # Entities are pairwise disjoint
    entities_per_part = {
        part: {r["entity"] for r in split[part]} for part in ("train", "val", "test")
    }
    assert entities_per_part["train"].isdisjoint(entities_per_part["val"])
    assert entities_per_part["train"].isdisjoint(entities_per_part["test"])
    assert entities_per_part["val"].isdisjoint(entities_per_part["test"])
    # Union covers all 183 entities
    union = (
        entities_per_part["train"]
        | entities_per_part["val"]
        | entities_per_part["test"]
    )
    assert len(union) == 183


def test_setup2_test_set_is_han(prepared_raw: tuple[Path, Path]) -> None:
    fs_dir, lf_dir = prepared_raw
    split = create_split("factscore_bio", 2, fs_dir, lf_dir, seed=42)
    han = load_han_test_entities(fs_dir)
    assert [r["entity"] for r in split["test"]] == [e["entity"] for e in han]


def test_setup3_topic_level_split(prepared_raw: tuple[Path, Path]) -> None:
    fs_dir, lf_dir = prepared_raw
    split = create_split("longfact", 3, fs_dir, lf_dir, seed=42)

    # 26 train topics × 30, 4 val × 30, 8 test × 30
    assert len(split["train"]) == 26 * 30
    assert len(split["val"]) == 4 * 30
    assert len(split["test"]) == 8 * 30

    # Topic-level disjointness: no topic appears in two parts.
    topics_per_part = {
        part: {r["topic"] for r in split[part]} for part in ("train", "val", "test")
    }
    assert topics_per_part["train"].isdisjoint(topics_per_part["val"])
    assert topics_per_part["train"].isdisjoint(topics_per_part["test"])
    assert topics_per_part["val"].isdisjoint(topics_per_part["test"])
    assert (
        len(topics_per_part["train"]) == 26
        and len(topics_per_part["val"]) == 4
        and len(topics_per_part["test"]) == 8
    )
    # Each topic contributes all 30 prompts to its split.
    for part in ("train", "val", "test"):
        topic_counts: dict[str, int] = {}
        for r in split[part]:
            topic_counts[r["topic"]] = topic_counts.get(r["topic"], 0) + 1
        assert all(c == 30 for c in topic_counts.values())


def test_setup1_cross_domain(prepared_raw: tuple[Path, Path]) -> None:
    fs_dir, lf_dir = prepared_raw
    split = create_split("longfact", 1, fs_dir, lf_dir, seed=42)

    assert len(split["train"]) == 34 * 30
    assert len(split["val"]) == 4 * 30
    assert len(split["test"]) == HAN_TEST_SIZE

    # Train+val together cover all 38 LongFact topics, disjointly.
    train_topics = {r["topic"] for r in split["train"]}
    val_topics = {r["topic"] for r in split["val"]}
    assert train_topics.isdisjoint(val_topics)
    assert len(train_topics) == 34 and len(val_topics) == 4
    # Test records come from FActScore.
    assert all(r["dataset"] == "factscore_bio" for r in split["test"])
    # Train/val from LongFact.
    assert all(r["dataset"] == "longfact" for r in split["train"])
    assert all(r["dataset"] == "longfact" for r in split["val"])


def test_split_is_reproducible(prepared_raw: tuple[Path, Path]) -> None:
    fs_dir, lf_dir = prepared_raw
    a = create_split("factscore_bio", 2, fs_dir, lf_dir, seed=42)
    b = create_split("factscore_bio", 2, fs_dir, lf_dir, seed=42)
    assert a == b


def test_different_seeds_produce_different_splits(
    prepared_raw: tuple[Path, Path],
) -> None:
    fs_dir, lf_dir = prepared_raw
    a = create_split("factscore_bio", 2, fs_dir, lf_dir, seed=42)
    b = create_split("factscore_bio", 2, fs_dir, lf_dir, seed=7)
    # Test set is fixed regardless of seed; train/val ordering should differ.
    assert a["test"] == b["test"]
    assert [r["entity"] for r in a["train"]] != [r["entity"] for r in b["train"]]


def test_create_split_validates_setup(prepared_raw: tuple[Path, Path]) -> None:
    fs_dir, lf_dir = prepared_raw
    with pytest.raises(ValueError):
        create_split("factscore_bio", setup=99, factscore_dir=fs_dir, longfact_dir=lf_dir)
    with pytest.raises(ValueError):
        create_split("unknown", setup=2, factscore_dir=fs_dir, longfact_dir=lf_dir)


def test_create_split_writes_save_file(
    tmp_path: Path, prepared_raw: tuple[Path, Path]
) -> None:
    fs_dir, lf_dir = prepared_raw
    out = tmp_path / "splits" / "setup_2.json"
    split = create_split(
        "factscore_bio", 2, fs_dir, lf_dir, save_path=out, seed=42
    )
    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == split


def test_summarise_split_mentions_setup_and_counts(
    prepared_raw: tuple[Path, Path],
) -> None:
    fs_dir, lf_dir = prepared_raw
    split = create_split("factscore_bio", 2, fs_dir, lf_dir, seed=42)
    text = summarise_split(split)
    assert "Setup 2" in text
    assert "120" in text  # train
    assert "33" in text  # val
    assert "30" in text  # test


# ---------------------------------------------------------------------------
# End-to-end helper
# ---------------------------------------------------------------------------


def test_prepare_all_and_split_setup2(
    tmp_path: Path,
    synthetic_factscore_source: Path,
    synthetic_longfact_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub the local-source lookup so prepare_all_and_split finds our synthetic data.
    import src.data.dataset as ds

    monkeypatch.setattr(ds, "_LOCAL_FACTSCORE_ENTITIES", (str(synthetic_factscore_source),))
    monkeypatch.setattr(ds, "_LOCAL_LONGFACT_DIRS", (str(synthetic_longfact_source),))

    fs_dir = tmp_path / "raw" / "factscore_bio"
    lf_dir = tmp_path / "raw" / "longfact"
    splits_dir = tmp_path / "splits"

    split = prepare_all_and_split(
        setup=2,
        factscore_dir=fs_dir,
        longfact_dir=lf_dir,
        splits_dir=splits_dir,
        seed=42,
    )
    assert (splits_dir / split_save_filename(2)).exists()
    assert split["setup"] == 2
    assert len(split["train"]) == 120
    assert len(split["val"]) == 33
    assert len(split["test"]) == HAN_TEST_SIZE


# ---------------------------------------------------------------------------
# Sanity check on the real local mirrors (skipped if absent)
# ---------------------------------------------------------------------------


def test_local_mirrors_available_if_present(tmp_path: Path) -> None:
    """If the bundled local mirrors exist, prep + split must complete and have
    the canonical sizes. Otherwise the test is skipped."""
    fs_local = "/home/ys971217/LLM_S_UQ/long-form-factuality-main/third_party/factscore/labeled_data/prompt_entities.txt"
    lf_local = "/home/ys971217/LLM_S_UQ/long-form-factuality-main/longfact/longfact-objects_gpt4_01-12-2024_noduplicates"
    if not (os.path.exists(fs_local) and os.path.isdir(lf_local)):
        pytest.skip("Local FActScore / LongFact mirrors not present in this env")

    fs_dir = tmp_path / "raw" / "factscore_bio"
    lf_dir = tmp_path / "raw" / "longfact"
    prepare_factscore_bio(fs_dir)  # no source_path -> use local mirror
    prepare_longfact_objects(lf_dir)

    entities = load_factscore_bio_entities(fs_dir)
    topics = load_longfact_topics(lf_dir)
    prompts = load_longfact_prompts(lf_dir)
    assert len(entities) == 183
    assert len(topics) == 38
    assert len(prompts) == 38 * 30

    for setup in SETUPS:
        split = create_split(
            "factscore_bio" if setup == 2 else "longfact",
            setup,
            fs_dir,
            lf_dir,
            seed=42,
        )
        # Splits must be non-empty in all three parts.
        assert split["train"] and split["val"] and split["test"]
