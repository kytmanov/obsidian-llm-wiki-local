"""Tests for compare/corpus.py loader + mode detection + hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_llm_wiki.compare.corpus import (
    CorpusError,
    CorpusMode,
    detect_mode,
    load_corpus,
    notes_set_hash,
)

BUILTIN = Path(__file__).parent / "compare_corpus"


# ── Mode detection ────────────────────────────────────────────────────────────


def test_detect_mode_default_curated():
    assert detect_mode(None, None, None) is CorpusMode.CURATED


def test_detect_mode_byo():
    assert detect_mode(None, Path("/tmp/x"), None) is CorpusMode.BYO


def test_detect_mode_baseline():
    assert detect_mode(None, None, Path("/tmp/vault")) is CorpusMode.BASELINE


def test_detect_mode_conflict_rejected():
    with pytest.raises(CorpusError):
        detect_mode(Path("/a"), Path("/b"), None)


# ── Curated loader ────────────────────────────────────────────────────────────


def test_curated_corpus_loads():
    c = load_corpus(corpus_path=BUILTIN)
    assert c.mode is CorpusMode.CURATED
    assert c.version == "1.0"
    assert len(c.notes) == 10
    assert c.has_queries
    assert c.has_ground_truth_concepts
    assert len(c.notes_set_hash) == 64


def test_curated_corpus_default_builtin_loads():
    c = load_corpus()
    assert c.mode is CorpusMode.CURATED
    assert c.notes


def test_curated_corpus_known_concepts_lowercased():
    c = load_corpus(corpus_path=BUILTIN)
    for n in c.notes:
        assert all(kc == kc.lower() for kc in n.known_concepts)


def test_curated_corpus_references_real_files():
    c = load_corpus(corpus_path=BUILTIN)
    for n in c.notes:
        assert n.path.exists(), f"missing {n.file}"


def test_curated_corpus_sample_n_caps_notes():
    c = load_corpus(corpus_path=BUILTIN, sample_n=3)
    assert len(c.notes) == 3


def test_curated_corpus_queries_valid():
    c = load_corpus(corpus_path=BUILTIN)
    assert len(c.queries) == 8
    ids = [q.id for q in c.queries]
    assert len(set(ids)) == len(ids)  # unique
    refusals = [q for q in c.queries if q.expected_refusal]
    assert len(refusals) == 2


def test_curated_corpus_missing_dir(tmp_path):
    with pytest.raises(CorpusError, match="not found"):
        load_corpus(corpus_path=tmp_path / "nope")


def test_curated_corpus_missing_toml(tmp_path):
    with pytest.raises(CorpusError, match="Missing corpus.toml"):
        load_corpus(corpus_path=tmp_path)


# ── BYO loader ────────────────────────────────────────────────────────────────


def _make_byo_dir(tmp_path: Path, n: int) -> Path:
    d = tmp_path / "byo"
    d.mkdir()
    for i in range(n):
        (d / f"note_{i}.md").write_text(f"# Note {i}\n\nBody {i}.\n")
    return d


def test_byo_loads_three_notes(tmp_path):
    d = _make_byo_dir(tmp_path, 3)
    c = load_corpus(notes_path=d)
    assert c.mode is CorpusMode.BYO
    assert len(c.notes) == 3
    assert not c.has_ground_truth_concepts


def test_byo_rejects_fewer_than_three(tmp_path):
    d = _make_byo_dir(tmp_path, 2)
    with pytest.raises(CorpusError, match="at least 3"):
        load_corpus(notes_path=d)


def test_byo_ignores_hidden_files(tmp_path):
    d = _make_byo_dir(tmp_path, 3)
    (d / ".hidden.md").write_text("x")
    c = load_corpus(notes_path=d)
    assert all(not n.file.startswith(".") for n in c.notes)


def test_byo_rejects_file_arg(tmp_path):
    f = tmp_path / "single.md"
    f.write_text("hi")
    with pytest.raises(CorpusError, match="must be a directory"):
        load_corpus(notes_path=f)


# ── Baseline loader ───────────────────────────────────────────────────────────


def test_baseline_loads_from_raw(tmp_path):
    vault = tmp_path / "vault"
    raw = vault / "raw"
    raw.mkdir(parents=True)
    (vault / "wiki.toml").write_text('[models]\nfast="f"\nheavy="h"\n')
    for i in range(4):
        (raw / f"n{i}.md").write_text(f"# {i}\n")
    c = load_corpus(baseline_vault_path=vault)
    assert c.mode is CorpusMode.BASELINE
    assert len(c.notes) == 4


def test_baseline_requires_raw_dir(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    with pytest.raises(CorpusError, match="missing raw"):
        load_corpus(baseline_vault_path=vault)


def test_baseline_requires_wiki_toml(tmp_path):
    vault = tmp_path / "vault"
    raw = vault / "raw"
    raw.mkdir(parents=True)
    for i in range(3):
        (raw / f"n{i}.md").write_text("x")
    with pytest.raises(CorpusError, match="missing wiki.toml"):
        load_corpus(baseline_vault_path=vault)


def test_baseline_requires_min_notes(tmp_path):
    vault = tmp_path / "vault"
    raw = vault / "raw"
    raw.mkdir(parents=True)
    (vault / "wiki.toml").write_text('[models]\nfast="f"\nheavy="h"\n')
    (raw / "only.md").write_text("x")
    with pytest.raises(CorpusError, match="at least 3"):
        load_corpus(baseline_vault_path=vault)


# ── notes_set_hash ────────────────────────────────────────────────────────────


def test_notes_hash_stable_across_load(tmp_path):
    d = _make_byo_dir(tmp_path, 3)
    h1 = load_corpus(notes_path=d).notes_set_hash
    h2 = load_corpus(notes_path=d).notes_set_hash
    assert h1 == h2


def test_notes_hash_changes_on_edit(tmp_path):
    d = _make_byo_dir(tmp_path, 3)
    h1 = load_corpus(notes_path=d).notes_set_hash
    (d / "note_0.md").write_text("edited")
    h2 = load_corpus(notes_path=d).notes_set_hash
    assert h1 != h2


def test_notes_hash_changes_on_rename(tmp_path):
    d = _make_byo_dir(tmp_path, 3)
    h1 = load_corpus(notes_path=d).notes_set_hash
    (d / "note_0.md").rename(d / "renamed.md")
    h2 = load_corpus(notes_path=d).notes_set_hash
    assert h1 != h2


def test_notes_hash_deterministic_regardless_of_fs_order(tmp_path):
    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    # Create in opposite order; hash should match because inputs are sorted.
    (d1 / "aaa.md").write_text("1")
    (d1 / "bbb.md").write_text("2")
    (d1 / "ccc.md").write_text("3")
    (d2 / "ccc.md").write_text("3")
    (d2 / "bbb.md").write_text("2")
    (d2 / "aaa.md").write_text("1")
    from obsidian_llm_wiki.compare.corpus import Note

    def _notes_of(d: Path) -> list[Note]:
        return load_corpus(notes_path=d).notes

    assert notes_set_hash(_notes_of(d1)) == notes_set_hash(_notes_of(d2))
