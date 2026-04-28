"""Tests for state.py SQLite tracking."""

from __future__ import annotations

import pytest

from obsidian_llm_wiki.models import (
    ItemMentionRecord,
    KnowledgeItemRecord,
    RawNoteRecord,
    WikiArticleRecord,
)
from obsidian_llm_wiki.state import _CURRENT_SCHEMA_VERSION, StateDB


@pytest.fixture
def db(tmp_path):
    return StateDB(tmp_path / ".olw" / "state.db")


def test_upsert_and_get_raw(db):
    r = RawNoteRecord(path="raw/note.md", content_hash="abc123", status="new")
    db.upsert_raw(r)
    got = db.get_raw("raw/note.md")
    assert got is not None
    assert got.content_hash == "abc123"
    assert got.status == "new"


def test_dedup_by_hash(db):
    r1 = RawNoteRecord(path="raw/a.md", content_hash="samehash", status="new")
    r2 = RawNoteRecord(path="raw/b.md", content_hash="samehash", status="new")
    db.upsert_raw(r1)
    db.upsert_raw(r2)
    existing = db.get_raw_by_hash("samehash")
    assert existing is not None
    # Should find first occurrence
    assert existing.path == "raw/a.md"


def test_mark_ingested(db):
    db.upsert_raw(RawNoteRecord(path="raw/n.md", content_hash="h1"))
    db.mark_raw_status("raw/n.md", "ingested")
    got = db.get_raw("raw/n.md")
    assert got.status == "ingested"
    assert got.ingested_at is not None


def test_mark_failed_with_error(db):
    db.upsert_raw(RawNoteRecord(path="raw/n.md", content_hash="h2"))
    db.mark_raw_status("raw/n.md", "failed", error="LLM timeout")
    got = db.get_raw("raw/n.md")
    assert got.status == "failed"
    assert got.error == "LLM timeout"


def test_list_raw_by_status(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="new"))
    ingested = db.list_raw(status="ingested")
    assert len(ingested) == 1
    assert ingested[0].path == "raw/a.md"


def test_article_upsert_and_draft(db):
    a = WikiArticleRecord(
        path="wiki/.drafts/test.md",
        title="Test Article",
        sources=["raw/note.md"],
        content_hash="contenthash",
        is_draft=True,
    )
    db.upsert_article(a)
    got = db.get_article("wiki/.drafts/test.md")
    assert got is not None
    assert got.is_draft is True
    assert got.title == "Test Article"


def test_stats_counts_on_disk_drafts(tmp_path):
    db = StateDB(tmp_path / ".olw" / "state.db")
    drafts_dir = tmp_path / "wiki" / ".drafts"
    drafts_dir.mkdir(parents=True)
    (drafts_dir / "Untracked.md").write_text("---\ntitle: Untracked\n---\nBody.")

    assert db.stats(tmp_path)["drafts"] == 1


def test_publish_article(db):
    a = WikiArticleRecord(
        path="wiki/.drafts/test.md",
        title="Test",
        sources=[],
        content_hash="h",
        is_draft=True,
    )
    db.upsert_article(a)
    db.publish_article("wiki/.drafts/test.md", "wiki/test.md")
    got = db.get_article("wiki/test.md")
    assert got is not None
    assert got.is_draft is False


def test_publish_article_republish_no_unique_violation(db):
    """Re-publishing a concept that was already published must not raise UNIQUE error."""
    # Simulate first publish: existing row at wiki/test.md
    existing = WikiArticleRecord(
        path="wiki/test.md",
        title="Test",
        sources=[],
        content_hash="old",
        is_draft=False,
    )
    db.upsert_article(existing)
    # New draft for same concept
    draft = WikiArticleRecord(
        path="wiki/.drafts/test.md",
        title="Test",
        sources=[],
        content_hash="new",
        is_draft=True,
    )
    db.upsert_article(draft)
    # Should not raise sqlite3.IntegrityError
    db.publish_article("wiki/.drafts/test.md", "wiki/test.md")
    got = db.get_article("wiki/test.md")
    assert got is not None
    assert got.is_draft is False


# ── Concepts ──────────────────────────────────────────────────────────────────


def test_upsert_concepts_and_list(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Quantum Computing", "Qubit", "Shor's Algorithm"])
    names = db.list_all_concept_names()
    assert "Quantum Computing" in names
    assert "Qubit" in names
    assert len(names) == 3


def test_upsert_concepts_idempotent(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Quantum Computing", "Quantum Computing"])
    assert db.list_all_concept_names().count("Quantum Computing") == 1


def test_get_sources_for_concept(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Quantum Computing"])
    db.upsert_concepts("raw/b.md", ["Quantum Computing", "Machine Learning"])
    srcs = db.get_sources_for_concept("Quantum Computing")
    assert set(srcs) == {"raw/a.md", "raw/b.md"}
    ml_srcs = db.get_sources_for_concept("Machine Learning")
    assert ml_srcs == ["raw/b.md"]


def test_get_sources_case_insensitive(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Quantum Computing"])
    srcs = db.get_sources_for_concept("quantum computing")
    assert srcs == ["raw/a.md"]


def test_upsert_concepts_backfills_knowledge_items(db):
    db.upsert_concepts("raw/a.md", ["Quantum Computing"])
    item = db.get_item("Quantum Computing")
    assert item is not None
    assert item.kind == "concept"
    assert item.status == "confirmed"


def test_concepts_needing_compile(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="compiled"))
    db.upsert_concepts("raw/a.md", ["New Concept"])
    db.upsert_concepts("raw/b.md", ["Old Concept"])
    db.mark_concept_compile_state("Old Concept", ["raw/b.md"], "compiled")
    needing = db.concepts_needing_compile()
    assert "New Concept" in needing
    assert "Old Concept" not in needing  # source already compiled


def test_concepts_needing_compile_empty_when_all_compiled(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="compiled"))
    db.upsert_concepts("raw/a.md", ["Done Concept"])
    db.mark_concept_compile_state("Done Concept", ["raw/a.md"], "compiled")
    assert db.concepts_needing_compile() == []


def test_knowledge_item_crud(db):
    db.upsert_item(
        KnowledgeItemRecord(
            name="Example Reference",
            kind="ambiguous",
            subtype="named_reference",
            status="candidate",
            confidence=0.6,
        )
    )
    item = db.get_item("example reference")
    assert item is not None
    assert item.name == "Example Reference"
    assert item.subtype == "named_reference"
    assert db.list_items(kind="ambiguous")[0].name == "Example Reference"


def test_item_mentions_idempotent(db):
    mention = ItemMentionRecord(
        item_name="Example Reference",
        source_path="raw/talk.md",
        mention_text="Example Reference",
        context="A note about Example Reference",
        evidence_level="title_supported",
        confidence=0.7,
    )
    db.add_item_mention(mention)
    db.add_item_mention(mention)
    mentions = db.get_item_mentions("Example Reference")
    assert len(mentions) == 1
    assert mentions[0].source_path == "raw/talk.md"


def test_stats(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="new"))
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/.drafts/x.md", title="X", sources=[], content_hash="hx", is_draft=True
        )
    )
    s = db.stats()
    assert s["raw"]["ingested"] == 1
    assert s["raw"]["new"] == 1
    assert s["drafts"] == 1
    assert s["published"] == 0


# ── v0.2: Rejections ─────────────────────────────────────────────────────────


def test_add_and_get_rejection(db):
    db.add_rejection("Quantum Computing", "Too vague", body="draft body here")
    rejections = db.get_rejections("Quantum Computing")
    assert len(rejections) == 1
    assert rejections[0]["feedback"] == "Too vague"
    assert rejections[0]["body"] == "draft body here"
    assert rejections[0]["rejected_at"] is not None


def test_get_rejections_newest_first(db):
    db.add_rejection("Topic", "First feedback")
    db.add_rejection("Topic", "Second feedback")
    rejections = db.get_rejections("Topic")
    assert rejections[0]["feedback"] == "Second feedback"
    assert rejections[1]["feedback"] == "First feedback"


def test_get_rejections_limit(db):
    for i in range(5):
        db.add_rejection("Topic", f"feedback {i}")
    rejections = db.get_rejections("Topic", limit=3)
    assert len(rejections) == 3


def test_rejection_count(db):
    assert db.rejection_count("Topic") == 0
    db.add_rejection("Topic", "bad")
    db.add_rejection("Topic", "still bad")
    assert db.rejection_count("Topic") == 2


def test_rejection_cap_blocks_concept(db):
    """After 5 rejections, concept is auto-blocked."""
    for i in range(5):
        db.add_rejection("Tricky Topic", f"feedback {i}")
    assert db.is_concept_blocked("Tricky Topic")


def test_rejection_below_cap_no_block(db):
    for i in range(4):
        db.add_rejection("Easy Topic", f"feedback {i}")
    assert not db.is_concept_blocked("Easy Topic")


# ── v0.2: Blocked Concepts ────────────────────────────────────────────────────


def test_block_and_unblock(db):
    db.mark_concept_blocked("Some Concept")
    assert db.is_concept_blocked("Some Concept")
    db.unblock_concept("Some Concept")
    assert not db.is_concept_blocked("Some Concept")


def test_list_blocked_concepts(db):
    db.mark_concept_blocked("Alpha")
    db.mark_concept_blocked("Beta")
    blocked = db.list_blocked_concepts()
    assert "Alpha" in blocked
    assert "Beta" in blocked


def test_blocked_concept_excluded_from_needing_compile(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Blocked Topic"])
    db.mark_concept_blocked("Blocked Topic")
    needing = db.concepts_needing_compile()
    assert "Blocked Topic" not in needing


# ── v0.2: Stubs ───────────────────────────────────────────────────────────────


def test_add_and_has_stub(db):
    db.add_stub("Orphan Concept")
    assert db.has_stub("Orphan Concept")
    assert not db.has_stub("Other Concept")


def test_delete_stub(db):
    db.add_stub("Orphan Concept")
    db.delete_stub("Orphan Concept")
    assert not db.has_stub("Orphan Concept")


def test_stub_appears_in_needing_compile(db):
    db.add_stub("Stub Concept")
    needing = db.concepts_needing_compile()
    assert "Stub Concept" in needing


def test_stub_superseded_by_real_source(db):
    """Once real source added for stub's concept, stub is excluded from UNION."""
    db.add_stub("Shared Concept")
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="compiled"))
    db.upsert_concepts("raw/a.md", ["Shared Concept"])
    db.mark_concept_compile_state("Shared Concept", ["raw/a.md"], "compiled")
    # Stub should not appear since real source already compiled
    needing = db.concepts_needing_compile()
    assert "Shared Concept" not in needing


def test_stub_blocked_excluded(db):
    db.add_stub("Stub Topic")
    db.mark_concept_blocked("Stub Topic")
    needing = db.concepts_needing_compile()
    assert "Stub Topic" not in needing


def test_get_stubs(db):
    db.add_stub("A")
    db.add_stub("B")
    stubs = db.get_stubs()
    assert set(stubs) == {"A", "B"}


def test_add_stub_idempotent(db):
    db.add_stub("Topic")
    db.add_stub("Topic")  # should not raise or duplicate
    assert db.get_stubs().count("Topic") == 1


# ── v0.2: get_concepts_for_sources ───────────────────────────────────────────


def test_get_concepts_for_sources(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha", "Beta"])
    db.upsert_concepts("raw/b.md", ["Beta", "Gamma"])
    result = db.get_concepts_for_sources(["raw/a.md"])
    assert set(result) == {"Alpha", "Beta"}


def test_get_concepts_for_sources_empty(db):
    assert db.get_concepts_for_sources([]) == []


# ── v0.2: quality_stats ───────────────────────────────────────────────────────


def test_quality_stats(db):
    db.upsert_raw(
        RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested", quality="high")
    )  # noqa: E501
    db.upsert_raw(
        RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested", quality="low")
    )  # noqa: E501
    db.upsert_raw(
        RawNoteRecord(path="raw/c.md", content_hash="h3", status="ingested", quality="low")
    )  # noqa: E501
    stats = db.quality_stats()
    assert stats["high"] == 1
    assert stats["low"] == 2
    assert stats["medium"] == 0


# ── v0.2: approve_article ─────────────────────────────────────────────────────


def test_approve_article_sets_timestamp(db):
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/test.md", title="Test", sources=[], content_hash="h", is_draft=False
        )
    )
    db.approve_article("wiki/test.md", notes="Looks great")
    art = db.get_article("wiki/test.md")
    assert art is not None
    assert art.approved_at is not None
    assert art.approval_notes == "Looks great"


# ── v0.2: schema versioning ───────────────────────────────────────────────────


def test_schema_version_set_on_fresh_db(db):
    """Fresh DB should have schema_version = current."""
    row = db._conn.execute("SELECT version FROM schema_version").fetchone()
    assert row is not None
    assert row[0] == _CURRENT_SCHEMA_VERSION


def test_schema_version_idempotent(tmp_path):
    """Opening the same DB twice should not change schema_version."""
    path = tmp_path / ".olw" / "state.db"
    db1 = StateDB(path)
    db1.close()
    db2 = StateDB(path)
    row = db2._conn.execute("SELECT version FROM schema_version").fetchone()
    assert row[0] == _CURRENT_SCHEMA_VERSION


def test_legacy_migration_from_v0(tmp_path):
    """DB with no schema_version row and no approved_at column → migrates to v3."""
    import sqlite3

    path = tmp_path / ".olw" / "state.db"
    path.parent.mkdir(parents=True)

    # Simulate a v0.1 DB: tables exist but no schema_version row, no approved_at
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS raw_notes (
            path TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new', summary TEXT, quality TEXT,
            ingested_at TEXT, compiled_at TEXT, error TEXT
        );
        CREATE TABLE IF NOT EXISTS wiki_articles (
            path TEXT PRIMARY KEY, title TEXT NOT NULL,
            sources TEXT NOT NULL, content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            is_draft INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS concepts (
            name TEXT NOT NULL, source_path TEXT NOT NULL,
            PRIMARY KEY (name, source_path)
        );
        CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
    """)
    # Insert version=0 to trigger migration (no approved_at column present)
    conn.execute("INSERT INTO schema_version VALUES (0)")
    conn.commit()
    conn.close()

    # Opening via StateDB should apply v1 + v2 + v3 migrations
    db = StateDB(path)
    row = db._conn.execute("SELECT version FROM schema_version").fetchone()
    assert row[0] == _CURRENT_SCHEMA_VERSION

    # Verify v0.2 tables exist after migration
    tables = {
        r[0]
        for r in db._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "rejections" in tables
    assert "stubs" in tables
    assert "blocked_concepts" in tables

    # Verify approved_at column added to wiki_articles (v2)
    cols = {r[1] for r in db._conn.execute("PRAGMA table_info(wiki_articles)").fetchall()}
    assert "approved_at" in cols
    assert "approval_notes" in cols

    # Verify language column added to raw_notes (v3)
    raw_cols = {r[1] for r in db._conn.execute("PRAGMA table_info(raw_notes)").fetchall()}
    assert "language" in raw_cols


def test_upsert_note_stores_language(db):
    r = RawNoteRecord(path="raw/note.md", content_hash="abc", status="ingested", language="fr")
    db.upsert_raw(r)
    got = db.get_raw("raw/note.md")
    assert got.language == "fr"
    assert db.get_note_language("raw/note.md") == "fr"


def test_upsert_note_language_none(db):
    r = RawNoteRecord(path="raw/note.md", content_hash="abc", status="ingested", language=None)
    db.upsert_raw(r)
    assert db.get_note_language("raw/note.md") is None


def test_get_note_language_missing_path(db):
    assert db.get_note_language("raw/nonexistent.md") is None


def test_replace_concepts_for_source_removes_stale_rows(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha", "Beta"])
    db.mark_concept_compile_state("Alpha", ["raw/a.md"], "compiled")

    db.replace_concepts_for_source("raw/a.md", ["Beta", "Gamma"])

    assert set(db.get_concepts_for_sources(["raw/a.md"])) == {"Beta", "Gamma"}
    assert db.get_compile_state("Alpha", "raw/a.md") is None
    gamma_state = db.get_compile_state("Gamma", "raw/a.md")
    assert gamma_state is not None
    assert gamma_state["status"] == "pending"


def test_mark_concept_compile_state_refreshes_raw_status(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha", "Beta"])

    db.mark_concept_compile_state("Alpha", ["raw/a.md"], "compiled")
    assert db.get_raw("raw/a.md").status == "ingested"

    db.mark_concept_compile_state("Beta", ["raw/a.md"], "compiled")
    assert db.get_raw("raw/a.md").status == "compiled"


def test_list_failed_concepts_returns_distinct_names(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Alpha", "Beta"])
    db.mark_concept_compile_state("Alpha", ["raw/a.md", "raw/b.md"], "failed", error="bad json")
    db.mark_concept_compile_state("Beta", ["raw/b.md"], "failed", error="timeout")

    assert db.list_failed_concepts() == ["Alpha", "Beta"]


def test_ingest_chunk_crud(db):
    db.upsert_ingest_chunk(
        "raw/a.md",
        "hash",
        0,
        3,
        100,
        '{"summary":"s","concepts":[],"suggested_topics":[],"named_references":[],"quality":"high","language":null}',
    )
    rows = db.list_ingest_chunks("raw/a.md", "hash", 3, 100)
    assert len(rows) == 1
    assert rows[0]["chunk_index"] == 0

    db.delete_ingest_chunks("raw/a.md", "hash", 3, 100)
    assert db.list_ingest_chunks("raw/a.md", "hash", 3, 100) == []
