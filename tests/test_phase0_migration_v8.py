"""Tests for the v8 schema migration (Phase 0).

Verifies:
- Fresh DB creation sets schema_version to 8 with all v8 columns.
- A v7 DB shape (with all v7-era tables and rows) upgrades to v8 cleanly.
- All v7-era indexes survive the upgrade.
- Existing Pydantic models (RawNoteRecord) still load from the new schema.
- Existing v0.8 functionality is unaffected.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from obsidian_llm_wiki.models import RawNoteRecord
from obsidian_llm_wiki.state import StateDB


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_indexes(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,)
    ).fetchall()
    return {row[0] for row in rows if not row[0].startswith("sqlite_")}


V8_NEW_COLUMNS = {
    "source_type",
    "origin_uri",
    "imported_at",
    "normalized_hash",
    "extractor_version",
    "prompt_version",
}

V7_ERA_TABLES = {
    "schema_version",
    "raw_notes",
    "concepts",
    "wiki_articles",
    "rejections",
    "stubs",
    "blocked_concepts",
    "concept_aliases",
    "knowledge_items",
    "item_mentions",
    "ingest_chunks",
    "concept_compile_state",
}

V7_ERA_INDEXES = {
    "idx_raw_hash",
    "idx_raw_status",
    "idx_concept_name",
    "idx_ingest_chunks_source",
    "idx_concept_compile_status",
    "idx_concept_compile_name",
    "idx_rejections_concept",
    "idx_alias_lookup",
    "idx_items_kind",
    "idx_items_status",
    "idx_mentions_item",
    "idx_mentions_source",
    "idx_wiki_articles_kind",
    "idx_wiki_articles_question_hash",
}


def test_fresh_db_is_at_v8(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert version == 8

    cols = _table_columns(conn, "raw_notes")
    assert V8_NEW_COLUMNS.issubset(cols), f"missing columns: {V8_NEW_COLUMNS - cols}"
    conn.close()


def test_fresh_db_has_all_v7_tables(tmp_path: Path) -> None:
    """Fresh-DB creation must produce every v7-era table (no regression)."""
    db_path = tmp_path / "state.db"
    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = {row[0] for row in rows}
    missing = V7_ERA_TABLES - tables
    assert not missing, f"missing tables in fresh DB: {missing}"
    conn.close()


def test_fresh_db_default_source_type(tmp_path: Path) -> None:
    db_path = tmp_path / "state.db"
    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO raw_notes (path, content_hash, status) VALUES (?, ?, ?)",
        ("raw/note.md", "abc123", "new"),
    )
    conn.commit()

    row = conn.execute(
        "SELECT source_type, origin_uri FROM raw_notes WHERE path = 'raw/note.md'"
    ).fetchone()
    assert row[0] == "notes"
    assert row[1] is None
    conn.close()


def _build_v7_db(db_path: Path) -> None:
    """Build a realistic v7-shaped DB with ALL v7-era tables + indexes."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (
            id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL
        );
        CREATE TABLE raw_notes (
            path        TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'new',
            summary     TEXT,
            quality     TEXT,
            language    TEXT,
            ingested_at TEXT,
            compiled_at TEXT,
            error       TEXT
        );
        CREATE TABLE concepts (
            name TEXT NOT NULL, source_path TEXT NOT NULL,
            PRIMARY KEY (name, source_path)
        );
        CREATE TABLE wiki_articles (
            path TEXT PRIMARY KEY, title TEXT NOT NULL, sources TEXT NOT NULL,
            content_hash TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, is_draft INTEGER NOT NULL DEFAULT 1,
            approved_at TEXT, approval_notes TEXT,
            kind TEXT NOT NULL DEFAULT 'concept', question_hash TEXT,
            synthesis_sources TEXT, synthesis_source_hashes TEXT
        );
        CREATE TABLE rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT NOT NULL, feedback TEXT NOT NULL,
            rejected_body TEXT, rejected_at TEXT NOT NULL
        );
        CREATE TABLE stubs (
            concept TEXT PRIMARY KEY, created_at TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'auto'
        );
        CREATE TABLE blocked_concepts (
            concept TEXT PRIMARY KEY, blocked_at TEXT NOT NULL
        );
        CREATE TABLE concept_aliases (
            concept_name TEXT NOT NULL, alias TEXT NOT NULL,
            PRIMARY KEY (concept_name, alias)
        );
        CREATE TABLE knowledge_items (
            name TEXT PRIMARY KEY, kind TEXT NOT NULL DEFAULT 'ambiguous',
            subtype TEXT, status TEXT NOT NULL DEFAULT 'candidate',
            confidence REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE item_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT NOT NULL, source_path TEXT NOT NULL,
            mention_text TEXT NOT NULL, context TEXT,
            evidence_level TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            UNIQUE(item_name, source_path, mention_text, evidence_level)
        );
        CREATE TABLE ingest_chunks (
            source_path TEXT NOT NULL, content_hash TEXT NOT NULL,
            chunk_index INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
            chunk_size INTEGER NOT NULL, checkpoint_schema INTEGER NOT NULL,
            result_json TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (
                source_path,
                content_hash,
                chunk_index,
                chunk_count,
                chunk_size,
                checkpoint_schema
            )
        );
        CREATE TABLE concept_compile_state (
            concept_name TEXT NOT NULL, source_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', error TEXT,
            compiled_at TEXT, updated_at TEXT NOT NULL,
            PRIMARY KEY (concept_name, source_path),
            CHECK (
                status IN (
                    'pending', 'failed', 'compiled', 'deferred_draft', 'deferred_manual_edit'
                )
            )
        );

        CREATE INDEX idx_raw_hash ON raw_notes(content_hash);
        CREATE INDEX idx_raw_status ON raw_notes(status);
        CREATE INDEX idx_concept_name ON concepts(name);
        CREATE INDEX idx_ingest_chunks_source ON ingest_chunks(source_path, content_hash);
        CREATE INDEX idx_concept_compile_status ON concept_compile_state(status, source_path);
        CREATE INDEX idx_concept_compile_name ON concept_compile_state(lower(concept_name));
        CREATE INDEX idx_rejections_concept ON rejections(concept);
        CREATE INDEX idx_alias_lookup ON concept_aliases(lower(alias));
        CREATE INDEX idx_items_kind ON knowledge_items(kind);
        CREATE INDEX idx_items_status ON knowledge_items(status);
        CREATE INDEX idx_mentions_item ON item_mentions(item_name);
        CREATE INDEX idx_mentions_source ON item_mentions(source_path);
        CREATE INDEX idx_wiki_articles_kind ON wiki_articles(kind);
        CREATE UNIQUE INDEX idx_wiki_articles_question_hash
            ON wiki_articles(question_hash) WHERE question_hash IS NOT NULL;

        INSERT INTO schema_version (id, version) VALUES (1, 7);

        INSERT INTO raw_notes (path, content_hash, status, summary, quality, language)
            VALUES ('raw/old.md', 'h1', 'ingested', 'pre-existing note', 'high', 'en');
        INSERT INTO concepts (name, source_path) VALUES ('Test Concept', 'raw/old.md');
        INSERT INTO wiki_articles
            (path, title, sources, content_hash, created_at, updated_at, is_draft, kind)
            VALUES ('wiki/Test.md', 'Test Concept', '["raw/old.md"]', 'wh1',
                    '2024-01-01T00:00:00', '2024-01-01T00:00:00', 0, 'concept');
        """
    )
    conn.commit()
    conn.close()


def test_v7_to_v8_upgrade_preserves_rows(tmp_path: Path) -> None:
    """Simulate a realistic v7 DB, then verify v8 upgrade preserves rows."""
    db_path = tmp_path / "state.db"
    _build_v7_db(db_path)

    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert version == 8

    cols = _table_columns(conn, "raw_notes")
    assert V8_NEW_COLUMNS.issubset(cols)

    row = conn.execute(
        "SELECT source_type, summary FROM raw_notes WHERE path = 'raw/old.md'"
    ).fetchone()
    assert row[0] == "notes"
    assert row[1] == "pre-existing note"

    n_concepts = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    assert n_concepts == 1
    n_articles = conn.execute("SELECT COUNT(*) FROM wiki_articles").fetchone()[0]
    assert n_articles == 1
    conn.close()


def test_v7_to_v8_upgrade_preserves_indexes(tmp_path: Path) -> None:
    """All v7-era indexes must survive the upgrade to v8."""
    db_path = tmp_path / "state.db"
    _build_v7_db(db_path)

    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    all_indexes = set()
    for table in V7_ERA_TABLES:
        all_indexes |= _table_indexes(conn, table)
    missing = V7_ERA_INDEXES - all_indexes
    assert not missing, f"v8 upgrade dropped indexes: {missing}"
    conn.close()


def test_raw_note_record_loads_from_v8_schema(tmp_path: Path) -> None:
    """RawNoteRecord must still validate when loaded from a v8 row dict."""
    db_path = tmp_path / "state.db"
    StateDB(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO raw_notes (path, content_hash, status, summary, quality, language) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("raw/x.md", "abc", "ingested", "summary", "high", "en"),
    )
    conn.commit()

    row = conn.execute("SELECT * FROM raw_notes WHERE path = 'raw/x.md'").fetchone()
    assert row is not None
    row_dict = dict(row)

    record = RawNoteRecord.model_validate(row_dict)
    assert record.path == "raw/x.md"
    assert record.summary == "summary"
    conn.close()


def test_existing_test_suite_unaffected(tmp_path: Path) -> None:
    """Smoke check: StateDB still constructs without error against tmp_path."""
    db = StateDB(tmp_path / "state.db")
    assert db is not None
