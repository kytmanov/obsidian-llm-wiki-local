"""
SQLite-backed state tracking for the pipeline.

Tracks raw note processing status and wiki article lineage.
Handles: dedup via content hash, partial failure recovery, resume.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .models import RawNoteRecord, WikiArticleRecord

_CURRENT_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_notes (
    path        TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'new',
    summary     TEXT,
    quality     TEXT,
    ingested_at TEXT,
    compiled_at TEXT,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS concepts (
    name        TEXT NOT NULL,
    source_path TEXT NOT NULL,
    PRIMARY KEY (name, source_path)
);

CREATE TABLE IF NOT EXISTS wiki_articles (
    path         TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    sources      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    is_draft     INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_hash ON raw_notes(content_hash);
CREATE INDEX IF NOT EXISTS idx_raw_status ON raw_notes(status);
CREATE INDEX IF NOT EXISTS idx_concept_name ON concepts(name);
"""

# Each value is a list of SQL statements to run when upgrading to that version.
# Key 2 → upgrade from v1 to v2 (add summary/quality columns).
_MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE raw_notes ADD COLUMN summary TEXT",
        "ALTER TABLE raw_notes ADD COLUMN quality TEXT",
    ],
}


class StateDB:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._ensure_schema_version()

    def _ensure_schema_version(self) -> None:
        """Check schema version, run needed migrations, and guard against future versions."""
        row = self._conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            # Fresh DB or pre-versioning DB — run all applicable migrations then stamp.
            self._run_migrations_from(1)
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (_CURRENT_VERSION,),
            )
            self._conn.commit()
            return

        version = row[0]
        if version > _CURRENT_VERSION:
            raise RuntimeError(
                f"Database schema version {version} is newer than supported"
                f" version {_CURRENT_VERSION}."
                " Please upgrade obsidian-llm-wiki."
            )
        if version < _CURRENT_VERSION:
            self._run_migrations_from(version)
            self._conn.execute("UPDATE schema_version SET version = ?", (_CURRENT_VERSION,))
            self._conn.commit()

    def _run_migrations_from(self, from_version: int) -> None:
        """Apply all migration steps from *from_version* up to _CURRENT_VERSION."""
        for target in range(from_version + 1, _CURRENT_VERSION + 1):
            for stmt in _MIGRATIONS.get(target, []):
                try:
                    self._conn.execute(stmt)
                    self._conn.commit()
                except sqlite3.OperationalError:
                    pass  # e.g. column already exists

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ── Raw Notes ─────────────────────────────────────────────────────────────

    def upsert_raw(self, record: RawNoteRecord) -> None:
        with self._tx():
            self._conn.execute(
                """INSERT INTO raw_notes
                       (path, content_hash, status, summary, quality,
                        ingested_at, compiled_at, error)
                   VALUES
                       (:path, :content_hash, :status, :summary, :quality,
                        :ingested_at, :compiled_at, :error)
                   ON CONFLICT(path) DO UPDATE SET
                       content_hash=excluded.content_hash,
                       status=excluded.status,
                       summary=excluded.summary,
                       quality=excluded.quality,
                       ingested_at=excluded.ingested_at,
                       compiled_at=excluded.compiled_at,
                       error=excluded.error""",
                {
                    "path": record.path,
                    "content_hash": record.content_hash,
                    "status": record.status,
                    "summary": record.summary,
                    "quality": record.quality,
                    "ingested_at": record.ingested_at.isoformat() if record.ingested_at else None,
                    "compiled_at": record.compiled_at.isoformat() if record.compiled_at else None,
                    "error": record.error,
                },
            )

    def get_raw(self, path: str) -> RawNoteRecord | None:
        row = self._conn.execute("SELECT * FROM raw_notes WHERE path = ?", (path,)).fetchone()
        return _row_to_raw(row) if row else None

    def get_raw_by_hash(self, content_hash: str) -> RawNoteRecord | None:
        row = self._conn.execute(
            "SELECT * FROM raw_notes WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return _row_to_raw(row) if row else None

    def list_raw(self, status: str | None = None) -> list[RawNoteRecord]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM raw_notes WHERE status = ?", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM raw_notes").fetchall()
        return [_row_to_raw(r) for r in rows]

    def mark_raw_status(self, path: str, status: str, error: str | None = None) -> None:
        now = datetime.now().isoformat()
        with self._tx():
            if status == "ingested":
                self._conn.execute(
                    "UPDATE raw_notes SET status=?, ingested_at=?, error=NULL WHERE path=?",
                    (status, now, path),
                )
            elif status == "compiled":
                self._conn.execute(
                    "UPDATE raw_notes SET status=?, compiled_at=?, error=NULL WHERE path=?",
                    (status, now, path),
                )
            else:
                self._conn.execute(
                    "UPDATE raw_notes SET status=?, error=? WHERE path=?",
                    (status, error, path),
                )

    # ── Concepts ──────────────────────────────────────────────────────────────

    def upsert_concepts(self, source_path: str, concept_names: list[str]) -> None:
        """Link concept names to a source note (idempotent)."""
        with self._tx():
            for name in concept_names:
                name = name.strip()
                if not name:
                    continue
                self._conn.execute(
                    "INSERT OR IGNORE INTO concepts (name, source_path) VALUES (?, ?)",
                    (name, source_path),
                )

    def list_all_concept_names(self) -> list[str]:
        """All unique canonical concept names, sorted."""
        rows = self._conn.execute("SELECT DISTINCT name FROM concepts ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def get_sources_for_concept(self, name: str) -> list[str]:
        """Raw note paths linked to a concept (case-insensitive match)."""
        rows = self._conn.execute(
            "SELECT DISTINCT source_path FROM concepts WHERE lower(name) = lower(?)",
            (name,),
        ).fetchall()
        return [r[0] for r in rows]

    def concepts_needing_compile(self) -> list[str]:
        """Concepts where any linked source has status='ingested' (pending compile)."""
        rows = self._conn.execute(
            """SELECT DISTINCT c.name
               FROM concepts c
               JOIN raw_notes r ON c.source_path = r.path
               WHERE r.status = 'ingested'
               ORDER BY c.name"""
        ).fetchall()
        return [r[0] for r in rows]

    # ── Wiki Articles ─────────────────────────────────────────────────────────

    def upsert_article(self, record: WikiArticleRecord) -> None:
        with self._tx():
            self._conn.execute(
                """INSERT INTO wiki_articles
                       (path, title, sources, content_hash, created_at, updated_at, is_draft)
                   VALUES (:path, :title, :sources, :content_hash,
                           :created_at, :updated_at, :is_draft)
                   ON CONFLICT(path) DO UPDATE SET
                       title=excluded.title,
                       sources=excluded.sources,
                       content_hash=excluded.content_hash,
                       updated_at=excluded.updated_at,
                       is_draft=excluded.is_draft""",
                {
                    "path": record.path,
                    "title": record.title,
                    "sources": json.dumps(record.sources),
                    "content_hash": record.content_hash,
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                    "is_draft": int(record.is_draft),
                },
            )

    def get_article(self, path: str) -> WikiArticleRecord | None:
        row = self._conn.execute("SELECT * FROM wiki_articles WHERE path = ?", (path,)).fetchone()
        return _row_to_article(row) if row else None

    def list_articles(self, drafts_only: bool = False) -> list[WikiArticleRecord]:
        if drafts_only:
            rows = self._conn.execute("SELECT * FROM wiki_articles WHERE is_draft = 1").fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM wiki_articles").fetchall()
        return [_row_to_article(r) for r in rows]

    def publish_article(self, old_path: str, new_path: str) -> None:
        with self._tx():
            self._conn.execute(
                "UPDATE wiki_articles SET path=?, is_draft=0, updated_at=? WHERE path=?",
                (new_path, datetime.now().isoformat(), old_path),
            )

    def delete_article(self, path: str) -> None:
        with self._tx():
            self._conn.execute("DELETE FROM wiki_articles WHERE path = ?", (path,))

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        raw_counts = {
            row["status"]: row["cnt"]
            for row in self._conn.execute(
                "SELECT status, COUNT(*) as cnt FROM raw_notes GROUP BY status"
            ).fetchall()
        }
        draft_count = self._conn.execute(
            "SELECT COUNT(*) FROM wiki_articles WHERE is_draft=1"
        ).fetchone()[0]
        pub_count = self._conn.execute(
            "SELECT COUNT(*) FROM wiki_articles WHERE is_draft=0"
        ).fetchone()[0]
        return {
            "raw": raw_counts,
            "drafts": draft_count,
            "published": pub_count,
        }


# ── Row converters ────────────────────────────────────────────────────────────


def _row_to_raw(row: sqlite3.Row) -> RawNoteRecord:
    return RawNoteRecord(
        path=row["path"],
        content_hash=row["content_hash"],
        status=row["status"],
        summary=row["summary"] if "summary" in row.keys() else None,
        quality=row["quality"] if "quality" in row.keys() else None,
        ingested_at=datetime.fromisoformat(row["ingested_at"]) if row["ingested_at"] else None,
        compiled_at=datetime.fromisoformat(row["compiled_at"]) if row["compiled_at"] else None,
        error=row["error"],
    )


def _row_to_article(row: sqlite3.Row) -> WikiArticleRecord:
    return WikiArticleRecord(
        path=row["path"],
        title=row["title"],
        sources=json.loads(row["sources"]),
        content_hash=row["content_hash"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        is_draft=bool(row["is_draft"]),
    )
