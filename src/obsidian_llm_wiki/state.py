"""
SQLite-backed state tracking for the pipeline.

Tracks raw note processing status and wiki article lineage.
Handles: dedup via content hash, partial failure recovery, resume.

Schema versioning: schema_version table tracks migration level.
  v1 — initial (summary/quality columns on raw_notes)
  v2 — rejections, stubs, blocked_concepts tables; approved_at/approval_notes on wiki_articles
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .models import RawNoteRecord, WikiArticleRecord

_CURRENT_SCHEMA_VERSION = 2

# Full current schema — idempotent (CREATE IF NOT EXISTS).
# Fresh DBs get all tables + columns from here. Existing DBs use _VERSIONED_MIGRATIONS.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    id      INTEGER PRIMARY KEY CHECK(id = 1),
    version INTEGER NOT NULL
);

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
    path           TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    sources        TEXT NOT NULL,
    content_hash   TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    is_draft       INTEGER NOT NULL DEFAULT 1,
    approved_at    TEXT,
    approval_notes TEXT
);

CREATE TABLE IF NOT EXISTS rejections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    concept       TEXT NOT NULL,
    feedback      TEXT NOT NULL,
    rejected_body TEXT,
    rejected_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stubs (
    concept    TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS blocked_concepts (
    concept    TEXT PRIMARY KEY,
    blocked_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_hash ON raw_notes(content_hash);
CREATE INDEX IF NOT EXISTS idx_raw_status ON raw_notes(status);
CREATE INDEX IF NOT EXISTS idx_concept_name ON concepts(name);
CREATE INDEX IF NOT EXISTS idx_rejections_concept ON rejections(concept);
"""

# Migrations keyed by version they bring the DB to.
_VERSIONED_MIGRATIONS: dict[int, list[str]] = {
    1: [
        # v0.1: add summary/quality columns to raw_notes (were missing in earliest schema)
        "ALTER TABLE raw_notes ADD COLUMN summary TEXT",
        "ALTER TABLE raw_notes ADD COLUMN quality TEXT",
    ],
    2: [
        # v0.2: new tables and columns
        """CREATE TABLE IF NOT EXISTS rejections (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               concept TEXT NOT NULL,
               feedback TEXT NOT NULL,
               rejected_body TEXT,
               rejected_at TEXT NOT NULL
           )""",
        "CREATE INDEX IF NOT EXISTS idx_rejections_concept ON rejections(concept)",
        """CREATE TABLE IF NOT EXISTS stubs (
               concept TEXT PRIMARY KEY,
               created_at TEXT NOT NULL,
               source TEXT NOT NULL DEFAULT 'auto'
           )""",
        """CREATE TABLE IF NOT EXISTS blocked_concepts (
               concept TEXT PRIMARY KEY,
               blocked_at TEXT NOT NULL
           )""",
        "ALTER TABLE wiki_articles ADD COLUMN approved_at TEXT",
        "ALTER TABLE wiki_articles ADD COLUMN approval_notes TEXT",
    ],
}


class StateDB:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Apply schema migrations in version order. Idempotent."""
        # Upgrade schema_version table if it lacks the id column (pre-v0.2 DBs).
        sv_cols = {r[1] for r in self._conn.execute("PRAGMA table_info(schema_version)").fetchall()}
        if sv_cols and "id" not in sv_cols:
            # Read the current version from the old single-column table, then
            # recreate it with the proper constraint.
            old_row = self._conn.execute(
                "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            old_version = old_row[0] if old_row else None
            self._conn.executescript(
                "DROP TABLE schema_version;"
                "CREATE TABLE schema_version "
                "(id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER NOT NULL);"
            )
            if old_version is not None:
                self._conn.execute(
                    "INSERT INTO schema_version (id, version) VALUES (1, ?)", (old_version,)
                )
            self._conn.commit()

        # Use ORDER BY rowid DESC LIMIT 1 to be robust against legacy DBs that
        # accumulated multiple rows before the id=1 uniqueness constraint was added.
        row = self._conn.execute(
            "SELECT version FROM schema_version ORDER BY rowid DESC LIMIT 1"
        ).fetchone()

        if row is None:
            # No version record yet. Determine starting state by inspecting schema:
            # If wiki_articles already has approved_at, this is a fresh DB created
            # by the current _SCHEMA — all tables/columns exist, just record version.
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(wiki_articles)").fetchall()}
            if "approved_at" in cols:
                with self._tx():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, ?)",
                        (_CURRENT_SCHEMA_VERSION,),
                    )
                return
            # Existing DB with no version tracking — start from 0, apply all migrations.
            with self._tx():
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, 0)"
                )
            current_version = 0
        else:
            current_version = row[0]

        if current_version >= _CURRENT_SCHEMA_VERSION:
            return

        for version, stmts in sorted(_VERSIONED_MIGRATIONS.items()):
            if current_version >= version:
                continue
            for stmt in stmts:
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            with self._tx():
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, ?)",
                    (version,),
                )
            current_version = version

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

    def get_concepts_for_sources(self, source_paths: list[str]) -> list[str]:
        """Concept names linked to any of the given source paths."""
        if not source_paths:
            return []
        placeholders = ",".join("?" * len(source_paths))
        rows = self._conn.execute(
            f"SELECT DISTINCT name FROM concepts WHERE source_path IN ({placeholders})",
            source_paths,
        ).fetchall()
        return [r[0] for r in rows]

    def concepts_needing_compile(self) -> list[str]:
        """Concepts where any linked source has status='ingested', plus stub concepts.

        Excludes blocked concepts from both sets.
        """
        rows = self._conn.execute(
            """
            SELECT DISTINCT c.name
            FROM concepts c
            JOIN raw_notes r ON c.source_path = r.path
            WHERE r.status = 'ingested'
              AND lower(c.name) NOT IN (SELECT lower(concept) FROM blocked_concepts)

            UNION

            SELECT s.concept FROM stubs s
            WHERE s.concept NOT IN (
                SELECT DISTINCT c2.name FROM concepts c2
                JOIN raw_notes r2 ON c2.source_path = r2.path
                WHERE r2.status IN ('ingested', 'compiled')
            )
            AND lower(s.concept) NOT IN (SELECT lower(concept) FROM blocked_concepts)

            ORDER BY 1
            """
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
            # Guard: draft row must exist before we touch anything.
            # Without this, the DELETE below would silently destroy the previously
            # published row when the draft was never recorded in wiki_articles.
            if not self._conn.execute(
                "SELECT 1 FROM wiki_articles WHERE path = ?", (old_path,)
            ).fetchone():
                return
            # Remove existing published row at target path (re-publish scenario)
            if old_path != new_path:
                self._conn.execute("DELETE FROM wiki_articles WHERE path = ?", (new_path,))
            self._conn.execute(
                "UPDATE wiki_articles SET path=?, is_draft=0, updated_at=? WHERE path=?",
                (new_path, datetime.now().isoformat(), old_path),
            )

    def approve_article(self, path: str, notes: str = "") -> None:
        """Record approval timestamp and optional notes on a published article."""
        with self._tx():
            self._conn.execute(
                "UPDATE wiki_articles SET approved_at=?, approval_notes=? WHERE path=?",
                (datetime.now().isoformat(), notes or None, path),
            )

    def delete_article(self, path: str) -> None:
        with self._tx():
            self._conn.execute("DELETE FROM wiki_articles WHERE path = ?", (path,))

    # ── Rejections ────────────────────────────────────────────────────────────

    _REJECTION_CAP = 5

    def add_rejection(self, concept: str, feedback: str, body: str = "") -> None:
        """Store a rejection record. Auto-blocks concept after _REJECTION_CAP rejections."""
        with self._tx():
            self._conn.execute(
                """INSERT INTO rejections (concept, feedback, rejected_body, rejected_at)
                   VALUES (?, ?, ?, ?)""",
                (concept, feedback, body or None, datetime.now().isoformat()),
            )
        if self.rejection_count(concept) >= self._REJECTION_CAP:
            self.mark_concept_blocked(concept)

    def get_rejections(self, concept: str, limit: int = 3) -> list[dict]:
        """Return most recent rejections for a concept, newest first."""
        rows = self._conn.execute(
            """SELECT feedback, rejected_body, rejected_at
               FROM rejections WHERE concept = ?
               ORDER BY rejected_at DESC LIMIT ?""",
            (concept, limit),
        ).fetchall()
        return [
            {"feedback": r["feedback"], "body": r["rejected_body"], "rejected_at": r["rejected_at"]}
            for r in rows
        ]

    def rejection_count(self, concept: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM rejections WHERE concept = ?", (concept,)
        ).fetchone()
        return row[0] if row else 0

    # ── Blocked Concepts ──────────────────────────────────────────────────────

    def mark_concept_blocked(self, concept: str) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO blocked_concepts (concept, blocked_at) VALUES (?, ?)",
                (concept, datetime.now().isoformat()),
            )

    def is_concept_blocked(self, concept: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM blocked_concepts WHERE lower(concept) = lower(?)", (concept,)
        ).fetchone()
        return row is not None

    def unblock_concept(self, concept: str) -> None:
        with self._tx():
            self._conn.execute("DELETE FROM blocked_concepts WHERE concept = ?", (concept,))

    def list_blocked_concepts(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT concept FROM blocked_concepts ORDER BY concept"
        ).fetchall()
        return [r[0] for r in rows]

    # ── Stubs ─────────────────────────────────────────────────────────────────

    def add_stub(self, concept: str, source: str = "auto") -> None:
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO stubs (concept, created_at, source) VALUES (?, ?, ?)",
                (concept, datetime.now().isoformat(), source),
            )

    def delete_stub(self, concept: str) -> None:
        with self._tx():
            self._conn.execute("DELETE FROM stubs WHERE concept = ?", (concept,))

    def has_stub(self, concept: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM stubs WHERE concept = ?", (concept,)).fetchone()
        return row is not None

    def get_stubs(self) -> list[str]:
        rows = self._conn.execute("SELECT concept FROM stubs ORDER BY concept").fetchall()
        return [r[0] for r in rows]

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

    def quality_stats(self) -> dict[str, int]:
        """Distribution of source quality levels."""
        rows = self._conn.execute(
            "SELECT quality, COUNT(*) as cnt FROM raw_notes "
            "WHERE quality IS NOT NULL GROUP BY quality"
        ).fetchall()
        result: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for row in rows:
            if row["quality"] in result:
                result[row["quality"]] = row["cnt"]
        return result


# ── Row converters ────────────────────────────────────────────────────────────


def _row_to_raw(row: sqlite3.Row) -> RawNoteRecord:
    keys = row.keys()
    return RawNoteRecord(
        path=row["path"],
        content_hash=row["content_hash"],
        status=row["status"],
        summary=row["summary"] if "summary" in keys else None,
        quality=row["quality"] if "quality" in keys else None,
        ingested_at=datetime.fromisoformat(row["ingested_at"]) if row["ingested_at"] else None,
        compiled_at=datetime.fromisoformat(row["compiled_at"]) if row["compiled_at"] else None,
        error=row["error"],
    )


def _row_to_article(row: sqlite3.Row) -> WikiArticleRecord:
    keys = row.keys()
    return WikiArticleRecord(
        path=row["path"],
        title=row["title"],
        sources=json.loads(row["sources"]),
        content_hash=row["content_hash"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        is_draft=bool(row["is_draft"]),
        approved_at=(
            datetime.fromisoformat(row["approved_at"])
            if "approved_at" in keys and row["approved_at"]
            else None
        ),
        approval_notes=row["approval_notes"] if "approval_notes" in keys else None,
    )
