"""Contract tests: verify data flows between pipeline stages.

Uses real StateDB (not mocked) with fixture data to ensure stage outputs
match what downstream stages expect.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord, WikiArticleRecord
from obsidian_llm_wiki.pipeline.compile import approve_drafts, compile_concepts
from obsidian_llm_wiki.pipeline.ingest import ingest_note
from obsidian_llm_wiki.pipeline.lint import run_lint
from obsidian_llm_wiki.pipeline.query import _find_page, _load_pages
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.vault import parse_note, write_note

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".olw").mkdir()
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


def _analysis_json(
    concepts: list[str] | None = None,
    quality: str = "high",
    summary: str = "A summary.",
) -> str:
    return json.dumps(
        {
            "summary": summary,
            "key_concepts": concepts or ["Quantum Computing", "Qubit"],
            "suggested_topics": ["Quantum Computing"],
            "quality": quality,
        }
    )


def _article_json(
    title: str = "Quantum Computing",
    content: str = "## Overview\n\nQuantum computing uses qubits.",
    tags: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "title": title,
            "content": content,
            "tags": tags or ["quantum", "computing"],
        }
    )


def _write_raw(vault: Path, name: str, content: str) -> Path:
    p = vault / "raw" / name
    p.write_text(content, encoding="utf-8")
    return p


# ── 1. Ingest → Compile contract ─────────────────────────────────────────────


def test_ingest_compile_contract(vault: Path, config: Config, db: StateDB) -> None:
    """After ingest stores concepts+raw_notes, compile reads them correctly."""
    # Ingest phase
    path = _write_raw(vault, "quantum.md", "# Quantum\n\nQubits are cool.")
    ingest_client = MagicMock()
    ingest_client.generate.return_value = _analysis_json(
        concepts=["Quantum Computing"],
        quality="high",
        summary="Note about quantum computing.",
    )
    result = ingest_note(path, config, ingest_client, db)
    assert result is not None

    # Verify ingest stored what compile expects
    rec = db.get_raw("raw/quantum.md")
    assert rec is not None
    assert rec.status == "ingested"
    assert rec.summary == "Note about quantum computing."
    assert rec.quality == "high"

    concepts = db.list_all_concept_names()
    assert "Quantum Computing" in concepts

    pending = db.concepts_needing_compile()
    assert "Quantum Computing" in pending

    sources = db.get_sources_for_concept("Quantum Computing")
    assert "raw/quantum.md" in sources

    # Compile phase: reads what ingest stored
    compile_client = MagicMock()
    compile_client.generate.return_value = _article_json(
        title="Quantum Computing",
        content="## Overview\n\n[[Qubit]]s enable quantum computing.",
    )
    drafts, failed = compile_concepts(config, compile_client, db)

    assert len(drafts) == 1
    assert failed == []
    assert drafts[0].exists()

    # Draft has expected structure
    meta, body = parse_note(drafts[0])
    assert meta["title"] == "Quantum Computing"
    assert meta["status"] == "draft"
    assert "sources" in meta
    assert isinstance(meta["sources"], list)


# ── 2. Compile → Approve contract ────────────────────────────────────────────


def test_compile_approve_contract(vault: Path, config: Config, db: StateDB) -> None:
    """Draft frontmatter has correct fields; approve moves to wiki."""
    # Set up a draft via compile
    _write_raw(vault, "note.md", "# Note\n\nSome content.")
    db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="abc", status="ingested"))
    db.upsert_concepts("raw/note.md", ["Test Concept"])

    client = MagicMock()
    client.generate.return_value = _article_json(
        title="Test Concept",
        content="## Overview\n\nTest concept explained.",
        tags=["test"],
    )
    drafts, _ = compile_concepts(config, client, db)
    assert len(drafts) == 1

    draft_path = drafts[0]
    meta, body = parse_note(draft_path)

    # Verify draft frontmatter has fields approve expects
    assert meta["status"] == "draft"
    assert "title" in meta
    assert "tags" in meta
    assert "sources" in meta
    assert "confidence" in meta

    # Approve
    published = approve_drafts(config, db, [draft_path])

    assert len(published) == 1
    assert published[0].exists()
    assert not draft_path.exists()

    pub_meta, pub_body = parse_note(published[0])
    assert pub_meta["status"] == "published"
    assert pub_meta["title"] == "Test Concept"

    # DB record updated
    pub_rel = str(published[0].relative_to(vault))
    art = db.get_article(pub_rel)
    assert art is not None
    assert art.is_draft is False


# ── 3. Approve → Lint contract ────────────────────────────────────────────────


def test_approve_lint_contract(vault: Path, config: Config, db: StateDB) -> None:
    """Published article structure is valid for lint checks."""
    # Create and approve a draft
    draft_path = config.drafts_dir / "Test Article.md"
    write_note(
        draft_path,
        {
            "title": "Test Article",
            "tags": ["test"],
            "status": "draft",
            "sources": ["raw/note.md"],
            "confidence": 0.75,
        },
        "## Overview\n\nTest content about the article.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Test Article",
            sources=["raw/note.md"],
            content_hash="h",
            is_draft=True,
        )
    )

    published = approve_drafts(config, db, [draft_path])
    assert len(published) == 1

    # Lint the result — should pass all structural checks
    result = run_lint(config, db)

    # No missing_frontmatter since title/tags/status are all present
    fm_issues = [i for i in result.issues if i.issue_type == "missing_frontmatter"]
    assert not fm_issues

    # No low_confidence since confidence=0.75 > 0.3 threshold
    low = [i for i in result.issues if i.issue_type == "low_confidence"]
    assert not low


# ── 4. Approve → Query contract ──────────────────────────────────────────────


def test_approve_query_contract(vault: Path, config: Config, db: StateDB) -> None:
    """Published articles are findable by query's _find_page and _load_pages."""
    # Create a published article
    page_path = config.wiki_dir / "Quantum Computing.md"
    write_note(
        page_path,
        {
            "title": "Quantum Computing",
            "tags": ["quantum"],
            "status": "published",
        },
        "## Overview\n\nQuantum computing uses qubits for computation.",
    )

    # _find_page should locate it
    found = _find_page(config, "Quantum Computing")
    assert found is not None
    assert found == page_path

    # _load_pages should load its content
    content = _load_pages(config, ["Quantum Computing"])
    assert "qubits" in content.lower()
    assert "Quantum Computing" in content


# ── 5. Full pipeline flow ────────────────────────────────────────────────────


def test_full_pipeline_ingest_compile_approve_lint_query(
    vault: Path, config: Config, db: StateDB
) -> None:
    """Full pipeline: ingest → compile → approve → lint → query (mocked LLM)."""
    # Step 1: Ingest
    path = _write_raw(vault, "physics.md", "# Physics\n\nNewton's laws of motion.")
    ingest_client = MagicMock()
    ingest_client.generate.return_value = _analysis_json(
        concepts=["Newtons Laws"],
        quality="high",
        summary="Note about Newton's laws.",
    )
    result = ingest_note(path, config, ingest_client, db)
    assert result is not None
    assert db.get_raw("raw/physics.md").status == "ingested"

    # Step 2: Compile
    compile_client = MagicMock()
    compile_client.generate.return_value = _article_json(
        title="Newtons Laws",
        content="## Overview\n\nNewton described three laws of motion.",
        tags=["physics", "mechanics"],
    )
    drafts, failed = compile_concepts(config, compile_client, db)
    assert len(drafts) == 1
    assert not failed

    # Step 3: Approve
    published = approve_drafts(config, db, drafts)
    assert len(published) == 1
    assert published[0].exists()
    pub_meta, _ = parse_note(published[0])
    assert pub_meta["status"] == "published"

    # Step 4: Lint
    lint_result = run_lint(config, db)
    fm_issues = [i for i in lint_result.issues if i.issue_type == "missing_frontmatter"]
    assert not fm_issues

    # Step 5: Query
    # Write an index so query can route
    index_path = config.wiki_dir / "index.md"
    index_path.write_text(
        "# Wiki Index\n\n## Concepts\n- [[Newtons Laws]]\n",
        encoding="utf-8",
    )

    found = _find_page(config, "Newtons Laws")
    assert found is not None

    content = _load_pages(config, ["Newtons Laws"])
    assert "Newton" in content


# ── 6. State DB round-trip ────────────────────────────────────────────────────


def test_raw_note_record_roundtrip(vault: Path, config: Config, db: StateDB) -> None:
    """RawNoteRecord → upsert → get preserves all fields."""
    from datetime import datetime

    now = datetime.now()
    original = RawNoteRecord(
        path="raw/test.md",
        content_hash="sha256hash",
        status="ingested",
        summary="A comprehensive summary of the note.",
        quality="high",
        ingested_at=now,
        compiled_at=None,
        error=None,
    )
    db.upsert_raw(original)

    loaded = db.get_raw("raw/test.md")
    assert loaded is not None
    assert loaded.path == original.path
    assert loaded.content_hash == original.content_hash
    assert loaded.status == original.status
    assert loaded.summary == original.summary
    assert loaded.quality == original.quality
    assert loaded.ingested_at is not None
    assert loaded.error is None


def test_raw_note_record_update_preserves_fields(vault: Path, config: Config, db: StateDB) -> None:
    """Updating a record preserves changed fields."""
    db.upsert_raw(
        RawNoteRecord(
            path="raw/a.md",
            content_hash="h1",
            status="new",
        )
    )
    db.upsert_raw(
        RawNoteRecord(
            path="raw/a.md",
            content_hash="h2",
            status="ingested",
            summary="Updated summary",
            quality="medium",
        )
    )
    rec = db.get_raw("raw/a.md")
    assert rec.content_hash == "h2"
    assert rec.status == "ingested"
    assert rec.summary == "Updated summary"
    assert rec.quality == "medium"


# ── 7. WikiArticleRecord sources serialization ───────────────────────────────


def test_wiki_article_sources_roundtrip(vault: Path, config: Config, db: StateDB) -> None:
    """Sources list survives JSON serialization in DB."""
    sources = ["raw/note1.md", "raw/note2.md", "raw/subdir/note3.md"]
    record = WikiArticleRecord(
        path="wiki/Test.md",
        title="Test Article",
        sources=sources,
        content_hash="abc123",
        is_draft=False,
    )
    db.upsert_article(record)

    loaded = db.get_article("wiki/Test.md")
    assert loaded is not None
    assert loaded.sources == sources
    assert isinstance(loaded.sources, list)
    assert len(loaded.sources) == 3


def test_wiki_article_empty_sources_roundtrip(vault: Path, config: Config, db: StateDB) -> None:
    """Empty sources list survives serialization."""
    record = WikiArticleRecord(
        path="wiki/Orphan.md",
        title="Orphan",
        sources=[],
        content_hash="xyz",
        is_draft=True,
    )
    db.upsert_article(record)

    loaded = db.get_article("wiki/Orphan.md")
    assert loaded is not None
    assert loaded.sources == []
    assert loaded.is_draft is True


def test_wiki_article_special_chars_in_sources(vault: Path, config: Config, db: StateDB) -> None:
    """Source paths with special characters survive serialization."""
    sources = [
        "raw/note with spaces.md",
        "raw/note-with-dashes.md",
        "raw/note_underscores.md",
    ]
    record = WikiArticleRecord(
        path="wiki/Special.md",
        title="Special",
        sources=sources,
        content_hash="def",
        is_draft=False,
    )
    db.upsert_article(record)

    loaded = db.get_article("wiki/Special.md")
    assert loaded.sources == sources
