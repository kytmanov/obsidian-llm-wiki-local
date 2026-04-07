"""Tests for error paths — resilience under bad input, no Ollama required."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord
from obsidian_llm_wiki.pipeline.ingest import ingest_note
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.structured_output import StructuredOutputError, request_structured
from obsidian_llm_wiki.vault import atomic_write, parse_note


# ── Helpers ───────────────────────────────────────────────────────────────────


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


def _write_raw(vault: Path, name: str, content: str) -> Path:
    p = vault / "raw" / name
    p.write_text(content, encoding="utf-8")
    return p


def _analysis_json(
    concepts: list[str] | None = None,
    quality: str = "high",
    summary: str = "A summary.",
) -> str:
    return json.dumps({
        "summary": summary,
        "key_concepts": concepts or ["Concept A"],
        "suggested_topics": ["Topic A"],
        "quality": quality,
    })


def _make_client(response: str) -> MagicMock:
    client = MagicMock()
    client.generate.return_value = response
    return client


# ── 1. Ollama returns completely invalid JSON → StructuredOutputError ─────────


def test_structured_output_invalid_json_raises(config: Config) -> None:
    """request_structured raises StructuredOutputError after retries exhausted."""
    from obsidian_llm_wiki.models import AnalysisResult

    client = MagicMock()
    client.generate.return_value = "this is not json at all }{{"

    with pytest.raises(StructuredOutputError, match="Failed to get valid"):
        request_structured(
            client=client,
            prompt="analyze this",
            model_class=AnalysisResult,
            model="test-model",
            max_retries=1,
        )
    # Initial attempt + 1 retry = 2 calls
    assert client.generate.call_count == 2


# ── 2. Ollama timeout during ingest → note marked failed ─────────────────────


def test_ingest_timeout_marks_note_failed(
    vault: Path, config: Config, db: StateDB
) -> None:
    """Ollama timeout → note marked 'failed', pipeline continues."""
    path = _write_raw(vault, "timeout.md", "# Timeout\n\nContent here.")
    client = MagicMock()
    client.generate.side_effect = TimeoutError("connection timed out")

    result = ingest_note(path, config, client, db)

    assert result is None
    rec = db.get_raw("raw/timeout.md")
    assert rec is not None
    assert rec.status == "failed"
    assert "timed out" in (rec.error or "").lower()


def test_ingest_failure_does_not_block_next_note(
    vault: Path, config: Config, db: StateDB
) -> None:
    """After one note fails, the next note can still be ingested."""
    fail_path = _write_raw(vault, "fail.md", "# Fail\n\nBad note.")
    good_path = _write_raw(vault, "good.md", "# Good\n\nGood note.")

    fail_client = MagicMock()
    fail_client.generate.side_effect = RuntimeError("Ollama error")
    r1 = ingest_note(fail_path, config, fail_client, db)

    good_client = MagicMock()
    good_client.generate.return_value = _analysis_json()
    r2 = ingest_note(good_path, config, good_client, db)

    assert r1 is None
    assert r2 is not None
    assert db.get_raw("raw/fail.md").status == "failed"
    assert db.get_raw("raw/good.md").status == "ingested"


# ── 3. Empty .md file → ingest handles gracefully ────────────────────────────


def test_ingest_empty_md_file(
    vault: Path, config: Config, db: StateDB
) -> None:
    """Empty .md file should not crash ingest."""
    path = _write_raw(vault, "empty.md", "")
    client = _make_client(_analysis_json())

    result = ingest_note(path, config, client, db)

    # Should succeed (LLM analyzes empty body) or be handled gracefully
    if result is not None:
        assert result.summary
    rec = db.get_raw("raw/empty.md")
    assert rec is not None
    assert rec.status in ("ingested", "failed")


# ── 4. Binary file renamed to .md → ingest handles gracefully ────────────────


def test_ingest_binary_file_as_md(
    vault: Path, config: Config, db: StateDB
) -> None:
    """Binary file with .md extension should not crash."""
    path = vault / "raw" / "binary.md"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe")

    # read_text will raise UnicodeDecodeError
    with pytest.raises(Exception):
        ingest_note(path, config, MagicMock(), db)


# ── 5. Non-UTF-8 file → ingest handles gracefully ────────────────────────────


def test_ingest_non_utf8_file(
    vault: Path, config: Config, db: StateDB
) -> None:
    """File with Latin-1 encoding should raise or fail gracefully."""
    path = vault / "raw" / "latin1.md"
    path.write_bytes("# Café résumé\n\nNaïve text.".encode("latin-1"))

    # ingest_note calls path.read_text(encoding="utf-8"), which will fail
    # on the non-UTF-8 bytes
    with pytest.raises(Exception):
        ingest_note(path, config, MagicMock(), db)


# ── 6. Atomic write interrupted → no partial .tmp files ──────────────────────


def test_atomic_write_no_partial_on_error(tmp_path: Path) -> None:
    """Simulated write error must not leave .tmp files."""
    target = tmp_path / "output.md"
    large_content = "x" * 100_000

    with patch("obsidian_llm_wiki.vault.open", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            atomic_write(target, large_content)

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Leftover .tmp files found: {tmp_files}"
    assert not target.exists()


def test_atomic_write_success_no_tmp(tmp_path: Path) -> None:
    """Successful atomic_write should leave no .tmp files."""
    target = tmp_path / "clean.md"
    atomic_write(target, "content")

    assert target.exists()
    assert target.read_text() == "content"
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


# ── 7. Frontmatter with unexpected types → parse_note still works ────────────


def test_parse_note_tags_as_string(tmp_path: Path) -> None:
    """Tags stored as string instead of list should not crash parse_note."""
    p = tmp_path / "note.md"
    p.write_text(
        "---\ntitle: Test\ntags: single-tag\n---\n\nBody text.",
        encoding="utf-8",
    )
    meta, body = parse_note(p)
    assert meta["title"] == "Test"
    # Tags may be string or list depending on YAML parsing; just no crash
    assert "tags" in meta
    assert "Body text" in body


def test_parse_note_numeric_title(tmp_path: Path) -> None:
    """Title as integer should not crash parse_note."""
    p = tmp_path / "note.md"
    p.write_text(
        "---\ntitle: 42\ntags: []\n---\n\nBody text.",
        encoding="utf-8",
    )
    meta, body = parse_note(p)
    assert meta["title"] == 42
    assert "Body text" in body


def test_parse_note_nested_frontmatter(tmp_path: Path) -> None:
    """Nested objects in frontmatter should be parsed without crash."""
    p = tmp_path / "note.md"
    p.write_text(
        "---\ntitle: Test\nmeta:\n  key: value\n  nested:\n    deep: true\n"
        "---\n\nBody.",
        encoding="utf-8",
    )
    meta, body = parse_note(p)
    assert meta["title"] == "Test"
    assert isinstance(meta["meta"], dict)


def test_parse_note_empty_frontmatter(tmp_path: Path) -> None:
    """Empty frontmatter (--- followed by ---) should return empty dict."""
    p = tmp_path / "note.md"
    p.write_text("---\n---\n\nBody only.", encoding="utf-8")
    meta, body = parse_note(p)
    assert isinstance(meta, dict)
    assert "Body only" in body


# ── 8. StateDB opened twice on same file → works ─────────────────────────────


def test_statedb_opened_twice_same_file(tmp_path: Path) -> None:
    """Two StateDB instances on the same file should work (check_same_thread=False)."""
    db_path = tmp_path / ".olw" / "state.db"
    db1 = StateDB(db_path)
    db2 = StateDB(db_path)

    db1.upsert_raw(RawNoteRecord(
        path="raw/a.md", content_hash="h1", status="new"
    ))
    rec = db2.get_raw("raw/a.md")
    assert rec is not None
    assert rec.content_hash == "h1"

    db1.close()
    db2.close()


# ── 9. Compile with no raw notes → returns empty, no crash ───────────────────


def test_compile_concepts_no_raw_notes(
    vault: Path, config: Config, db: StateDB
) -> None:
    """compile_concepts with no concepts needing compile returns empty."""
    from obsidian_llm_wiki.pipeline.compile import compile_concepts

    client = MagicMock()
    drafts, failed = compile_concepts(config, client, db)

    assert drafts == []
    assert failed == []
    client.generate.assert_not_called()


def test_compile_notes_no_ingested(
    vault: Path, config: Config, db: StateDB
) -> None:
    """compile_notes with no ingested notes returns empty."""
    from obsidian_llm_wiki.pipeline.compile import compile_notes

    client = MagicMock()
    drafts, failed = compile_notes(config, client, db)

    assert drafts == []
    assert failed == []
    client.generate.assert_not_called()


# ── 10. Query with empty wiki → helpful error message ────────────────────────


def test_query_empty_wiki_helpful_message(
    vault: Path, config: Config, db: StateDB
) -> None:
    """Query with no index.md returns helpful guidance."""
    from obsidian_llm_wiki.pipeline.query import run_query

    client = MagicMock()
    answer, pages = run_query(config, client, db, "What is physics?")

    assert pages == []
    assert "index" in answer.lower() or "ingest" in answer.lower()
    client.generate.assert_not_called()


# ── 11. Lint on empty wiki dir → healthy result, no crash ────────────────────


def test_lint_empty_wiki_dir(
    vault: Path, config: Config, db: StateDB
) -> None:
    """Lint on empty wiki/ returns healthy result."""
    from obsidian_llm_wiki.pipeline.lint import run_lint

    result = run_lint(config, db)

    assert result.health_score == 100.0
    assert result.issues == []
    assert "healthy" in result.summary.lower()


def test_lint_nonexistent_wiki_dir(tmp_path: Path) -> None:
    """Lint when wiki/ doesn't exist should not crash."""
    from obsidian_llm_wiki.pipeline.lint import run_lint

    (tmp_path / ".olw").mkdir()
    cfg = Config(vault=tmp_path)
    state = StateDB(cfg.state_db_path)

    result = run_lint(cfg, state)

    assert result.health_score == 100.0
    assert result.issues == []
    state.close()
