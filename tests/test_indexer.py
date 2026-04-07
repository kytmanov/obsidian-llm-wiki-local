"""Tests for indexer — index generation and log appending."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.indexer import append_log, generate_index
from obsidian_llm_wiki.models import WikiArticleRecord
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.vault import write_note


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


# ── generate_index ────────────────────────────────────────────────────────────


def test_generate_index_creates_file(config: Config, db: StateDB):
    path = generate_index(config, db)
    assert path.exists()
    assert path.name == "index.md"


def test_generate_index_has_frontmatter(config: Config, db: StateDB):
    path = generate_index(config, db)
    content = path.read_text()
    assert "---" in content
    assert "title: Wiki Index" in content
    assert "tags:" in content


def test_generate_index_with_concept_articles(config: Config, db: StateDB):
    # Register an article in DB
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/Quantum Computing.md",
            title="Quantum Computing",
            sources=["raw/quantum.md"],
            content_hash="abc123",
            is_draft=False,
        )
    )
    # Create file on disk
    write_note(
        config.wiki_dir / "Quantum Computing.md",
        {"title": "Quantum Computing", "status": "published"},
        "Content about quantum computing.",
    )

    path = generate_index(config, db)
    content = path.read_text()
    assert "[[Quantum Computing]]" in content
    assert "## Concepts" in content


def test_generate_index_with_source_pages(config: Config, db: StateDB):
    # Create a source page on disk
    write_note(
        config.sources_dir / "My Source.md",
        {"title": "My Source", "quality": "high", "source_file": "raw/source.md"},
        "Summary of source.",
    )

    path = generate_index(config, db)
    content = path.read_text()
    assert "[[My Source]]" in content
    assert "## Sources" in content


def test_generate_index_entry_count(config: Config, db: StateDB):
    # Add 3 concept articles
    for i in range(3):
        db.upsert_article(
            WikiArticleRecord(
                path=f"wiki/Concept {i}.md",
                title=f"Concept {i}",
                sources=[f"raw/source{i}.md"],
                content_hash=f"hash{i}",
                is_draft=False,
            )
        )
        write_note(
            config.wiki_dir / f"Concept {i}.md",
            {"title": f"Concept {i}", "status": "published"},
            f"Content {i}.",
        )

    path = generate_index(config, db)
    content = path.read_text()
    assert "3 entries" in content


def test_generate_index_excludes_drafts(config: Config, db: StateDB):
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/.drafts/Draft Article.md",
            title="Draft Article",
            sources=["raw/s.md"],
            content_hash="draft_hash",
            is_draft=True,
        )
    )

    path = generate_index(config, db)
    content = path.read_text()
    assert "Draft Article" not in content


def test_generate_index_picks_up_untracked_wiki_files(config: Config, db: StateDB):
    """Files on disk but not in DB should still appear in index."""
    write_note(
        config.wiki_dir / "Untracked.md",
        {"title": "Untracked"},
        "Not in DB.",
    )

    path = generate_index(config, db)
    content = path.read_text()
    assert "[[Untracked]]" in content


def test_generate_index_skips_index_and_log(config: Config, db: StateDB):
    """index.md and log.md should not appear as entries."""
    write_note(config.wiki_dir / "log.md", {"title": "Log"}, "Log entries.")

    path = generate_index(config, db)
    content = path.read_text()
    # Should not list index.md or log.md as entries
    lines = [line for line in content.split("\n") if line.startswith("- [[")]
    assert not any("index" in line.lower() for line in lines)
    assert not any("[[Log]]" in line for line in lines)


# ── append_log ────────────────────────────────────────────────────────────────


def test_append_log_creates_file(config: Config):
    path = append_log(config, "test entry")
    assert path.exists()
    assert path.name == "log.md"


def test_append_log_has_header_on_first_write(config: Config):
    path = append_log(config, "first entry")
    content = path.read_text()
    assert "# Operation Log" in content
    assert "first entry" in content


def test_append_log_appends_multiple(config: Config):
    append_log(config, "entry 1")
    path = append_log(config, "entry 2")
    content = path.read_text()
    assert "entry 1" in content
    assert "entry 2" in content


def test_append_log_has_timestamp(config: Config):
    path = append_log(config, "timestamped")
    content = path.read_text()
    # Should contain a date-like pattern
    import re

    assert re.search(r"\d{4}-\d{2}-\d{2}", content)


def test_append_log_creates_parent_dirs(tmp_path: Path):
    """If wiki/ doesn't exist, append_log should create it."""
    config = Config(vault=tmp_path)
    # Don't create wiki/ — let append_log handle it
    (tmp_path / "wiki").mkdir(parents=True, exist_ok=True)
    path = append_log(config, "auto-created")
    assert path.exists()
