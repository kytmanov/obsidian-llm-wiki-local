"""Tests for the `olw reject` CLI command — single file and --all flag."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from obsidian_llm_wiki.cli import cli
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord, WikiArticleRecord
from obsidian_llm_wiki.state import StateDB


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".olw").mkdir()
    return tmp_path


@pytest.fixture
def config(vault):
    return Config(vault=vault)


@pytest.fixture
def db(config):
    return StateDB(config.state_db_path)


@pytest.fixture
def runner():
    return CliRunner()


def _make_draft(vault: Path, title: str, content: str = "# Draft\n\nBody.") -> Path:
    """Write a minimal draft markdown file."""
    draft = vault / "wiki" / ".drafts" / f"{title}.md"
    draft.write_text(f"---\ntitle: {title}\n---\n{content}")
    return draft


def _register_draft(db: StateDB, vault: Path, draft: Path, source: str = "raw/note.md") -> None:
    """Register a draft article in the state DB."""
    db.upsert_raw(RawNoteRecord(path=source, content_hash="h1", status="ingested"))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft.relative_to(vault)),
            title=draft.stem,
            sources=[source],
            content_hash="dh1",
            is_draft=True,
        )
    )


def test_reject_single_file(vault, config, db, runner):
    draft = _make_draft(vault, "My Concept")
    _register_draft(db, vault, draft)

    result = runner.invoke(
        cli,
        ["reject", "--vault", str(vault), "--feedback", "too vague", str(draft)],
    )

    assert result.exit_code == 0
    assert not draft.exists()
    assert "Draft rejected" in result.output


def test_reject_relative_filename_resolves_under_drafts(vault, config, db, runner):
    draft = _make_draft(vault, "My Concept")
    _register_draft(db, vault, draft)

    result = runner.invoke(
        cli,
        ["reject", "--vault", str(vault), "--feedback", "too vague", "My Concept.md"],
    )

    assert result.exit_code == 0
    assert not draft.exists()


def test_reject_all_rejects_all_drafts(vault, config, db, runner):
    drafts = [_make_draft(vault, f"Concept {i}") for i in range(3)]
    for i, d in enumerate(drafts):
        _register_draft(db, vault, d, source=f"raw/note{i}.md")

    result = runner.invoke(
        cli,
        ["reject", "--vault", str(vault), "--all", "--feedback", "batch reject"],
        input="\n",  # confirm any prompts
    )

    assert result.exit_code == 0
    for d in drafts:
        assert not d.exists()
    assert "Rejected 3 draft(s)" in result.output


def test_reject_all_empty_drafts_dir(vault, config, db, runner):
    result = runner.invoke(
        cli,
        ["reject", "--vault", str(vault), "--all", "--feedback", "test"],
        input="\n",
    )

    assert result.exit_code == 0
    assert "No drafts" in result.output


def test_reject_all_with_feedback_flag(vault, config, db, runner):
    draft = _make_draft(vault, "Topic A")
    _register_draft(db, vault, draft)

    runner.invoke(
        cli,
        ["reject", "--vault", str(vault), "--all", "--feedback", "bad quality"],
        input="\n",
    )

    rejections = db.get_rejections("Topic A", limit=5)
    assert len(rejections) == 1
    assert rejections[0]["feedback"] == "bad quality"


def test_reject_no_args_shows_error(vault, config, db, runner):
    result = runner.invoke(
        cli,
        ["reject", "--vault", str(vault)],
        input="\n",
    )

    assert result.exit_code != 0


def test_reject_all_and_files_all_wins(vault, config, db, runner):
    draft_a = _make_draft(vault, "Concept A")
    draft_b = _make_draft(vault, "Concept B")
    _register_draft(db, vault, draft_a, source="raw/a.md")
    _register_draft(db, vault, draft_b, source="raw/b.md")

    # Pass --all AND an extra file — --all should process all drafts
    result = runner.invoke(
        cli,
        ["reject", "--vault", str(vault), "--all", "--feedback", "x", str(draft_a)],
        input="\n",
    )

    assert result.exit_code == 0
    # Both drafts rejected because --all wins
    assert not draft_a.exists()
    assert not draft_b.exists()
