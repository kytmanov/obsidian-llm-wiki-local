"""Integration tests for reject --all → recompile with feedback loop."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from obsidian_llm_wiki.cli import cli
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord, WikiArticleRecord
from obsidian_llm_wiki.pipeline.compile import compile_concepts
from obsidian_llm_wiki.state import StateDB


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
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


def _write_raw(vault: Path, name: str, content: str = "# Note\n\nBody.") -> Path:
    p = vault / "raw" / name
    p.write_text(content)
    return p


def _make_draft(vault: Path, title: str) -> Path:
    draft = vault / "wiki" / ".drafts" / f"{title}.md"
    draft.write_text(f"---\ntitle: {title}\n---\n# {title}\n\nDraft content.")
    return draft


def _register_draft_and_source(db: StateDB, vault: Path, draft: Path, source: str) -> None:
    db.upsert_raw(RawNoteRecord(path=source, content_hash="h1", status="compiled"))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft.relative_to(vault)),
            title=draft.stem,
            sources=[source],
            content_hash="dh",
            is_draft=True,
        )
    )


@pytest.mark.integration
def test_reject_all_then_recompile_uses_feedback(vault, config, db, runner):
    """
    Full loop: create 2 drafts → reject --all with feedback →
    verify drafts deleted + rejections in DB → compile again →
    verify feedback injected into compile prompt.
    """
    # Setup: 2 source notes and their drafts
    _write_raw(vault, "note_a.md", "# Note A\n\nContent A.")
    _write_raw(vault, "note_b.md", "# Note B\n\nContent B.")
    draft_a = _make_draft(vault, "Concept A")
    draft_b = _make_draft(vault, "Concept B")
    _register_draft_and_source(db, vault, draft_a, "raw/note_a.md")
    _register_draft_and_source(db, vault, draft_b, "raw/note_b.md")
    db.upsert_concepts("raw/note_a.md", ["Concept A"])
    db.upsert_concepts("raw/note_b.md", ["Concept B"])

    # Reject all
    result = runner.invoke(
        cli,
        ["reject", "--vault", str(vault), "--all", "--feedback", "wrong language"],
        input="\n",
    )
    assert result.exit_code == 0
    assert not draft_a.exists()
    assert not draft_b.exists()

    # Rejections stored in DB
    rej_a = db.get_rejections("Concept A", limit=3)
    rej_b = db.get_rejections("Concept B", limit=3)
    assert len(rej_a) == 1 and rej_a[0]["feedback"] == "wrong language"
    assert len(rej_b) == 1 and rej_b[0]["feedback"] == "wrong language"

    # Recompile — mock LLM, capture prompts
    article_json = json.dumps({
        "title": "Concept A",
        "content": "Rewritten content.",
        "tags": ["concept-a"],
    })
    client = MagicMock()
    client.generate.return_value = article_json

    # Mark sources as ingested so compile picks them up
    db.mark_raw_status("raw/note_a.md", "ingested")
    db.mark_raw_status("raw/note_b.md", "ingested")

    compile_concepts(config=config, client=client, db=db)

    # At least one compile call prompt should contain the feedback
    all_prompts = [call.kwargs.get("prompt", "") for call in client.generate.call_args_list]
    assert any("wrong language" in p for p in all_prompts), (
        "Expected rejection feedback in at least one compile prompt"
    )
