"""Tests for compile pipeline — mocked LLM, no Ollama required."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord
from obsidian_llm_wiki.pipeline.compile import (
    _write_concept_prompt,
    _write_prompt_legacy,
    approve_drafts,
    compile_concepts,
    compile_notes,
    reject_draft,
)
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


def _make_client(plan_json: str, article_json: str):
    """Mock client: first call returns plan, subsequent return article."""
    client = MagicMock()
    call_count = [0]

    def generate_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return plan_json
        return article_json

    client.generate.side_effect = generate_side_effect
    return client


def test_compile_creates_draft(vault, config, db, fixtures_dir):
    # Setup: ingested raw note
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("---\ntitle: Note\n---\n\nQuantum entanglement content.")
    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="abc",
            status="ingested",
        )
    )

    plan_json = (fixtures_dir / "compile_plan_valid.json").read_text()
    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_client(plan_json, article_json)

    drafts, failed = compile_notes(config=config, client=client, db=db)

    assert len(drafts) == 1
    assert len(failed) == 0
    assert drafts[0].exists()
    assert drafts[0].parent == config.drafts_dir


def test_draft_has_correct_frontmatter(vault, config, db, fixtures_dir):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("# Note\n\nContent here.")
    db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="h", status="ingested"))

    plan_json = (fixtures_dir / "compile_plan_valid.json").read_text()
    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_client(plan_json, article_json)

    drafts, _ = compile_notes(config=config, client=client, db=db)
    assert drafts

    from obsidian_llm_wiki.vault import parse_note

    meta, body = parse_note(drafts[0])
    assert meta["status"] == "draft"
    assert "title" in meta
    assert "tags" in meta
    assert 0.0 <= meta["confidence"] <= 1.0


def test_dry_run_writes_nothing(vault, config, db, fixtures_dir):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(RawNoteRecord(path="raw/note.md", content_hash="h", status="ingested"))

    plan_json = (fixtures_dir / "compile_plan_valid.json").read_text()
    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_client(plan_json, article_json)

    drafts, _ = compile_notes(config=config, client=client, db=db, dry_run=True)
    assert drafts == []
    assert list(config.drafts_dir.glob("*.md")) == []


def test_approve_moves_draft_to_wiki(vault, config, db):
    from obsidian_llm_wiki.models import WikiArticleRecord
    from obsidian_llm_wiki.vault import write_note

    draft_path = config.drafts_dir / "article.md"
    write_note(draft_path, {"title": "Article", "status": "draft", "tags": []}, "Body.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Article",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    published = approve_drafts(config, db, [draft_path])
    assert len(published) == 1
    assert published[0].exists()
    assert published[0].parent == config.wiki_dir
    assert not draft_path.exists()

    # State updated
    record = db.get_article(str(published[0].relative_to(vault)))
    assert record is not None
    assert record.is_draft is False


def test_reject_deletes_draft(vault, config, db):
    from obsidian_llm_wiki.models import WikiArticleRecord
    from obsidian_llm_wiki.vault import write_note

    draft_path = config.drafts_dir / "bad.md"
    write_note(draft_path, {"title": "Bad", "status": "draft"}, "Wrong content.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="Bad",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    reject_draft(draft_path, config, db, feedback="Hallucinated content")
    assert not draft_path.exists()


# ── Concept-driven compile tests ───────────────────────────────────────────────


def _make_concept_client(article_json: str):
    """Mock client that returns a single article for any generate() call."""
    client = MagicMock()
    client.generate.return_value = article_json
    return client


def test_compile_concepts_creates_draft(vault, config, db, fixtures_dir):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("---\ntitle: Note\n---\n\nQuantum entanglement content.")
    db.upsert_raw(
        __import__("obsidian_llm_wiki.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="ingested"
        )
    )
    db.upsert_concepts("raw/note.md", ["Quantum Entanglement"])

    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_concept_client(article_json)

    drafts, failed, _ = compile_concepts(config=config, client=client, db=db)

    assert len(drafts) == 1
    assert len(failed) == 0
    assert drafts[0].exists()
    assert drafts[0].parent == config.drafts_dir


def test_compile_concepts_skips_when_no_concepts_needing_compile(vault, config, db):
    db.upsert_raw(
        __import__("obsidian_llm_wiki.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="compiled"
        )
    )
    db.upsert_concepts("raw/note.md", ["Some Concept"])

    client = MagicMock()
    drafts, failed, _ = compile_concepts(config=config, client=client, db=db)

    assert drafts == []
    assert failed == []
    client.generate.assert_not_called()


def test_compile_concepts_dry_run(vault, config, db, fixtures_dir, capsys):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        __import__("obsidian_llm_wiki.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="ingested"
        )
    )
    db.upsert_concepts("raw/note.md", ["Concept A"])

    client = MagicMock()
    drafts, _, _ = compile_concepts(config=config, client=client, db=db, dry_run=True)

    assert drafts == []
    assert list(config.drafts_dir.glob("*.md")) == []
    captured = capsys.readouterr()
    assert "Concept A" in captured.out


def test_compile_concepts_manual_edit_protection(vault, config, db, fixtures_dir):
    """Article with content_hash mismatch (manually edited) should be skipped."""
    from obsidian_llm_wiki.models import WikiArticleRecord

    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        __import__("obsidian_llm_wiki.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="ingested"
        )
    )
    db.upsert_concepts("raw/note.md", ["Quantum Entanglement"])

    # Simulate published article with a DIFFERENT content_hash than what's on disk
    wiki_path = config.wiki_dir / "Quantum Entanglement.md"
    wiki_path.write_text("---\ntitle: Quantum Entanglement\n---\n\nManually edited content.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(wiki_path.relative_to(vault)),
            title="Quantum Entanglement",
            sources=["raw/note.md"],
            content_hash="original_hash_before_edit",  # differs from file on disk
            is_draft=False,
        )
    )

    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_concept_client(article_json)

    drafts, failed, _ = compile_concepts(config=config, client=client, db=db)

    # Should skip the manually-edited article
    assert drafts == []
    client.generate.assert_not_called()


def test_compile_concepts_force_overrides_edit_protection(vault, config, db, fixtures_dir):
    """--force should recompile even manually-edited articles."""
    from obsidian_llm_wiki.models import WikiArticleRecord

    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        __import__("obsidian_llm_wiki.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="ingested"
        )
    )
    db.upsert_concepts("raw/note.md", ["Quantum Entanglement"])

    wiki_path = config.wiki_dir / "Quantum Entanglement.md"
    wiki_path.write_text("---\ntitle: Quantum Entanglement\n---\n\nManually edited.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(wiki_path.relative_to(vault)),
            title="Quantum Entanglement",
            sources=["raw/note.md"],
            content_hash="old_hash",
            is_draft=False,
        )
    )

    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_concept_client(article_json)

    drafts, failed, _ = compile_concepts(config=config, client=client, db=db, force=True)

    assert len(drafts) == 1


def test_write_concept_prompt_has_tag_instructions():
    prompt = _write_concept_prompt("Quantum Computing", "source text", [])
    assert "hyphen-separated" in prompt
    assert "machine-learning" in prompt


def test_write_prompt_legacy_has_tag_instructions():
    from obsidian_llm_wiki.models import ArticlePlan

    plan = ArticlePlan(
        title="Test",
        action="create",
        path="test.md",
        reasoning="needed",
        source_paths=[],
    )
    prompt = _write_prompt_legacy(plan, "source text", [])
    assert "hyphen-separated" in prompt


def test_compile_concepts_marks_sources_compiled(vault, config, db, fixtures_dir):
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        __import__("obsidian_llm_wiki.models", fromlist=["RawNoteRecord"]).RawNoteRecord(
            path="raw/note.md", content_hash="abc", status="ingested"
        )
    )
    db.upsert_concepts("raw/note.md", ["Concept A"])

    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_concept_client(article_json)

    compile_concepts(config=config, client=client, db=db)

    record = db.get_raw("raw/note.md")
    assert record.status == "compiled"
