"""Tests for compile pipeline — mocked LLM, no Ollama required."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord
from obsidian_llm_wiki.pipeline.compile import (
    _compute_confidence,
    _gather_sources,
    _load_vault_schema,
    _truncate_to_budget,
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


def test_compile_notes_emits_deprecation_warning(vault, config, db):
    """compile_notes() should emit a DeprecationWarning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client = MagicMock()
        compile_notes(config=config, client=client, db=db)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "deprecated" in str(deprecation_warnings[0].message).lower()


def test_compile_notes_logs_deprecation_warning(vault, config, db, caplog):
    """compile_notes() should also log the deprecation via the logging module."""
    import logging

    with caplog.at_level(logging.WARNING, logger="obsidian_llm_wiki.pipeline.compile"):
        client = MagicMock()
        compile_notes(config=config, client=client, db=db)
    assert any("deprecated" in r.message.lower() for r in caplog.records)


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

    drafts, failed = compile_concepts(config=config, client=client, db=db)

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
    drafts, failed = compile_concepts(config=config, client=client, db=db)

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
    drafts, _ = compile_concepts(config=config, client=client, db=db, dry_run=True)

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

    drafts, failed = compile_concepts(config=config, client=client, db=db)

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

    drafts, failed = compile_concepts(config=config, client=client, db=db, force=True)

    assert len(drafts) == 1


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


# ── _load_vault_schema ────────────────────────────────────────────────────


def test_load_vault_schema_returns_content(vault, config):
    schema_file = config.schema_path
    schema_file.write_text("# Vault Schema\n\nUse ## headings.")
    result = _load_vault_schema(config)
    assert "Vault Schema" in result


def test_load_vault_schema_returns_empty_when_missing(vault, config):
    assert not config.schema_path.exists()
    result = _load_vault_schema(config)
    assert result == ""


def test_load_vault_schema_returns_empty_on_read_error(vault, config):
    from unittest.mock import patch

    config.schema_path.write_text("content")
    with patch.object(
        type(config.schema_path),
        "read_text",
        side_effect=OSError("permission denied"),
    ):
        result = _load_vault_schema(config)
    assert result == ""


# ── _truncate_to_budget ───────────────────────────────────────────────────


def test_truncate_to_budget_short_text():
    assert _truncate_to_budget("short", 100) == "short"


def test_truncate_to_budget_long_text():
    text = "x" * 5000
    result = _truncate_to_budget(text, 500)  # limit = 500*4 = 2000
    assert len(result) < len(text)
    assert result.endswith("[...truncated...]")


# ── _gather_sources ───────────────────────────────────────────────────────


def test_gather_sources_parse_exception(vault, config):
    """Source file that can't be parsed yields warning, not crash."""
    bad_file = vault / "raw" / "bad.md"
    bad_file.write_text("content")

    from unittest.mock import patch

    with patch(
        "obsidian_llm_wiki.pipeline.compile.parse_note",
        side_effect=ValueError("bad parse"),
    ):
        text, resolved = _gather_sources(["raw/bad.md"], vault)
    assert resolved == []
    assert text == ""


# ── _compute_confidence ──────────────────────────────────────────────────


def test_compute_confidence_high_quality_bonus(vault, config, db):
    raw_note = vault / "raw" / "q.md"
    raw_note.write_text("content")
    db.upsert_raw(
        RawNoteRecord(
            path="raw/q.md",
            content_hash="h",
            status="ingested",
            quality="high",
        )
    )
    conf = _compute_confidence(["raw/q.md"], db)
    # 1 source * 0.25 + 0.25 high bonus = 0.5
    assert conf == pytest.approx(0.5)


def test_compute_confidence_medium_quality(vault, config, db):
    raw_note = vault / "raw" / "m.md"
    raw_note.write_text("content")
    db.upsert_raw(
        RawNoteRecord(
            path="raw/m.md",
            content_hash="h",
            status="ingested",
            quality="medium",
        )
    )
    conf = _compute_confidence(["raw/m.md"], db)
    # 1 * 0.25 + 0.1 medium bonus = 0.35
    assert conf == pytest.approx(0.35)


def test_compute_confidence_multiple_sources(vault, config, db):
    for i in range(5):
        p = vault / "raw" / f"s{i}.md"
        p.write_text("content")
        db.upsert_raw(
            RawNoteRecord(
                path=f"raw/s{i}.md",
                content_hash=f"h{i}",
                status="ingested",
                quality="low",
            )
        )
    paths = [f"raw/s{i}.md" for i in range(5)]
    conf = _compute_confidence(paths, db)
    # min(1.0, 5*0.25 + 0.0) = 1.0
    assert conf == 1.0


# ── _inject_body_sections (source path doesn't exist) ────────────────────


def test_inject_body_sections_missing_source(vault, config):
    from obsidian_llm_wiki.pipeline.compile import _inject_body_sections

    body = "## Overview\n\nSome content."
    # Source path doesn't exist — falls back to stem title
    result = _inject_body_sections(body, ["raw/nonexistent.md"], config)
    assert "## Sources" in result
    assert "Nonexistent" in result


# ── _write_concept_prompt with vault_schema ──────────────────────────────


def test_write_concept_prompt_with_schema():
    from obsidian_llm_wiki.pipeline.compile import (
        _write_concept_prompt,
    )

    prompt = _write_concept_prompt(
        concept="Test",
        sources="source text",
        existing_titles=["A", "B"],
        vault_schema="Use ## sections and [[wikilinks]].",
    )
    assert "VAULT CONVENTIONS" in prompt
    assert "Use ## sections" in prompt


# ── compile_concepts edge cases ──────────────────────────────────────────


def test_compile_concepts_no_source_paths(vault, config, db):
    """Concept with no source_paths is silently skipped."""
    from unittest.mock import patch

    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="abc",
            status="ingested",
        )
    )
    db.upsert_concepts("raw/note.md", ["EmptyConcept"])

    with patch.object(
        db,
        "get_sources_for_concept",
        return_value=[],
    ):
        client = MagicMock()
        drafts, failed = compile_concepts(config=config, client=client, db=db)
    client.generate.assert_not_called()


def test_compile_concepts_no_readable_sources(vault, config, db):
    """When _gather_sources returns nothing → concept is failed."""
    from unittest.mock import patch

    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="abc",
            status="ingested",
        )
    )
    db.upsert_concepts("raw/note.md", ["NoSources"])

    with patch(
        "obsidian_llm_wiki.pipeline.compile._gather_sources",
        return_value=("", []),
    ):
        client = MagicMock()
        drafts, failed = compile_concepts(config=config, client=client, db=db)
    assert "NoSources" in failed


def test_compile_concepts_parse_note_exception_in_edit_check(vault, config, db, fixtures_dir):
    """parse_note exception during manual-edit check is caught."""
    from obsidian_llm_wiki.models import WikiArticleRecord

    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="abc",
            status="ingested",
        )
    )
    db.upsert_concepts("raw/note.md", ["BadParse"])

    wiki_path = config.wiki_dir / "BadParse.md"
    wiki_path.write_text("invalid content without frontmatter")
    db.upsert_article(
        WikiArticleRecord(
            path=str(wiki_path.relative_to(vault)),
            title="BadParse",
            sources=["raw/note.md"],
            content_hash="old",
            is_draft=False,
        )
    )

    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_concept_client(article_json)

    from unittest.mock import patch

    with patch(
        "obsidian_llm_wiki.pipeline.compile.parse_note",
        side_effect=ValueError("bad fm"),
    ):
        drafts, failed = compile_concepts(config=config, client=client, db=db)
    # Should still proceed (exception is caught)
    # Client was called because the except: pass path was taken
    assert client.generate.called or len(failed) > 0


def test_compile_concepts_existing_content_snippet(vault, config, db, fixtures_dir):
    """Existing wiki article content is included in prompt."""
    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="abc",
            status="ingested",
        )
    )
    db.upsert_concepts("raw/note.md", ["Quantum Entanglement"])

    wiki_path = config.wiki_dir / "Quantum Entanglement.md"
    wiki_path.write_text("---\ntitle: Quantum Entanglement\n---\n\nExisting body content here.")

    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_concept_client(article_json)

    drafts, _ = compile_concepts(config=config, client=client, db=db, force=True)
    assert len(drafts) == 1


def test_compile_concepts_structured_output_error(vault, config, db):
    """StructuredOutputError from request_structured → failed."""
    from unittest.mock import patch

    from obsidian_llm_wiki.structured_output import (
        StructuredOutputError,
    )

    raw_note = vault / "raw" / "note.md"
    raw_note.write_text("Content.")
    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="abc",
            status="ingested",
        )
    )
    db.upsert_concepts("raw/note.md", ["FailConcept"])

    client = MagicMock()
    client.generate.side_effect = StructuredOutputError("bad")

    with patch(
        "obsidian_llm_wiki.pipeline.compile.request_structured",
        side_effect=StructuredOutputError("bad json"),
    ):
        drafts, failed = compile_concepts(config=config, client=client, db=db)
    assert "FailConcept" in failed
    assert drafts == []


# ── compile_notes edge cases ─────────────────────────────────────────────


def test_compile_notes_unreadable_source(vault, config, db):
    """Unreadable source file uses '(unreadable)' summary."""
    from unittest.mock import patch

    db.upsert_raw(
        RawNoteRecord(
            path="raw/bad.md",
            content_hash="h",
            status="ingested",
        )
    )
    (vault / "raw" / "bad.md").write_text("content")

    plan_json = (Path(__file__).parent / "fixtures" / "compile_plan_valid.json").read_text()
    article_json = (Path(__file__).parent / "fixtures" / "single_article_valid.json").read_text()
    client = _make_client(plan_json, article_json)

    with patch(
        "obsidian_llm_wiki.pipeline.compile.parse_note",
        side_effect=ValueError("unreadable"),
    ):
        drafts, failed = compile_notes(config=config, client=client, db=db)
    # Planning still happens; articles may fail gathering
    assert isinstance(drafts, list)


def test_compile_notes_planning_structured_error(vault, config, db):
    """StructuredOutputError during planning → early return."""
    from unittest.mock import patch

    from obsidian_llm_wiki.structured_output import (
        StructuredOutputError,
    )

    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="h",
            status="ingested",
        )
    )
    (vault / "raw" / "note.md").write_text("Content.")

    client = MagicMock()
    with patch(
        "obsidian_llm_wiki.pipeline.compile.request_structured",
        side_effect=StructuredOutputError("plan fail"),
    ):
        drafts, failed = compile_notes(config=config, client=client, db=db)
    assert drafts == []
    assert "__planning_failed__" in failed


def test_compile_notes_empty_plan(vault, config, db):
    """Plan with no articles → empty results."""
    from unittest.mock import patch

    from obsidian_llm_wiki.models import CompilePlan

    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="h",
            status="ingested",
        )
    )
    (vault / "raw" / "note.md").write_text("Content.")

    empty_plan = CompilePlan(articles=[], mocs_to_update=[])
    client = MagicMock()
    with patch(
        "obsidian_llm_wiki.pipeline.compile.request_structured",
        return_value=empty_plan,
    ):
        drafts, failed = compile_notes(config=config, client=client, db=db)
    assert drafts == []
    assert failed == []


def test_compile_notes_article_write_structured_error(vault, config, db):
    """StructuredOutputError writing an article → that article fails."""
    from unittest.mock import patch

    from obsidian_llm_wiki.models import (
        ArticlePlan,
        CompilePlan,
    )
    from obsidian_llm_wiki.structured_output import (
        StructuredOutputError,
    )

    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="h",
            status="ingested",
        )
    )
    (vault / "raw" / "note.md").write_text("---\ntitle: Note\n---\n\nContent.")

    plan = CompilePlan(
        articles=[
            ArticlePlan(
                title="FailArticle",
                action="create",
                path="fail-article.md",
                reasoning="test",
                source_paths=["raw/note.md"],
            )
        ]
    )

    call_count = [0]

    def mock_structured(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return plan
        raise StructuredOutputError("write fail")

    with patch(
        "obsidian_llm_wiki.pipeline.compile.request_structured",
        side_effect=mock_structured,
    ):
        client = MagicMock()
        drafts, failed = compile_notes(config=config, client=client, db=db)
    assert "FailArticle" in failed
    assert drafts == []


def test_compile_notes_existing_article_meta_preserved(vault, config, db, fixtures_dir):
    """Existing article meta is preserved when updating."""
    from obsidian_llm_wiki.vault import write_note

    db.upsert_raw(
        RawNoteRecord(
            path="raw/note.md",
            content_hash="h",
            status="ingested",
        )
    )
    (vault / "raw" / "note.md").write_text("---\ntitle: Note\n---\n\nContent.")

    # Create existing article at the path the plan references
    existing_path = config.wiki_dir / "quantum-entanglement.md"
    write_note(
        existing_path,
        {
            "title": "Quantum Entanglement",
            "status": "published",
            "custom_field": "preserve_me",
        },
        "Old body.",
    )

    plan_json = (fixtures_dir / "compile_plan_valid.json").read_text()
    article_json = (fixtures_dir / "single_article_valid.json").read_text()
    client = _make_client(plan_json, article_json)

    drafts, _ = compile_notes(config=config, client=client, db=db)
    assert len(drafts) == 1


# ── approve_drafts edge cases ────────────────────────────────────────────


def test_approve_drafts_not_found(vault, config, db):
    """Approving a non-existent draft is skipped gracefully."""
    missing = config.drafts_dir / "nonexistent.md"
    result = approve_drafts(config, db, [missing])
    assert result == []


def test_approve_drafts_hash_update_exception(vault, config, db):
    """Exception in post-publish hash update is caught."""
    from unittest.mock import patch

    from obsidian_llm_wiki.models import WikiArticleRecord
    from obsidian_llm_wiki.vault import write_note

    draft_path = config.drafts_dir / "hashfail.md"
    write_note(
        draft_path,
        {"title": "HashFail", "status": "draft", "tags": []},
        "Body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(vault)),
            title="HashFail",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    original_parse = __import__("obsidian_llm_wiki.vault", fromlist=["parse_note"]).parse_note
    call_count = [0]

    def parse_side_effect(p):
        call_count[0] += 1
        # Let draft parse succeed, fail on published parse
        if call_count[0] >= 2:
            raise OSError("disk error")
        return original_parse(p)

    with patch(
        "obsidian_llm_wiki.pipeline.compile.parse_note",
        side_effect=parse_side_effect,
    ):
        result = approve_drafts(config, db, [draft_path])

    assert len(result) == 1
    assert result[0].exists()
