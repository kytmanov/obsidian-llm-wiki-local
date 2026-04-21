"""Tests for v0.2 compile.py additions: annotations, rejection feedback, selective compile."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord, WikiArticleRecord
from obsidian_llm_wiki.ollama_client import OllamaClient
from obsidian_llm_wiki.pipeline.compile import (
    _build_olw_annotations,
    _gather_sources,
    _inject_body_sections,
    _strip_olw_annotations,
    _write_concept_prompt,
    approve_drafts,
    compile_concepts,
    reject_draft,
)
from obsidian_llm_wiki.state import StateDB


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".olw").mkdir()
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


def make_mock_client(response: str = "{}") -> OllamaClient:
    client = MagicMock(spec=OllamaClient)
    client.generate.return_value = response
    return client


# ── Annotations ───────────────────────────────────────────────────────────────


def test_build_annotations_low_confidence(db):
    annotations = _build_olw_annotations(0.3, ["raw/a.md"], db)
    assert any("low-confidence" in a for a in annotations)


def test_build_annotations_single_source(db):
    annotations = _build_olw_annotations(0.8, ["raw/a.md"], db)
    assert any("single-source" in a for a in annotations)


def test_build_no_annotations_above_threshold(db):
    rec_a = RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested", quality="high")
    rec_b = RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested", quality="high")
    db.upsert_raw(rec_a)
    db.upsert_raw(rec_b)
    annotations = _build_olw_annotations(0.75, ["raw/a.md", "raw/b.md"], db)
    # High confidence, multiple sources, high quality → no annotations
    assert annotations == []


def test_build_annotations_all_low_quality(db):
    rec_a = RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested", quality="low")
    rec_b = RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested", quality="low")
    db.upsert_raw(rec_a)
    db.upsert_raw(rec_b)
    annotations = _build_olw_annotations(0.6, ["raw/a.md", "raw/b.md"], db)
    assert any("low-quality" in a for a in annotations)


def test_strip_annotations_removes_olw_auto(db):
    body = "<!-- olw-auto: low-confidence -->\n\n## Overview\n\nContent."
    stripped = _strip_olw_annotations(body)
    assert "olw-auto" not in stripped
    assert "## Overview" in stripped
    assert "Content." in stripped


def test_strip_annotations_leaves_user_html_comments(db):
    body = "<!-- user comment -->\n\n<!-- olw-auto: single-source -->\n\nContent."
    stripped = _strip_olw_annotations(body)
    assert "<!-- user comment -->" in stripped
    assert "olw-auto" not in stripped


def test_strip_annotations_no_annotations_unchanged(db):
    body = "## Title\n\nNo annotations here."
    assert _strip_olw_annotations(body) == body


def test_inject_body_sections_always_includes_sources_heading(config):
    body = _inject_body_sections("## Overview\n\nContent.", [], config)
    assert "## Sources" in body


def test_annotations_stripped_on_approve(config, db, tmp_path):
    """Annotations must be absent in published articles."""
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write, parse_note

    annotated_body = "<!-- olw-auto: low-confidence (0.25) -->\n\n## Overview\n\nContent."
    meta = {
        "title": "Test Article",
        "status": "draft",
        "tags": [],
        "sources": [],
        "confidence": 0.25,
        "created": "2024-01-01",
        "updated": "2024-01-01",
    }
    draft_path = config.drafts_dir / "Test_Article.md"
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post(annotated_body, **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Test Article",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    published = approve_drafts(config, db, [draft_path])
    assert len(published) == 1
    _, pub_body = parse_note(published[0])
    assert "olw-auto" not in pub_body
    assert "## Overview" in pub_body


# ── Rejection feedback in prompt ──────────────────────────────────────────────


def test_write_concept_prompt_no_rejection_history():
    prompt = _write_concept_prompt("Quantum Computing", "source text", [])
    assert "PREVIOUS REJECTIONS" not in prompt


def test_write_concept_prompt_with_rejection_history():
    history = ["Too vague", "Wrong format"]
    prompt = _write_concept_prompt(
        "Quantum Computing", "source text", [], rejection_history=history
    )
    assert "PREVIOUS REJECTIONS" in prompt
    assert "Too vague" in prompt
    assert "Wrong format" in prompt


def test_write_concept_prompt_deduplicates_feedback():
    history = ["Same feedback", "Same feedback", "Different feedback"]
    prompt = _write_concept_prompt("Topic", "source", [], rejection_history=history)
    # Should appear only once
    assert prompt.count("Same feedback") == 1
    assert "Different feedback" in prompt


# ── reject_draft stores feedback and body ────────────────────────────────────


def test_reject_draft_stores_feedback(config, db, tmp_path):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    draft_path = config.drafts_dir / "Test.md"
    meta = {"title": "Test Topic", "status": "draft", "tags": [], "sources": [], "confidence": 0.5}
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("Draft body.", **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Test Topic",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    reject_draft(draft_path, config, db, feedback="Needs more detail")

    assert not draft_path.exists()
    rejections = db.get_rejections("Test Topic")
    assert len(rejections) == 1
    assert rejections[0]["feedback"] == "Needs more detail"
    assert rejections[0]["body"] == "Draft body."


def test_reject_draft_no_feedback_no_rejection(config, db):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    draft_path = config.drafts_dir / "Test.md"
    meta = {"title": "Test", "status": "draft", "tags": [], "sources": [], "confidence": 0.5}
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("body", **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Test",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    reject_draft(draft_path, config, db, feedback="")
    assert db.rejection_count("Test") == 0


def test_reject_draft_blocks_after_cap(config, db):
    """5th rejection auto-blocks the concept."""
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    for i in range(4):
        db.add_rejection("Blocked Topic", f"feedback {i}", body="body")

    # 5th rejection via reject_draft
    draft_path = config.drafts_dir / "Blocked.md"
    meta = {
        "title": "Blocked Topic",
        "status": "draft",
        "tags": [],
        "sources": [],
        "confidence": 0.5,
    }  # noqa: E501
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("body", **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Blocked Topic",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    reject_draft(draft_path, config, db, feedback="Still wrong")
    assert db.is_concept_blocked("Blocked Topic")


# ── Selective compile (concepts= param) ───────────────────────────────────────


def test_compile_concepts_selective(config, db, tmp_path):
    """compile_concepts(concepts=['Alpha']) should only compile Alpha, not Beta."""
    import json

    # Set up two concepts needing compile
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])

    # Write source files
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent about Alpha.")
    (config.vault / "raw" / "b.md").write_text("---\ntitle: B\n---\nContent about Beta.")

    mock_response = json.dumps({"title": "Alpha", "content": "Alpha content.", "tags": []})
    client = make_mock_client(mock_response)

    drafts, failed, _ = compile_concepts(config, client, db, concepts=["Alpha"])
    assert len(drafts) == 1
    assert "Alpha" in drafts[0].name or "alpha" in drafts[0].name.lower()
    # Beta should still be needing compile
    needing = db.concepts_needing_compile()
    assert "Beta" in needing


def test_compile_concepts_none_compiles_all(config, db):
    """concepts=None falls back to all needing compile."""
    import json

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Topic"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent.")

    mock_response = json.dumps({"title": "Topic", "content": "Content.", "tags": []})
    client = make_mock_client(mock_response)

    drafts, failed, _ = compile_concepts(config, client, db, concepts=None)
    assert len(drafts) == 1


# ── Approval records approved_at ─────────────────────────────────────────────


def test_approve_drafts_sets_approved_at(config, db):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    draft_path = config.drafts_dir / "Article.md"
    meta = {"title": "Article", "status": "draft", "tags": [], "sources": [], "confidence": 0.7}
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("## Body\n\nContent.", **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Article",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    published = approve_drafts(config, db, [draft_path])
    assert len(published) == 1
    art = db.get_article(str(published[0].relative_to(config.vault)))
    assert art is not None
    assert art.approved_at is not None


# ── _gather_sources ───────────────────────────────────────────────────────────


def test_gather_sources_missing_file_skipped(vault):
    """Source path that doesn't exist → skipped, no crash."""
    text, resolved = _gather_sources(["raw/nonexistent.md"], vault)
    assert text == ""
    assert resolved == []


def test_gather_sources_unreadable_file_skipped(vault):
    """Source file that can't be parsed → skipped, no crash."""
    bad = vault / "raw" / "bad.md"
    bad.write_bytes(b"\xff\xfe")  # invalid UTF-8 triggers parse error
    text, resolved = _gather_sources(["raw/bad.md"], vault)
    # May or may not resolve depending on parse_note tolerance — just no exception
    assert isinstance(text, str)
    assert isinstance(resolved, list)


def test_gather_sources_combines_multiple(vault):
    (vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent A.")
    (vault / "raw" / "b.md").write_text("---\ntitle: B\n---\nContent B.")
    text, resolved = _gather_sources(["raw/a.md", "raw/b.md"], vault)
    assert "Content A." in text
    assert "Content B." in text
    assert len(resolved) == 2


def test_gather_sources_bare_filename_resolved(vault):
    """Model sometimes returns bare filename without raw/ prefix."""
    (vault / "raw" / "note.md").write_text("---\ntitle: Note\n---\nBody.")
    text, resolved = _gather_sources(["note.md"], vault)
    assert "Body." in text
    assert len(resolved) == 1


# ── Stub compile path ─────────────────────────────────────────────────────────


def test_compile_concepts_stub_produces_draft(config, db):
    """Stub concept → fast model called → draft created → stub removed from DB."""
    import json

    db.add_stub("Orphan Topic")
    mock_response = json.dumps({"title": "Orphan Topic", "content": "Stub content.", "tags": []})
    client = make_mock_client(mock_response)

    drafts, failed, _ = compile_concepts(config, client, db)
    assert len(drafts) == 1
    assert not db.has_stub("Orphan Topic")
    assert "Orphan" in drafts[0].name or "orphan" in drafts[0].name.lower()


def test_compile_concepts_stub_failure_adds_to_failed(config, db):
    """StructuredOutputError during stub compile → concept added to failed."""

    db.add_stub("Bad Stub")
    client = make_mock_client("not valid json at all {{{")
    # structured_output will exhaust retries and raise StructuredOutputError
    # Mock client to always return garbage
    client.generate.return_value = "garbage"

    drafts, failed, _ = compile_concepts(config, client, db)
    assert "Bad Stub" in failed


def test_compile_concepts_stub_llm_bad_request_adds_to_failed(config, db):
    """LLMBadRequestError during stub compile → concept in failed, no crash."""
    from obsidian_llm_wiki.openai_compat_client import LLMBadRequestError

    db.add_stub("Context Overflow Stub")
    client = make_mock_client()
    client.generate.side_effect = LLMBadRequestError("HTTP 400: context too long")

    drafts, failed, _ = compile_concepts(config, client, db)
    assert "Context Overflow Stub" in failed
    assert drafts == []


def test_compile_concepts_article_llm_bad_request_adds_to_failed(config, db):
    """LLMBadRequestError during article compile → concept in failed, no crash."""
    from obsidian_llm_wiki.openai_compat_client import LLMBadRequestError

    db.upsert_raw(RawNoteRecord(path="raw/src.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/src.md", ["Heavy Concept"])
    (config.vault / "raw" / "src.md").write_text(
        "---\ntitle: Source\n---\nContent about Heavy Concept."
    )

    client = make_mock_client()
    client.generate.side_effect = LLMBadRequestError("HTTP 400: n_keep > ctx")

    drafts, failed, _ = compile_concepts(config, client, db)
    assert "Heavy Concept" in failed
    assert drafts == []
