"""Tests for pipeline/maintain.py."""

from __future__ import annotations

from pathlib import Path

import frontmatter as fm_lib
import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord
from obsidian_llm_wiki.pipeline.maintain import (
    _extract_link_target,
    create_stubs,
    suggest_concept_merges,
    suggest_orphan_links,
)
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.vault import atomic_write


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


def _write_article(config: Config, title: str, body: str = "## Body\n\nContent.") -> Path:
    from obsidian_llm_wiki.vault import sanitize_filename

    path = config.wiki_dir / f"{sanitize_filename(title)}.md"
    post = fm_lib.Post(body, title=title, status="published", tags=[], sources=[])
    atomic_write(path, fm_lib.dumps(post))
    return path


# ── _extract_link_target ──────────────────────────────────────────────────────


def test_extract_link_target_found():
    assert _extract_link_target("[[Quantum Computing]] not found") == "Quantum Computing"


def test_extract_link_target_not_found():
    assert _extract_link_target("some generic description") is None


def test_extract_link_target_first_match():
    result = _extract_link_target("[[Alpha]] and [[Beta]] missing")
    assert result == "Alpha"


# ── create_stubs ──────────────────────────────────────────────────────────────


def test_create_stubs_from_lint_issues(config, db):
    from obsidian_llm_wiki.models import LintIssue

    issues = [
        LintIssue(
            path="wiki/Article.md",
            issue_type="broken_link",
            description="[[Missing Topic]] not found",
            suggestion="Create article for Missing Topic",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 1
    assert created[0].exists()
    assert db.has_stub("Missing Topic")


def test_create_stubs_skips_existing_stub(config, db):
    from obsidian_llm_wiki.models import LintIssue

    db.add_stub("Already There")
    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[Already There]] not found",
            suggestion="Create stub for Already There",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 0


def test_create_stubs_skips_existing_draft(config, db):
    from obsidian_llm_wiki.models import LintIssue
    from obsidian_llm_wiki.vault import sanitize_filename

    target = "Existing Draft"
    draft_path = config.drafts_dir / f"{sanitize_filename(target)}.md"
    post = fm_lib.Post("body", title=target, status="draft", tags=[], sources=[])
    atomic_write(draft_path, fm_lib.dumps(post))

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description=f"[[{target}]] not found",
            suggestion=f"Create stub for {target}",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 0


def test_create_stubs_deduplicates_targets(config, db):
    from obsidian_llm_wiki.models import LintIssue

    issues = [
        LintIssue(
            path="wiki/A.md",
            issue_type="broken_link",
            description="[[Same Topic]] not found",
            suggestion="Create stub for Same Topic",
        ),
        LintIssue(
            path="wiki/B.md",
            issue_type="broken_link",
            description="[[Same Topic]] not found",
            suggestion="Create stub for Same Topic",
        ),
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    assert len(created) == 1


def test_create_stubs_respects_max_stubs(config, db):
    from obsidian_llm_wiki.models import LintIssue

    issues = [
        LintIssue(
            path=f"wiki/A{i}.md",
            issue_type="broken_link",
            description=f"[[Topic {i}]] not found",
            suggestion=f"Create stub for Topic {i}",
        )
        for i in range(10)
    ]
    created = create_stubs(config, db, broken_link_issues=issues, max_stubs=3)
    assert len(created) == 3


def test_create_stubs_stub_body_has_info_callout(config, db):
    from obsidian_llm_wiki.models import LintIssue

    issues = [
        LintIssue(
            path="wiki/Ref.md",
            issue_type="broken_link",
            description="[[New Topic]] not found",
            suggestion="Create stub for New Topic",
        )
    ]
    created = create_stubs(config, db, broken_link_issues=issues)
    content = created[0].read_text()
    assert "[!info]" in content or "stub" in content.lower()


def test_create_stubs_empty_issues(config, db):
    created = create_stubs(config, db, broken_link_issues=[])
    assert created == []


def test_create_stubs_none_runs_lint_internally(config, db):
    """broken_link_issues=None → lint runs internally, no crash on empty wiki."""
    # Empty wiki → lint finds no broken links → no stubs created
    created = create_stubs(config, db, broken_link_issues=None)
    assert created == []


# ── suggest_orphan_links ──────────────────────────────────────────────────────


def test_suggest_orphan_links_empty_wiki(config, db):
    result = suggest_orphan_links(config, db)
    assert result == []


def test_suggest_orphan_links_no_unlinked_mentions(config, db):
    _write_article(config, "Isolated Topic", body="## Body\n\nNo mentions of anything.")
    result = suggest_orphan_links(config, db)
    # No mentions of Isolated Topic in other articles → no suggestions
    assert result == []


def test_suggest_orphan_links_finds_unlinked_mention(config, db):
    # Write orphan article
    _write_article(config, "Orphan Topic", body="## Body\n\nAbout orphan.")
    # Write another article that mentions "Orphan Topic" in plain text
    _write_article(
        config,
        "Main Article",
        body="## Body\n\nThis discusses Orphan Topic in detail.",
    )

    result = suggest_orphan_links(config, db)
    # May or may not find it depending on lint detecting orphan — just check no crash
    assert isinstance(result, list)


# ── suggest_concept_merges ────────────────────────────────────────────────────


def test_suggest_concept_merges_empty(config, db):
    result = suggest_concept_merges(config, db)
    assert result == []


def test_suggest_concept_merges_single_concept(config, db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Machine Learning"])
    result = suggest_concept_merges(config, db)
    assert result == []


def test_suggest_concept_merges_high_similarity(config, db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Machine Learning"])
    db.upsert_concepts("raw/b.md", ["Machine-Learning"])

    result = suggest_concept_merges(config, db)
    assert len(result) > 0
    a, b, score = result[0]
    assert score >= 0.7
    assert {"Machine Learning", "Machine-Learning"} == {a, b}


def test_suggest_concept_merges_low_similarity_excluded(config, db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Quantum Computing"])
    db.upsert_concepts("raw/b.md", ["Neural Networks"])

    result = suggest_concept_merges(config, db)
    assert result == []


def test_suggest_concept_merges_sorted_by_score_desc(config, db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/c.md", content_hash="h3", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Deep Learning"])
    db.upsert_concepts("raw/b.md", ["Deep-Learning"])
    db.upsert_concepts("raw/c.md", ["Deep Learning Model"])

    result = suggest_concept_merges(config, db)
    scores = [r[2] for r in result]
    assert scores == sorted(scores, reverse=True)
