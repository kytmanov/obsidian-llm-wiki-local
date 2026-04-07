"""Tests for pipeline/lint.py — no LLM, no Ollama required."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import WikiArticleRecord
from obsidian_llm_wiki.pipeline.lint import run_lint
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.vault import write_note


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


def _write_page(
    config: Config, title: str, body: str = "", meta_override: dict | None = None
) -> Path:
    meta = {"title": title, "tags": ["test"], "status": "published"}
    if meta_override:
        meta.update(meta_override)
    path = config.wiki_dir / f"{title}.md"
    write_note(path, meta, body or f"Content about {title}.")
    return path


# ── Health score ──────────────────────────────────────────────────────────────


def test_no_pages_returns_healthy(vault, config, db):
    result = run_lint(config, db)
    assert result.health_score == 100.0
    assert result.issues == []


def test_clean_wiki_scores_100(vault, config, db):
    _write_page(config, "Quantum Computing", "See also [[Machine Learning]].")
    _write_page(config, "Machine Learning", "Related to [[Quantum Computing]].")
    result = run_lint(config, db)
    # Both pages link to each other — no orphans; no broken links; all fields present
    orphan_issues = [i for i in result.issues if i.issue_type == "orphan"]
    broken_issues = [i for i in result.issues if i.issue_type == "broken_link"]
    assert not orphan_issues
    assert not broken_issues


# ── Missing frontmatter ───────────────────────────────────────────────────────


def test_missing_frontmatter_detected(vault, config, db):
    # Write a page without frontmatter
    path = config.wiki_dir / "Bare.md"
    path.write_text("Just a body, no frontmatter.", encoding="utf-8")

    result = run_lint(config, db)
    types = [i.issue_type for i in result.issues]
    assert "missing_frontmatter" in types


def test_missing_fields_reported(vault, config, db):
    # Write page missing 'tags' and 'status'
    path = config.wiki_dir / "NoTags.md"
    write_note(path, {"title": "NoTags"}, "Content.")

    result = run_lint(config, db)
    missing_issues = [i for i in result.issues if i.issue_type == "missing_frontmatter"]
    assert missing_issues
    assert any("tags" in i.description or "status" in i.description for i in missing_issues)


def test_fix_mode_adds_missing_fields(vault, config, db):
    path = config.wiki_dir / "NoStatus.md"
    write_note(path, {"title": "NoStatus", "tags": []}, "Body.")

    run_lint(config, db, fix=True)

    import frontmatter

    post = frontmatter.load(str(path))
    assert "status" in post.metadata


# ── Orphan detection ──────────────────────────────────────────────────────────


def test_orphan_detected(vault, config, db):
    _write_page(config, "Isolated Page", "No links to or from anywhere.")
    result = run_lint(config, db)
    orphans = [i for i in result.issues if i.issue_type == "orphan"]
    assert orphans
    assert "Isolated Page" in orphans[0].path


def test_orphan_not_flagged_when_linked(vault, config, db):
    _write_page(config, "Alpha", "See [[Beta]].")
    _write_page(config, "Beta", "See [[Alpha]].")
    result = run_lint(config, db)
    orphans = [i for i in result.issues if i.issue_type == "orphan"]
    assert not orphans


def test_index_md_not_checked(vault, config, db):
    """index.md and log.md are system files — skip them."""
    (config.wiki_dir / "index.md").write_text("# Index\n", encoding="utf-8")
    (config.wiki_dir / "log.md").write_text("# Log\n", encoding="utf-8")
    result = run_lint(config, db)
    paths = [i.path for i in result.issues]
    assert not any("index.md" in p or "log.md" in p for p in paths)


# ── Broken links ──────────────────────────────────────────────────────────────


def test_broken_wikilink_detected(vault, config, db):
    _write_page(config, "Alpha", "See [[Ghost Page]] for details.")
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert broken
    assert "Ghost Page" in broken[0].description


def test_valid_wikilink_not_broken(vault, config, db):
    _write_page(config, "Alpha", "See [[Beta]] for details.")
    _write_page(config, "Beta", "Linked from Alpha.")
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert not broken


# ── Low confidence ────────────────────────────────────────────────────────────


def test_low_confidence_detected(vault, config, db):
    _write_page(
        config,
        "Weak",
        meta_override={"confidence": 0.1, "title": "Weak", "tags": [], "status": "published"},
    )
    result = run_lint(config, db)
    low = [i for i in result.issues if i.issue_type == "low_confidence"]
    assert low


def test_high_confidence_not_flagged(vault, config, db):
    _write_page(
        config,
        "Strong",
        meta_override={"confidence": 0.8, "title": "Strong", "tags": [], "status": "published"},
    )
    result = run_lint(config, db)
    low = [i for i in result.issues if i.issue_type == "low_confidence"]
    assert not low


# ── Stale (manually edited) ───────────────────────────────────────────────────


def test_stale_detected_on_hash_mismatch(vault, config, db):
    path = _write_page(config, "Edited")
    rel = str(path.relative_to(vault))
    # Register with a WRONG hash
    db.upsert_article(
        WikiArticleRecord(
            path=rel,
            title="Edited",
            sources=[],
            content_hash="wrong_hash",
            is_draft=False,
        )
    )

    result = run_lint(config, db)
    stale = [i for i in result.issues if i.issue_type == "stale"]
    assert stale


def test_not_stale_when_hash_matches(vault, config, db):
    import hashlib

    path = _write_page(config, "Fresh")
    rel = str(path.relative_to(vault))
    correct_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    db.upsert_article(
        WikiArticleRecord(
            path=rel,
            title="Fresh",
            sources=[],
            content_hash=correct_hash,
            is_draft=False,
        )
    )

    result = run_lint(config, db)
    stale = [i for i in result.issues if i.issue_type == "stale"]
    assert not stale


# ── Summary string ────────────────────────────────────────────────────────────


def test_summary_mentions_issue_counts(vault, config, db):
    _write_page(config, "Solo", "No links.")  # orphan
    result = run_lint(config, db)
    assert "orphan" in result.summary


def test_summary_healthy_when_no_issues(vault, config, db):
    result = run_lint(config, db)
    assert "healthy" in result.summary.lower()


# ── Draft files skipped in indexes ────────────────────────────────────────


def test_drafts_skipped_in_title_index(vault, config, db):
    """Files under .drafts/ must not appear in the title index."""
    draft = config.drafts_dir / "Draft Note.md"
    write_note(draft, {"title": "Draft Note", "tags": [], "status": "draft"}, "Draft.")
    _write_page(config, "Real Page", "See [[Draft Note]].")
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    # Draft is NOT in title index → [[Draft Note]] is broken
    assert any("Draft Note" in i.description for i in broken)


def test_drafts_skipped_in_inbound_index(vault, config, db):
    """Links from draft pages must not count as inbound links."""
    draft = config.drafts_dir / "Linker.md"
    write_note(
        draft,
        {"title": "Linker", "tags": [], "status": "draft"},
        "See [[Orphan Target]].",
    )
    _write_page(config, "Orphan Target", "Content.")
    result = run_lint(config, db)
    orphans = [i for i in result.issues if i.issue_type == "orphan"]
    assert any("Orphan Target" in i.path for i in orphans)


# ── Parse exceptions in index builders ────────────────────────────────────


def test_title_index_parse_exception(vault, config, db):
    """Binary file in wiki/ should not crash _build_title_index."""
    (config.wiki_dir / "Binary.md").write_bytes(b"\x80\x81\x82")
    # Should not raise; the bad file is silently skipped
    result = run_lint(config, db)
    # Binary.md is a concept page → parse fails → missing_frontmatter
    fm = [i for i in result.issues if i.issue_type == "missing_frontmatter"]
    assert any("Failed to parse" in i.description for i in fm)


def test_inbound_index_parse_exception(vault, config, db):
    """Binary file should not crash _build_inbound_index."""
    (config.wiki_dir / "Bad.md").write_bytes(b"\x80\x81\x82")
    _write_page(config, "Good Page", "Content.")
    # Should complete without error
    result = run_lint(config, db)
    assert result is not None


# ── _concept_pages: wiki_dir doesn't exist ────────────────────────────────


def test_concept_pages_missing_wiki_dir(tmp_path):
    """If wiki_dir doesn't exist, _concept_pages returns []."""
    from obsidian_llm_wiki.pipeline.lint import _concept_pages

    cfg = Config(vault=tmp_path)
    # wiki/ directory not created
    assert _concept_pages(cfg) == []


# ── Parse exception on concept page (lines 103-113) ──────────────────────


def test_parse_exception_yields_missing_frontmatter(vault, config, db):
    """Unparseable concept page → missing_frontmatter issue."""
    (config.wiki_dir / "Corrupt.md").write_bytes(b"\x80\x81\x82")
    result = run_lint(config, db)
    fm = [i for i in result.issues if i.issue_type == "missing_frontmatter"]
    match = [i for i in fm if "Failed to parse" in i.description]
    assert match
    assert not match[0].auto_fixable


# ── Fix mode: title and tags fields ──────────────────────────────────────


def test_fix_mode_adds_title_and_tags(vault, config, db):
    """Fix mode should add title (from stem) and tags ([])."""
    import frontmatter

    path = config.wiki_dir / "NeedsFix.md"
    write_note(path, {"status": "published"}, "Some body.")

    run_lint(config, db, fix=True)

    post = frontmatter.load(str(path))
    assert post.metadata["title"] == "NeedsFix"
    assert post.metadata["tags"] == []
    assert post.metadata["status"] == "published"


# ── Non-numeric confidence ────────────────────────────────────────────────


def test_non_numeric_confidence_ignored(vault, config, db):
    """confidence: 'not_a_number' should not raise or flag."""
    _write_page(
        config,
        "WeirdConf",
        meta_override={
            "confidence": "not_a_number",
            "title": "WeirdConf",
            "tags": [],
            "status": "published",
        },
    )
    result = run_lint(config, db)
    low = [i for i in result.issues if i.issue_type == "low_confidence"]
    assert not low
