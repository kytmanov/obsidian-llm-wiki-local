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


def test_url_wikilinks_not_broken(vault, config, db):
    """[[https://example.com]] and domain/path links must not trigger broken_link."""
    body = "See [[https://example.com/page]] and [[scrummasters.com.ua/book]]."
    _write_page(config, "Alpha", body)
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert not broken


def test_vault_path_fragments_not_broken(vault, config, db):
    """LLM sometimes writes [[wiki/]], [[raw/]], [[source]] as links — not real pages."""
    body = "See [[wiki/]] and [[raw/]] and [[source]] and [[sources]] and [[wiki/.drafts/]]."
    _write_page(config, "Alpha", body)
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert not broken


def test_duplicate_broken_links_deduplicated(vault, config, db):
    """Same broken target appearing multiple times in one page → only one issue."""
    body = "See [[Ghost]] here. Also [[Ghost]] there. And [[Ghost]] again."
    _write_page(config, "Alpha", body)
    result = run_lint(config, db)
    broken = [i for i in result.issues if i.issue_type == "broken_link"]
    assert len(broken) == 1
    assert "Ghost" in broken[0].description


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
    from obsidian_llm_wiki.vault import parse_note

    _, body = parse_note(path)
    correct_hash = hashlib.sha256(body.encode()).hexdigest()
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


# ── Invalid tags ─────────────────────────────────────────────────────────────


def test_invalid_tag_detected(vault, config, db):
    _write_page(
        config,
        "BadTags",
        meta_override={"tags": ["bad tag", "C++"], "status": "published"},
    )
    result = run_lint(config, db)
    tag_issues = [i for i in result.issues if i.issue_type == "invalid_tag"]
    assert tag_issues
    assert "bad tag" in tag_issues[0].description


def test_valid_tags_no_issue(vault, config, db):
    _write_page(
        config,
        "GoodTags",
        meta_override={"tags": ["physics", "machine-learning"], "status": "published"},
    )
    result = run_lint(config, db)
    tag_issues = [i for i in result.issues if i.issue_type == "invalid_tag"]
    assert not tag_issues


def test_fix_mode_sanitizes_tags(vault, config, db):
    import frontmatter as fm

    path = _write_page(
        config,
        "FixTags",
        meta_override={"tags": ["bad tag", "physics"], "status": "published"},
    )
    run_lint(config, db, fix=True)
    post = fm.load(str(path))
    assert "bad-tag" in post.metadata["tags"]
    assert "bad tag" not in post.metadata["tags"]


def test_lint_checks_source_pages(vault, config, db):
    """Tags in wiki/sources/ pages are also checked."""
    sources_dir = config.wiki_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    src_path = sources_dir / "MySource.md"
    write_note(src_path, {"title": "MySource", "tags": ["bad tag"], "status": "published"}, "Body.")
    result = run_lint(config, db)
    tag_issues = [i for i in result.issues if i.issue_type == "invalid_tag"]
    assert any("sources" in i.path for i in tag_issues)


# ── Summary string ────────────────────────────────────────────────────────────


def test_summary_mentions_issue_counts(vault, config, db):
    _write_page(config, "Solo", "No links.")  # orphan
    result = run_lint(config, db)
    assert "orphan" in result.summary


def test_summary_healthy_when_no_issues(vault, config, db):
    result = run_lint(config, db)
    assert "healthy" in result.summary.lower()
