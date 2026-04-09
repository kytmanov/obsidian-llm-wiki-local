"""Tests for pipeline/review.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_llm_wiki.pipeline.review import (
    compute_diff,
    compute_rejection_diff,
    list_drafts,
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
def config(vault: Path):
    from obsidian_llm_wiki.config import Config

    return Config(vault=vault)


@pytest.fixture
def db(config):
    return StateDB(config.state_db_path)


def _write_draft(config, title: str, confidence: float = 0.5, sources: list | None = None):
    """Helper: write a draft file with frontmatter."""
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    meta = {
        "title": title,
        "status": "draft",
        "tags": [],
        "sources": sources or [],
        "confidence": confidence,
        "created": "2024-01-01",
        "updated": "2024-01-01",
    }
    body = f"## Overview\n\nThis is a draft about {title}."
    post = fm_lib.Post(body, **meta)
    draft_path = config.drafts_dir / f"{title.replace(' ', '_')}.md"
    atomic_write(draft_path, fm_lib.dumps(post))
    return draft_path


# ── list_drafts ───────────────────────────────────────────────────────────────


def test_list_drafts_empty(config, db):
    assert list_drafts(config, db) == []


def test_list_drafts_returns_summaries(config, db):
    _write_draft(config, "Alpha", confidence=0.8, sources=["raw/a.md", "raw/b.md"])
    _write_draft(config, "Beta", confidence=0.3, sources=["raw/c.md"])
    summaries = list_drafts(config, db)
    assert len(summaries) == 2
    titles = {s.title for s in summaries}
    assert titles == {"Alpha", "Beta"}


def test_list_drafts_confidence_and_sources(config, db):
    _write_draft(config, "Topic", confidence=0.75, sources=["raw/a.md", "raw/b.md"])
    summaries = list_drafts(config, db)
    assert summaries[0].confidence == 0.75
    assert summaries[0].source_count == 2


def test_list_drafts_rejection_count(config, db):
    _write_draft(config, "Rejected Topic")
    db.add_rejection("Rejected Topic", "Too vague")
    db.add_rejection("Rejected Topic", "Wrong tone")
    summaries = list_drafts(config, db)
    assert summaries[0].rejection_count == 2


def test_list_drafts_sorted_by_rejections_desc(config, db):
    _write_draft(config, "High Rejections")
    _write_draft(config, "Low Rejections")
    db.add_rejection("High Rejections", "r1")
    db.add_rejection("High Rejections", "r2")
    db.add_rejection("High Rejections", "r3")
    db.add_rejection("Low Rejections", "r1")
    summaries = list_drafts(config, db)
    assert summaries[0].title == "High Rejections"
    assert summaries[1].title == "Low Rejections"


def test_list_drafts_has_annotations_detected(config, db):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    meta = {
        "title": "Annotated",
        "status": "draft",
        "tags": [],
        "sources": [],
        "confidence": 0.2,
        "created": "2024-01-01",
        "updated": "2024-01-01",
    }
    body = "<!-- olw-auto: low-confidence -->\n\n## Body\n\nContent."
    post = fm_lib.Post(body, **meta)
    atomic_write(config.drafts_dir / "Annotated.md", fm_lib.dumps(post))

    summaries = list_drafts(config, db)
    assert summaries[0].has_annotations is True


def test_list_drafts_has_published_version(config, db):
    _write_draft(config, "Published Article")
    # Create a fake published version — use sanitize_filename to match review.py logic
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write, sanitize_filename

    wiki_path = config.wiki_dir / f"{sanitize_filename('Published Article')}.md"
    post = fm_lib.Post("body", title="Published Article", status="published", tags=[])
    atomic_write(wiki_path, fm_lib.dumps(post))

    summaries = list_drafts(config, db)
    assert summaries[0].has_published_version is True


# ── compute_diff ──────────────────────────────────────────────────────────────


def test_compute_diff_no_published_version(config, db):
    draft_path = _write_draft(config, "New Article")
    wiki_path = config.wiki_dir / "New_Article.md"
    result = compute_diff(draft_path, wiki_path)
    assert result is None


def test_compute_diff_returns_unified_diff(config, db):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    # Write published version
    wiki_path = config.wiki_dir / "Topic.md"
    post = fm_lib.Post(
        "Old content.", title="Topic", status="published", tags=[], sources=[], confidence=0.5
    )
    atomic_write(wiki_path, fm_lib.dumps(post))

    # Write draft with different content
    draft_path = _write_draft(config, "Topic")

    result = compute_diff(draft_path, wiki_path)
    assert result is not None
    assert "---" in result or "@@ " in result or "no differences" in result


def test_compute_diff_no_differences(config, db):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    body = "## Overview\n\nThis is a draft about Same Content."
    meta = {
        "title": "Same Content",
        "status": "published",
        "tags": [],
        "sources": [],
        "confidence": 0.5,
        "created": "2024-01-01",
        "updated": "2024-01-01",
    }
    wiki_path = config.wiki_dir / "Same_Content.md"
    post = fm_lib.Post(body, **meta)
    atomic_write(wiki_path, fm_lib.dumps(post))

    draft_path = _write_draft(config, "Same Content")

    result = compute_diff(draft_path, wiki_path)
    # Both have same body content → "no differences"
    assert result == "(no differences)"


# ── compute_rejection_diff ────────────────────────────────────────────────────


def test_compute_rejection_diff_no_rejections(config, db):
    draft_path = _write_draft(config, "Topic")
    result = compute_rejection_diff(draft_path, db, "Topic")
    assert result is None


def test_compute_rejection_diff_no_body_stored(config, db):
    draft_path = _write_draft(config, "Topic")
    db.add_rejection("Topic", "Bad feedback")  # no body stored
    result = compute_rejection_diff(draft_path, db, "Topic")
    assert result is None


def test_compute_rejection_diff_with_body(config, db):
    draft_path = _write_draft(config, "Topic")
    old_body = "## Old\n\nThis is the old content."
    db.add_rejection("Topic", "Needs improvement", body=old_body)

    result = compute_rejection_diff(draft_path, db, "Topic")
    assert result is not None
    # Should contain diff markers
    assert "---" in result or "@@ " in result or "no differences" in result


# ── compute_diff exception path ───────────────────────────────────────────────


def test_review_menu_uses_v_not_R_for_rejection_diff():
    """Menu string must use [v]iew not [R] — [R] was dead code (lowercased input)."""
    import inspect

    from obsidian_llm_wiki import cli as cli_mod

    src = inspect.getsource(cli_mod)
    assert "[v]iew rejection diff" in src or 'action == "v"' in src
    # The old dead branch must be gone
    assert '"R", "shift+r"' not in src
    assert "shift+r" not in src


def test_compute_diff_unreadable_draft_returns_none(config, db):
    """parse_note failure on draft → returns None, no exception raised."""
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    # Write published version (valid)
    wiki_path = config.wiki_dir / "Broken.md"
    post = fm_lib.Post("body", title="Broken", status="published", tags=[], sources=[])
    atomic_write(wiki_path, fm_lib.dumps(post))

    # Draft path that doesn't exist → parse_note will fail
    draft_path = config.drafts_dir / "Broken.md"
    # Don't create it — file missing triggers exception inside compute_diff

    result = compute_diff(draft_path, wiki_path)
    assert result is None
