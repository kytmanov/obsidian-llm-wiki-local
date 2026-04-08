"""Tests for pipeline/ingest.py — no Ollama required (mocked client)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.pipeline.ingest import (
    _build_analysis_prompt,
    _normalize_concept_names,
    _preprocess_web_clip,
    ingest_note,
)
from obsidian_llm_wiki.state import StateDB

# ── Fixtures ──────────────────────────────────────────────────────────────────


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


def _make_client(analysis_json: str) -> MagicMock:
    client = MagicMock()
    client.generate.return_value = analysis_json
    return client


def _write_raw(vault: Path, name: str, content: str) -> Path:
    p = vault / "raw" / name
    p.write_text(content, encoding="utf-8")
    return p


# ── _preprocess_web_clip ──────────────────────────────────────────────────────


def test_preprocess_strips_html_tags():
    content = (
        "<nav>Skip Navigation Menu</nav>\n\n"
        "# Real Content\n\n"
        "Full paragraph with enough words to pass the filter."
    )
    result = _preprocess_web_clip(content)
    assert "<nav>" not in result
    assert "Real Content" in result
    assert "Full paragraph" in result


def test_preprocess_strips_short_header_lines():
    # Short plain-text lines in first 30 lines (nav/banner) should be stripped
    # But markdown headings (starting with #) must be kept even if short
    lines = [
        "Home",
        "About",
        "Contact",
        "",
        "# Article Title",
        "",
        "This is a full substantive paragraph with many words that will not be stripped.",
    ]
    result = _preprocess_web_clip("\n".join(lines))
    assert "Home" not in result
    assert "Article Title" in result
    assert "substantive paragraph" in result


def test_preprocess_preserves_short_body_lines():
    """Short lines AFTER line 30 must NOT be stripped (bullets, code comments, etc.)."""
    header = ["Nav item"] * 31  # push past the 30-line scan window
    body = ["- Key insight", "- Another bullet", "Short sentence."]
    content = "\n".join(header + body)
    result = _preprocess_web_clip(content)
    assert "Key insight" in result
    assert "Another bullet" in result


def test_preprocess_preserves_body_html():
    """HTML after line 30 (body content) must be preserved."""
    header = ["Nav item"] * 31  # push past the 30-line scan window
    body = [
        "<details><summary>Collapse me</summary>",
        "Hidden content here.",
        "</details>",
        "Use <kbd>Ctrl+C</kbd> to copy.",
    ]
    content = "\n".join(header + body)
    result = _preprocess_web_clip(content)
    assert "<details>" in result
    assert "<kbd>Ctrl+C</kbd>" in result


def test_preprocess_strips_header_html():
    """HTML tags in first 30 lines must be stripped."""
    content = "<nav>Skip Navigation</nav>\n\n# Real Title\n\nBody content here."
    result = _preprocess_web_clip(content)
    assert "<nav>" not in result
    assert "Real Title" in result


def test_preprocess_preserves_blank_lines():
    content = "Home\n\n# Title\n\nContent."
    result = _preprocess_web_clip(content)
    assert "Title" in result


# ── _build_analysis_prompt ────────────────────────────────────────────────────


def test_build_prompt_includes_body():
    prompt = _build_analysis_prompt("Some content here.", [])
    assert "Some content here" in prompt


def test_build_prompt_includes_existing_concepts():
    prompt = _build_analysis_prompt("content", ["Quantum Computing", "Machine Learning"])
    assert "Quantum Computing" in prompt
    assert "Machine Learning" in prompt


def test_build_prompt_truncates_long_body():
    long_body = "x " * 5000  # way over 4000 chars
    prompt = _build_analysis_prompt(long_body, [])
    # Prompt body portion should not be full 10000+ chars
    assert len(prompt) < 6000


def test_build_prompt_warns_on_truncation(caplog):
    import logging

    long_body = "word " * 1000  # ~5000 chars
    with caplog.at_level(logging.WARNING, logger="obsidian_llm_wiki.pipeline.ingest"):
        _build_analysis_prompt(long_body, [], path_name="test.md")
    assert "truncated" in caplog.text.lower()


def test_build_prompt_no_warning_for_short_body(caplog):
    import logging

    short_body = "word " * 100
    with caplog.at_level(logging.WARNING, logger="obsidian_llm_wiki.pipeline.ingest"):
        _build_analysis_prompt(short_body, [], path_name="test.md")
    assert "truncated" not in caplog.text.lower()


# ── _normalize_concept_names ──────────────────────────────────────────────────


def test_normalize_reuses_canonical_case(vault, config, db):
    db.upsert_concepts("raw/a.md", ["Quantum Computing"])
    result = _normalize_concept_names(["quantum computing"], db)
    assert result == ["Quantum Computing"]  # canonical form preserved


def test_normalize_deduplicates(vault, config, db):
    result = _normalize_concept_names(["ML", "ML", "Machine Learning"], db)
    assert len(result) == 2
    assert "ML" in result


def test_normalize_strips_empty(vault, config, db):
    result = _normalize_concept_names(["", "  ", "Neural Networks"], db)
    assert "" not in result
    assert "  " not in result
    assert "Neural Networks" in result


# ── ingest_note ───────────────────────────────────────────────────────────────


def _analysis_json(concepts=None, quality="high", summary="A summary."):
    return json.dumps(
        {
            "summary": summary,
            "key_concepts": concepts or ["Quantum Computing", "Qubit"],
            "suggested_topics": ["Quantum Computing"],
            "quality": quality,
        }
    )


def test_ingest_note_returns_analysis_result(vault, config, db):
    path = _write_raw(vault, "quantum.md", "# Quantum Computing\n\nQubits are awesome.")
    client = _make_client(_analysis_json())
    result = ingest_note(path, config, client, db)
    assert result is not None
    assert result.quality == "high"
    assert len(result.key_concepts) >= 1


def test_ingest_note_stores_status_ingested(vault, config, db):
    path = _write_raw(vault, "note.md", "# Note\n\nSome content here.")
    client = _make_client(_analysis_json())
    ingest_note(path, config, client, db)
    rec = db.get_raw("raw/note.md")
    assert rec is not None
    assert rec.status == "ingested"


def test_ingest_note_skip_already_ingested(vault, config, db):
    path = _write_raw(vault, "dup.md", "# Dup\n\nContent.")
    client = _make_client(_analysis_json())
    ingest_note(path, config, client, db)
    # Second call without force — should skip
    result = ingest_note(path, config, client, db)
    assert result is None
    # Client called only once (for first ingest)
    assert client.generate.call_count == 1


def test_ingest_note_force_reingest(vault, config, db):
    path = _write_raw(vault, "forceme.md", "# Force\n\nContent.")
    client = _make_client(_analysis_json())
    ingest_note(path, config, client, db)
    result = ingest_note(path, config, client, db, force=True)
    assert result is not None
    assert client.generate.call_count == 2


def test_ingest_note_dedup_by_hash(vault, config, db):
    """Same content in two files → second skipped as duplicate."""
    content = "# Same\n\nIdentical body content here."
    p1 = _write_raw(vault, "first.md", content)
    p2 = _write_raw(vault, "second.md", content)
    client = _make_client(_analysis_json())
    ingest_note(p1, config, client, db)
    result = ingest_note(p2, config, client, db)
    assert result is None
    assert client.generate.call_count == 1


def test_ingest_note_stores_concepts(vault, config, db):
    path = _write_raw(vault, "ml.md", "# ML\n\nNeural networks and backprop.")
    client = _make_client(_analysis_json(concepts=["Neural Networks", "Backpropagation"]))
    ingest_note(path, config, client, db)
    names = db.list_all_concept_names()
    assert "Neural Networks" in names
    assert "Backpropagation" in names


def test_ingest_note_failure_marks_db_status(vault, config, db):
    path = _write_raw(vault, "fail.md", "# Fail\n\nContent.")
    client = MagicMock()
    client.generate.side_effect = RuntimeError("Ollama timeout")
    result = ingest_note(path, config, client, db)
    assert result is None
    rec = db.get_raw("raw/fail.md")
    assert rec is not None
    assert rec.status == "failed"
    assert "timeout" in (rec.error or "").lower()


def test_ingest_note_creates_source_summary_page(vault, config, db):
    path = _write_raw(vault, "quantum.md", "# Quantum\n\nSuperposition and entanglement.")
    client = _make_client(_analysis_json(concepts=["Superposition", "Entanglement"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources, "Source summary page should be created"


def test_source_page_yaml_with_colon_title(vault, config, db):
    """Source page title containing ':' must not break YAML parsing."""
    # Raw note uses quoted title (valid YAML) — the colon in title flows to source page
    path = _write_raw(vault, "guide.md", "---\ntitle: 'Python: A Guide'\n---\n\nContent here.")
    client = _make_client(_analysis_json(concepts=["Python"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    from obsidian_llm_wiki.vault import parse_note

    meta, _ = parse_note(sources[0])
    assert meta["title"] == "Python: A Guide"


def test_source_page_aliases_are_list(vault, config, db):
    """Aliases must be a proper YAML list, not Python repr string."""
    path = _write_raw(vault, "ml.md", "# ML\n\nMachine Learning (ML) basics.")
    client = _make_client(_analysis_json(concepts=["Machine Learning"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    from obsidian_llm_wiki.vault import parse_note

    meta, _ = parse_note(sources[0])
    assert isinstance(meta.get("aliases", []), list)


def test_source_page_roundtrip(vault, config, db):
    """Source page has all required fields with correct types."""
    path = _write_raw(vault, "q.md", "# Quantum\n\nContent.")
    client = _make_client(_analysis_json(concepts=["Qubits"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    from obsidian_llm_wiki.vault import parse_note

    meta, body = parse_note(sources[0])
    assert "title" in meta
    assert meta["status"] == "published"
    assert meta["tags"] == ["source"]
    assert isinstance(meta["aliases"], list)
    assert "## Summary" in body
    assert "## Concepts" in body


def test_source_page_media_section(vault, config, db):
    """Raw note with images produces ## Media section in source page."""
    content = (
        "# Note\n\nSee ![[diagram.png]] for the architecture.\n"
        "Also ![Photo](http://example.com/photo.jpg) is relevant."
    )
    path = _write_raw(vault, "media-note.md", content)
    client = _make_client(_analysis_json(concepts=["Architecture"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    source_text = sources[0].read_text()
    assert "## Media" in source_text
    assert "diagram.png" in source_text
    assert "photo.jpg" in source_text


def test_source_page_no_media_section_when_none(vault, config, db):
    """Raw note without media produces no ## Media section."""
    path = _write_raw(vault, "text-only.md", "# Note\n\nJust text, no images.")
    client = _make_client(_analysis_json(concepts=["Text"]))
    ingest_note(path, config, client, db)
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    source_text = sources[0].read_text()
    assert "## Media" not in source_text


def test_ingest_note_respects_max_concepts_per_source(vault, config, db):
    config2 = Config(vault=vault, pipeline={"max_concepts_per_source": 2})
    path = _write_raw(vault, "many.md", "# Many\n\nLots of concepts.")
    client = _make_client(_analysis_json(concepts=["A", "B", "C", "D", "E"]))
    ingest_note(path, config2, client, db)
    names = db.list_all_concept_names()
    # Only first 2 should be stored
    assert len(names) <= 2
