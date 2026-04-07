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
    ingest_all,
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


def test_ingest_note_respects_max_concepts_per_source(vault, config, db):
    config2 = Config(vault=vault, pipeline={"max_concepts_per_source": 2})
    path = _write_raw(vault, "many.md", "# Many\n\nLots of concepts.")
    client = _make_client(_analysis_json(concepts=["A", "B", "C", "D", "E"]))
    ingest_note(path, config2, client, db)
    names = db.list_all_concept_names()
    # Only first 2 should be stored
    assert len(names) <= 2


# ── Web-clip preprocessing (source_url in frontmatter) ────────────────────


def test_ingest_web_clip_triggers_preprocess(vault, config, db):
    """Notes with source/url in frontmatter trigger _preprocess_web_clip."""
    content = (
        "---\ntitle: Clip\nsource: http://example.com\n---\n\n"
        "Home\nAbout\n\n# Real Content\n\n"
        "Full paragraph with enough words to pass the filter."
    )
    path = _write_raw(vault, "clip.md", content)
    client = _make_client(_analysis_json())
    result = ingest_note(path, config, client, db)
    assert result is not None
    # Source summary page should include source_url
    sources = list((vault / "wiki" / "sources").glob("*.md"))
    assert sources
    text = sources[0].read_text()
    assert "source_url: http://example.com" in text
    assert "**URL:** http://example.com" in text


# ── parse_note fallback (no valid frontmatter) ───────────────────────────


def test_ingest_note_no_frontmatter_fallback(vault, config, db):
    """When parse_note raises, body_for_hash falls back to raw content."""
    # Write a file with invalid frontmatter to trigger the except path
    # The file just has no --- delimiters at all, but parse_note may
    # still handle it. We mock parse_note to raise on first call only.
    from unittest.mock import patch

    path = _write_raw(vault, "bad_fm.md", "No frontmatter content here.")
    call_count = [0]
    original_parse = __import__(
        "obsidian_llm_wiki.vault", fromlist=["parse_note"]
    ).parse_note

    def parse_side_effect(p):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("bad frontmatter")
        return original_parse(p)

    with patch(
        "obsidian_llm_wiki.pipeline.ingest.parse_note",
        side_effect=parse_side_effect,
    ):
        client = _make_client(_analysis_json())
        result = ingest_note(path, config, client, db)
    assert result is not None


# ── RAG code path ─────────────────────────────────────────────────────────


def test_ingest_note_with_rag(vault, config, db):
    """When rag is not None, chunk+embed is called."""
    path = _write_raw(vault, "rag.md", "# RAG Test\n\nContent for RAG.")
    client = _make_client(_analysis_json())
    client.embed_batch.return_value = [[0.1] * 768]

    rag = MagicMock()
    result = ingest_note(path, config, client, db, rag=rag)

    assert result is not None
    client.embed_batch.assert_called_once()
    rag.add_document.assert_called_once()
    call_kwargs = rag.add_document.call_args
    assert call_kwargs[1]["doc_id"] == "raw/rag.md"


# ── Source summary page failure ───────────────────────────────────────────


def test_ingest_note_source_summary_failure(vault, config, db):
    """Exception in _create_source_summary_page is caught gracefully."""
    from unittest.mock import patch

    path = _write_raw(vault, "sumfail.md", "# Sum\n\nContent here.")
    client = _make_client(_analysis_json())

    with patch(
        "obsidian_llm_wiki.pipeline.ingest._create_source_summary_page",
        side_effect=OSError("disk full"),
    ):
        result = ingest_note(path, config, client, db)

    # Should still succeed despite summary page failure
    assert result is not None
    rec = db.get_raw("raw/sumfail.md")
    assert rec.status == "ingested"


# ── ingest_all ────────────────────────────────────────────────────────────


def test_ingest_all_processes_multiple_files(vault, config, db):
    """ingest_all finds and processes all .md files in raw/."""
    _write_raw(vault, "a.md", "# Alpha\n\nAlpha content.")
    _write_raw(vault, "b.md", "# Beta\n\nBeta content.")
    client = _make_client(_analysis_json())

    results = ingest_all(config, client, db)

    assert len(results) == 2
    assert all(r is not None for _, r in results)
    assert client.generate.call_count == 2


def test_ingest_all_skips_processed_subfolder(vault, config, db):
    """Files under raw/processed/ should be excluded."""
    _write_raw(vault, "good.md", "# Good\n\nKeep me.")
    proc_dir = vault / "raw" / "processed"
    proc_dir.mkdir()
    (proc_dir / "old.md").write_text("# Old\n\nSkip me.")
    client = _make_client(_analysis_json())

    results = ingest_all(config, client, db)
    assert len(results) == 1


def test_ingest_all_skips_dotfiles(vault, config, db):
    """Files starting with '.' should be excluded."""
    _write_raw(vault, "real.md", "# Real\n\nKeep.")
    _write_raw(vault, ".hidden.md", "# Hidden\n\nSkip.")
    client = _make_client(_analysis_json())

    results = ingest_all(config, client, db)
    assert len(results) == 1
