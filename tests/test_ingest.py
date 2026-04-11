"""Tests for pipeline/ingest.py — no Ollama required (mocked client)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import AnalysisResult
from obsidian_llm_wiki.pipeline.ingest import (
    _analyze_body,
    _build_analysis_prompt,
    _merge_chunk_results,
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


def test_build_prompt_includes_full_body():
    # No truncation in _build_analysis_prompt — chunking happens in _analyze_body
    body = "x " * 5000  # ~10000 chars
    prompt = _build_analysis_prompt(body, [])
    assert body in prompt


def test_build_prompt_includes_chunk_label():
    prompt = _build_analysis_prompt("content", [], chunk_label="[part 2/4]")
    assert "[part 2/4]" in prompt


def test_build_prompt_no_chunk_label_by_default():
    prompt = _build_analysis_prompt("content", [])
    assert "[part" not in prompt


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


# ── _merge_chunk_results ──────────────────────────────────────────────────────


def _make_result(concepts, summary="Summary.", quality="high", topics=None):
    return AnalysisResult(
        summary=summary,
        key_concepts=concepts,
        suggested_topics=topics or ["Topic"],
        quality=quality,
    )


def test_merge_single_chunk_returns_unchanged():
    r = _make_result(["A", "B"])
    assert _merge_chunk_results([r]) is r


def test_merge_unions_concepts():
    r1 = _make_result(["A", "B"])
    r2 = _make_result(["B", "C"])
    merged = _merge_chunk_results([r1, r2])
    assert merged.key_concepts == ["A", "B", "C"]  # B deduped


def test_merge_concept_dedup_case_insensitive():
    r1 = _make_result(["Machine Learning"])
    r2 = _make_result(["machine learning", "Deep Learning"])
    merged = _merge_chunk_results([r1, r2])
    names_lower = [c.lower() for c in merged.key_concepts]
    assert names_lower.count("machine learning") == 1
    assert "deep learning" in names_lower


def test_merge_summary_from_first_chunk():
    r1 = _make_result(["A"], summary="First summary.")
    r2 = _make_result(["B"], summary="Second summary.")
    merged = _merge_chunk_results([r1, r2])
    assert merged.summary == "First summary."


def test_merge_quality_is_minimum():
    r1 = _make_result(["A"], quality="high")
    r2 = _make_result(["B"], quality="low")
    r3 = _make_result(["C"], quality="medium")
    merged = _merge_chunk_results([r1, r2, r3])
    assert merged.quality == "low"


def test_merge_unions_topics():
    r1 = _make_result(["A"], topics=["Topic A"])
    r2 = _make_result(["B"], topics=["Topic B", "Topic A"])
    merged = _merge_chunk_results([r1, r2])
    assert len(merged.suggested_topics) == 2


# ── _analyze_body ──────────────────────────────────────────────────────────────


def test_analyze_body_single_call_for_short_note(vault, config, db):
    """Body <= fast_ctx // 2 → exactly one generate call."""
    client = _make_client(_analysis_json())
    body = "Short note content."
    _analyze_body(body, [], "test.md", client, config)
    assert client.generate.call_count == 1


def test_analyze_body_multi_call_for_long_note(vault, config, db):
    """Body > fast_ctx // 2 → one call per chunk."""
    config2 = Config(vault=vault, ollama={"fast_ctx": 100})  # tiny ctx for test
    chunk_size = 100 // 2  # = 50 chars per chunk
    body = "x" * 200  # 200 chars → 4 chunks
    client = _make_client(_analysis_json())
    result = _analyze_body(body, [], "long.md", client, config2)
    expected_chunks = -(-200 // chunk_size)  # ceiling division
    assert client.generate.call_count == expected_chunks
    assert isinstance(result, AnalysisResult)


def test_analyze_body_chunk_labels_in_prompt(vault, config, db):
    """Multi-chunk prompts include [part N/M] labels."""
    config2 = Config(vault=vault, ollama={"fast_ctx": 100})
    body = "x" * 200
    client = _make_client(_analysis_json())
    _analyze_body(body, [], "long.md", client, config2)
    prompts = [call.kwargs["prompt"] for call in client.generate.call_args_list]
    assert any("[part 1/" in p for p in prompts)
    assert any("[part 2/" in p for p in prompts)


def test_analyze_body_parallel_mode(vault):
    """ingest_parallel=True still produces one call per chunk, results merged."""
    config2 = Config(
        vault=vault,
        ollama={"fast_ctx": 100},
        pipeline={"ingest_parallel": True},
    )
    body = "x" * 200
    client = _make_client(_analysis_json(concepts=["A"]))
    result = _analyze_body(body, [], "long.md", client, config2)
    assert client.generate.call_count == -(-200 // 50)  # same chunk count
    assert isinstance(result, AnalysisResult)
