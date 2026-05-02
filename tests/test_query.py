"""Tests for pipeline/query.py — mocked OllamaClient, no Ollama required."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.indexer import generate_index
from obsidian_llm_wiki.models import WikiArticleRecord
from obsidian_llm_wiki.pipeline.query import _find_page, _load_pages, run_query
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.vault import write_note


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


def _make_client(selection_json: str, answer_json: str) -> MagicMock:
    """Mock client: 1st call → page selection, 2nd → answer."""
    client = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        return selection_json if call_count[0] == 1 else answer_json

    client.generate.side_effect = side_effect
    return client


def _write_index(config: Config, content: str) -> None:
    (config.wiki_dir / "index.md").write_text(content, encoding="utf-8")


def _write_concept_page(config: Config, title: str, body: str = "") -> Path:
    path = config.wiki_dir / f"{title}.md"
    write_note(
        path, {"title": title, "tags": [], "status": "published"}, body or f"Content about {title}."
    )
    return path


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_run_query_returns_answer(vault, config, db):
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Quantum Computing]]\n")
    _write_concept_page(config, "Quantum Computing", "Qubits exploit superposition.")

    selection_json = json.dumps({"pages": ["Quantum Computing"]})
    answer_json = json.dumps({"answer": "[[Quantum Computing]] uses qubits."})
    client = _make_client(selection_json, answer_json)

    result = run_query(config, client, db, "What is quantum computing?")

    assert "qubits" in result.answer.lower()
    assert "Quantum Computing" in result.selected_pages
    assert client.generate.call_count == 2


def test_run_query_no_index_returns_helpful_message(vault, config, db):
    client = MagicMock()
    result = run_query(config, client, db, "Any question")

    assert "index" in result.answer.lower()
    assert result.selected_pages == []
    client.generate.assert_not_called()


def test_run_query_missing_page_skipped(vault, config, db):
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Nonexistent Page]]\n")
    # No actual page file created

    selection_json = json.dumps({"pages": ["Nonexistent Page"]})
    answer_json = json.dumps({"answer": "General answer."})
    client = _make_client(selection_json, answer_json)

    result = run_query(config, client, db, "question?")
    # Still gets an answer (with fallback context)
    assert result.answer == "General answer."


def test_run_query_save_creates_file(vault, config, db):
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer about Topic."})
    client = _make_client(selection_json, answer_json)

    run_query(config, client, db, "Tell me about Topic", save=True)

    queries = list(config.queries_dir.glob("*.md"))
    assert len(queries) == 1
    assert queries[0].read_text(encoding="utf-8").strip() != ""


def test_find_page_by_filename(vault, config):
    _write_concept_page(config, "Machine Learning")
    found = _find_page(config, "Machine Learning")
    assert found is not None
    assert found.stem == "Machine Learning"


def test_find_page_by_explicit_sources_path(vault, config):
    path = config.sources_dir / "Source Note.md"
    write_note(path, {"title": "Source Note", "tags": ["source"]}, "Source body.")

    found = _find_page(config, "sources/Source Note")

    assert found == path


def test_find_page_by_frontmatter_title(vault, config):
    # File named differently from its frontmatter title
    path = config.wiki_dir / "ml.md"
    write_note(path, {"title": "Machine Learning", "tags": []}, "Content.")
    found = _find_page(config, "Machine Learning")
    assert found == path


def test_find_page_not_found_returns_none(vault, config):
    assert _find_page(config, "Does Not Exist") is None


def test_find_page_prefers_concept_over_synthesis(vault, config, db):
    concept = config.wiki_dir / "Topic.md"
    synthesis = config.synthesis_dir / "Topic.md"
    write_note(concept, {"title": "Topic", "tags": [], "status": "published"}, "Concept body.")
    write_note(
        synthesis,
        {"title": "Topic", "tags": ["synthesis"], "kind": "synthesis", "status": "published"},
        "Synthesis body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Topic.md",
            title="Topic",
            sources=[],
            content_hash="hash",
            is_draft=False,
            kind="synthesis",
            question_hash="qh",
        )
    )

    found = _find_page(config, "Topic", db=db)

    assert found == concept


def test_generate_index_lists_synthesis_from_db_only(vault, config, db):
    write_note(
        config.synthesis_dir / "Tracked.md",
        {"title": "Tracked", "tags": ["synthesis"], "kind": "synthesis", "status": "published"},
        "Tracked body.",
    )
    write_note(
        config.synthesis_dir / "Orphan.md",
        {"title": "Orphan", "tags": ["synthesis"], "kind": "synthesis", "status": "published"},
        "Orphan body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Tracked.md",
            title="Tracked",
            sources=[],
            content_hash="hash",
            is_draft=False,
            kind="synthesis",
            question_hash="qh",
        )
    )

    index_path = generate_index(config, db)
    index_text = index_path.read_text(encoding="utf-8")

    assert "## Synthesis" in index_text
    assert "Tracked" in index_text
    assert "Orphan" not in index_text


def test_generate_index_caps_synthesis_section(vault, config, db):
    for i in range(27):
        title = f"Topic {i:02d}"
        path = config.synthesis_dir / f"{title}.md"
        write_note(
            path,
            {"title": title, "tags": ["synthesis"], "kind": "synthesis", "status": "published"},
            "Body.",
        )
        db.upsert_article(
            WikiArticleRecord(
                path=str(path.relative_to(config.vault)),
                title=title,
                sources=[],
                content_hash=f"hash-{i}",
                is_draft=False,
                kind="synthesis",
                question_hash=f"qh-{i}",
            )
        )

    index_text = generate_index(config, db).read_text(encoding="utf-8")

    assert index_text.count("[[Topic ") == 25
    assert "_(2 more synthesis pages not shown)_" in index_text


def test_load_pages_truncates_to_max_chars(vault, config):
    # Write a very long page
    long_body = "word " * 10_000  # way over 8000 chars
    _write_concept_page(config, "Long Page", long_body)
    result = _load_pages(config, ["Long Page"])
    # Result should not contain all words
    assert len(result) < len(long_body)


def test_run_query_passes_index_to_first_call(vault, config, db):
    """Fast model prompt must include index content."""
    index_text = "# Wiki Index\n\n## Concepts\n- [[Special Topic]]\n"
    _write_index(config, index_text)
    _write_concept_page(config, "Special Topic")

    selection_json = json.dumps({"pages": ["Special Topic"]})
    answer_json = json.dumps({"answer": "Answer."})
    client = _make_client(selection_json, answer_json)

    run_query(config, client, db, "question?")

    first_call_prompt = client.generate.call_args_list[0].kwargs.get("prompt", "")
    assert "Special Topic" in first_call_prompt


def test_query_answer_prompt_has_language_instruction(vault, config, db):
    """Answer prompt must tell LLM to match user's question language."""
    index_text = "# Wiki Index\n\n## Concepts\n- [[Topic]]\n"
    _write_index(config, index_text)
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer."})
    client = _make_client(selection_json, answer_json)

    run_query(config, client, db, "What is Topic?")

    second_call_prompt = client.generate.call_args_list[1].kwargs.get("prompt", "")
    assert "same language as the user's question" in second_call_prompt


def test_query_answer_prompt_limits_wikilinks_to_existing_pages(vault, config, db):
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Scrum]]\n")
    _write_concept_page(config, "Scrum", "Product Backlog is mentioned but has no page.")

    selection_json = json.dumps({"pages": ["Scrum"]})
    answer_json = json.dumps({"answer": "Answer."})
    client = _make_client(selection_json, answer_json)

    run_query(config, client, db, "What is Scrum?")

    second_call_prompt = client.generate.call_args_list[1].kwargs.get("prompt", "")
    assert "Use [[wikilinks]] only for existing wiki pages" in second_call_prompt
    assert "Scrum" in second_call_prompt
    assert "Product Backlog," not in second_call_prompt
