"""Integration tests for language detection and config override across the pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord
from obsidian_llm_wiki.pipeline.compile import _resolve_language, _write_concept_prompt
from obsidian_llm_wiki.pipeline.ingest import ingest_note
from obsidian_llm_wiki.pipeline.query import run_query
from obsidian_llm_wiki.state import StateDB


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


@pytest.mark.integration
def test_ingest_detects_and_stores_language(vault, config, db):
    """Mock LLM returns language='fr' → DB stores it on the raw note."""
    path = vault / "raw" / "french.md"
    path.write_text("# Bonjour\n\nCeci est une note en français.")

    analysis = json.dumps({
        "summary": "A French note.",
        "key_concepts": ["Bonjour"],
        "suggested_topics": ["Salutations"],
        "quality": "high",
        "language": "fr",
    })
    client = MagicMock()
    client.generate.return_value = analysis

    ingest_note(path, config, client, db)
    assert db.get_note_language("raw/french.md") == "fr"


@pytest.mark.integration
def test_compile_prompt_uses_detected_language(config, db):
    """Source note has language='fr' in DB and no config → compile prompt contains 'fr'."""
    r = RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested", language="fr")
    db.upsert_raw(r)

    lang = _resolve_language(["raw/a.md"], db, config)
    assert lang == "fr"

    prompt = _write_concept_prompt("Topic", "source material", [], language=lang)
    assert "Output language: fr" in prompt


@pytest.mark.integration
def test_compile_prompt_config_wins_over_detected(config, db):
    """config.pipeline.language='de', detected 'fr' → prompt contains 'de'."""
    r = RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested", language="fr")
    db.upsert_raw(r)
    config.pipeline.language = "de"

    lang = _resolve_language(["raw/a.md"], db, config)
    assert lang == "de"

    prompt = _write_concept_prompt("Topic", "source material", [], language=lang)
    assert "Output language: de" in prompt
    assert "fr" not in prompt


@pytest.mark.integration
def test_query_answer_prompt_has_language_instruction(vault, config, db):
    """Answer prompt tells LLM to respond in user's question language."""
    index = vault / "wiki" / "index.md"
    index.write_text("# Index\n\n- [[Topic]]\n")
    (vault / "wiki" / "Topic.md").write_text("---\ntitle: Topic\n---\nContent.")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Réponse."})
    client = MagicMock()
    client.generate.side_effect = [selection_json, answer_json]

    run_query(config, client, db, "Qu'est-ce que Topic?")

    second_call_prompt = client.generate.call_args_list[1].kwargs.get("prompt", "")
    assert "same language as the user's question" in second_call_prompt
