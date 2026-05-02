"""Tests for synthesis-oriented query helpers and result shape."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import WikiArticleRecord
from obsidian_llm_wiki.pipeline.query import _body_hash, _derive_synthesis_title, run_query
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.telemetry import app_event_sink
from obsidian_llm_wiki.vault import parse_note, write_note


def _make_client(selection_json: str, answer_json: str) -> MagicMock:
    client = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        return selection_json if call_count[0] == 1 else answer_json

    client.generate.side_effect = side_effect
    return client


def _make_vault(tmp_path: Path) -> tuple[Path, Config, StateDB]:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".olw").mkdir()
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)
    return tmp_path, config, db


def _write_index(config: Config, content: str) -> None:
    (config.wiki_dir / "index.md").write_text(content, encoding="utf-8")


def _write_concept_page(config: Config, title: str, body: str = "") -> Path:
    path = config.wiki_dir / f"{title}.md"
    write_note(
        path, {"title": title, "tags": [], "status": "published"}, body or f"Content about {title}."
    )
    return path


def test_derive_synthesis_title_uses_model_title():
    assert _derive_synthesis_title("what is topic?", "Topic Overview") == "Topic Overview"


def test_derive_synthesis_title_falls_back_for_blank_title():
    assert _derive_synthesis_title("what is topic?", "   ") == "What Is Topic"


def test_derive_synthesis_title_falls_back_for_oversized_title():
    long_title = "one two three four five six seven eight nine ten eleven twelve thirteen"
    assert (
        _derive_synthesis_title("how do manual edits work?", long_title)
        == "How Do Manual Edits Work"
    )


def test_derive_synthesis_title_handles_non_ascii_question():
    assert _derive_synthesis_title("Qu'est-ce que Topic?", None) == "Qu'Est-Ce Que Topic"


def test_derive_synthesis_title_falls_back_when_model_title_sanitizes_empty():
    assert _derive_synthesis_title("what is topic?", "???") == "What Is Topic"


def test_derive_synthesis_title_uses_untitled_when_question_empty():
    assert _derive_synthesis_title(" ? ", None) == "untitled-synthesis"


def test_run_query_returns_result_object_with_no_save_metadata(tmp_path):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer.", "title": "Topic"})
    client = _make_client(selection_json, answer_json)

    result = run_query(config, client, db, "What is Topic?")

    assert result.answer == "Answer."
    assert result.selected_pages == ["Topic"]
    assert result.query_save is None
    assert result.synthesis is None


def test_run_query_save_returns_query_save_metadata(tmp_path):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer.", "title": "Topic"})
    client = _make_client(selection_json, answer_json)

    result = run_query(config, client, db, "What is Topic?", save=True)

    assert result.query_save is not None
    assert result.query_save.resolution == "saved_new"
    assert result.query_save.path.parent == config.queries_dir
    assert result.synthesis is None


def test_query_answer_prompt_requests_title(tmp_path):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer.", "title": "Topic"})
    client = _make_client(selection_json, answer_json)

    run_query(config, client, db, "What is Topic?")

    second_call_prompt = client.generate.call_args_list[1].kwargs.get("prompt", "")
    assert "short topic title" in second_call_prompt
    assert '"title": "short title"' in second_call_prompt


def test_run_query_synthesize_creates_synthesis_file_and_db_row(tmp_path):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer.", "title": "Topic Overview"})
    client = _make_client(selection_json, answer_json)

    with app_event_sink() as events:
        result = run_query(config, client, db, "What is Topic?", synthesize=True)

    assert result.synthesis is not None
    assert result.synthesis.path.parent == config.synthesis_dir
    meta, body = parse_note(result.synthesis.path)
    assert meta["title"] == "Topic Overview"
    assert meta["tags"] == ["synthesis"]
    assert meta["kind"] == "synthesis"
    assert meta["source_question"] == "What is Topic?"
    assert meta["source_pages"] == ["Topic"]
    assert meta["question_hash"]
    assert meta["content_hash"]
    assert meta["status"] == "published"
    assert meta["source_page_hashes"] == [
        {"path": "wiki/Topic.md", "hash": meta["source_page_hashes"][0]["hash"]}
    ]
    assert "## Sources" in body
    assert "[[Topic]]" in body

    article = db.get_article(str(result.synthesis.path.relative_to(config.vault)))
    assert article is not None
    assert article.kind == "synthesis"
    assert article.question_hash == meta["question_hash"]
    assert article.synthesis_sources == ["wiki/Topic.md"]
    assert article.synthesis_source_hashes == [
        ["wiki/Topic.md", meta["source_page_hashes"][0]["hash"]]
    ]
    assert len(events) == 1
    assert events[0].name == "query_synthesize"
    assert events[0].payload == {
        "question_hash": meta["question_hash"],
        "resolution": "saved_new",
        "file_written": True,
        "path": str(result.synthesis.path),
        "duplicate_detected": False,
        "source_page_count": 1,
        "error": None,
    }


def test_run_query_synthesize_keeps_existing_duplicate(tmp_path):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer.", "title": "Topic Overview"})
    client = _make_client(selection_json, answer_json)

    first = run_query(config, client, db, "What is Topic?", synthesize=True)
    second_client = _make_client(selection_json, answer_json)
    with app_event_sink() as events:
        second = run_query(config, second_client, db, "What is Topic?", synthesize=True)

    assert first.synthesis is not None
    assert second.synthesis is not None
    assert second.synthesis.resolution == "kept_existing"
    assert second.synthesis.duplicate_detected is True
    assert second.synthesis.path == first.synthesis.path
    assert len(list(config.synthesis_dir.glob("*.md"))) == 1
    assert len(events) == 1
    assert events[0].payload["resolution"] == "kept_existing"
    assert events[0].payload["duplicate_detected"] is True
    assert events[0].payload["file_written"] is False


def test_run_query_synthesize_duplicate_can_save_with_suffix(tmp_path):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer.", "title": "Topic Overview"})
    first = run_query(
        config, _make_client(selection_json, answer_json), db, "What is Topic?", synthesize=True
    )

    with app_event_sink() as events:
        second = run_query(
            config,
            _make_client(selection_json, answer_json),
            db,
            "What is Topic?",
            synthesize=True,
            duplicate_strategy="save_with_suffix",
        )

    assert first.synthesis is not None
    assert second.synthesis is not None
    assert second.synthesis.resolution == "saved_with_suffix"
    assert second.synthesis.duplicate_detected is True
    assert second.synthesis.path != first.synthesis.path
    assert len(list(config.synthesis_dir.glob("*.md"))) == 2
    assert events[0].payload["resolution"] == "saved_with_suffix"


def test_run_query_synthesize_duplicate_can_update_in_place(tmp_path):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    first_answer = json.dumps({"answer": "Original answer.", "title": "Topic Overview"})
    updated_answer = json.dumps({"answer": "Updated answer.", "title": "Topic Overview"})
    first = run_query(
        config, _make_client(selection_json, first_answer), db, "What is Topic?", synthesize=True
    )

    with app_event_sink() as events:
        second = run_query(
            config,
            _make_client(selection_json, updated_answer),
            db,
            "What is Topic?",
            synthesize=True,
            duplicate_strategy="update_in_place",
        )

    assert first.synthesis is not None
    assert second.synthesis is not None
    assert second.synthesis.resolution == "updated_in_place"
    assert second.synthesis.path == first.synthesis.path
    _, body = parse_note(second.synthesis.path)
    meta, _ = parse_note(second.synthesis.path)
    article = db.get_article(str(second.synthesis.path.relative_to(config.vault)))
    assert "Updated answer." in body
    assert meta["title"] == "Topic Overview"
    assert article is not None
    assert article.title == "Topic Overview"
    assert len(list(config.synthesis_dir.glob("*.md"))) == 1
    assert events[0].payload["resolution"] == "updated_in_place"


def test_run_query_synthesize_update_in_place_rejects_manual_edit_drift(tmp_path):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Original answer.", "title": "Topic Overview"})
    first = run_query(
        config, _make_client(selection_json, answer_json), db, "What is Topic?", synthesize=True
    )
    assert first.synthesis is not None

    path = first.synthesis.path
    meta, _ = parse_note(path)
    write_note(path, meta, "Edited by hand.")

    with app_event_sink() as events:
        with pytest.raises(ValueError, match="manually edited"):
            run_query(
                config,
                _make_client(
                    selection_json,
                    json.dumps({"answer": "Updated answer.", "title": "Retitled Topic"}),
                ),
                db,
                "What is Topic?",
                synthesize=True,
                duplicate_strategy="update_in_place",
            )

    refreshed_meta, refreshed_body = parse_note(path)
    article = db.get_article(str(path.relative_to(config.vault)))
    assert refreshed_meta["title"] == "Topic Overview"
    assert refreshed_body == "Edited by hand."
    assert article is not None
    assert article.title == "Topic Overview"
    assert len(events) == 1
    assert events[0].payload["resolution"] == "manual_edit_conflict"
    assert events[0].payload["duplicate_detected"] is True
    assert events[0].payload["file_written"] is False


def test_run_query_synthesize_race_duplicate_can_save_with_suffix(tmp_path, monkeypatch):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer.", "title": "Topic Overview"})

    original_insert = db.insert_synthesis_atomic
    first_call = {"value": True}

    def race_insert(record):
        if first_call["value"]:
            first_call["value"] = False
            db.upsert_article(
                WikiArticleRecord(
                    path="wiki/synthesis/Existing Topic.md",
                    title="Existing Topic",
                    sources=[],
                    content_hash="existing-hash",
                    is_draft=False,
                    kind="synthesis",
                    question_hash=record.question_hash,
                )
            )
        return original_insert(record)

    monkeypatch.setattr(db, "insert_synthesis_atomic", race_insert)

    with app_event_sink() as events:
        result = run_query(
            config,
            _make_client(selection_json, answer_json),
            db,
            "What is Topic?",
            synthesize=True,
            duplicate_strategy="save_with_suffix",
        )

    assert result.synthesis is not None
    assert result.synthesis.resolution == "saved_with_suffix"
    assert result.synthesis.duplicate_detected is True
    assert result.synthesis.path.name == "Topic Overview.md"
    assert events[0].payload["resolution"] == "saved_with_suffix"


def test_run_query_synthesize_stale_db_row_missing_file_uses_suffix_without_loop(
    tmp_path, monkeypatch
):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    stale_path = config.synthesis_dir / "Topic Overview.md"
    db.upsert_article(
        WikiArticleRecord(
            path=str(stale_path.relative_to(config.vault)),
            title="Topic Overview",
            sources=[],
            content_hash="missing-file-hash",
            is_draft=False,
            kind="synthesis",
            question_hash=None,
        )
    )

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Fresh answer.", "title": "Topic Overview"})

    original_insert = db.insert_synthesis_atomic
    attempts: list[str] = []

    def fail_on_repeat(record):
        attempts.append(record.path)
        if attempts.count(record.path) > 1:
            raise AssertionError(f"repeated synthesis path attempt: {record.path}")
        return original_insert(record)

    monkeypatch.setattr(db, "insert_synthesis_atomic", fail_on_repeat)

    result = run_query(
        config,
        _make_client(selection_json, answer_json),
        db,
        "A different question altogether",
        synthesize=True,
    )

    assert result.synthesis is not None
    assert result.synthesis.path == config.synthesis_dir / "Topic Overview-2.md"
    assert result.synthesis.resolution == "saved_with_suffix"
    assert attempts == ["wiki/synthesis/Topic Overview-2.md"]


def test_run_query_synthesize_orphan_file_uses_suffix_resolution(tmp_path):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    write_note(
        config.synthesis_dir / "Topic Overview.md",
        {
            "title": "Topic Overview",
            "tags": ["synthesis"],
            "kind": "synthesis",
            "status": "published",
        },
        "Orphan body.",
    )

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Fresh answer.", "title": "Topic Overview"})

    result = run_query(
        config,
        _make_client(selection_json, answer_json),
        db,
        "What is topic freshness?",
        synthesize=True,
    )

    assert result.synthesis is not None
    assert result.synthesis.path == config.synthesis_dir / "Topic Overview-2.md"
    assert result.synthesis.resolution == "saved_with_suffix"


def test_run_query_synthesize_race_duplicate_can_update_in_place(tmp_path, monkeypatch):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Concepts\n- [[Topic]]\n")
    _write_concept_page(config, "Topic")

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Updated answer.", "title": "Topic Overview"})

    original_insert = db.insert_synthesis_atomic
    first_call = {"value": True}
    raced_path = config.synthesis_dir / "Existing Topic.md"
    race_question = "What is Topic?"

    def race_insert(record):
        if first_call["value"]:
            first_call["value"] = False
            write_note(
                raced_path,
                {
                    "title": "Existing Topic",
                    "tags": ["synthesis"],
                    "kind": "synthesis",
                    "source_question": race_question,
                    "source_pages": ["Topic"],
                    "source_page_hashes": [{"path": "wiki/Topic.md", "hash": "seed"}],
                    "question_hash": record.question_hash,
                    "content_hash": "Original hash",
                    "created": "2026-05-02",
                    "status": "published",
                },
                "Original answer.\n\n## Sources\n\n- [[Topic]]",
            )
            _, existing_body = parse_note(raced_path)
            db.upsert_article(
                WikiArticleRecord(
                    path=str(raced_path.relative_to(config.vault)),
                    title="Existing Topic",
                    sources=[],
                    content_hash=_body_hash(existing_body),
                    is_draft=False,
                    kind="synthesis",
                    question_hash=record.question_hash,
                )
            )
        return original_insert(record)

    monkeypatch.setattr(db, "insert_synthesis_atomic", race_insert)

    with app_event_sink() as events:
        result = run_query(
            config,
            _make_client(selection_json, answer_json),
            db,
            race_question,
            synthesize=True,
            duplicate_strategy="update_in_place",
        )

    assert result.synthesis is not None
    assert result.synthesis.resolution == "updated_in_place"
    assert result.synthesis.path == raced_path
    meta, body = parse_note(raced_path)
    article = db.get_article(str(raced_path.relative_to(config.vault)))
    assert meta["title"] == "Topic Overview"
    assert "Updated answer." in body
    assert article is not None
    assert article.title == "Topic Overview"
    assert events[0].payload["resolution"] == "updated_in_place"


def test_run_query_synthesize_rejects_synthesis_source_chain(tmp_path):
    _, config, db = _make_vault(tmp_path)
    _write_index(config, "# Wiki Index\n\n## Synthesis\n- [[Old Synthesis]]\n")
    synthesis_path = config.synthesis_dir / "Old Synthesis.md"
    write_note(
        synthesis_path,
        {
            "title": "Old Synthesis",
            "tags": ["synthesis"],
            "kind": "synthesis",
            "status": "published",
        },
        "Prior answer.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Old Synthesis.md",
            title="Old Synthesis",
            sources=[],
            content_hash="hash",
            is_draft=False,
            kind="synthesis",
            question_hash="old-hash",
        )
    )

    selection_json = json.dumps({"pages": ["Old Synthesis"]})
    answer_json = json.dumps({"answer": "Answer.", "title": "Topic Overview"})
    client = _make_client(selection_json, answer_json)

    with app_event_sink() as events:
        with pytest.raises(ValueError, match="cannot include another synthesis"):
            run_query(config, client, db, "What is Topic?", synthesize=True)

    assert len(list(config.synthesis_dir.glob("*.md"))) == 1
    assert len(events) == 1
    assert events[0].payload["resolution"] == "rejected_synthesis_chain"
    assert events[0].payload["file_written"] is False
    assert events[0].payload["error"] == "Synthesis sources cannot include another synthesis page"
