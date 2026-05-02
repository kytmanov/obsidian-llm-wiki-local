from __future__ import annotations

from types import SimpleNamespace

import click
from click.testing import CliRunner

from obsidian_llm_wiki.cli import cli
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import WikiArticleRecord
from obsidian_llm_wiki.pipeline.query import (
    QueryRunResult,
    QuerySaveResult,
    SynthesisChainError,
    _question_hash,
)
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.vault import write_note


def _make_vault(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".olw").mkdir()
    return tmp_path


def test_query_cli_reports_synthesis_saved(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    config = Config(vault=vault)
    db = StateDB(config.state_db_path)

    monkeypatch.setattr("obsidian_llm_wiki.cli._load_deps", lambda cfg: (object(), db))
    monkeypatch.setattr(
        "obsidian_llm_wiki.pipeline.query.run_query",
        lambda *args, **kwargs: QueryRunResult(
            answer="Answer.",
            selected_pages=["Topic"],
            synthesis=QuerySaveResult(path=config.synthesis_dir / "Topic.md"),
        ),
    )

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    result = CliRunner().invoke(
        cli, ["query", "--vault", str(vault), "--synthesize", "What is Topic?"]
    )

    assert result.exit_code == 0
    assert "Synthesis saved to" in result.output


def test_query_cli_reports_existing_synthesis(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    config = Config(vault=vault)
    db = StateDB(config.state_db_path)

    monkeypatch.setattr("obsidian_llm_wiki.cli._load_deps", lambda cfg: (object(), db))
    monkeypatch.setattr(
        "obsidian_llm_wiki.pipeline.query.run_query",
        lambda *args, **kwargs: QueryRunResult(
            answer="Answer.",
            selected_pages=["Topic"],
            synthesis=QuerySaveResult(
                path=config.synthesis_dir / "Topic.md",
                resolution="kept_existing",
                duplicate_detected=True,
                file_written=False,
            ),
        ),
    )

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    result = CliRunner().invoke(
        cli, ["query", "--vault", str(vault), "--synthesize", "What is Topic?"]
    )

    assert result.exit_code == 0
    assert "Existing synthesis kept" in result.output


def test_query_cli_returns_nonzero_on_synthesis_chain_error(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    config = Config(vault=vault)
    db = StateDB(config.state_db_path)

    monkeypatch.setattr("obsidian_llm_wiki.cli._load_deps", lambda cfg: (object(), db))

    def _raise(*args, **kwargs):
        raise SynthesisChainError("Synthesis sources cannot include another synthesis page")

    monkeypatch.setattr("obsidian_llm_wiki.pipeline.query.run_query", _raise)

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    result = CliRunner().invoke(
        cli, ["query", "--vault", str(vault), "--synthesize", "What is Topic?"]
    )

    assert result.exit_code == 1
    assert "cannot include another synthesis page" in result.output


def test_query_cli_only_prompts_when_duplicate_exists(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    config = Config(vault=vault)
    db = StateDB(config.state_db_path)
    prompted = {"value": False}
    captured = {}

    monkeypatch.setattr("obsidian_llm_wiki.cli._load_deps", lambda cfg: (object(), db))

    def fake_run_query(*args, **kwargs):
        captured["duplicate_strategy"] = kwargs["duplicate_strategy"]
        return QueryRunResult(answer="Answer.", selected_pages=["Topic"])

    def fail_prompt(*args, **kwargs):
        prompted["value"] = True
        raise AssertionError("prompt should not be called")

    monkeypatch.setattr("obsidian_llm_wiki.pipeline.query.run_query", fake_run_query)
    monkeypatch.setattr(click, "prompt", fail_prompt)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    result = CliRunner().invoke(
        cli, ["query", "--vault", str(vault), "--synthesize", "What is Topic?"]
    )

    assert result.exit_code == 0
    assert prompted["value"] is False
    assert captured["duplicate_strategy"] == "keep_existing"


def test_query_cli_prompts_when_duplicate_exists(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    config = Config(vault=vault)
    db = StateDB(config.state_db_path)
    synthesis_path = config.synthesis_dir / "Topic.md"
    write_note(
        synthesis_path,
        {"title": "Topic", "tags": ["synthesis"], "kind": "synthesis", "status": "published"},
        "Body.",
    )
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/synthesis/Topic.md",
            title="Topic",
            sources=[],
            content_hash="hash",
            is_draft=False,
            kind="synthesis",
            question_hash=_question_hash("What is Topic?"),
        )
    )
    captured = {}

    monkeypatch.setattr("obsidian_llm_wiki.cli._load_deps", lambda cfg: (object(), db))

    def fake_run_query(*args, **kwargs):
        captured["duplicate_strategy"] = kwargs["duplicate_strategy"]
        return QueryRunResult(answer="Answer.", selected_pages=["Topic"])

    monkeypatch.setattr("obsidian_llm_wiki.pipeline.query.run_query", fake_run_query)
    monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: "update")
    tty_stream = SimpleNamespace(isatty=lambda: True)
    monkeypatch.setattr(
        "obsidian_llm_wiki.cli.sys",
        SimpleNamespace(stdin=tty_stream, stdout=tty_stream),
    )

    result = CliRunner().invoke(
        cli, ["query", "--vault", str(vault), "--synthesize", "What is Topic?"]
    )

    assert result.exit_code == 0
    assert captured["duplicate_strategy"] == "update_in_place"
