"""Tests for the `olw compile` CLI command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from click.testing import CliRunner

from obsidian_llm_wiki.cli import cli
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord
from obsidian_llm_wiki.state import StateDB


def test_compile_cli_concept_alias_resolution(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".olw").mkdir()
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Product Backlog"])
    db.upsert_aliases("Product Backlog", ["Backlog"])
    (tmp_path / "raw" / "a.md").write_text("Body.")

    client = MagicMock()
    client.generate.return_value = json.dumps(
        {"title": "Product Backlog", "content": "Body.", "tags": []}
    )

    monkeypatch.setattr("obsidian_llm_wiki.cli._load_deps", lambda cfg: (client, db))

    result = CliRunner().invoke(
        cli,
        ["compile", "--vault", str(tmp_path), "--concept", "Backlog"],
    )

    assert result.exit_code == 0
    assert (tmp_path / "wiki" / ".drafts" / "Product Backlog.md").exists()


def test_compile_cli_retry_failed_retries_failed_concepts(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".olw").mkdir()
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.mark_concept_compile_state("Alpha", ["raw/a.md"], "failed", error="bad json")
    (tmp_path / "raw" / "a.md").write_text("Body.")

    client = MagicMock()
    client.generate.return_value = json.dumps({"title": "Alpha", "content": "Body.", "tags": []})

    monkeypatch.setattr("obsidian_llm_wiki.cli._load_deps", lambda cfg: (client, db))

    result = CliRunner().invoke(
        cli,
        ["compile", "--vault", str(tmp_path), "--retry-failed"],
    )

    assert result.exit_code == 0
    assert (tmp_path / "wiki" / ".drafts" / "Alpha.md").exists()
