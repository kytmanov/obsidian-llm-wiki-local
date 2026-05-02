from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner

from obsidian_llm_wiki.cli import cli
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.vault import write_note


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)


def _make_client(selection_json: str, answer_json: str) -> MagicMock:
    client = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        return selection_json if call_count[0] == 1 else answer_json

    client.generate.side_effect = side_effect
    client.close = MagicMock()
    return client


def test_query_synthesize_creates_no_new_olw_commit(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".olw").mkdir()
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)

    _run(["git", "init"], cwd=tmp_path)
    _run(["git", "config", "user.name", "Test User"], cwd=tmp_path)
    _run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)

    (config.wiki_dir / "index.md").write_text(
        "# Wiki Index\n\n## Concepts\n- [[Topic]]\n", encoding="utf-8"
    )
    write_note(
        config.wiki_dir / "Topic.md", {"title": "Topic", "tags": [], "status": "published"}, "Body."
    )
    write_note(
        config.wiki_dir / "log.md",
        {"title": "Operation Log", "tags": ["meta"]},
        "# Operation Log\n",
    )
    _run(["git", "add", "."], cwd=tmp_path)
    _run(["git", "commit", "-m", "baseline"], cwd=tmp_path)

    selection_json = json.dumps({"pages": ["Topic"]})
    answer_json = json.dumps({"answer": "Answer.", "title": "Topic Overview"})
    client = _make_client(selection_json, answer_json)

    monkeypatch.setattr("obsidian_llm_wiki.cli._load_deps", lambda cfg: (client, db))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    result = CliRunner().invoke(
        cli,
        ["query", "--vault", str(tmp_path), "--synthesize", "What is Topic?"],
    )

    assert result.exit_code == 0
    log_after = _run(["git", "log", "--format=%s", "-1"], cwd=tmp_path).stdout.strip()
    assert log_after == "baseline"
    status = _run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=tmp_path).stdout
    assert "wiki/index.md" in status
    assert "wiki/log.md" in status
    assert "wiki/synthesis/Topic Overview.md" in status
