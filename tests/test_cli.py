"""Tests for CLI commands via Click's CliRunner (no Ollama needed)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from obsidian_llm_wiki.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def vault_dir(tmp_path: Path) -> Path:
    """Pre-initialised vault for CLI tests."""
    for d in ["raw", "wiki", "wiki/.drafts", "wiki/sources", ".olw"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "wiki.toml").write_text(
        '[models]\nfast = "test:1b"\nheavy = "test:7b"\n'
        "[ollama]\n"
        'url = "http://localhost:11434"\n'
        "timeout = 10\n"
    )
    # Init git
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    (tmp_path / "README.md").write_text("init")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    return tmp_path


# ── init ──────────────────────────────────────────────────────────────────────


def test_init_creates_structure(runner: CliRunner, tmp_path: Path):
    vault = tmp_path / "new-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0
    assert (vault / "raw").is_dir()
    assert (vault / "wiki").is_dir()
    assert (vault / "wiki" / ".drafts").is_dir()
    assert (vault / "wiki" / "sources").is_dir()
    assert (vault / ".olw").is_dir()
    assert (vault / "wiki.toml").exists()
    assert (vault / ".git").exists()


def test_init_idempotent(runner: CliRunner, tmp_path: Path):
    vault = tmp_path / "vault2"
    runner.invoke(cli, ["init", str(vault)])
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0


# ── status ────────────────────────────────────────────────────────────────────


def test_status_shows_table(runner: CliRunner, vault_dir: Path):
    result = runner.invoke(cli, ["status", "--vault", str(vault_dir)])
    assert result.exit_code == 0
    assert "Vault Status" in result.output


def test_status_shows_failed_flag(runner: CliRunner, vault_dir: Path):
    result = runner.invoke(cli, ["status", "--vault", str(vault_dir), "--failed"])
    assert result.exit_code == 0


# ── clean ─────────────────────────────────────────────────────────────────────


def test_clean_with_yes(runner: CliRunner, vault_dir: Path):
    # Create state.db so there's something to delete
    (vault_dir / ".olw" / "state.db").write_text("")
    (vault_dir / "wiki" / "article.md").write_text("content")

    result = runner.invoke(cli, ["clean", "--vault", str(vault_dir), "--yes"])
    assert result.exit_code == 0
    assert "Clean complete" in result.output
    # wiki/ structure should be recreated (empty)
    assert (vault_dir / "wiki").is_dir()
    assert (vault_dir / "wiki" / ".drafts").is_dir()
    assert (vault_dir / "wiki" / "sources").is_dir()
    # raw/ should still exist
    assert (vault_dir / "raw").is_dir()


# ── undo ──────────────────────────────────────────────────────────────────────


def test_undo_no_commits(runner: CliRunner, vault_dir: Path):
    result = runner.invoke(cli, ["undo", "--vault", str(vault_dir)])
    assert result.exit_code == 0
    assert "commits" in result.output and "revert" in result.output


# ── lint (no Ollama needed) ───────────────────────────────────────────────────


def test_lint_empty_wiki(runner: CliRunner, vault_dir: Path):
    result = runner.invoke(cli, ["lint", "--vault", str(vault_dir)])
    assert result.exit_code == 0
    assert "Health" in result.output


def test_lint_with_article(runner: CliRunner, vault_dir: Path):
    (vault_dir / "wiki" / "Test Article.md").write_text(
        "---\ntitle: Test Article\ntags: [test]\nstatus: published\n"
        "confidence: 0.8\n---\n\nContent about testing.\n"
    )
    result = runner.invoke(cli, ["lint", "--vault", str(vault_dir)])
    assert result.exit_code == 0


# ── ingest (requires mocked Ollama) ──────────────────────────────────────────


def test_ingest_no_args(runner: CliRunner, vault_dir: Path):
    """Without --all or paths, should print error."""
    with _mock_ollama():
        result = runner.invoke(cli, ["ingest", "--vault", str(vault_dir)])
    assert result.exit_code != 0


def test_ingest_all_empty_raw(runner: CliRunner, vault_dir: Path):
    """No notes in raw/ — should print 'No notes found'."""
    with _mock_ollama():
        result = runner.invoke(cli, ["ingest", "--vault", str(vault_dir), "--all"])
    assert result.exit_code == 0
    assert "No notes found" in result.output


# ── doctor (requires mocked Ollama) ──────────────────────────────────────────


def test_doctor_shows_checks(runner: CliRunner, vault_dir: Path):
    with _mock_ollama():
        result = runner.invoke(cli, ["doctor", "--vault", str(vault_dir)])
    assert result.exit_code == 0
    assert "olw doctor" in result.output


# ── approve ──────────────────────────────────────────────────────────────────


def test_approve_no_drafts(runner: CliRunner, vault_dir: Path):
    result = runner.invoke(cli, ["approve", "--vault", str(vault_dir), "--all"])
    assert result.exit_code == 0
    assert "No drafts" in result.output


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_ollama():
    """Context manager that mocks OllamaClient for CLI commands that need it."""
    mock_client = MagicMock()
    mock_client.healthcheck.return_value = True
    mock_client.require_healthy.return_value = None
    mock_client.list_models.return_value = ["test:1b", "test:7b"]
    mock_client.generate.return_value = (
        '{"summary":"test","key_concepts":[],"suggested_topics":[],"quality":"medium"}'
    )

    return patch(
        "obsidian_llm_wiki.ollama_client.OllamaClient",
        return_value=mock_client,
    )
