"""Tests for git_ops — auto-commit and safe undo via git revert."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from obsidian_llm_wiki.git_ops import git_commit, git_init, git_log_olw, git_undo


@pytest.fixture
def git_vault(tmp_path: Path) -> Path:
    """Vault with initialised git repo and initial commit."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".olw").mkdir()
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
    # Create initial commit so HEAD exists
    (tmp_path / "README.md").write_text("init")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    return tmp_path


# ── git_init ──────────────────────────────────────────────────────────────────


def test_git_init_creates_repo(tmp_path: Path):
    git_init(tmp_path)
    assert (tmp_path / ".git").exists()


def test_git_init_noop_if_already_exists(git_vault: Path):
    # Should not raise or re-init
    git_init(git_vault)
    assert (git_vault / ".git").exists()


# ── git_commit ────────────────────────────────────────────────────────────────


def test_git_commit_creates_olw_commit(git_vault: Path):
    (git_vault / "wiki" / "test.md").write_text("hello")
    committed = git_commit(git_vault, "test commit", paths=["wiki/"])
    assert committed is True
    result = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        cwd=git_vault,
        capture_output=True,
        text=True,
    )
    assert "[olw]" in result.stdout
    assert "test commit" in result.stdout


def test_git_commit_nothing_to_commit(git_vault: Path):
    # No changes — should return False
    committed = git_commit(git_vault, "empty commit", paths=["wiki/"])
    assert committed is False


def test_git_commit_custom_paths(git_vault: Path):
    (git_vault / "wiki" / "article.md").write_text("content")
    committed = git_commit(git_vault, "custom paths", paths=["wiki/"])
    assert committed is True


def test_git_commit_returns_false_on_error(tmp_path: Path):
    # Not a git repo — should fail gracefully
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / "test.md").write_text("hello")
    committed = git_commit(tmp_path, "will fail")
    assert committed is False


# ── git_log_olw ───────────────────────────────────────────────────────────────


def test_git_log_olw_returns_olw_commits(git_vault: Path):
    # Create an [olw] commit
    (git_vault / "wiki" / "a.md").write_text("a")
    git_commit(git_vault, "first", paths=["wiki/"])
    (git_vault / "wiki" / "b.md").write_text("b")
    git_commit(git_vault, "second", paths=["wiki/"])

    commits = git_log_olw(git_vault, n=10)
    assert len(commits) == 2
    assert "[olw]" in commits[0]["message"]
    assert "second" in commits[0]["message"]


def test_git_log_olw_skips_non_olw_commits(git_vault: Path):
    # Create a regular commit
    (git_vault / "wiki" / "a.md").write_text("a")
    subprocess.run(["git", "add", "."], cwd=git_vault, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "regular commit"],
        cwd=git_vault,
        capture_output=True,
        check=True,
    )
    commits = git_log_olw(git_vault)
    assert len(commits) == 0


def test_git_log_olw_limits_results(git_vault: Path):
    for i in range(5):
        (git_vault / f"wiki/f{i}.md").write_text(f"content {i}")
        git_commit(git_vault, f"commit-{i}", paths=["wiki/"])
    commits = git_log_olw(git_vault, n=2)
    assert len(commits) == 2


def test_git_log_olw_returns_empty_on_non_repo(tmp_path: Path):
    commits = git_log_olw(tmp_path)
    assert commits == []


# ── git_undo ──────────────────────────────────────────────────────────────────


def test_git_undo_reverts_last_commit(git_vault: Path):
    (git_vault / "wiki" / "a.md").write_text("original")
    git_commit(git_vault, "create article", paths=["wiki/"])

    reverted = git_undo(git_vault, steps=1)
    assert len(reverted) == 1
    assert "create article" in reverted[0]

    # File should be removed (reverted the addition)
    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=git_vault,
        capture_output=True,
        text=True,
    )
    assert "Revert" in result.stdout


def test_git_undo_no_olw_commits(git_vault: Path):
    reverted = git_undo(git_vault, steps=1)
    assert reverted == []


def test_git_undo_multiple_steps(git_vault: Path):
    (git_vault / "wiki" / "a.md").write_text("a")
    git_commit(git_vault, "first", paths=["wiki/"])
    (git_vault / "wiki" / "b.md").write_text("b")
    git_commit(git_vault, "second", paths=["wiki/"])

    reverted = git_undo(git_vault, steps=2)
    assert len(reverted) == 2
