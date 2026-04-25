"""Tests for pipeline/lock.py."""

from __future__ import annotations

import platform
import threading
from pathlib import Path

import pytest

from obsidian_llm_wiki.pipeline.lock import has_invalid_lock_file, lock_holder_pid, pipeline_lock


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / ".olw").mkdir()
    return tmp_path


# ── Basic lock acquisition ────────────────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_acquired_yields_true(vault):
    with pipeline_lock(vault) as acquired:
        assert acquired is True


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_held_yields_false(vault):
    with pipeline_lock(vault) as acquired:
        assert acquired is True
        # Second non-blocking attempt while first is held
        with pipeline_lock(vault, block=False) as acquired2:
            assert acquired2 is False


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_released_after_context(vault):
    with pipeline_lock(vault) as acquired:
        assert acquired is True
    # After context exits, lock should be acquirable again
    with pipeline_lock(vault) as acquired:
        assert acquired is True


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_released_on_exception(vault):
    try:
        with pipeline_lock(vault) as acquired:
            assert acquired is True
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    # Lock must be released even though exception was raised
    with pipeline_lock(vault) as acquired:
        assert acquired is True


# ── Lock file creation ────────────────────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_file_created(vault):
    with pipeline_lock(vault):
        assert (vault / ".olw" / "pipeline.lock").exists()


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_file_contains_pid(vault):
    import os

    with pipeline_lock(vault):
        pid = lock_holder_pid(vault)
        assert pid == os.getpid()


# ── lock_holder_pid ───────────────────────────────────────────────────────────


def test_lock_holder_pid_no_file(vault):
    # No lock file exists yet
    assert lock_holder_pid(vault) is None


def test_lock_holder_pid_unreadable(vault):
    # Write garbage
    lock_path = vault / ".olw" / "pipeline.lock"
    lock_path.write_text("not-a-pid")
    assert lock_holder_pid(vault) is None


def test_has_invalid_lock_file_detects_garbage(vault):
    lock_path = vault / ".olw" / "pipeline.lock"
    lock_path.write_text("not-a-pid")

    assert has_invalid_lock_file(vault) is True


# ── Thread safety ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_only_one_thread_acquires_lock(vault):
    """Concurrent non-blocking attempts: exactly one succeeds."""
    results: list[bool] = []
    barrier = threading.Barrier(2)

    def try_lock():
        with pipeline_lock(vault, block=False) as acquired:
            barrier.wait()  # both threads enter before either exits
            results.append(acquired)

    t1 = threading.Thread(target=try_lock)
    t2 = threading.Thread(target=try_lock)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert results.count(True) == 1
    assert results.count(False) == 1


# ── Stale PID detection ───────────────────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_holder_pid_returns_none_after_release(vault):
    """lock_holder_pid returns None once the flock is released (not just stale file)."""
    with pipeline_lock(vault) as acquired:
        assert acquired is True
        pid_during = lock_holder_pid(vault)
        assert pid_during is not None  # lock held → pid visible
    # After context exit flock is released; file still exists but lock is free
    pid_after = lock_holder_pid(vault)
    assert pid_after is None  # stale file should not report lock as held


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_lock_holder_pid_returns_pid_while_held(vault):
    import os

    with pipeline_lock(vault) as acquired:
        assert acquired is True
        pid = lock_holder_pid(vault)
        assert pid == os.getpid()


# ── PID written after flock acquired ─────────────────────────────────────────


@pytest.mark.skipif(platform.system() == "Windows", reason="flock POSIX only")
def test_pid_written_after_lock_acquired(vault):
    """Lock file should contain our PID only after the flock is held."""
    import os

    with pipeline_lock(vault) as acquired:
        assert acquired is True
        lock_path = vault / ".olw" / "pipeline.lock"
        pid_in_file = int(lock_path.read_text().strip())
        assert pid_in_file == os.getpid()


# ── Windows fallback ──────────────────────────────────────────────────────────


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
def test_windows_yields_true_without_lock(vault):
    with pipeline_lock(vault) as acquired:
        assert acquired is True
