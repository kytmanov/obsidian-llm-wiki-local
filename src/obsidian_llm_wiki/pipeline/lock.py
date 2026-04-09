"""
Pipeline concurrency lock.

Prevents concurrent pipeline runs (olw watch + olw compile, etc.) from
racing on the same StateDB. Uses fcntl.flock() — advisory, auto-released
on process death (no stale-lock problem).

POSIX only (Linux/macOS). On Windows, locking is skipped with a warning.
Vault must be on a local filesystem — flock() is unreliable on NFS/Dropbox.
"""

from __future__ import annotations

import contextlib
import logging
import platform
from pathlib import Path

log = logging.getLogger(__name__)

_IS_POSIX = platform.system() != "Windows"

# Known sync directories that indicate a remote/synced vault
_SYNC_DIRS = {"Dropbox", "OneDrive", "iCloud Drive", "Google Drive"}


def _warn_if_synced(vault: Path) -> None:
    parts = set(vault.parts)
    for sync_dir in _SYNC_DIRS:
        if sync_dir in parts:
            log.warning(
                "Vault is inside '%s' — pipeline lock (flock) may be unreliable on synced "
                "filesystems. Ensure .olw/ is on a local path.",
                sync_dir,
            )
            break


@contextlib.contextmanager
def pipeline_lock(vault: Path, block: bool = False):
    """
    Acquire an exclusive pipeline lock for the vault.

    Yields True if the lock was acquired, False if it was already held.
    The lock is released on context exit, including on exceptions.

    Usage::

        with pipeline_lock(config.vault) as acquired:
            if not acquired:
                console.print("⚠ pipeline already running")
                return
            # ... do pipeline work ...
    """
    if not _IS_POSIX:
        log.warning("Pipeline lock not supported on Windows — proceeding without lock")
        yield True
        return

    import fcntl

    _warn_if_synced(vault)

    lock_path = vault / ".olw" / "pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "w") as f:
        import os

        f.write(str(os.getpid()))
        f.flush()
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def lock_holder_pid(vault: Path) -> int | None:
    """Return the PID written in the lock file, or None if absent/unreadable."""
    lock_path = vault / ".olw" / "pipeline.lock"
    try:
        return int(lock_path.read_text().strip())
    except Exception:
        return None
