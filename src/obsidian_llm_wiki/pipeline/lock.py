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

    # Open with "a+" (create if absent, no truncation) so a competing process
    # that fails to acquire the lock does not clear the incumbent's PID.
    # We truncate and write the PID ourselves only after the lock is held.
    with open(lock_path, "a+") as f:
        import os

        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        # We now hold the lock — overwrite with our PID.
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def lock_holder_pid(vault: Path) -> int | None:
    """Return PID if the pipeline lock is actively held, None otherwise.

    Verifies the lock is actually held (not just a stale lock file) by
    attempting a non-blocking exclusive acquire. If that succeeds the lock
    is free; if it raises BlockingIOError the lock is live.
    """
    lock_path = vault / ".olw" / "pipeline.lock"
    if not lock_path.exists():
        return None
    try:
        pid = int(lock_path.read_text().strip())
    except Exception:
        return None
    if not _IS_POSIX:
        return pid
    import fcntl

    try:
        with open(lock_path) as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
        return None  # acquired → nobody holding it
    except BlockingIOError:
        return pid  # lock is live
