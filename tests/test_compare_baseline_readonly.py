"""Baseline-vault read-only invariant test.

Plan pitfall: runner must never mutate a user's baseline vault. This
test snapshots the vault directory's contents hash before + after a run
and asserts equality.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.compare.corpus import load_corpus
from obsidian_llm_wiki.compare.models import ContestantSpec
from obsidian_llm_wiki.compare.runner import run_compare
from obsidian_llm_wiki.pipeline.orchestrator import PipelineReport


def _dir_hash(root: Path) -> str:
    """Content-hash every file under root; stable under rename-free edits."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        h.update(str(rel).encode())
        h.update(b"\x00")
        if p.is_file():
            h.update(p.read_bytes())
        h.update(b"\x01")
    return h.hexdigest()


@pytest.fixture
def patched_pipeline(monkeypatch):
    fake_client = MagicMock()
    fake_client.close = MagicMock()
    monkeypatch.setattr("obsidian_llm_wiki.client_factory.build_client", lambda cfg: fake_client)

    fake_db = MagicMock()
    fake_db.close = MagicMock()
    monkeypatch.setattr("obsidian_llm_wiki.state.StateDB", lambda _p: fake_db)

    def fake_run(self, auto_approve=True, max_rounds=2):
        return PipelineReport(
            ingested=3,
            compiled=3,
            failed=[],
            published=3,
            lint_issues=0,
            stubs_created=0,
            rounds=1,
            timings={},
            concept_timings={},
        )

    monkeypatch.setattr(
        "obsidian_llm_wiki.pipeline.orchestrator.PipelineOrchestrator.run", fake_run
    )


def _fake_baseline_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "user_vault"
    raw = vault / "raw"
    raw.mkdir(parents=True)
    for i in range(3):
        (raw / f"real_note_{i}.md").write_text(f"# Real {i}\n\nSensitive content {i}.\n")
    (vault / "wiki").mkdir()
    (vault / "wiki" / "existing.md").write_text("# Existing article\n")
    (vault / "wiki.toml").write_text('[models]\nfast = "gemma4:e4b"\nheavy = "gemma4:e4b"\n')
    return vault


def test_baseline_vault_unchanged_after_run(tmp_path, patched_pipeline):
    vault = _fake_baseline_vault(tmp_path)
    hash_before = _dir_hash(vault)

    corpus = load_corpus(baseline_vault_path=vault)
    run_compare(
        contestants=[ContestantSpec(name="c", fast_model="m", heavy_model="m")],
        corpus=corpus,
        out_dir=tmp_path / "out",
        seeds=1,
    )

    hash_after = _dir_hash(vault)
    assert hash_before == hash_after, "baseline vault was mutated by the runner"


def test_baseline_vault_mtimes_unchanged(tmp_path, patched_pipeline):
    """Even read patterns should not update atime-but-mtime invariants."""
    vault = _fake_baseline_vault(tmp_path)
    mtimes_before = {p: p.stat().st_mtime_ns for p in vault.rglob("*") if p.is_file()}

    corpus = load_corpus(baseline_vault_path=vault)
    run_compare(
        contestants=[ContestantSpec(name="c", fast_model="m", heavy_model="m")],
        corpus=corpus,
        out_dir=tmp_path / "out",
        seeds=1,
    )

    mtimes_after = {p: p.stat().st_mtime_ns for p in vault.rglob("*") if p.is_file()}
    assert mtimes_before == mtimes_after
