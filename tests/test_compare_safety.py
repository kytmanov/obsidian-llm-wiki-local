"""Safety tests for compare MVP."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.compare.runner import run_compare
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.pipeline.orchestrator import PipelineReport


def _dir_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if rel.parts[:2] == (".olw", "compare"):
            continue
        h.update(str(rel).encode())
        h.update(b"\x00")
        if p.is_file():
            h.update(p.read_bytes())
        h.update(b"\x01")
    return h.hexdigest()


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    raw = vault / "raw"
    wiki = vault / "wiki"
    olw = vault / ".olw"
    raw.mkdir(parents=True)
    wiki.mkdir()
    olw.mkdir()
    for i in range(3):
        (raw / f"n{i}.md").write_text(f"# Note {i}\n\nBody {i}.\n")
    (wiki / "existing.md").write_text("---\ntitle: Existing\n---\nBody\n")
    (vault / "wiki.toml").write_text(
        '[models]\nfast = "base-fast"\nheavy = "base-heavy"\n\n[ollama]\nurl = "http://localhost:11434"\n'
    )
    return vault


@pytest.fixture
def patched_pipeline(monkeypatch):
    fake_client = MagicMock()
    fake_client.close = MagicMock()
    monkeypatch.setattr("obsidian_llm_wiki.client_factory.build_client", lambda cfg: fake_client)

    fake_db = MagicMock()
    fake_db.close = MagicMock()
    monkeypatch.setattr("obsidian_llm_wiki.state.StateDB", lambda _path: fake_db)

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

    from obsidian_llm_wiki.models import LintResult

    monkeypatch.setattr(
        "obsidian_llm_wiki.pipeline.lint.run_lint",
        lambda config, db, fix=False: LintResult(issues=[], health_score=100.0, summary="clean"),
    )


def test_compare_does_not_mutate_active_vault(tmp_path, patched_pipeline):
    vault = _make_vault(tmp_path)
    before = _dir_hash(vault)
    current = Config.from_vault(vault)
    challenger = Config.from_vault(vault, models={"heavy": "new-heavy"})
    run_compare(current, challenger, vault / ".olw" / "compare")
    after = _dir_hash(vault)
    assert before == after


def test_compare_never_creates_active_queries_dir(tmp_path, patched_pipeline):
    vault = _make_vault(tmp_path)
    queries = tmp_path / "queries.toml"
    queries.write_text('[query]\nid = "q1"\nquestion = "What?"\n')
    current = Config.from_vault(vault)
    challenger = Config.from_vault(vault, models={"heavy": "new-heavy"})
    run_compare(current, challenger, vault / ".olw" / "compare", queries_path=queries)
    assert not (vault / "wiki" / "queries").exists()


def test_compare_rejects_programmatic_out_dir_inside_raw(tmp_path, patched_pipeline):
    vault = _make_vault(tmp_path)
    current = Config.from_vault(vault)
    challenger = Config.from_vault(vault, models={"heavy": "new-heavy"})

    with pytest.raises(ValueError, match="inside active vault raw/ or wiki/"):
        run_compare(current, challenger, vault / "raw" / "compare")


def test_compare_rejects_programmatic_out_dir_inside_vault_outside_compare(
    tmp_path, patched_pipeline
):
    vault = _make_vault(tmp_path)
    current = Config.from_vault(vault)
    challenger = Config.from_vault(vault, models={"heavy": "new-heavy"})

    with pytest.raises(ValueError, match=r"must be under \.olw/compare/"):
        run_compare(current, challenger, vault / "reports")


def test_compare_preserves_active_vault_schema_file(tmp_path, patched_pipeline):
    vault = _make_vault(tmp_path)
    schema = vault / "vault-schema.md"
    schema.write_text("# Active schema\n")
    current = Config.from_vault(vault)
    challenger = Config.from_vault(vault, models={"heavy": "new-heavy"})

    run_compare(current, challenger, vault / ".olw" / "compare")

    assert schema.read_text() == "# Active schema\n"
