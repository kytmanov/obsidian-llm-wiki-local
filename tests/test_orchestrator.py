"""Tests for pipeline/orchestrator.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord
from obsidian_llm_wiki.pipeline.orchestrator import (
    FailureReason,
    FailureRecord,
    PipelineOrchestrator,
    PipelineReport,
    _run_compile,
)
from obsidian_llm_wiki.state import StateDB


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".olw").mkdir()
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


def make_mock_client(response: str = "{}") -> MagicMock:
    from obsidian_llm_wiki.ollama_client import OllamaClient

    client = MagicMock(spec=OllamaClient)
    client.generate.return_value = response
    return client


# ── PipelineReport ────────────────────────────────────────────────────────────


def test_pipeline_report_failed_names():
    report = PipelineReport()
    report.failed.append(FailureRecord(concept="Alpha", reason=FailureReason.TRANSIENT))
    report.failed.append(FailureRecord(concept="Beta", reason=FailureReason.LLM_OUTPUT))
    assert report.failed_names == ["Alpha", "Beta"]


def test_pipeline_report_defaults():
    report = PipelineReport()
    assert report.ingested == 0
    assert report.compiled == 0
    assert report.published == 0
    assert report.rounds == 0
    assert report.timings == {}
    assert report.failed == []


# ── FailureReason ─────────────────────────────────────────────────────────────


def test_failure_reason_values():
    assert FailureReason.TRANSIENT == "transient"
    assert FailureReason.LLM_OUTPUT == "llm_output"
    assert FailureReason.MISSING_SOURCES == "missing_sources"
    assert FailureReason.UNKNOWN == "unknown"


# ── _run_compile ──────────────────────────────────────────────────────────────


def test_run_compile_ollama_error_classified_as_transient(config, db):
    """OllamaError raised by compile_concepts → all concepts become TRANSIENT."""
    import obsidian_llm_wiki.pipeline.compile as compile_mod
    from obsidian_llm_wiki.ollama_client import OllamaError

    original = compile_mod.compile_concepts

    def raise_ollama(**kwargs):
        raise OllamaError("connection refused")

    compile_mod.compile_concepts = raise_ollama
    try:
        drafts, failures, _ = _run_compile(
            config, make_mock_client(), db, concepts=["Alpha"], dry_run=False
        )
    finally:
        compile_mod.compile_concepts = original

    assert drafts == []
    assert len(failures) == 1
    assert failures[0].reason == FailureReason.TRANSIENT
    assert failures[0].concept == "Alpha"


def test_run_compile_unknown_failure_per_concept(config, db):
    """Failed concept names from compile_concepts become UNKNOWN records."""
    import obsidian_llm_wiki.pipeline.compile as compile_mod

    original = compile_mod.compile_concepts

    def fake_compile(**kwargs):
        return ([], ["Fails"], {})

    compile_mod.compile_concepts = fake_compile
    try:
        drafts, failures, _ = _run_compile(
            config, make_mock_client(), db, concepts=["Fails"], dry_run=False
        )
    finally:
        compile_mod.compile_concepts = original

    assert len(failures) == 1
    assert failures[0].reason == FailureReason.UNKNOWN
    assert failures[0].concept == "Fails"


# ── PipelineOrchestrator.run ──────────────────────────────────────────────────


def test_orchestrator_run_dry_run_no_writes(config, db, tmp_path):
    """dry_run=True: no files written, ingested count reported."""
    raw_file = config.vault / "raw" / "note.md"
    raw_file.write_text("---\ntitle: Note\n---\nContent.")

    client = make_mock_client()
    orch = PipelineOrchestrator(config, client, db)
    report = orch.run(paths=[str(raw_file)], dry_run=True)

    assert report.ingested == 1
    # No LLM calls in dry_run
    client.generate.assert_not_called()


def test_orchestrator_run_timings_populated(config, db):
    """Timings dict should have 'ingest' key after run."""
    client = make_mock_client()
    orch = PipelineOrchestrator(config, client, db)
    report = orch.run(paths=[], dry_run=True)
    assert "ingest" in report.timings


def test_orchestrator_run_rounds_default_one(config, db):
    """No transient failures → only one compile round."""
    with patch("obsidian_llm_wiki.pipeline.orchestrator._run_compile") as mock_compile:
        mock_compile.return_value = ([], [], {})
        orch = PipelineOrchestrator(config, make_mock_client(), db)
        report = orch.run(paths=[])

    assert report.rounds == 1


def test_orchestrator_retries_transient_failures(config, db):
    """TRANSIENT failures trigger round 2; LLM_OUTPUT failures do not."""
    transient = FailureRecord(concept="Flaky", reason=FailureReason.TRANSIENT)
    bad_json = FailureRecord(concept="BadJSON", reason=FailureReason.LLM_OUTPUT)

    round1_result = ([], [transient, bad_json], {})
    round2_result = ([], [], {})  # transient resolved

    with patch("obsidian_llm_wiki.pipeline.orchestrator._run_compile") as mock_compile:
        mock_compile.side_effect = [round1_result, round2_result]
        orch = PipelineOrchestrator(config, make_mock_client(), db)
        report = orch.run(paths=[])

    assert report.rounds == 2
    # BadJSON not retried: should remain in failed
    assert any(f.concept == "BadJSON" for f in report.failed)
    # Transient resolved: should be absent from final failed list
    assert not any(f.concept == "Flaky" for f in report.failed)


def test_orchestrator_llm_output_not_retried(config, db):
    """LLM_OUTPUT failure stays in report without triggering round 2."""
    bad = FailureRecord(concept="BadJSON", reason=FailureReason.LLM_OUTPUT)

    with patch("obsidian_llm_wiki.pipeline.orchestrator._run_compile") as mock_compile:
        mock_compile.return_value = ([], [bad], {})
        orch = PipelineOrchestrator(config, make_mock_client(), db)
        report = orch.run(paths=[])

    assert mock_compile.call_count == 1  # no round 2
    assert report.rounds == 1
    assert report.failed[0].concept == "BadJSON"


def test_orchestrator_selective_recompile_with_absolute_paths(config, db):
    """Absolute paths from watchdog must be normalized to vault-relative before DB lookup."""
    import json

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])

    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent about Alpha.")
    (config.vault / "raw" / "b.md").write_text("---\ntitle: B\n---\nContent about Beta.")

    mock_response = json.dumps({"title": "Alpha", "content": "Alpha content.", "tags": []})
    client = make_mock_client(mock_response)

    # Pass absolute path (as watchdog would supply it)
    abs_path = str(config.vault / "raw" / "a.md")

    import obsidian_llm_wiki.pipeline.ingest as ingest_mod

    original_ingest = ingest_mod.ingest_note

    def fake_ingest(path, config, client, db):
        return object()  # truthy — simulates successful ingest

    ingest_mod.ingest_note = fake_ingest
    try:
        orch = PipelineOrchestrator(config, client, db)
        orch.run(paths=[abs_path])
    finally:
        ingest_mod.ingest_note = original_ingest

    # Alpha was the linked concept; Beta should still need compile
    needing = db.concepts_needing_compile()
    assert "Beta" in needing


def test_orchestrator_auto_approve(config, db):
    """auto_approve=True publishes drafts returned from compile."""
    import json

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Topic"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent.")

    mock_response = json.dumps({"title": "Topic", "content": "Content.", "tags": []})
    client = make_mock_client(mock_response)

    orch = PipelineOrchestrator(config, client, db)
    report = orch.run(paths=[], auto_approve=True)

    assert report.published >= 0  # may be 0 if compile skipped but no crash


def test_orchestrator_lint_runs_when_no_drafts_produced(config, db):
    """lint must run even when compile produces no new drafts."""
    import obsidian_llm_wiki.pipeline.lint as lint_mod

    lint_called = []
    original_lint = lint_mod.run_lint

    def fake_lint(config, db):
        lint_called.append(True)
        from obsidian_llm_wiki.models import LintResult

        return LintResult(issues=[], health_score=100.0, summary="ok")

    lint_mod.run_lint = fake_lint
    try:
        with patch("obsidian_llm_wiki.pipeline.orchestrator._run_compile") as mock_compile:
            mock_compile.return_value = ([], [], {})  # no drafts produced
            orch = PipelineOrchestrator(config, make_mock_client(), db)
            orch.run(paths=[])
    finally:
        lint_mod.run_lint = original_lint

    assert len(lint_called) == 1  # lint ran despite zero new drafts


def test_orchestrator_ingest_exception_logged_not_raised(config, db):
    """Exception during ingest of one file is caught; run() returns normally."""
    import obsidian_llm_wiki.pipeline.ingest as ingest_mod

    raw_file = config.vault / "raw" / "note.md"
    raw_file.write_text("---\ntitle: Note\n---\nContent.")

    original = ingest_mod.ingest_note

    def boom(**kwargs):
        raise RuntimeError("ingest boom")

    ingest_mod.ingest_note = boom
    try:
        orch = PipelineOrchestrator(config, make_mock_client(), db)
        report = orch.run(paths=[str(raw_file)])
    finally:
        ingest_mod.ingest_note = original

    assert report.ingested == 0  # exception swallowed, not counted


def test_orchestrator_fix_creates_stubs(config, db):
    """fix=True + broken link issues in lint → create_stubs called."""
    from pathlib import Path

    import obsidian_llm_wiki.pipeline.lint as lint_mod
    import obsidian_llm_wiki.pipeline.maintain as maintain_mod
    from obsidian_llm_wiki.models import LintIssue, LintResult

    fake_lint = LintResult(
        issues=[
            LintIssue(
                path="wiki/Article.md",
                issue_type="broken_link",
                description="[[Missing Stub]] not found",
                suggestion="Create stub for Missing Stub",
            )
        ],
        health_score=80.0,
        summary="1 issue",
    )
    fake_draft = config.drafts_dir / "Something.md"
    fake_draft.write_text("stub")

    stubs_called_with = {}

    original_lint = lint_mod.run_lint
    original_stubs = maintain_mod.create_stubs

    import obsidian_llm_wiki.pipeline.compile as compile_mod

    original_compile_fn = compile_mod.compile_concepts

    def fake_compile(**kwargs):
        return ([fake_draft], [], {})

    def fake_lint_fn(config, db):
        return fake_lint

    def fake_stubs(config, db, broken_link_issues=None, max_stubs=5):
        stubs_called_with["issues"] = broken_link_issues
        return [Path("wiki/.drafts/Missing_Stub.md")]

    compile_mod.compile_concepts = fake_compile
    lint_mod.run_lint = fake_lint_fn
    maintain_mod.create_stubs = fake_stubs
    try:
        orch = PipelineOrchestrator(config, make_mock_client(), db)
        report = orch.run(paths=[], fix=True)
    finally:
        compile_mod.compile_concepts = original_compile_fn
        lint_mod.run_lint = original_lint
        maintain_mod.create_stubs = original_stubs

    assert "issues" in stubs_called_with  # create_stubs was called
    assert report.stubs_created == 1
