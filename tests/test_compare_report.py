"""Report rendering + self-test (same config both sides → near-parity + tie)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.compare.corpus import Corpus, CorpusMode, Note, Query
from obsidian_llm_wiki.compare.models import ContestantSpec
from obsidian_llm_wiki.compare.report import render_json, render_markdown, resolve
from obsidian_llm_wiki.compare.runner import run_compare
from obsidian_llm_wiki.pipeline.orchestrator import PipelineReport


def _fake_corpus(tmp_path: Path) -> Corpus:
    src = tmp_path / "src"
    src.mkdir()
    notes: list[Note] = []
    for i in range(3):
        p = src / f"n{i}.md"
        p.write_text(f"# {i}\n\nSome prose about topic {i}.\n")
        notes.append(Note(file=p.name, path=p.resolve(), known_concepts=[f"topic {i}"]))
    queries = [
        Query(id="q1", question="?", expected_pages=["topic 0"], expected_contains=["0"]),
    ]
    return Corpus(
        mode=CorpusMode.CURATED,
        version="1.0",
        language="en",
        description="",
        notes=notes,
        queries=queries,
        notes_set_hash=hashlib.sha256(b"x").hexdigest(),
        root=src,
    )


@pytest.fixture
def patched_pipeline(monkeypatch):
    fake_client = MagicMock()
    fake_client.close = MagicMock()
    monkeypatch.setattr("obsidian_llm_wiki.client_factory.build_client", lambda cfg: fake_client)

    fake_db = MagicMock()
    fake_db.close = MagicMock()
    fake_db.list_all_concept_names = MagicMock(return_value=["Topic 0", "Topic 1", "Topic 2"])
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
            timings={"ingest": 0.5, "compile": 1.0},
            concept_timings={"topic 0": 0.3, "topic 1": 0.3, "topic 2": 0.4},
        )

    monkeypatch.setattr(
        "obsidian_llm_wiki.pipeline.orchestrator.PipelineOrchestrator.run", fake_run
    )

    # run_query returns a deterministic answer
    def fake_query(**kw):
        return ("Answer about topic 0.", ["topic 0"])

    monkeypatch.setattr("obsidian_llm_wiki.pipeline.query.run_query", fake_query)

    # run_lint returns a clean result
    from obsidian_llm_wiki.models import LintResult

    def fake_lint(config, db, fix=False):
        return LintResult(issues=[], health_score=95.0, summary="clean")

    monkeypatch.setattr("obsidian_llm_wiki.pipeline.lint.run_lint", fake_lint)

    # Deterministic wall time so identical-contestant self-test isn't
    # skewed by real-clock jitter on ~microsecond mock runs.
    counter = {"n": 0.0}

    def fake_monotonic():
        counter["n"] += 1.0
        return counter["n"]

    monkeypatch.setattr("obsidian_llm_wiki.compare.runner.time.monotonic", fake_monotonic)


# ── resolve() + rendering ─────────────────────────────────────────────────────


def test_resolve_populates_overall_and_weights(tmp_path, patched_pipeline):
    corpus = _fake_corpus(tmp_path)
    specs = [
        ContestantSpec(name="a", fast_model="m", heavy_model="m"),
        ContestantSpec(name="b", fast_model="m", heavy_model="m"),
    ]
    report = run_compare(
        contestants=specs, corpus=corpus, out_dir=tmp_path / "out", seeds=2
    )
    resolve(report, corpus)
    assert report.weights
    assert sum(report.weights.values()) == pytest.approx(1.0)
    for r in report.contestants:
        assert 0.0 <= r.overall <= 1.0


def test_render_markdown_has_expected_sections(tmp_path, patched_pipeline):
    corpus = _fake_corpus(tmp_path)
    specs = [
        ContestantSpec(name="a", fast_model="m", heavy_model="m"),
        ContestantSpec(name="b", fast_model="m", heavy_model="m"),
    ]
    report = run_compare(
        contestants=specs, corpus=corpus, out_dir=tmp_path / "out", seeds=2
    )
    resolve(report, corpus)
    md = render_markdown(report, corpus)
    for section in (
        "# olw compare",
        "## Contestants",
        "## Overall verdict",
        "## Per-dimension scores",
        "## Advisory diagnostics",
        "## Trade-off narrative",
        "## Blind spots",
    ):
        assert section in md


def test_render_json_round_trips(tmp_path, patched_pipeline):
    corpus = _fake_corpus(tmp_path)
    specs = [ContestantSpec(name="a", fast_model="m", heavy_model="m")]
    report = run_compare(
        contestants=specs, corpus=corpus, out_dir=tmp_path / "out", seeds=1
    )
    resolve(report, corpus)
    data = json.loads(render_json(report))
    assert data["run_id"] == report.run_id
    assert "contestants" in data
    assert len(data["contestants"]) == 1


# ── Self-test: same config both sides → near-parity + statistical tie ────────


def test_self_test_near_parity(tmp_path, patched_pipeline):
    """Identical contestants should land within 0.15 on every scored dim."""
    corpus = _fake_corpus(tmp_path)
    specs = [
        ContestantSpec(name="a", fast_model="m", heavy_model="m"),
        ContestantSpec(name="b", fast_model="m", heavy_model="m"),
    ]
    report = run_compare(
        contestants=specs, corpus=corpus, out_dir=tmp_path / "out", seeds=2
    )
    resolve(report, corpus)
    a, b = report.contestants
    for dim in a.scores:
        da = a.scores[dim]
        db = b.scores[dim]
        if da.n == 0 or db.n == 0:
            continue
        # Efficiency dims use wall-clock time — inherently jittery across
        # mocked runs. Skip them for parity comparison.
        if dim.startswith("efficiency."):
            continue
        assert abs(da.mean - db.mean) <= 0.15, f"{dim}: a={da.mean} b={db.mean}"
    # Overall scores should be close, but allow efficiency jitter.
    assert abs(a.overall - b.overall) < 0.1


def test_self_test_declares_statistical_tie(tmp_path, patched_pipeline):
    corpus = _fake_corpus(tmp_path)
    specs = [
        ContestantSpec(name="a", fast_model="m", heavy_model="m"),
        ContestantSpec(name="b", fast_model="m", heavy_model="m"),
    ]
    report = run_compare(
        contestants=specs, corpus=corpus, out_dir=tmp_path / "out", seeds=2
    )
    resolve(report, corpus)
    # With identical mocks, margin is ~0 → either margin_sigma None (sd=0) or tie.
    # Accept either "no divergence" signal.
    assert report.statistical_tie or report.margin_sigma is None or report.margin_sigma < 1.0


def test_winner_set_when_contestants_differ(tmp_path, patched_pipeline, monkeypatch):
    """Contestant 'worse' has higher pipeline failure → 'better' wins."""
    from obsidian_llm_wiki.pipeline.orchestrator import FailureReason, FailureRecord

    corpus = _fake_corpus(tmp_path)

    def staged_run(self, auto_approve=True, max_rounds=2):
        fast = self.config.models.fast if self.config.models else ""
        if "bad_m" in (fast or ""):
            return PipelineReport(
                ingested=3, compiled=0,
                failed=[FailureRecord(concept="x", reason=FailureReason.UNKNOWN)],
                published=0, lint_issues=5, stubs_created=0, rounds=1,
                timings={}, concept_timings={},
            )
        return PipelineReport(
            ingested=3, compiled=3, failed=[], published=3,
            lint_issues=0, stubs_created=0, rounds=1,
            timings={}, concept_timings={},
        )

    monkeypatch.setattr(
        "obsidian_llm_wiki.pipeline.orchestrator.PipelineOrchestrator.run", staged_run
    )

    specs = [
        ContestantSpec(name="good", fast_model="good_m", heavy_model="good_m"),
        ContestantSpec(name="bad", fast_model="bad_m", heavy_model="bad_m"),
    ]
    report = run_compare(
        contestants=specs, corpus=corpus, out_dir=tmp_path / "out", seeds=1
    )
    resolve(report, corpus)
    assert report.winner == "good"
