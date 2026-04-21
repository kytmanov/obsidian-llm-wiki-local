"""Tests for compare/runner.py — orchestrates contestants × seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.compare.corpus import Corpus, CorpusMode, Note, Query
from obsidian_llm_wiki.compare.models import ContestantSpec
from obsidian_llm_wiki.compare.runner import _synthesize_wiki_toml, run_compare
from obsidian_llm_wiki.pipeline.orchestrator import PipelineReport


def _fake_corpus(tmp_path: Path, *, with_query: bool = True) -> Corpus:
    src = tmp_path / "src_notes"
    src.mkdir()
    notes: list[Note] = []
    for i in range(3):
        p = src / f"note_{i}.md"
        p.write_text(f"# Note {i}\n\nBody {i}.\n")
        notes.append(Note(file=p.name, path=p.resolve(), category="byo"))
    queries = (
        [Query(id="q1", question="What?", expected_pages=[], expected_contains=[])]
        if with_query
        else []
    )
    return Corpus(
        mode=CorpusMode.BYO,
        version="test",
        language="",
        description="",
        notes=notes,
        queries=queries,
        notes_set_hash=hashlib.sha256(b"x").hexdigest(),
        root=src,
    )


def _spec(name: str = "c1") -> ContestantSpec:
    return ContestantSpec(name=name, fast_model="m1", heavy_model="m2")


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Stub client, StateDB, orchestrator, and run_query — no real LLM calls."""
    fake_client = MagicMock()
    fake_client.close = MagicMock()
    monkeypatch.setattr("obsidian_llm_wiki.client_factory.build_client", lambda cfg: fake_client)

    fake_db = MagicMock()
    fake_db.close = MagicMock()
    monkeypatch.setattr("obsidian_llm_wiki.state.StateDB", lambda _path: fake_db)

    def fake_run(self, auto_approve=True, max_rounds=2):
        return PipelineReport(
            ingested=3,
            compiled=2,
            failed=[],
            published=2,
            lint_issues=0,
            stubs_created=0,
            rounds=1,
            timings={"ingest": 1.0, "compile": 2.0},
            concept_timings={"A": 0.5, "B": 0.6},
        )

    monkeypatch.setattr(
        "obsidian_llm_wiki.pipeline.orchestrator.PipelineOrchestrator.run", fake_run
    )

    def fake_query(**_kw):
        return ("Answer.", ["page1"])

    monkeypatch.setattr("obsidian_llm_wiki.pipeline.query.run_query", fake_query)

    return fake_client, fake_db


# ── _synthesize_wiki_toml ─────────────────────────────────────────────────────


def test_synth_toml_ollama_default():
    toml = _synthesize_wiki_toml(_spec())
    assert "[ollama]" in toml
    assert "[models]" in toml
    assert 'fast = "m1"' in toml
    assert 'heavy = "m2"' in toml
    assert "auto_commit = false" in toml
    assert "auto_approve = true" in toml


def test_synth_toml_non_ollama_provider():
    spec = ContestantSpec(
        name="c1",
        fast_model="m1",
        heavy_model="m2",
        provider_name="groq",
        provider_url="https://api.groq.com/openai/v1",
    )
    toml = _synthesize_wiki_toml(spec)
    assert "[provider]" in toml
    assert "[ollama]" not in toml
    assert 'name = "groq"' in toml


# ── run_compare ───────────────────────────────────────────────────────────────


def test_run_compare_rejects_empty(tmp_path, patched_pipeline):
    with pytest.raises(ValueError, match="At least one"):
        run_compare(contestants=[], corpus=_fake_corpus(tmp_path), out_dir=tmp_path, seeds=1)


def test_run_compare_rejects_zero_seeds(tmp_path, patched_pipeline):
    with pytest.raises(ValueError, match=">= 1"):
        run_compare(contestants=[_spec()], corpus=_fake_corpus(tmp_path), out_dir=tmp_path, seeds=0)


def test_run_compare_happy_path(tmp_path, patched_pipeline):
    specs = [_spec("a"), _spec("b")]
    report = run_compare(
        contestants=specs,
        corpus=_fake_corpus(tmp_path),
        out_dir=tmp_path,
        seeds=2,
        keep_artifacts=True,
    )
    assert report.seeds == 2
    assert report.mode == "byo"
    assert len(report.contestants) == 2
    for r in report.contestants:
        assert not r.partial
        assert set(r.seed_events.keys()) == {0, 1}
        assert set(r.seed_artifacts.keys()) == {0, 1}


def test_run_compare_writes_raw_report_json(tmp_path, patched_pipeline):
    report = run_compare(
        contestants=[_spec()],
        corpus=_fake_corpus(tmp_path),
        out_dir=tmp_path,
        seeds=1,
    )
    raw = tmp_path / report.run_id / "results" / "raw_report.json"
    assert raw.exists()
    data = json.loads(raw.read_text())
    assert data["run_id"] == report.run_id
    assert data["mode"] == "byo"
    assert len(data["contestants"]) == 1


def test_run_compare_per_seed_checkpoint(tmp_path, patched_pipeline):
    report = run_compare(
        contestants=[_spec("solo")],
        corpus=_fake_corpus(tmp_path),
        out_dir=tmp_path,
        seeds=2,
    )
    ckpt = tmp_path / report.run_id / "results" / "solo" / "seed_0.json"
    assert ckpt.exists()
    payload = json.loads(ckpt.read_text())
    assert payload["contestant"] == "solo"
    assert payload["seed"] == 0
    assert payload["pipeline_report"]["compiled"] == 2


def test_run_compare_cleans_vaults_by_default(tmp_path, patched_pipeline):
    report = run_compare(
        contestants=[_spec("x")],
        corpus=_fake_corpus(tmp_path),
        out_dir=tmp_path,
        seeds=1,
    )
    vault = tmp_path / report.run_id / "vaults" / "x" / "0"
    assert not vault.exists()


def test_run_compare_keeps_vaults_with_flag(tmp_path, patched_pipeline):
    report = run_compare(
        contestants=[_spec("x")],
        corpus=_fake_corpus(tmp_path),
        out_dir=tmp_path,
        seeds=1,
        keep_artifacts=True,
    )
    vault = tmp_path / report.run_id / "vaults" / "x" / "0"
    assert vault.exists()
    assert (vault / "raw").is_dir()
    assert (vault / "wiki.toml").exists()


def test_run_compare_pipeline_crash_marks_partial(tmp_path, monkeypatch, patched_pipeline):
    """Pipeline crash on one seed → contestant flagged partial, run continues."""

    def boom(self, auto_approve=True, max_rounds=2):
        raise RuntimeError("simulated pipeline crash")

    monkeypatch.setattr("obsidian_llm_wiki.pipeline.orchestrator.PipelineOrchestrator.run", boom)

    report = run_compare(
        contestants=[_spec("crashy")],
        corpus=_fake_corpus(tmp_path),
        out_dir=tmp_path,
        seeds=1,
    )
    assert report.contestants[0].partial is True


def test_run_compare_skip_queries_flag(tmp_path, monkeypatch, patched_pipeline):
    called = {"n": 0}

    def tracking_query(**_kw):
        called["n"] += 1
        return ("x", [])

    monkeypatch.setattr("obsidian_llm_wiki.pipeline.query.run_query", tracking_query)

    run_compare(
        contestants=[_spec()],
        corpus=_fake_corpus(tmp_path, with_query=True),
        out_dir=tmp_path,
        seeds=1,
        skip_queries=True,
    )
    assert called["n"] == 0


def test_run_compare_vault_collision_refused(tmp_path, patched_pipeline):
    """Pre-existing ephemeral vault path must abort (pipeline-lock safety)."""
    specs = [_spec("collide")]
    run_id = "fixed-run"
    # Pre-create the would-be ephemeral vault path.
    (tmp_path / run_id / "vaults" / "collide" / "0").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already exists"):
        run_compare(
            contestants=specs,
            corpus=_fake_corpus(tmp_path),
            out_dir=tmp_path,
            seeds=1,
            run_id=run_id,
        )


def test_run_compare_populates_version_pins(tmp_path, patched_pipeline):
    report = run_compare(
        contestants=[_spec()],
        corpus=_fake_corpus(tmp_path),
        out_dir=tmp_path,
        seeds=1,
    )
    assert report.olw_version  # non-empty
    assert len(report.pipeline_prompt_hash) == 16
    assert report.notes_set_hash
