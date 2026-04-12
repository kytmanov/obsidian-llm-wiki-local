"""
Offline micro-benchmarks for non-LLM hot paths.

LLM calls are mocked — these benchmarks measure Python overhead only:
prompt construction, chunking, merging, DB ops, file writes.

Run:
    uv run pytest tests/test_benchmarks.py --benchmark-only -v
    uv run pytest tests/test_benchmarks.py --benchmark-compare  # compare to stored baseline

Store baseline:
    uv run pytest tests/test_benchmarks.py --benchmark-save=baseline
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import AnalysisResult
from obsidian_llm_wiki.pipeline.ingest import (
    _analyze_body,
    _build_analysis_prompt,
    _merge_chunk_results,
    ingest_note,
)
from obsidian_llm_wiki.state import StateDB

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".olw").mkdir()
    return tmp_path


@pytest.fixture
def config(vault):
    return Config(vault=vault)


@pytest.fixture
def db(config):
    return StateDB(config.state_db_path)


def _analysis_json(n_concepts: int = 4) -> str:
    return json.dumps(
        {
            "summary": "A concise two-sentence summary of the note content.",
            "key_concepts": [f"Concept {i}" for i in range(n_concepts)],
            "suggested_topics": ["Topic A", "Topic B"],
            "quality": "high",
        }
    )


def _mock_client(n_concepts: int = 4) -> MagicMock:
    c = MagicMock()
    c.generate.return_value = _analysis_json(n_concepts)
    return c


def _make_result(n: int = 4) -> AnalysisResult:
    return AnalysisResult(
        summary="Summary.",
        key_concepts=[f"Concept {i}" for i in range(n)],
        suggested_topics=["Topic"],
        quality="high",
    )


# ── Prompt construction ───────────────────────────────────────────────────────


def test_bench_prompt_build_short(benchmark):
    body = "word " * 200  # ~1K chars
    concepts = [f"Concept {i}" for i in range(30)]
    benchmark(_build_analysis_prompt, body, concepts, "note.md")


def test_bench_prompt_build_chunk_size(benchmark, config):
    """Prompt build at exactly fast_ctx // 2 chars (single-chunk boundary)."""
    body = "x " * (config.ollama.fast_ctx // 4)  # chars, not tokens
    concepts = [f"Concept {i}" for i in range(30)]
    benchmark(_build_analysis_prompt, body, concepts, "note.md")


# ── Chunk splitting ───────────────────────────────────────────────────────────


def test_bench_chunk_split_25k(benchmark, config):
    """Split a 25K note into chunks of fast_ctx // 2."""
    body = "word " * 5000  # ~25K chars
    chunk_size = config.ollama.fast_ctx // 2

    def split():
        return [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]

    benchmark(split)


def test_bench_chunk_split_100k(benchmark, config):
    body = "word " * 20000  # ~100K chars
    chunk_size = config.ollama.fast_ctx // 2

    def split():
        return [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]

    benchmark(split)


# ── Merge results ─────────────────────────────────────────────────────────────


def test_bench_merge_2_chunks(benchmark):
    results = [_make_result(4), _make_result(4)]
    benchmark(_merge_chunk_results, results)


def test_bench_merge_8_chunks(benchmark):
    results = [_make_result(8) for _ in range(8)]
    benchmark(_merge_chunk_results, results)


def test_bench_merge_20_chunks(benchmark):
    """Stress: 20 chunks × 8 concepts = up to 160 candidates to deduplicate."""
    results = [_make_result(8) for _ in range(20)]
    benchmark(_merge_chunk_results, results)


# ── Analyze body (mocked LLM) ─────────────────────────────────────────────────


def test_bench_analyze_body_single_chunk(benchmark, config):
    """Short note — single LLM call, no chunking overhead."""
    body = "word " * 200
    client = _mock_client()
    benchmark(_analyze_body, body, [], "note.md", client, config)


def test_bench_analyze_body_multi_chunk(benchmark, vault):
    """Long note — 4 chunks, sequential. Measures chunking + merge overhead."""
    config = Config(vault=vault, ollama={"fast_ctx": 400})  # small ctx → 4 chunks for 400 chars
    body = "word " * 80  # ~400 chars
    client = _mock_client()
    benchmark(_analyze_body, body, [], "note.md", client, config)


# ── DB operations ─────────────────────────────────────────────────────────────


def test_bench_db_upsert_concepts(benchmark, db):
    concepts = [f"Concept {i}" for i in range(8)]
    i = 0

    def upsert():
        nonlocal i
        db.upsert_concepts(f"raw/note{i}.md", concepts)
        i += 1

    benchmark(upsert)


def test_bench_db_concepts_needing_compile(benchmark, db):
    # Seed 50 ingested notes with concepts
    from obsidian_llm_wiki.models import RawNoteRecord

    for i in range(50):
        rec = RawNoteRecord(path=f"raw/note{i}.md", content_hash=f"h{i}", status="ingested")
        db.upsert_raw(rec)
        db.upsert_concepts(f"raw/note{i}.md", [f"Concept {i}", "Shared Concept"])
    benchmark(db.concepts_needing_compile)


def test_bench_db_get_sources_for_concept(benchmark, db):
    from obsidian_llm_wiki.models import RawNoteRecord

    for i in range(20):
        db.upsert_raw(
            RawNoteRecord(path=f"raw/note{i}.md", content_hash=f"h{i}", status="ingested")
        )
        db.upsert_concepts(f"raw/note{i}.md", ["Shared Concept"])
    benchmark(db.get_sources_for_concept, "Shared Concept")


# ── Full ingest_note (mocked LLM, real DB + file I/O) ────────────────────────


def test_bench_ingest_note(benchmark, vault, config, db):
    path = vault / "raw" / "bench.md"
    path.write_text("# Benchmark Note\n\n" + "Content word. " * 200, encoding="utf-8")
    client = _mock_client()

    def run():
        db_fresh = StateDB(config.state_db_path)
        ingest_note(path, config, client, db_fresh, force=True)

    benchmark(run)
