"""Pure metric function tests — synthetic inputs, exact expected outputs."""

from __future__ import annotations

import hashlib

import pytest

from obsidian_llm_wiki.compare.corpus import Corpus, CorpusMode, Note, Query
from obsidian_llm_wiki.compare.metrics import (
    _broken_link_rate,
    _concept_coverage,
    _concept_precision,
    _final_success_rate,
    _lint_health,
    _orphan_rate,
    _pipeline_failure_rate,
    _query_answer_containment,
    _query_page_f1,
    _query_refusal,
    _retry_rate,
    _tag_validity,
    _wikilink_density,
    score_report,
)
from obsidian_llm_wiki.compare.models import CompareReport, ContestantResult, ContestantSpec
from obsidian_llm_wiki.compare.weights import build_flat_weights, compute_overall


def _corpus_with(notes: list[Note], queries: list[Query] | None = None) -> Corpus:
    return Corpus(
        mode=CorpusMode.CURATED,
        version="test",
        language="en",
        description="",
        notes=notes,
        queries=queries or [],
        notes_set_hash=hashlib.sha256(b"x").hexdigest(),
        root=None,  # type: ignore[arg-type]
    )


# ── Reliability ───────────────────────────────────────────────────────────────


def test_final_success_rate_all_success():
    events = [{"tier": 1, "retries": 0}, {"tier": 2, "retries": 1}]
    assert _final_success_rate(events) == 1.0


def test_final_success_rate_mixed():
    events = [{"tier": 1}, {"tier": -1}, {"tier": 3}]
    assert _final_success_rate(events) == pytest.approx(2 / 3)


def test_final_success_rate_empty_is_none():
    assert _final_success_rate([]) is None


def test_retry_rate_no_retries():
    assert _retry_rate([{"tier": 1, "retries": 0}, {"tier": 1, "retries": 0}]) == 1.0


def test_retry_rate_with_retries():
    events = [{"retries": 0}, {"retries": 3}, {"retries": 1}]
    # avg = 4/3; 1 - (4/3)/3 = 1 - 0.444 = 0.555
    assert _retry_rate(events) == pytest.approx(1 - (4 / 3) / 3)


def test_pipeline_failure_rate():
    pr = {"compiled": 8, "failed": [{"concept": "x"}, {"concept": "y"}]}
    assert _pipeline_failure_rate(pr) == pytest.approx(1 - 2 / 10)


def test_pipeline_failure_rate_no_attempts():
    assert _pipeline_failure_rate({"compiled": 0, "failed": []}) is None


def test_pipeline_failure_rate_none():
    assert _pipeline_failure_rate(None) is None


# ── Structure ─────────────────────────────────────────────────────────────────


def _fake_note(kc: list[str]) -> Note:
    # Use __new__ to bypass frozen-check on path existence
    return Note(file="x.md", path=None, known_concepts=kc)  # type: ignore[arg-type]


def test_concept_coverage_perfect():
    corpus = _corpus_with([_fake_note(["a", "b", "c"])])
    diag = {"extracted_concepts": ["a", "b", "c"]}
    assert _concept_coverage(diag, corpus) == 1.0


def test_concept_coverage_partial():
    corpus = _corpus_with([_fake_note(["a", "b", "c", "d"])])
    diag = {"extracted_concepts": ["a", "b"]}
    assert _concept_coverage(diag, corpus) == 0.5


def test_concept_coverage_byo_returns_none():
    corpus = _corpus_with([_fake_note([])])
    assert _concept_coverage({"extracted_concepts": ["x"]}, corpus) is None


def test_concept_precision_mild_surplus_gets_full_credit():
    corpus = _corpus_with([_fake_note(["a", "b"])])
    diag = {"extracted_concepts": ["a", "b", "c"]}  # 3 <= 2*2
    assert _concept_precision(diag, corpus) == 1.0


def test_concept_precision_over_extraction_penalized():
    corpus = _corpus_with([_fake_note(["a", "b"])])
    diag = {"extracted_concepts": ["a", "b", "c", "d", "e"]}  # 5 > 2*2
    assert _concept_precision(diag, corpus) == pytest.approx(2 / 5)


def test_tag_validity():
    diag = {"total_tags": 10, "issue_counts": {"invalid_tag": 2}}
    assert _tag_validity(diag) == 0.8


def test_tag_validity_zero_tags_none():
    assert _tag_validity({"total_tags": 0}) is None


def test_wikilink_density_in_range():
    # 1500 words, 15 links → density = 10/1000w; in [5,15] plateau
    diag = {"total_words": 1500, "total_wikilinks": 15}
    assert _wikilink_density(diag) == 1.0


def test_wikilink_density_too_low():
    # 1000 words, 1 link → density=1.0; gap from 5 = 4; 1 - 4/10 = 0.6
    diag = {"total_words": 1000, "total_wikilinks": 1}
    assert _wikilink_density(diag) == pytest.approx(0.6)


def test_wikilink_density_too_high():
    # 1000 words, 30 links → density=30; gap from 15 = 15 > falloff(10) → 0
    diag = {"total_words": 1000, "total_wikilinks": 30}
    assert _wikilink_density(diag) == 0.0


def test_wikilink_density_too_few_words():
    assert _wikilink_density({"total_words": 50, "total_wikilinks": 2}) is None


def test_orphan_rate():
    diag = {"total_pages": 10, "issue_counts": {"orphan": 2}}
    assert _orphan_rate(diag) == 0.8


def test_broken_link_rate():
    diag = {"total_wikilinks": 20, "issue_counts": {"broken_link": 5}}
    assert _broken_link_rate(diag) == 0.75


def test_lint_health():
    assert _lint_health({"lint_health": 85.0}) == 0.85
    assert _lint_health({"lint_health": None}) is None


# ── Query ─────────────────────────────────────────────────────────────────────


def test_query_page_f1_perfect():
    corpus = _corpus_with(
        [_fake_note([])],
        [Query(id="q1", question="?", expected_pages=["A", "B"])],
    )
    queries = [{"id": "q1", "pages": ["A", "B"], "answer": ""}]
    assert _query_page_f1(queries, corpus) == 1.0


def test_query_page_f1_partial():
    corpus = _corpus_with(
        [_fake_note([])],
        [Query(id="q1", question="?", expected_pages=["A", "B"])],
    )
    queries = [{"id": "q1", "pages": ["A", "C"], "answer": ""}]
    # TP=1, precision=0.5, recall=0.5, F1=0.5
    assert _query_page_f1(queries, corpus) == 0.5


def test_query_page_f1_refusal_excluded():
    corpus = _corpus_with(
        [_fake_note([])],
        [Query(id="q1", question="?", expected_refusal=True)],
    )
    queries = [{"id": "q1", "pages": [], "answer": ""}]
    assert _query_page_f1(queries, corpus) is None


def test_query_answer_containment():
    corpus = _corpus_with(
        [_fake_note([])],
        [Query(id="q1", question="?", expected_contains=["gradient", "chain rule"])],
    )
    queries = [{"id": "q1", "pages": [], "answer": "It uses gradient descent."}]
    assert _query_answer_containment(queries, corpus) == 0.5


def test_query_refusal_correct():
    corpus = _corpus_with(
        [_fake_note([])],
        [Query(id="q1", question="?", expected_refusal=True)],
    )
    queries = [{"id": "q1", "pages": [], "answer": ""}]
    assert _query_refusal(queries, corpus) == 1.0


def test_query_refusal_hallucinated():
    corpus = _corpus_with(
        [_fake_note([])],
        [Query(id="q1", question="?", expected_refusal=True)],
    )
    queries = [{"id": "q1", "pages": ["Lima"], "answer": "Lima is the capital"}]
    assert _query_refusal(queries, corpus) == 0.0


def test_query_refusal_via_phrase():
    corpus = _corpus_with(
        [_fake_note([])],
        [Query(id="q1", question="?", expected_refusal=True)],
    )
    queries = [{"id": "q1", "pages": ["Something"], "answer": "This is not in wiki."}]
    assert _query_refusal(queries, corpus) == 1.0


# ── Full scoring pipeline ─────────────────────────────────────────────────────


def _full_contestant(
    name: str, *, seeds: int = 2, issue_counts: dict | None = None
) -> ContestantResult:
    r = ContestantResult(spec=ContestantSpec(name=name, fast_model="m", heavy_model="m"))
    for s in range(seeds):
        r.seed_events[s] = [
            {
                "tier": 1,
                "retries": 0,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "num_ctx": 8192,
            }
            for _ in range(10)
        ]
        r.seed_pipeline_reports[s] = {
            "compiled": 5,
            "failed": [],
            "published": 5,
            "timings": {},
            "concept_timings": {},
        }
        r.seed_queries[s] = []
        r.seed_diagnostics[s] = {
            "lint_health": 90.0,
            "issue_counts": issue_counts or {},
            "extracted_concepts": ["a", "b", "c"],
            "total_pages": 5,
            "total_words": 1500,
            "total_wikilinks": 15,
            "total_tags": 10,
            "total_chars": 6000,
            "fidelity_source_overlap": 0.45,
        }
        r.seed_wall_seconds[s] = 20.0
    return r


def test_score_report_end_to_end():
    corpus = _corpus_with([_fake_note(["a", "b", "c"])])
    report = CompareReport(
        run_id="r",
        mode="curated",
        seeds=2,
        olw_version="x",
        pipeline_prompt_hash="h",
        contestants=[_full_contestant("a"), _full_contestant("b")],
    )
    score_report(report, corpus)
    for r in report.contestants:
        # coverage = 1.0, tag_validity = 1.0, fidelity = 0.45
        assert r.scores["structure.concept_coverage"].mean == 1.0
        assert r.scores["fidelity.source_overlap"].mean == pytest.approx(0.45)
        assert r.scores["reliability.final_success_rate"].mean == 1.0
        # Two seeds with identical values → HIGH confidence
        assert r.scores["structure.concept_coverage"].confidence == "HIGH"


def test_score_report_consistency_derives_from_variance():
    corpus = _corpus_with([_fake_note(["a", "b", "c"])])
    # Construct contestant with variable lint_health across seeds
    r = _full_contestant("x")
    r.seed_diagnostics[0]["lint_health"] = 50.0
    r.seed_diagnostics[1]["lint_health"] = 90.0
    report = CompareReport(
        run_id="r", mode="curated", seeds=2, olw_version="", pipeline_prompt_hash="",
        contestants=[r],
    )
    score_report(report, corpus)
    # Variance on lint_health alone shifts consistency below 1.0
    assert r.scores["consistency.seed_variance"].mean < 1.0


def test_compute_overall_redistributes_missing_dims():
    """A dim with n=0 should not count; remaining dims scale up."""
    from obsidian_llm_wiki.compare.models import DimScore

    scores = {
        "reliability.final_success_rate": DimScore(mean=1.0, n=2),
        "reliability.retry_rate": DimScore(mean=0.5, n=2),
        "query.page_selection_f1": DimScore(mean=0.0, n=0, note="n/a"),
    }
    weights = {
        "reliability.final_success_rate": 0.4,
        "reliability.retry_rate": 0.4,
        "query.page_selection_f1": 0.2,
    }
    overall, active = compute_overall(scores, weights)
    # Query dim dropped; reliability dims scale up equally → 0.5*(1.0)+0.5*(0.5)=0.75
    assert overall == pytest.approx(0.75)
    assert "query.page_selection_f1" not in active


def test_weights_sum_to_one_curated():
    w = build_flat_weights(CorpusMode.CURATED)
    assert sum(w.values()) == pytest.approx(1.0)


def test_weights_sum_to_one_baseline():
    w = build_flat_weights(CorpusMode.BASELINE)
    assert sum(w.values()) == pytest.approx(1.0)


def test_weights_nested_override(tmp_path):
    toml = tmp_path / "w.toml"
    toml.write_text(
        "[weights.structure]\n"
        "orphan_rate = 0.5\n"
    )
    w = build_flat_weights(CorpusMode.CURATED, override_path=toml)
    # orphan_rate explicit, remaining structure sub-dims share (0.25 - 0.5) clamped to 0
    # Because 0.5 > 0.25, remaining share is 0; sum still normalizes.
    assert sum(w.values()) == pytest.approx(1.0)
