"""Tests for pairwise compare metrics."""

from __future__ import annotations

from obsidian_llm_wiki.compare.metrics import (
    _broken_link_rate,
    _lint_health,
    _orphan_rate,
    build_reasons,
    compute_advisor_metrics,
    decide_verdict,
    score_query_result,
)
from obsidian_llm_wiki.compare.models import (
    AdvisorVerdict,
    CompareReport,
    ContestantRunResult,
    PageDiffSummary,
    QueryDiff,
    QueryResult,
    QuerySpec,
)


def _report() -> CompareReport:
    return CompareReport(
        run_id="rid",
        vault_path="/vault",
        out_dir="/vault/.olw/compare/rid",
        current_config_summary={},
        challenger_config_summary={},
        current=ContestantRunResult(
            role="current",
            fast_model="f1",
            heavy_model="h1",
            provider_name="ollama",
            provider_url="http://localhost:11434",
            diagnostics={
                "lint_health": 90.0,
                "issue_counts": {"broken_link": 2, "orphan": 1},
                "total_wikilinks": 10,
                "total_pages": 5,
            },
        ),
        challenger=ContestantRunResult(
            role="challenger",
            fast_model="f2",
            heavy_model="h2",
            provider_name="ollama",
            provider_url="http://localhost:11434",
            diagnostics={
                "lint_health": 95.0,
                "issue_counts": {"broken_link": 1, "orphan": 0},
                "total_wikilinks": 10,
                "total_pages": 5,
            },
        ),
        page_diff=PageDiffSummary(changed=["A"]),
    )


def test_score_query_result_pages_and_contains():
    spec = QuerySpec(id="q1", question="?", expected_pages=["A"], expected_contains=["alpha"])
    result = QueryResult(id="q1", answer="alpha", pages=["A"])
    assert score_query_result(result, spec) == 1.0


def test_score_query_result_refusal():
    spec = QuerySpec(id="q1", question="?", expected_refusal=True)
    result = QueryResult(id="q1", answer="not in wiki", pages=[])
    assert score_query_result(result, spec) == 1.0


def test_rate_helpers():
    diag = {
        "issue_counts": {"broken_link": 2, "orphan": 1},
        "total_wikilinks": 10,
        "total_pages": 5,
        "lint_health": 85.0,
    }
    assert _broken_link_rate(diag) == 0.8
    assert _orphan_rate(diag) == 0.8
    assert _lint_health(diag) == 0.85


def test_decide_verdict_switch_on_improvement():
    report = _report()
    report.query_diffs = [QueryDiff("q1", "?", [], [], "", "", 0.5, 0.9, 0.4)]
    compute_advisor_metrics(report)
    decide_verdict(report)
    assert report.verdict == AdvisorVerdict.SWITCH


def test_decide_verdict_keep_current_on_regression():
    report = _report()
    report.query_diffs = [QueryDiff("q1", "?", [], [], "", "", 0.9, 0.4, -0.5)]
    compute_advisor_metrics(report)
    decide_verdict(report)
    assert report.verdict == AdvisorVerdict.KEEP_CURRENT


def test_build_reasons_populates():
    report = _report()
    report.query_diffs = [QueryDiff("q1", "?", [], [], "", "", 0.5, 0.9, 0.4)]
    compute_advisor_metrics(report)
    decide_verdict(report)
    build_reasons(report)
    assert report.reasons
