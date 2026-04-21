"""Tests for advisor-first compare reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from obsidian_llm_wiki.compare.metrics import load_queries
from obsidian_llm_wiki.compare.models import (
    AdvisorVerdict,
    CompareReport,
    ContestantRunResult,
    PageDiffSummary,
)
from obsidian_llm_wiki.compare.report import (
    render_json,
    render_markdown,
    render_summary_json,
    resolve,
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
            diagnostics={"lint_health": 90.0, "link_health": 0.8},
            wall_time_seconds=10.0,
        ),
        challenger=ContestantRunResult(
            role="challenger",
            fast_model="f2",
            heavy_model="h2",
            provider_name="ollama",
            provider_url="http://localhost:11434",
            diagnostics={"lint_health": 95.0, "link_health": 0.9},
            wall_time_seconds=12.0,
        ),
        page_diff=PageDiffSummary(changed=["A"], added=["B"], removed=[]),
    )


def test_resolve_populates_verdict_and_reasons():
    report = _report()
    resolve(report)
    assert report.verdict in {
        AdvisorVerdict.SWITCH,
        AdvisorVerdict.KEEP_CURRENT,
        AdvisorVerdict.MANUAL_REVIEW,
    }
    assert report.reasons


def test_render_markdown_has_expected_sections():
    report = _report()
    resolve(report)
    md = render_markdown(report)
    for section in (
        "# olw compare",
        "## Recommendation",
        "## Next Steps",
        "## Config Change",
        "## Query Summary",
        "## Vault Impact",
        "## Structure And Reliability",
        "## Representative Page Changes",
        "## Operational Cost",
        "## Caveats",
    ):
        assert section in md


def test_render_json_round_trips():
    report = _report()
    resolve(report)
    data = json.loads(render_json(report))
    assert data["run_id"] == "rid"


def test_render_summary_json_round_trips():
    report = _report()
    resolve(report)
    data = json.loads(render_summary_json(report))
    assert data["run_id"] == "rid"
    assert "verdict" in data


def test_render_markdown_switch_includes_provider_config():
    report = _report()
    report.verdict = AdvisorVerdict.SWITCH
    report.reasons = ["challenger preview is stronger"]
    report.challenger.provider_name = "groq"
    report.challenger.provider_url = "https://api.groq.com/openai/v1"

    md = render_markdown(report)

    assert "[provider]" in md
    assert 'name = "groq"' in md
    assert 'url = "https://api.groq.com/openai/v1"' in md


def test_load_queries_rejects_duplicate_ids(tmp_path: Path):
    queries = tmp_path / "queries.toml"
    queries.write_text(
        '[[query]]\nid = "q1"\nquestion = "one"\n\n[[query]]\nid = "q1"\nquestion = "two"\n'
    )

    with pytest.raises(ValueError, match="Duplicate query id: q1"):
        load_queries(queries)
