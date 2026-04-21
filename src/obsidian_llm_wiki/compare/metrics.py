"""Pairwise metrics and verdict logic for compare MVP."""

from __future__ import annotations

import tomllib
from pathlib import Path

from .models import AdvisorVerdict, CompareReport, QueryResult, QuerySpec


def load_queries(path: Path) -> list[QuerySpec]:
    data = tomllib.loads(path.read_text())
    out: list[QuerySpec] = []
    seen_ids: set[str] = set()
    entries = data.get("query", [])
    if isinstance(entries, dict):
        entries = [entries]
    for entry in entries:
        qid = entry["id"]
        if qid in seen_ids:
            raise ValueError(f"Duplicate query id: {qid}")
        seen_ids.add(qid)
        out.append(
            QuerySpec(
                id=qid,
                question=entry["question"],
                expected_pages=list(entry.get("expected_pages", [])),
                expected_contains=list(entry.get("expected_contains", [])),
                expected_refusal=bool(entry.get("expected_refusal", False)),
            )
        )
    return out


def score_query_result(result: QueryResult, spec: QuerySpec) -> float | None:
    if result.error:
        return 0.0
    if spec.expected_refusal:
        answer = result.answer.lower()
        refused = not result.pages or any(
            s in answer
            for s in (
                "not in wiki",
                "not found",
                "no information",
                "cannot find",
                "don't have",
                "do not have",
            )
        )
        return 1.0 if refused else 0.0

    scores: list[float] = []
    if spec.expected_pages:
        expected = {p.lower() for p in spec.expected_pages}
        actual = {p.lower() for p in result.pages}
        if expected or actual:
            tp = len(expected & actual)
            precision = tp / len(actual) if actual else 0.0
            recall = tp / len(expected) if expected else 0.0
            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append(2 * precision * recall / (precision + recall))
    if spec.expected_contains:
        answer = result.answer.lower()
        hits = sum(1 for phrase in spec.expected_contains if phrase.lower() in answer)
        scores.append(hits / len(spec.expected_contains))
    if not scores:
        return None
    return sum(scores) / len(scores)


def compute_advisor_metrics(report: CompareReport) -> None:
    current = report.current
    challenger = report.challenger
    cur_diag = current.diagnostics
    ch_diag = challenger.diagnostics
    report.current.diagnostics["link_health"] = _link_health(cur_diag)
    report.challenger.diagnostics["link_health"] = _link_health(ch_diag)
    report.current.diagnostics["orphan_rate"] = _orphan_rate(cur_diag)
    report.challenger.diagnostics["orphan_rate"] = _orphan_rate(ch_diag)
    report.current.diagnostics["lint_health_norm"] = _lint_health(cur_diag)
    report.challenger.diagnostics["lint_health_norm"] = _lint_health(ch_diag)


def decide_verdict(report: CompareReport) -> None:
    current = report.current
    challenger = report.challenger
    cur_diag = current.diagnostics
    ch_diag = challenger.diagnostics
    query_deltas = [q.delta for q in report.query_diffs if q.delta is not None]
    avg_query_delta = _composite(query_deltas)

    if challenger.partial and not current.partial:
        report.verdict = AdvisorVerdict.KEEP_CURRENT
        return
    # Query checks are the strongest signal. A single large regression or a
    # sustained negative average should block a switch recommendation.
    if any(delta < -0.10 for delta in query_deltas) or (
        avg_query_delta is not None and avg_query_delta <= -0.05
    ):
        report.verdict = AdvisorVerdict.KEEP_CURRENT
        return

    link_delta = _delta(ch_diag.get("link_health"), cur_diag.get("link_health"))
    orphan_delta = _delta(ch_diag.get("orphan_rate"), cur_diag.get("orphan_rate"))
    lint_delta = _delta(ch_diag.get("lint_health_norm"), cur_diag.get("lint_health_norm"))
    if (
        (link_delta is not None and link_delta < -0.05)
        or (orphan_delta is not None and orphan_delta < -0.05)
        or (lint_delta is not None and lint_delta < -0.05)
    ):
        report.verdict = AdvisorVerdict.KEEP_CURRENT
        return

    if not report.query_diffs:
        report.verdict = AdvisorVerdict.MANUAL_REVIEW
        return

    if avg_query_delta is not None and avg_query_delta > 0.10:
        report.verdict = AdvisorVerdict.SWITCH
        return
    if report.page_diff.changed or report.page_diff.added or report.page_diff.removed:
        if any(delta > 0 for delta in query_deltas) or any(
            d is not None and d > 0 for d in (link_delta, orphan_delta, lint_delta)
        ):
            report.verdict = AdvisorVerdict.SWITCH
            return
    report.verdict = AdvisorVerdict.MANUAL_REVIEW


def build_reasons(report: CompareReport) -> None:
    reasons: list[str] = []
    current = report.current
    challenger = report.challenger
    cur_diag = current.diagnostics
    ch_diag = challenger.diagnostics

    if challenger.partial and not current.partial:
        reasons.append("challenger preview had partial failures while current completed cleanly")

    improved_queries = [q.id for q in report.query_diffs if q.delta is not None and q.delta > 0.10]
    regressed_queries = [
        q.id for q in report.query_diffs if q.delta is not None and q.delta < -0.10
    ]
    if improved_queries:
        reasons.append(f"query results improved on {len(improved_queries)} check(s)")
    if regressed_queries:
        reasons.append(f"query results regressed on {len(regressed_queries)} check(s)")

    link_delta = _delta(ch_diag.get("link_health"), cur_diag.get("link_health"))
    if link_delta is not None and link_delta > 0.05:
        reasons.append("link health improved")
    elif link_delta is not None and link_delta < -0.05:
        reasons.append("link health worsened")

    lint_delta = _delta(ch_diag.get("lint_health_norm"), cur_diag.get("lint_health_norm"))
    if lint_delta is not None and lint_delta > 0.05:
        reasons.append("lint health improved")
    elif lint_delta is not None and lint_delta < -0.05:
        reasons.append("lint health worsened")

    changed = len(report.page_diff.changed)
    if changed:
        reasons.append(f"{changed} page(s) changed materially")
    if not report.query_diffs:
        orphan_delta2 = _delta(ch_diag.get("orphan_rate"), cur_diag.get("orphan_rate"))
        structure_composite = _composite([lint_delta, orphan_delta2, link_delta])
        if structure_composite is not None and structure_composite >= 0.05:
            reasons.append(
                "structure metrics improved, but no explicit compare queries were provided"
            )
        else:
            reasons.append("no explicit compare queries were provided")

    report.reasons = reasons[:4]
    if not report.reasons:
        report.reasons = ["no strong signal was found; inspect page diffs manually"]


def _delta(challenger, current) -> float | None:
    if challenger is None or current is None:
        return None
    return challenger - current


def _composite(deltas: list[float | None]) -> float | None:
    values = [d for d in deltas if d is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _link_health(diag: dict) -> float | None:
    links = diag.get("total_wikilinks", 0) or 0
    if links == 0:
        return None
    broken = (diag.get("issue_counts") or {}).get("broken_link", 0)
    return max(0.0, 1.0 - broken / links)


def _orphan_rate(diag: dict) -> float | None:
    pages = diag.get("total_pages", 0) or 0
    if pages == 0:
        return None
    orphans = (diag.get("issue_counts") or {}).get("orphan", 0)
    return max(0.0, 1.0 - orphans / pages)


def _lint_health(diag: dict) -> float | None:
    h = diag.get("lint_health")
    if h is None:
        return None
    return max(0.0, min(1.0, h / 100.0))
