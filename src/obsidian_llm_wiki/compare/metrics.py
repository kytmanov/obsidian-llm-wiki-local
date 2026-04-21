"""
Scoring for olw compare.

Every dim function takes single-seed payload data and returns a float in
[0, 1] or None when the dim is not applicable (e.g. query dims in BYO
mode with no queries, precision when no concepts extracted).

Aggregation: per-contestant mean ± stdev across seeds + corpus items.
Confidence = HIGH if stdev < 0.1, else LOW. None values are dropped
before aggregation; a dim with zero applicable seeds gets note="n/a".

Consistency is the odd one: it is derived from the stdev of every
*other* dim, not from any per-seed value directly.
"""

from __future__ import annotations

from statistics import median, stdev

from .corpus import Corpus
from .models import CompareReport, ContestantResult, DimScore

# Max retries used by structured_output.request_structured
_MAX_RETRIES = 3
# Wikilink density target range (per 1000 words) — plateau + linear falloff
_WL_RANGE = (5.0, 15.0)
_WL_FALLOFF = 10.0  # density units beyond range where score → 0
# Refusal signal phrases
_REFUSAL_PHRASES = (
    "not in wiki",
    "not found",
    "no information",
    "cannot find",
    "don't have",
    "do not have",
    "does not mention",
)


# ── Public entry point ────────────────────────────────────────────────────────


def score_report(report: CompareReport, corpus: Corpus) -> tuple[float | None, float | None]:
    """Populate `scores` on every ContestantResult in place.

    Returns the cross-contestant reference values (ref_wall, ref_cpc) so
    the report layer can recompute per-seed overalls with the same scale.
    """
    ref_wall = _median_positive(
        r.seed_wall_seconds.get(s, 0.0)
        for r in report.contestants
        for s in r.seed_wall_seconds
    )
    cpc_samples: list[float] = []
    for r in report.contestants:
        for s, pr in r.seed_pipeline_reports.items():
            if pr is None:
                continue
            compiled = pr.get("compiled", 0) or 0
            chars = r.seed_diagnostics.get(s, {}).get("total_chars", 0) or 0
            if compiled > 0:
                cpc_samples.append(chars / compiled)
    ref_cpc = median(cpc_samples) if cpc_samples else None

    for r in report.contestants:
        _score_contestant(r, corpus, ref_wall=ref_wall, ref_cpc=ref_cpc)
    return ref_wall, ref_cpc


# ── Per-contestant aggregation ────────────────────────────────────────────────


def _score_contestant(
    r: ContestantResult,
    corpus: Corpus,
    *,
    ref_wall: float | None,
    ref_cpc: float | None,
) -> None:
    seeds = sorted(r.seed_events.keys())
    per_seed: dict[str, list[float]] = {}

    for s in seeds:
        vals = per_seed_scores(r, s, corpus, ref_wall=ref_wall, ref_cpc=ref_cpc)
        for key, v in vals.items():
            per_seed.setdefault(key, []).append(v)

    for key, values in per_seed.items():
        r.scores[key] = _aggregate(values)

    # Consistency: mean (1 - stdev) across all scored dims with ≥2 samples.
    variances: list[float] = []
    for key, values in per_seed.items():
        clean = [v for v in values if v is not None]
        if len(clean) >= 2:
            variances.append(stdev(clean))
    if variances:
        mean_sd = sum(variances) / len(variances)
        score = max(0.0, 1.0 - mean_sd)
        r.scores["consistency.seed_variance"] = DimScore(
            mean=score, stdev=0.0, n=len(variances), confidence="HIGH"
        )
    else:
        r.scores["consistency.seed_variance"] = DimScore(
            mean=0.0, n=0, confidence="LOW", note="n/a (single seed)"
        )


def per_seed_scores(
    r: ContestantResult,
    s: int,
    corpus: Corpus,
    *,
    ref_wall: float | None,
    ref_cpc: float | None,
) -> dict[str, float | None]:
    events = r.seed_events.get(s, [])
    diag = r.seed_diagnostics.get(s, {})
    pr = r.seed_pipeline_reports.get(s)
    queries = r.seed_queries.get(s, [])
    wall = r.seed_wall_seconds.get(s, 0.0)

    cpc = _chars_per_concept(pr, diag)

    return {
        "reliability.final_success_rate": _final_success_rate(events),
        "reliability.retry_rate": _retry_rate(events),
        "reliability.pipeline_failure_rate": _pipeline_failure_rate(pr),
        "efficiency.wall_time": _rel_smaller_is_better(wall, ref_wall),
        "efficiency.chars_per_concept": _rel_smaller_is_better(cpc, ref_cpc),
        "structure.concept_coverage": _concept_coverage(diag, corpus),
        "structure.concept_precision": _concept_precision(diag, corpus),
        "structure.tag_validity": _tag_validity(diag),
        "structure.wikilink_density": _wikilink_density(diag),
        "structure.orphan_rate": _orphan_rate(diag),
        "structure.broken_link_rate": _broken_link_rate(diag),
        "structure.lint_health": _lint_health(diag),
        "fidelity.source_overlap": diag.get("fidelity_source_overlap"),
        "query.page_selection_f1": _query_page_f1(queries, corpus),
        "query.answer_containment": _query_answer_containment(queries, corpus),
        "query.refusal_correctness": _query_refusal(queries, corpus),
    }


def _aggregate(values: list[float | None]) -> DimScore:
    clean = [v for v in values if v is not None]
    if not clean:
        return DimScore(mean=0.0, n=0, confidence="LOW", note="n/a")
    m = sum(clean) / len(clean)
    sd = stdev(clean) if len(clean) > 1 else 0.0
    conf = "HIGH" if sd < 0.1 else "LOW"
    return DimScore(mean=m, stdev=sd, n=len(clean), confidence=conf)


# ── Reliability ───────────────────────────────────────────────────────────────


def _final_success_rate(events: list[dict]) -> float | None:
    if not events:
        return None
    succeeded = sum(1 for ev in events if ev.get("tier", -1) in (1, 2, 3))
    return succeeded / len(events)


def _retry_rate(events: list[dict]) -> float | None:
    if not events:
        return None
    avg_retries = sum(ev.get("retries", 0) for ev in events) / len(events)
    return max(0.0, 1.0 - avg_retries / _MAX_RETRIES)


def _pipeline_failure_rate(pr: dict | None) -> float | None:
    if pr is None:
        return None
    failed = len(pr.get("failed", []) or [])
    compiled = pr.get("compiled", 0) or 0
    attempted = failed + compiled
    if attempted == 0:
        return None
    return 1.0 - failed / attempted


# ── Efficiency ────────────────────────────────────────────────────────────────


def _chars_per_concept(pr: dict | None, diag: dict) -> float | None:
    if pr is None:
        return None
    compiled = pr.get("compiled", 0) or 0
    chars = diag.get("total_chars", 0) or 0
    if compiled <= 0:
        return None
    return chars / compiled


def _rel_smaller_is_better(actual: float | None, ref: float | None) -> float | None:
    if actual is None or ref is None or actual <= 0 or ref <= 0:
        return None
    return min(1.0, ref / actual)


# ── Structure ─────────────────────────────────────────────────────────────────


def _known_concepts(corpus: Corpus) -> set[str]:
    out: set[str] = set()
    for n in corpus.notes:
        for kc in n.known_concepts:
            out.add(kc.lower())
    return out


def _concept_coverage(diag: dict, corpus: Corpus) -> float | None:
    known = _known_concepts(corpus)
    if not known:
        return None  # BYO / baseline — no ground truth
    extracted = set(diag.get("extracted_concepts", []) or [])
    return len(extracted & known) / len(known)


def _concept_precision(diag: dict, corpus: Corpus) -> float | None:
    known = _known_concepts(corpus)
    if not known:
        return None
    extracted = set(diag.get("extracted_concepts", []) or [])
    if not extracted:
        return None
    # Only penalize when contestant over-extracts dramatically — mild surplus
    # is free.
    if len(extracted) <= 2 * len(known):
        return 1.0
    return len(extracted & known) / len(extracted)


def _tag_validity(diag: dict) -> float | None:
    total = diag.get("total_tags", 0) or 0
    if total == 0:
        return None
    invalid = (diag.get("issue_counts") or {}).get("invalid_tag", 0)
    return max(0.0, 1.0 - invalid / total)


def _wikilink_density(diag: dict) -> float | None:
    words = diag.get("total_words", 0) or 0
    links = diag.get("total_wikilinks", 0) or 0
    if words < 100:
        return None  # too little content to judge density
    density = links / (words / 1000.0)
    lo, hi = _WL_RANGE
    if lo <= density <= hi:
        return 1.0
    gap = lo - density if density < lo else density - hi
    return max(0.0, 1.0 - gap / _WL_FALLOFF)


def _orphan_rate(diag: dict) -> float | None:
    pages = diag.get("total_pages", 0) or 0
    if pages == 0:
        return None
    orphans = (diag.get("issue_counts") or {}).get("orphan", 0)
    return max(0.0, 1.0 - orphans / pages)


def _broken_link_rate(diag: dict) -> float | None:
    links = diag.get("total_wikilinks", 0) or 0
    if links == 0:
        return None
    broken = (diag.get("issue_counts") or {}).get("broken_link", 0)
    return max(0.0, 1.0 - broken / links)


def _lint_health(diag: dict) -> float | None:
    h = diag.get("lint_health")
    if h is None:
        return None
    return max(0.0, min(1.0, h / 100.0))


# ── Query ─────────────────────────────────────────────────────────────────────


def _query_page_f1(queries: list[dict], corpus: Corpus) -> float | None:
    if not corpus.queries:
        return None
    expected_by_id = {q.id: q for q in corpus.queries}
    f1s: list[float] = []
    for q in queries:
        spec = expected_by_id.get(q.get("id", ""))
        if spec is None or spec.expected_refusal:
            continue
        expected = {p.lower() for p in spec.expected_pages}
        selected = {p.lower() for p in (q.get("pages") or [])}
        if not expected and not selected:
            continue
        if not expected or not selected:
            f1s.append(0.0)
            continue
        tp = len(expected & selected)
        precision = tp / len(selected) if selected else 0.0
        recall = tp / len(expected) if expected else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    if not f1s:
        return None
    return sum(f1s) / len(f1s)


def _query_answer_containment(queries: list[dict], corpus: Corpus) -> float | None:
    if not corpus.queries:
        return None
    expected_by_id = {q.id: q for q in corpus.queries}
    rates: list[float] = []
    for q in queries:
        spec = expected_by_id.get(q.get("id", ""))
        if spec is None or spec.expected_refusal or not spec.expected_contains:
            continue
        answer = (q.get("answer") or "").lower()
        hits = sum(1 for phrase in spec.expected_contains if phrase.lower() in answer)
        rates.append(hits / len(spec.expected_contains))
    if not rates:
        return None
    return sum(rates) / len(rates)


def _query_refusal(queries: list[dict], corpus: Corpus) -> float | None:
    if not corpus.queries:
        return None
    expected_by_id = {q.id: q for q in corpus.queries}
    refusal_specs = [q for q in corpus.queries if q.expected_refusal]
    if not refusal_specs:
        return None
    hits: list[int] = []
    for q in queries:
        spec = expected_by_id.get(q.get("id", ""))
        if spec is None or not spec.expected_refusal:
            continue
        pages = q.get("pages") or []
        answer = (q.get("answer") or "").lower()
        refused = not pages or any(p in answer for p in _REFUSAL_PHRASES)
        hits.append(1 if refused else 0)
    if not hits:
        return None
    return sum(hits) / len(hits)


# ── Small utilities ───────────────────────────────────────────────────────────


def _median_positive(vals) -> float | None:
    clean = [v for v in vals if v and v > 0]
    if not clean:
        return None
    return median(clean)
