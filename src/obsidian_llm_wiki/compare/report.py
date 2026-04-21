"""
Report rendering — markdown + JSON.

Two responsibilities:
  • resolve() — take a scored CompareReport + weights, compute overall
    scores, winner, margin_sigma, statistical_tie, and populate
    advisory diagnostics per contestant.
  • render_markdown() / render_json() — deterministic, template-driven
    output. No LLM. Trade-off narrative is a fixed rule: wins = top 2
    positive per-dim deltas vs best competitor, loses = bottom 2.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import stdev

from .corpus import Corpus
from .metrics import per_seed_scores, score_report
from .models import CompareReport, ContestantResult, DimScore
from .pricing import LAST_UPDATED as PRICING_AS_OF
from .pricing import estimate_cost_usd
from .weights import build_flat_weights, compute_overall

# ── Public API ────────────────────────────────────────────────────────────────


def resolve(
    report: CompareReport,
    corpus: Corpus,
    weights_override: Path | None = None,
) -> None:
    """Score, weight, decide winner + tie, populate diagnostics. In place."""
    ref_wall, ref_cpc = score_report(report, corpus)
    weights = build_flat_weights(corpus.mode, override_path=weights_override)
    report.weights = weights

    for r in report.contestants:
        overall, _active = compute_overall(r.scores, weights)
        total_events = sum(len(v) for v in r.seed_events.values())
        r.overall = 0.0 if (r.partial and total_events == 0) else overall
        r.per_seed_overall = _per_seed_overalls(
            r, corpus, weights, ref_wall=ref_wall, ref_cpc=ref_cpc
        )
        r.diagnostics.update(_advisory_diagnostics(r))

    _decide_winner(report)


def _per_seed_overalls(
    r: ContestantResult,
    corpus: Corpus,
    weights: dict[str, float],
    *,
    ref_wall: float | None,
    ref_cpc: float | None,
) -> list[float]:
    """Weighted overall per seed; skipped dims redistribute weight."""
    seeds = sorted(r.seed_events.keys())
    out: list[float] = []
    for s in seeds:
        vals = per_seed_scores(r, s, corpus, ref_wall=ref_wall, ref_cpc=ref_cpc)
        active = {k: w for k, w in weights.items() if vals.get(k) is not None}
        total = sum(active.values())
        if total <= 0:
            continue
        norm = {k: w / total for k, w in active.items()}
        out.append(sum(vals[k] * w for k, w in norm.items()))
    return out


def render_markdown(report: CompareReport, corpus: Corpus) -> str:
    out: list[str] = []
    out.append(f"# olw compare — run {report.run_id}")
    out.append("")
    out.append(
        f"Mode: **{report.mode}** · Seeds: **{report.seeds}** · "
        f"Wall time: {report.wall_time_seconds:.1f}s"
    )
    out.append(
        f"olw_version: `{report.olw_version}` · "
        f"pipeline_prompt_hash: `{report.pipeline_prompt_hash}`"
    )
    if report.notes_set_hash:
        out.append(f"notes_set_hash: `{report.notes_set_hash[:12]}…`")
    out.append("")

    out.append("## Contestants")
    out.append("")
    out.append("| Name | Fast | Heavy | Provider | Seed honored | Partial |")
    out.append("|------|------|-------|----------|--------------|---------|")
    for r in report.contestants:
        provider = r.spec.provider_name or "ollama"
        honored = "✓" if r.seed_honored else "—"
        partial = "✓" if r.partial else ""
        out.append(
            f"| {r.spec.name} | {r.spec.fast_model} | {r.spec.heavy_model} "
            f"| {provider} | {honored} | {partial} |"
        )
    out.append("")

    out.append("## Overall verdict")
    out.append("")
    if report.winner:
        verdict = (
            f"**Winner: {report.winner}** — overall "
            f"{_overall_for(report, report.winner):.3f}"
        )
        if report.margin_sigma is not None:
            verdict += f" · margin {report.margin_sigma:.2f}σ"
        if report.statistical_tie:
            verdict += " · **statistical tie**"
        out.append(verdict)
    else:
        out.append("No winner could be determined (all contestants partial / no scores).")
    for w in report.warnings:
        out.append(f"> ⚠ {w}")
    out.append("")

    out.append("## Per-dimension scores")
    out.append("")
    out.extend(_dim_table(report))
    out.append("")

    out.append("## Advisory diagnostics (not weighted)")
    out.append("")
    out.extend(_diagnostics_table(report))
    out.append("")

    out.append("## Trade-off narrative")
    out.append("")
    out.extend(_narrative(report))
    out.append("")

    out.append("## Blind spots")
    out.append("")
    out.append(
        "- Corpus bias reflects the built-in 10-note English set unless "
        "`--corpus` / `--notes` is supplied."
    )
    out.append("- Tier-1 rate and cost are advisory (provider capability, not model quality).")
    out.append(f"- Cost estimates use pricing snapshot from {PRICING_AS_OF}; cloud rates drift.")
    if report.seeds < 3:
        out.append(
            f"- Only {report.seeds} seed(s) — consistency confidence is low; "
            "use `--seeds 3` for a tighter read."
        )
    out.append("")

    return "\n".join(out)


def render_json(report: CompareReport) -> str:
    data = asdict(report)
    return json.dumps(data, indent=2, default=str)


# ── Advisory diagnostics ──────────────────────────────────────────────────────


def _advisory_diagnostics(r: ContestantResult) -> dict:
    tier1 = 0
    ctx_constrained = 0
    total_prompt = 0
    cost = 0.0
    cost_known = False
    provider = r.spec.provider_name or "ollama"

    for events in r.seed_events.values():
        for ev in events:
            if ev.get("tier") == 1:
                tier1 += 1
            pt = ev.get("prompt_tokens") or 0
            ct = ev.get("completion_tokens") or 0
            total_prompt += pt
            num_ctx = ev.get("num_ctx") or 0
            if num_ctx and pt > num_ctx * 0.9:
                ctx_constrained += 1
            model = ev.get("model", "")
            c = estimate_cost_usd(provider, model, pt, ct)
            if c is not None:
                cost += c
                cost_known = True

    n_events = sum(len(ev) for ev in r.seed_events.values())
    compiled = sum(
        (pr or {}).get("compiled", 0) or 0
        for pr in r.seed_pipeline_reports.values()
    )
    return {
        "tier1_rate": (tier1 / n_events) if n_events else None,
        "prompt_tokens_per_concept": (
            total_prompt / compiled if compiled else None
        ),
        "cost_usd_total": cost if cost_known else None,
        "ctx_constrained_calls": ctx_constrained,
        "total_calls": n_events,
    }


# ── Winner + tie determination ────────────────────────────────────────────────


def _decide_winner(report: CompareReport) -> None:
    scored = [r for r in report.contestants if not _all_na(r.scores)]
    if not scored:
        return
    sorted_by_overall = sorted(scored, key=lambda r: r.overall, reverse=True)
    winner = sorted_by_overall[0]
    report.winner = winner.spec.name

    if len(sorted_by_overall) < 2:
        return
    runner_up = sorted_by_overall[1]
    margin = winner.overall - runner_up.overall

    # σ = seed-to-seed stdev of the winner's overall. Captures run noise;
    # a tight margin vs. noisy winner → tie.
    seed_overalls = winner.per_seed_overall
    sd = stdev(seed_overalls) if len(seed_overalls) > 1 else 0.0
    if sd > 0:
        report.margin_sigma = margin / sd
        report.statistical_tie = report.margin_sigma < 1.0
    else:
        # Zero seed-to-seed variance — treat sub-0.01 margin as tie, else not.
        report.margin_sigma = None
        report.statistical_tie = margin < 0.01


def _all_na(scores: dict[str, DimScore]) -> bool:
    return not scores or all(d.n == 0 for d in scores.values())


def _overall_for(report: CompareReport, name: str) -> float:
    for r in report.contestants:
        if r.spec.name == name:
            return r.overall
    return 0.0


# ── Table + narrative rendering ───────────────────────────────────────────────


def _dim_table(report: CompareReport) -> list[str]:
    dims = _all_dim_keys(report)
    header = ["Dimension"] + [r.spec.name for r in report.contestants]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for dim in dims:
        row = [dim]
        for r in report.contestants:
            d = r.scores.get(dim)
            if d is None or d.n == 0:
                row.append("n/a")
            else:
                row.append(f"{d.mean:.2f}±{d.stdev:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("**Overall:** " + ", ".join(
        f"{r.spec.name}={r.overall:.3f}" for r in report.contestants
    ))
    return lines


def _diagnostics_table(report: CompareReport) -> list[str]:
    cols = ["tier1_rate", "prompt_tokens_per_concept", "cost_usd_total", "ctx_constrained_calls"]
    header = ["Contestant"] + cols
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in report.contestants:
        row = [r.spec.name]
        for col in cols:
            v = r.diagnostics.get(col)
            if v is None:
                row.append("—")
            elif isinstance(v, float):
                row.append(f"{v:.3f}" if col == "cost_usd_total" else f"{v:.2f}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _narrative(report: CompareReport) -> list[str]:
    if len(report.contestants) < 2:
        return ["_Narrative skipped — single contestant._"]
    lines: list[str] = []
    for r in report.contestants:
        others = [o for o in report.contestants if o.spec.name != r.spec.name]
        deltas: list[tuple[str, float]] = []
        for dim, score in r.scores.items():
            if score.n == 0:
                continue
            best_other_mean = max(
                (o.scores[dim].mean for o in others if dim in o.scores and o.scores[dim].n > 0),
                default=None,
            )
            if best_other_mean is None:
                continue
            deltas.append((dim, score.mean - best_other_mean))
        deltas.sort(key=lambda kv: kv[1], reverse=True)
        wins = [d for d, delta in deltas[:2] if delta > 0]
        losses = [d for d, delta in deltas[-2:] if delta < 0]
        pieces: list[str] = []
        if wins:
            pieces.append(f"wins on {', '.join(wins)}")
        if losses:
            pieces.append(f"loses on {', '.join(losses)}")
        if pieces:
            lines.append(f"- **{r.spec.name}** — " + "; ".join(pieces))
        else:
            lines.append(f"- **{r.spec.name}** — no standout dims")
    return lines


def _all_dim_keys(report: CompareReport) -> list[str]:
    seen: dict[str, None] = {}
    for r in report.contestants:
        for k in r.scores.keys():
            seen.setdefault(k, None)
    return list(seen.keys())
