"""Recommendation-first rendering for compare MVP."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum

from ..config import _toml_quote
from .metrics import build_reasons, compute_advisor_metrics, decide_verdict
from .models import AdvisorVerdict, CompareReport


def resolve(report: CompareReport) -> None:
    compute_advisor_metrics(report)
    decide_verdict(report)
    build_reasons(report)


def render_markdown(report: CompareReport) -> str:
    out: list[str] = []
    out.append("# olw compare — vault switch advisor")
    out.append("")
    out.append(
        "This comparison ran in isolated preview vaults. "
        "Your active raw/ and wiki/ were not modified."
    )
    out.append("")
    out.append("## Recommendation")
    out.append("")
    out.append(f"Verdict: **{report.verdict.value}**")
    out.append("")
    out.append("Reasons:")
    for reason in report.reasons:
        out.append(f"- {reason}")
    out.append("")
    out.append("## Next Steps")
    out.append("")
    if report.verdict == AdvisorVerdict.SWITCH:
        switch_toml = render_switch_config_toml(
            fast_model=report.challenger.fast_model,
            heavy_model=report.challenger.heavy_model,
            provider_name=report.challenger.provider_name,
            provider_url=report.challenger.provider_url,
        )
        out.append(
            "Edit `wiki.toml` in your vault and set the challenger config:\n"
            f"```toml\n{switch_toml}\n```"
        )
    elif report.verdict == AdvisorVerdict.KEEP_CURRENT:
        out.append("No change needed. Your current config performed better.")
    else:
        out.append(f"Review page diffs in `{report.out_dir}/diffs/` to decide manually.")
    out.append("")
    out.append("## Config Change")
    out.append("")
    out.append(
        f"Current: `{report.current.fast_model}` / `{report.current.heavy_model}` "
        f"via `{report.current.provider_name}`"
    )
    out.append(
        f"Challenger: `{report.challenger.fast_model}` / `{report.challenger.heavy_model}` "
        f"via `{report.challenger.provider_name}`"
    )
    out.append("")
    out.append("## Query Summary")
    out.append("")
    if report.query_diffs:
        for q in report.query_diffs:
            delta = "n/a" if q.delta is None else f"{q.delta:+.2f}"
            out.append(f"- `{q.id}`: delta {delta}")
    else:
        out.append("- No explicit compare queries were provided.")
    out.append("")
    out.append("## Vault Impact")
    out.append("")
    out.append(f"- Pages changed: {len(report.page_diff.changed)}")
    out.append(f"- Pages added: {len(report.page_diff.added)}")
    out.append(f"- Pages removed: {len(report.page_diff.removed)}")
    out.append("")
    out.append("## Structure And Reliability")
    out.append("")
    out.append(f"- Current lint health: {report.current.diagnostics.get('lint_health', 'n/a')}")
    out.append(
        f"- Challenger lint health: {report.challenger.diagnostics.get('lint_health', 'n/a')}"
    )
    out.append(f"- Current link health: {_fmt(report.current.diagnostics.get('link_health'))}")
    out.append(
        f"- Challenger link health: {_fmt(report.challenger.diagnostics.get('link_health'))}"
    )
    out.append("")
    out.append("## Representative Page Changes")
    out.append("")
    out.extend([f"- Changed: {p}" for p in report.page_diff.changed[:10]])
    out.extend([f"- Added: {p}" for p in report.page_diff.added[:10]])
    out.extend([f"- Removed: {p}" for p in report.page_diff.removed[:10]])
    if not report.page_diff.changed and not report.page_diff.added and not report.page_diff.removed:
        out.append("- No material page changes detected.")
    out.append("")
    out.append("## Operational Cost")
    out.append("")
    out.append(f"- Current wall time: {report.current.wall_time_seconds:.1f}s")
    out.append(f"- Challenger wall time: {report.challenger.wall_time_seconds:.1f}s")
    out.append("")
    out.append("## Caveats")
    out.append("")
    out.append(
        "- This is a preview of automated generated output, not a final curated vault outcome."
    )
    out.append(
        "- Output quality depends on the LLM used;"
        " spot-check key pages before committing to a switch."
    )
    return "\n".join(out) + "\n"


def render_json(report: CompareReport) -> str:
    return json.dumps(_jsonable_report(report), indent=2)


def render_summary_json(report: CompareReport) -> str:
    data = {
        "run_id": report.run_id,
        "verdict": report.verdict.value,
        "reasons": report.reasons,
        "vault_path": report.vault_path,
    }
    return json.dumps(data, indent=2)


def render_switch_config_toml(
    fast_model: str,
    heavy_model: str,
    provider_name: str,
    provider_url: str,
) -> str:
    lines = [
        "[models]",
        f"fast = {_toml_quote(fast_model)}",
        f"heavy = {_toml_quote(heavy_model)}",
        "",
    ]
    if provider_name == "ollama":
        lines.extend(
            [
                "[ollama]",
                f"url = {_toml_quote(provider_url)}",
            ]
        )
    else:
        lines.extend(
            [
                "[provider]",
                f"name = {_toml_quote(provider_name)}",
                f"url = {_toml_quote(provider_url)}",
            ]
        )
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _jsonable_report(report: CompareReport) -> dict:
    return _jsonable(asdict(report))


def _jsonable(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value
