"""Data models for the compare switch-advisor MVP."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AdvisorVerdict(StrEnum):
    SWITCH = "switch"
    KEEP_CURRENT = "keep_current"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class QuerySpec:
    id: str
    question: str
    expected_pages: list[str] = field(default_factory=list)
    expected_contains: list[str] = field(default_factory=list)
    expected_refusal: bool = False


@dataclass
class QueryResult:
    id: str
    answer: str
    pages: list[str]
    error: str | None = None


@dataclass
class QueryDiff:
    id: str
    question: str
    current_pages: list[str]
    challenger_pages: list[str]
    current_answer: str
    challenger_answer: str
    current_score: float | None = None
    challenger_score: float | None = None
    delta: float | None = None


@dataclass
class PageSnapshot:
    path: str
    title: str
    content_hash: str
    word_count: int
    wikilinks: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass
class PageDiffSummary:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)


@dataclass
class ContestantRunResult:
    role: str
    fast_model: str
    heavy_model: str
    provider_name: str
    provider_url: str
    partial: bool = False
    pipeline_report: dict | None = None
    queries: list[QueryResult] = field(default_factory=list)
    diagnostics: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    page_snapshots: list[PageSnapshot] = field(default_factory=list)
    artifact_dir: str = ""


@dataclass
class CompareReport:
    run_id: str
    vault_path: str
    out_dir: str
    current_config_summary: dict[str, str]
    challenger_config_summary: dict[str, str]
    current: ContestantRunResult
    challenger: ContestantRunResult
    page_diff: PageDiffSummary = field(default_factory=PageDiffSummary)
    query_diffs: list[QueryDiff] = field(default_factory=list)
    verdict: AdvisorVerdict = AdvisorVerdict.MANUAL_REVIEW
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
