"""
Dataclasses shared across compare/ submodules.

These are internal data shapes (not LLM I/O), so plain dataclasses are
preferred over Pydantic to keep construction cheap and serialization
predictable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContestantSpec:
    """One LLM configuration under test.

    Mirrors the subset of wiki.toml that differs between contestants:
    model selection plus provider override. Unspecified provider/url
    fields mean "inherit from the reference vault config".
    """

    name: str
    fast_model: str
    heavy_model: str
    provider_name: str | None = None
    provider_url: str | None = None
    api_key_env: str | None = None

    def overrides(self) -> dict:
        """Produce the kwargs dict for Config.from_vault()."""
        out: dict = {"models": {"fast": self.fast_model, "heavy": self.heavy_model}}
        provider: dict = {}
        if self.provider_name:
            provider["name"] = self.provider_name
        if self.provider_url:
            provider["url"] = self.provider_url
        if provider:
            out["provider"] = provider
        return out


@dataclass
class DimScore:
    """One dimension's aggregate score for one contestant.

    Mean + stdev captured across seeds × corpus items so the report can
    tag low-confidence scores and compute statistical-tie margins.
    """

    mean: float
    stdev: float = 0.0
    n: int = 0  # samples underlying mean/stdev
    confidence: str = "HIGH"  # HIGH if stdev < 0.1 for all sub-dims, else LOW
    note: str = ""  # e.g. "partial", "seed not honored", "n/a"


@dataclass
class ContestantResult:
    """All outputs from running one contestant across N seeds.

    `scores` keys are dotted paths ("reliability.final_success_rate",
    "structure.concept_coverage", ...) so aggregate + per-sub-dim live
    in the same flat dict and serialization stays trivial.
    """

    spec: ContestantSpec
    scores: dict[str, DimScore] = field(default_factory=dict)
    diagnostics: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    # seed_events: seed index → list of serialized LLMCallEvent dicts
    seed_events: dict[int, list[dict]] = field(default_factory=dict)
    # seed_artifacts: seed index → ephemeral vault path
    seed_artifacts: dict[int, str] = field(default_factory=dict)
    # seed_pipeline_reports: seed → serialized PipelineReport dict (or None on crash)
    seed_pipeline_reports: dict[int, dict | None] = field(default_factory=dict)
    # seed_queries: seed → list of {id, answer, pages, [error]}
    seed_queries: dict[int, list[dict]] = field(default_factory=dict)
    # seed_diagnostics: seed → lint/vault-scan summary (see runner._capture_diagnostics)
    seed_diagnostics: dict[int, dict] = field(default_factory=dict)
    # seed_wall_seconds: seed → pipeline wall time
    seed_wall_seconds: dict[int, float] = field(default_factory=dict)
    # per-seed overall weighted score — drives margin_sigma (seed-to-seed spread)
    per_seed_overall: list[float] = field(default_factory=list)
    seed_honored: bool = True
    partial: bool = False  # true if pipeline broke mid-run
    overall: float = 0.0


@dataclass
class CompareReport:
    """Top-level report structure serialized to report.json + report.md.

    Version pinning (olw_version + pipeline_prompt_hash + corpus_version
    + notes_set_hash + baseline_vault_hash) lives here; the report
    header warns if any cross-run comparison invalidates these pins.
    """

    run_id: str
    mode: str  # "curated" | "byo" | "baseline"
    seeds: int
    olw_version: str
    pipeline_prompt_hash: str
    corpus_version: str = ""
    notes_set_hash: str = ""
    baseline_vault_hash: str = ""
    wall_time_seconds: float = 0.0
    contestants: list[ContestantResult] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    winner: str | None = None
    margin_sigma: float | None = None
    statistical_tie: bool = False
