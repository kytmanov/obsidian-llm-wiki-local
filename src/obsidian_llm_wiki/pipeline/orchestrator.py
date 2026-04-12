"""
Pipeline orchestrator — runs the full ingest → compile → lint → approve sequence.

Used by `olw run` and `olw watch`. Handles:
  - Selective compile (only concepts linked to changed sources)
  - Transient-failure retry (one additional round)
  - Optional stub creation after lint
  - Timing instrumentation per step
  - Dry-run mode (no LLM calls, no file writes)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ..config import Config
from ..ollama_client import OllamaClient
from ..state import StateDB

log = logging.getLogger(__name__)


class FailureReason(StrEnum):
    TRANSIENT = "transient"  # timeout, connection reset — retry
    LLM_OUTPUT = "llm_output"  # bad JSON / schema mismatch — structured_output already retried 3×
    MISSING_SOURCES = "missing_sources"  # no readable source files
    UNKNOWN = "unknown"


@dataclass
class FailureRecord:
    concept: str
    reason: FailureReason
    error_msg: str = ""


@dataclass
class PipelineReport:
    ingested: int = 0
    compiled: int = 0
    failed: list[FailureRecord] = field(default_factory=list)
    published: int = 0
    lint_issues: int = 0
    stubs_created: int = 0
    rounds: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    concept_timings: dict[str, float] = field(default_factory=dict)

    @property
    def failed_names(self) -> list[str]:
        return [f.concept for f in self.failed]


class PipelineOrchestrator:
    """
    Orchestrates the full pipeline. Caller is responsible for acquiring the
    pipeline lock before calling run() — this class does NOT lock internally.
    """

    def __init__(self, config: Config, client: OllamaClient, db: StateDB) -> None:
        self.config = config
        self.client = client
        self.db = db

    def run(
        self,
        paths: list[str] | None = None,
        auto_approve: bool = False,
        fix: bool = False,
        max_rounds: int = 2,
        dry_run: bool = False,
    ) -> PipelineReport:
        """
        Run full pipeline: ingest → compile → lint → [stubs] → [approve].

        paths: specific raw note paths to ingest (None = ingest all changed notes)
        auto_approve: publish drafts immediately without manual review
        fix: create stubs for broken wikilinks after lint
        max_rounds: maximum compile rounds (round 2 retries transient failures only)
        dry_run: report what would happen; no LLM calls, no file writes
        """
        from ..git_ops import git_commit
        from ..indexer import append_log, generate_index
        from ..pipeline.compile import approve_drafts
        from ..pipeline.ingest import ingest_note
        from ..pipeline.lint import run_lint
        from ..pipeline.maintain import create_stubs

        config = self.config
        client = self.client
        db = self.db
        report = PipelineReport()

        # ── Round 1: Ingest ────────────────────────────────────────────────────
        t0 = time.monotonic()
        ingested_paths: list[str] = []

        if paths is not None:
            md_paths = [p for p in paths if p.endswith(".md")]
        else:
            md_paths = (
                [str(p) for p in config.raw_dir.rglob("*.md")] if config.raw_dir.exists() else []
            )

        log.info("── Ingest (%d note(s)) ──────────────────────────────────", len(md_paths))
        for raw_path_str in md_paths:
            p = Path(raw_path_str)
            if not p.exists():
                continue
            if dry_run:
                log.info("[dry-run] would ingest: %s", p.name)
                ingested_paths.append(raw_path_str)
                report.ingested += 1
                continue
            try:
                result = ingest_note(path=p, config=config, client=client, db=db)
                if result is not None:
                    report.ingested += 1
                    ingested_paths.append(raw_path_str)
            except Exception as e:
                log.error("Ingest failed for %s: %s", p.name, e)

        report.timings["ingest"] = time.monotonic() - t0

        if not dry_run and report.ingested > 0:
            generate_index(config, db)
            append_log(config, f"run | ingested {report.ingested} note(s)")

        # ── Round 1: Compile ──────────────────────────────────────────────────
        # Normalize ingested paths to vault-relative for DB lookup (watchdog supplies
        # absolute paths; DB stores relative paths like raw/note.md).
        priority_concepts: list[str] | None = None
        if ingested_paths:
            relative_ingested = []
            for p_str in ingested_paths:
                try:
                    relative_ingested.append(str(Path(p_str).relative_to(config.vault)))
                except ValueError:
                    relative_ingested.append(p_str)  # already relative
            priority_concepts = db.get_concepts_for_sources(relative_ingested) or None

        n_concepts = len(priority_concepts) if priority_concepts else "all"
        log.info("── Compile round 1 (%s concept(s)) ─────────────────────────", n_concepts)
        t1 = time.monotonic()
        draft_paths, round1_failed, r1_timings = _run_compile(
            config, client, db, concepts=priority_concepts, dry_run=dry_run
        )
        report.timings["compile_r1"] = time.monotonic() - t1
        report.compiled += len(draft_paths)
        report.failed.extend(round1_failed)
        report.concept_timings.update(r1_timings)
        report.rounds = 1

        # ── Lint ──────────────────────────────────────────────────────────────
        log.info("── Lint ─────────────────────────────────────────────────────")
        if not dry_run:
            lint_result = run_lint(config, db)
            report.lint_issues = len(lint_result.issues)
            broken_links = [i for i in lint_result.issues if i.issue_type == "broken_link"]

            if fix and broken_links:
                stubs = create_stubs(config, db, broken_link_issues=broken_links, max_stubs=3)
                report.stubs_created = len(stubs)

        # ── Round 2: Retry transient failures ─────────────────────────────────
        transient = [f for f in round1_failed if f.reason == FailureReason.TRANSIENT]
        if transient and report.rounds < max_rounds:
            log.info("── Compile round 2 (%d retries) ────────────────────────────", len(transient))
            transient_concepts = [f.concept for f in transient]
            t2 = time.monotonic()
            r2_drafts, r2_failed, r2_timings = _run_compile(
                config, client, db, concepts=transient_concepts, dry_run=dry_run
            )
            report.timings["compile_r2"] = time.monotonic() - t2
            report.compiled += len(r2_drafts)
            draft_paths = draft_paths + r2_drafts
            report.concept_timings.update(r2_timings)
            # Replace transient failures with round-2 results
            report.failed = [f for f in report.failed if f.reason != FailureReason.TRANSIENT]
            report.failed.extend(r2_failed)
            report.rounds = 2

        # ── Approve ────────────────────────────────────────────────────────────
        if auto_approve and draft_paths and not dry_run:
            log.info(
                "── Auto-approve (%d draft(s)) ───────────────────────────────", len(draft_paths)
            )  # noqa: E501
            published = approve_drafts(config, db, draft_paths)
            report.published = len(published)
            generate_index(config, db)
            append_log(config, f"run | {report.published} articles published")

        # ── Commit ─────────────────────────────────────────────────────────────
        if config.pipeline.auto_commit and not dry_run and (report.compiled or report.published):
            msg = f"run: {report.compiled} compiled"
            if report.published:
                msg += f", {report.published} published"
            git_commit(config.vault, msg, paths=["wiki/", ".olw/"])

        return report


def _run_compile(
    config: Config,
    client: OllamaClient,
    db: StateDB,
    concepts: list[str] | None,
    dry_run: bool,
) -> tuple[list[Path], list[FailureRecord], dict[str, float]]:
    """Run compile_concepts and classify failures by reason."""
    from ..ollama_client import OllamaError
    from ..pipeline.compile import compile_concepts

    try:
        draft_paths, failed_names, concept_timings = compile_concepts(
            config=config,
            client=client,
            db=db,
            dry_run=dry_run,
            concepts=concepts,
        )
    except OllamaError as e:
        # Connection-level failure — all concepts are transient
        log.error("Ollama connection error during compile: %s", e)
        all_concepts = concepts or db.concepts_needing_compile()
        return (
            [],
            [
                FailureRecord(concept=c, reason=FailureReason.TRANSIENT, error_msg=str(e))
                for c in all_concepts
            ],
            {},
        )

    # Classify individual concept failures
    # compile_concepts returns bare names — we can't know the exact reason
    # per-concept without changing its return type. Use UNKNOWN for now;
    # transient failures (timeouts) will bubble up as OllamaError above.
    failure_records = [
        FailureRecord(concept=name, reason=FailureReason.UNKNOWN) for name in failed_names
    ]
    return draft_paths, failure_records, concept_timings
