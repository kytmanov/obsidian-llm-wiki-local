"""
olw compare runner — executes contestants × seeds against an ephemeral
vault and captures raw telemetry + artifacts per run.

Scoring happens in compare/metrics.py; this module is intentionally
narrow: set up an isolated vault, wire the real pipeline, capture
events, snapshot artifacts, hand the raw data to the next stage.

Critical invariants:
  • Ephemeral vault paths are unique per (contestant, seed); assert
    non-existence before creating to prevent pipeline-lock collision.
  • Synthetic wiki.toml hard-codes pipeline.auto_commit = false so
    git_ops can't fail in a non-repo vault.
  • StateDB.close() runs before any rmtree so SQLite doesn't hold
    the journal file open on cleanup.
  • --baseline-vault path is NEVER written to; the runner copies
    raw/ into its own isolated working dir before any mutation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ..config import Config, _toml_quote
from ..telemetry import telemetry_sink
from .corpus import Corpus
from .models import CompareReport, ContestantResult, ContestantSpec

log = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────


def run_compare(
    contestants: list[ContestantSpec],
    corpus: Corpus,
    out_dir: Path,
    seeds: int = 2,
    keep_artifacts: bool = False,
    skip_queries: bool = False,
    run_id: str | None = None,
) -> CompareReport:
    """Execute all contestants across N seeds, return a populated report.

    Scoring is left to compare/metrics.py in Phase 3; this function
    returns a report with per-contestant `seed_events` + artifact
    paths populated but `scores` empty.
    """
    if not contestants:
        raise ValueError("At least one contestant required")
    if seeds < 1:
        raise ValueError("seeds must be >= 1")

    run_id = run_id or _make_run_id()
    run_dir = out_dir / run_id
    vaults_dir = run_dir / "vaults"
    results_dir = run_dir / "results"
    vaults_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("Compare run %s — %d contestant(s) × %d seed(s)", run_id, len(contestants), seeds)

    t0 = time.monotonic()
    results: list[ContestantResult] = []
    for spec in contestants:
        result = _run_contestant(
            spec=spec,
            corpus=corpus,
            seeds=seeds,
            vaults_dir=vaults_dir,
            results_dir=results_dir,
            keep_artifacts=keep_artifacts,
            skip_queries=skip_queries,
        )
        results.append(result)

    wall_seconds = time.monotonic() - t0

    report = CompareReport(
        run_id=run_id,
        mode=corpus.mode.value,
        seeds=seeds,
        olw_version=_olw_version(),
        pipeline_prompt_hash=_pipeline_prompt_hash(),
        corpus_version=corpus.version,
        notes_set_hash=corpus.notes_set_hash,
        wall_time_seconds=wall_seconds,
        contestants=results,
    )

    # Persist raw report JSON for inspection + downstream report.py
    _write_report_json(report, results_dir / "raw_report.json")
    return report


# ── Per-contestant driver ─────────────────────────────────────────────────────


def _run_contestant(
    spec: ContestantSpec,
    corpus: Corpus,
    seeds: int,
    vaults_dir: Path,
    results_dir: Path,
    keep_artifacts: bool,
    skip_queries: bool,
) -> ContestantResult:
    log.info("── Contestant: %s ─────────────", spec.name)
    result = ContestantResult(spec=spec)
    contestant_results_dir = results_dir / spec.name
    contestant_results_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(seeds):
        vault = vaults_dir / spec.name / str(seed)
        if vault.exists():
            raise RuntimeError(f"Ephemeral vault already exists — refusing to overwrite: {vault}")
        try:
            seed_payload = _run_single(
                spec=spec, corpus=corpus, vault=vault, seed=seed, skip_queries=skip_queries
            )
            result.seed_events[seed] = seed_payload["events"]
            result.seed_artifacts[seed] = str(vault)
            result.seed_pipeline_reports[seed] = seed_payload.get("pipeline_report")
            result.seed_queries[seed] = seed_payload.get("queries", [])
            result.seed_diagnostics[seed] = seed_payload.get("diagnostics", {})
            result.seed_wall_seconds[seed] = seed_payload.get("pipeline_wall_seconds", 0.0)
            if seed_payload.get("partial"):
                result.partial = True

            # Checkpoint per-seed JSON as soon as it's available
            ckpt = contestant_results_dir / f"seed_{seed}.json"
            ckpt.write_text(json.dumps(seed_payload, indent=2, default=str))
        except Exception as e:  # noqa: BLE001
            log.error("Contestant %s seed %d crashed: %s", spec.name, seed, e)
            result.partial = True
            result.seed_events[seed] = []
            result.seed_artifacts[seed] = str(vault)
        finally:
            if not keep_artifacts and vault.exists():
                try:
                    shutil.rmtree(vault)
                except OSError as e:
                    log.warning("Could not remove %s: %s", vault, e)

    return result


def _run_single(
    spec: ContestantSpec,
    corpus: Corpus,
    vault: Path,
    seed: int,
    skip_queries: bool,
) -> dict:
    """One (contestant, seed) pipeline run — returns a serializable payload."""
    from ..client_factory import build_client
    from ..pipeline.orchestrator import PipelineOrchestrator
    from ..pipeline.query import run_query
    from ..state import StateDB

    _materialize_vault(vault, corpus, spec)

    config = Config.from_vault(vault)
    client = build_client(config)
    db = StateDB(config.state_db_path)

    payload: dict = {
        "contestant": spec.name,
        "seed": seed,
        "events": [],
        "queries": [],
        "pipeline_report": None,
        "partial": False,
    }

    try:
        with telemetry_sink() as events:
            t0 = time.monotonic()
            try:
                orchestrator = PipelineOrchestrator(config, client, db)
                report = orchestrator.run(auto_approve=True, max_rounds=2)
                payload["pipeline_report"] = _serialize_pipeline_report(report)
            except Exception as e:  # noqa: BLE001
                log.error("Pipeline crashed for %s seed %d: %s", spec.name, seed, e)
                payload["partial"] = True
                payload["pipeline_error"] = str(e)
            payload["pipeline_wall_seconds"] = time.monotonic() - t0

            # Queries — best-effort; skip on flag or if no queries or pipeline broke
            if not skip_queries and corpus.queries and not payload["partial"]:
                for q in corpus.queries:
                    try:
                        answer, pages = run_query(
                            config=config,
                            client=client,
                            db=db,
                            question=q.question,
                            save=False,
                        )
                        payload["queries"].append(
                            {
                                "id": q.id,
                                "answer": answer,
                                "pages": pages,
                            }
                        )
                    except Exception as e:  # noqa: BLE001
                        log.warning("Query '%s' failed: %s", q.id, e)
                        payload["queries"].append(
                            {
                                "id": q.id,
                                "answer": "",
                                "pages": [],
                                "error": str(e),
                            }
                        )

            # Serialize captured events (LLMCallEvent dataclass → dict)
            payload["events"] = [asdict(ev) for ev in events]

            # Silent all-note failure: pipeline didn't crash but made zero LLM calls
            if not payload["events"] and not payload["partial"]:
                payload["partial"] = True

        # Capture diagnostics BEFORE closing DB / removing vault.
        try:
            payload["diagnostics"] = _capture_diagnostics(vault, db, config, corpus)
        except Exception as e:  # noqa: BLE001
            log.warning("Diagnostic capture failed for %s seed %d: %s", spec.name, seed, e)
            payload["diagnostics"] = {}
    finally:
        # Close DB before caller's rmtree so SQLite drops its journal
        try:
            db.close()
        except AttributeError:
            pass
        try:
            client.close()
        except AttributeError:
            pass

    return payload


# ── Vault materialization ─────────────────────────────────────────────────────


def _materialize_vault(vault: Path, corpus: Corpus, spec: ContestantSpec) -> None:
    """Create ephemeral vault with corpus notes + synthetic wiki.toml."""
    raw_dir = vault / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    (vault / "wiki").mkdir()
    (vault / ".olw").mkdir()

    for note in corpus.notes:
        dst = raw_dir / note.file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(note.path, dst)

    (vault / "wiki.toml").write_text(_synthesize_wiki_toml(spec))


def _synthesize_wiki_toml(spec: ContestantSpec) -> str:
    """Build a wiki.toml for this contestant.

    Key safety knob: pipeline.auto_commit = false so git_ops is a no-op
    inside the ephemeral (non-repo) vault. git_ops itself short-circuits
    when this flag is false.
    """
    fast = _toml_quote(spec.fast_model)
    heavy = _toml_quote(spec.heavy_model)

    provider_block: str
    if spec.provider_name and spec.provider_name != "ollama":
        name = _toml_quote(spec.provider_name)
        url = _toml_quote(spec.provider_url or "")
        provider_block = (
            f"[provider]\n"
            f"name = {name}\n"
            f"url = {url}\n"
            f"timeout = 600\n"
            f"fast_ctx = 8192\n"
            f"heavy_ctx = 16384\n"
        )
    else:
        url = _toml_quote(spec.provider_url or "http://localhost:11434")
        provider_block = (
            f"[ollama]\nurl = {url}\ntimeout = 600\nfast_ctx = 8192\nheavy_ctx = 16384\n"
        )

    return (
        f"[models]\n"
        f"fast = {fast}\n"
        f"heavy = {heavy}\n\n"
        f"{provider_block}\n"
        f"[pipeline]\n"
        f"auto_approve = true\n"
        f"auto_commit = false\n"
        f"max_concepts_per_source = 8\n"
        f"ingest_parallel = false\n"
    )


# ── Diagnostics capture (runs before db close + vault cleanup) ───────────────


def _capture_diagnostics(vault: Path, db, config: Config, corpus: Corpus) -> dict:
    """Compact summary of vault state — enough for metrics.py to score.

    Must run BEFORE db.close() + rmtree, since it reads state.db and the
    ephemeral vault's wiki/ directory.
    """
    from ..pipeline.lint import run_lint
    from ..vault import extract_wikilinks, parse_note

    issue_counts: dict[str, int] = {}
    lint_health: float | None = None
    try:
        lint_result = run_lint(config, db, fix=False)
        lint_health = lint_result.health_score
        for issue in lint_result.issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
    except Exception as e:  # noqa: BLE001
        log.warning("run_lint failed during diagnostics: %s", e)

    extracted_concepts: list[str] = []
    try:
        extracted_concepts = [c.lower() for c in db.list_all_concept_names()]
    except Exception as e:  # noqa: BLE001
        log.warning("concept enumeration failed: %s", e)

    total_pages = 0
    total_words = 0
    total_wikilinks = 0
    total_tags = 0
    total_chars = 0
    article_bodies: list[str] = []

    wiki_dir = vault / "wiki"
    if wiki_dir.is_dir():
        for md in sorted(wiki_dir.rglob("*.md")):
            if md.stem in ("index", "log"):
                continue
            try:
                meta, body = parse_note(md)
            except Exception:  # noqa: BLE001
                continue
            total_pages += 1
            total_words += len(body.split())
            total_chars += len(body)
            total_wikilinks += len(extract_wikilinks(body))
            tags = meta.get("tags") or []
            if isinstance(tags, list):
                total_tags += len([t for t in tags if isinstance(t, str)])
            article_bodies.append(body)

    # Fidelity: 3-gram Jaccard of article bodies vs source notes (mean per article)
    source_text = ""
    try:
        source_text = "\n\n".join(n.path.read_text(errors="ignore") for n in corpus.notes)
    except Exception:  # noqa: BLE001
        pass
    fidelity = _mean_article_source_overlap(article_bodies, source_text)

    return {
        "lint_health": lint_health,
        "issue_counts": issue_counts,
        "extracted_concepts": extracted_concepts,
        "total_pages": total_pages,
        "total_words": total_words,
        "total_wikilinks": total_wikilinks,
        "total_tags": total_tags,
        "total_chars": total_chars,
        "fidelity_source_overlap": fidelity,
    }


def _ngram_set(text: str, n: int = 3) -> set[str]:
    """Word-level n-gram set. Lowercased, punctuation-stripped-ish."""
    import re as _re

    words = _re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _mean_article_source_overlap(bodies: list[str], source_text: str) -> float | None:
    if not bodies or not source_text:
        return None
    src_grams = _ngram_set(source_text)
    if not src_grams:
        return None
    jaccards: list[float] = []
    for body in bodies:
        art_grams = _ngram_set(body)
        if not art_grams:
            continue
        inter = len(art_grams & src_grams)
        union = len(art_grams | src_grams)
        if union == 0:
            continue
        jaccards.append(inter / union)
    if not jaccards:
        return None
    return sum(jaccards) / len(jaccards)


# ── Serialization helpers ─────────────────────────────────────────────────────


def _serialize_pipeline_report(report) -> dict:
    """PipelineReport dataclass → plain dict (FailureRecord enum-safe)."""
    return {
        "ingested": report.ingested,
        "compiled": report.compiled,
        "published": report.published,
        "lint_issues": report.lint_issues,
        "stubs_created": report.stubs_created,
        "rounds": report.rounds,
        "failed": [
            {"concept": f.concept, "reason": f.reason.value, "error_msg": f.error_msg}
            for f in report.failed
        ],
        "timings": dict(report.timings),
        "concept_timings": dict(report.concept_timings),
    }


def _write_report_json(report: CompareReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # CompareReport contains ContestantResult with DimScore objects — rely on
    # asdict to walk the whole tree.
    path.write_text(json.dumps(asdict(report), indent=2, default=str))


# ── Metadata ──────────────────────────────────────────────────────────────────


def _make_run_id() -> str:
    """yyyymmdd-HHMMSS + short hash — stable per run, sortable."""
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    h = hashlib.sha256(ts.encode()).hexdigest()[:6]
    return f"{ts}-{h}"


def _olw_version() -> str:
    try:
        from importlib.metadata import version

        return version("obsidian-llm-wiki")
    except Exception:  # noqa: BLE001
        return "unknown"


def _pipeline_prompt_hash() -> str:
    """Hash of the pipeline prompt templates currently in use.

    Two runs can only be compared apples-to-apples if this matches.
    We hash the top-level system prompts from ingest + compile because
    they drive every scored stage.
    """
    try:
        from ..pipeline import compile as _compile
        from ..pipeline import ingest as _ingest
    except ImportError:
        return "unknown"

    prompts = [
        getattr(_ingest, "_SYSTEM", ""),
        getattr(_compile, "_PLAN_SYSTEM", ""),
        getattr(_compile, "_WRITE_SYSTEM", ""),
        getattr(_compile, "_STUB_WRITE_SYSTEM", ""),
    ]
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
