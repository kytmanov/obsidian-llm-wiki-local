"""Safe runner for current-vs-challenger vault preview compare."""

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
from ..vault import extract_wikilinks, parse_note
from .metrics import load_queries
from .models import CompareReport, ContestantRunResult, PageDiffSummary, PageSnapshot, QueryResult

log = logging.getLogger(__name__)


def run_compare(
    current_config: Config,
    challenger_config: Config,
    out_dir: Path,
    queries_path: Path | None = None,
    keep_artifacts: bool = False,
    sample_n: int | None = None,
) -> CompareReport:
    run_id = _make_run_id()
    run_root = _safe_child(out_dir.resolve(), run_id)
    _assert_compare_root_safe(run_root.parent, current_config.vault)

    results_dir = _safe_child(run_root, "results")
    current_dir = _safe_child(run_root, "current")
    challenger_dir = _safe_child(run_root, "challenger")
    diffs_dir = _safe_child(run_root, "diffs")
    vaults_dir = _safe_child(run_root, "vaults")
    for d in (results_dir, current_dir, challenger_dir, diffs_dir):
        d.mkdir(parents=True, exist_ok=True)
    if keep_artifacts:
        vaults_dir.mkdir(parents=True, exist_ok=True)

    queries = []
    if queries_path is not None:
        queries = load_queries(_validate_queries_path(queries_path))

    t0 = time.monotonic()
    current_result = _run_single_vault(
        source_config=current_config,
        effective_config=current_config,
        role="current",
        run_root=run_root,
        artifact_dir=current_dir,
        keep_artifacts=keep_artifacts,
        queries=queries,
        sample_n=sample_n,
    )
    challenger_result = _run_single_vault(
        source_config=current_config,
        effective_config=challenger_config,
        role="challenger",
        run_root=run_root,
        artifact_dir=challenger_dir,
        keep_artifacts=keep_artifacts,
        queries=queries,
        sample_n=sample_n,
    )
    wall = time.monotonic() - t0

    page_diff = _diff_pages(current_result.page_snapshots, challenger_result.page_snapshots)
    query_diffs = _diff_queries(queries, current_result.queries, challenger_result.queries)

    report = CompareReport(
        run_id=run_id,
        vault_path=str(current_config.vault),
        out_dir=str(run_root),
        current_config_summary=_config_summary(current_config),
        challenger_config_summary=_config_summary(challenger_config),
        current=current_result,
        challenger=challenger_result,
        page_diff=page_diff,
        query_diffs=query_diffs,
    )
    report.current.diagnostics.setdefault("compare_wall_seconds", wall)
    _write_json(results_dir / "raw_report.json", asdict(report))
    _write_json(diffs_dir / "pages_added.json", page_diff.added)
    _write_json(diffs_dir / "pages_removed.json", page_diff.removed)
    _write_json(diffs_dir / "pages_changed.json", page_diff.changed)
    _write_json(diffs_dir / "queries_diff.json", [asdict(q) for q in query_diffs])
    return report


def _run_single_vault(
    source_config: Config,
    effective_config: Config,
    role: str,
    run_root: Path,
    artifact_dir: Path,
    keep_artifacts: bool,
    queries,
    sample_n: int | None = None,
) -> ContestantRunResult:
    from ..client_factory import build_client
    from ..pipeline.orchestrator import PipelineOrchestrator
    from ..pipeline.query import run_query
    from ..state import StateDB

    temp_root = _safe_child(run_root, "vaults", role)
    if temp_root.exists():
        raise RuntimeError(f"ephemeral compare vault already exists: {temp_root}")

    _materialize_compare_vault(
        temp_root, source_config.raw_dir, effective_config, sample_n=sample_n
    )
    config = Config.from_vault(temp_root)
    client = build_client(config)
    db = StateDB(config.state_db_path)
    pipeline_report = None
    partial = False
    diagnostics: dict[str, float | int | str | bool | None] = {}
    query_results: list[QueryResult] = []
    wall = 0.0

    try:
        with telemetry_sink() as events:
            t0 = time.monotonic()
            try:
                pipeline_report = PipelineOrchestrator(config, client, db).run(
                    auto_approve=True, max_rounds=2
                )
            except Exception as e:  # noqa: BLE001
                log.error("Compare pipeline failed for %s: %s", role, e)
                partial = True
                diagnostics["pipeline_error"] = str(e)
            wall = time.monotonic() - t0

            if queries and not partial:
                for q in queries:
                    try:
                        answer, pages = run_query(
                            config=config,
                            client=client,
                            db=db,
                            question=q.question,
                            save=False,
                        )
                        query_results.append(QueryResult(id=q.id, answer=answer, pages=list(pages)))
                    except Exception as e:  # noqa: BLE001
                        query_results.append(
                            QueryResult(id=q.id, answer="", pages=[], error=str(e))
                        )
            diagnostics.update(_capture_diagnostics(temp_root, db, config, events))
            diagnostics["total_calls"] = len(events)
    finally:
        try:
            db.close()
        except AttributeError:
            pass
        try:
            client.close()
        except AttributeError:
            pass

    snapshots = _snapshot_wiki(config.wiki_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifact_dir / "stats.json", diagnostics)
    _write_json(artifact_dir / "queries.json", [asdict(q) for q in query_results])
    _write_json(artifact_dir / "pages.json", [asdict(p) for p in snapshots])
    _copy_wiki_snapshot(config.wiki_dir, artifact_dir / "wiki_snapshot")

    if not keep_artifacts and temp_root.exists():
        _safe_rmtree(temp_root, _safe_child(run_root, "vaults"))

    return ContestantRunResult(
        role=role,
        fast_model=effective_config.models.fast,
        heavy_model=effective_config.models.heavy,
        provider_name=effective_config.effective_provider.name,
        provider_url=effective_config.effective_provider.url,
        partial=partial,
        pipeline_report=_serialize_pipeline_report(pipeline_report),
        queries=query_results,
        diagnostics=diagnostics,
        wall_time_seconds=wall,
        page_snapshots=snapshots,
        artifact_dir=str(artifact_dir),
    )


def _materialize_compare_vault(
    vault: Path, raw_dir: Path, config: Config, sample_n: int | None = None
) -> None:
    if sample_n is not None and sample_n < 1:
        raise ValueError("sample_n must be at least 1")
    (vault / "raw").mkdir(parents=True, exist_ok=False)
    (vault / "wiki").mkdir()
    (vault / ".olw").mkdir()
    notes = _collect_raw_notes(raw_dir)
    if sample_n is not None:
        notes = notes[:sample_n]
    for note in notes:
        dst = vault / "raw" / note.relative_to(raw_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(note, dst)
    if config.schema_path.exists():
        shutil.copy2(config.schema_path, vault / config.schema_path.name)
    _write_effective_compare_toml(vault, config)


def _write_effective_compare_toml(vault: Path, config: Config) -> None:
    prov = config.effective_provider
    lines = [
        "[models]",
        f"fast = {_toml_quote(config.models.fast)}",
        f"heavy = {_toml_quote(config.models.heavy)}",
        "",
    ]
    if prov.name == "ollama":
        lines.extend(
            [
                "[ollama]",
                f"url = {_toml_quote(prov.url)}",
                f"timeout = {int(prov.timeout)}",
                f"fast_ctx = {prov.fast_ctx}",
                f"heavy_ctx = {prov.heavy_ctx}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "[provider]",
                f"name = {_toml_quote(prov.name)}",
                f"url = {_toml_quote(prov.url)}",
                f"timeout = {int(prov.timeout)}",
                f"fast_ctx = {prov.fast_ctx}",
                f"heavy_ctx = {prov.heavy_ctx}",
            ]
        )
        if prov.name == "azure":
            lines.append(f"azure_api_version = {_toml_quote(prov.azure_api_version)}")
        lines.append("")

    lines.extend(
        [
            "[pipeline]",
            "auto_approve = true",
            "auto_commit = false",
            f"auto_maintain = {str(config.pipeline.auto_maintain).lower()}",
            f"watch_debounce = {config.pipeline.watch_debounce}",
            f"max_concepts_per_source = {config.pipeline.max_concepts_per_source}",
            f"ingest_parallel = {str(config.pipeline.ingest_parallel).lower()}",
        ]
    )
    if config.pipeline.language:
        lines.append(f"language = {_toml_quote(config.pipeline.language)}")
    (vault / "wiki.toml").write_text("\n".join(lines) + "\n")


def _capture_diagnostics(vault: Path, db, config: Config, events) -> dict:
    from ..pipeline.lint import run_lint

    issue_counts: dict[str, int] = {}
    lint_health: float | None = None
    try:
        lint_result = run_lint(config, db, fix=False)
        lint_health = lint_result.health_score
        for issue in lint_result.issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
    except Exception as e:  # noqa: BLE001
        log.warning("compare lint diagnostics failed: %s", e)

    total_pages = 0
    total_wikilinks = 0
    total_words = 0
    for md in sorted((vault / "wiki").rglob("*.md")):
        if md.stem in ("index", "log"):
            continue
        try:
            _, body = parse_note(md)
        except Exception:  # noqa: BLE001
            continue
        total_pages += 1
        total_words += len(body.split())
        total_wikilinks += len(extract_wikilinks(body))

    return {
        "lint_health": lint_health,
        "issue_counts": issue_counts,
        "total_pages": total_pages,
        "total_wikilinks": total_wikilinks,
        "total_words": total_words,
        "total_calls": len(events),
    }


def _snapshot_wiki(wiki_dir: Path) -> list[PageSnapshot]:
    snapshots: list[PageSnapshot] = []
    for md in sorted(wiki_dir.rglob("*.md")):
        if md.stem in ("index", "log") or ".drafts" in md.parts:
            continue
        try:
            meta, body = parse_note(md)
        except Exception:  # noqa: BLE001
            continue
        snapshots.append(
            PageSnapshot(
                path=str(md.relative_to(wiki_dir)),
                title=str(meta.get("title") or md.stem),
                content_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                word_count=len(body.split()),
                wikilinks=sorted(set(extract_wikilinks(body))),
                tags=[t for t in (meta.get("tags") or []) if isinstance(t, str)],
                sources=[s for s in (meta.get("sources") or []) if isinstance(s, str)],
            )
        )
    return snapshots


def _diff_pages(
    current_pages: list[PageSnapshot], challenger_pages: list[PageSnapshot]
) -> PageDiffSummary:
    cur_by_path = {p.path: p for p in current_pages}
    cur_by_title = {p.title.lower(): p for p in current_pages}
    added: list[str] = []
    changed: list[str] = []
    matched_current: set[str] = set()

    for page in challenger_pages:
        current = cur_by_path.get(page.path) or cur_by_title.get(page.title.lower())
        if current is None:
            added.append(page.title)
            continue
        matched_current.add(current.path)
        if (
            current.content_hash != page.content_hash
            or current.tags != page.tags
            or current.wikilinks != page.wikilinks
        ):
            changed.append(page.title)

    removed = [p.title for p in current_pages if p.path not in matched_current]
    return PageDiffSummary(
        added=sorted(added),
        removed=sorted(removed),
        changed=sorted(changed),
    )


def _diff_queries(
    queries, current_results: list[QueryResult], challenger_results: list[QueryResult]
):
    from .metrics import score_query_result
    from .models import QueryDiff

    cur_by_id = {q.id: q for q in current_results}
    challenger_by_id = {q.id: q for q in challenger_results}
    diffs: list[QueryDiff] = []
    for q in queries:
        current = cur_by_id.get(q.id, QueryResult(id=q.id, answer="", pages=[]))
        challenger = challenger_by_id.get(q.id, QueryResult(id=q.id, answer="", pages=[]))
        current_score = score_query_result(current, q)
        challenger_score = score_query_result(challenger, q)
        delta = None
        if current_score is not None and challenger_score is not None:
            delta = challenger_score - current_score
        diffs.append(
            QueryDiff(
                id=q.id,
                question=q.question,
                current_pages=current.pages,
                challenger_pages=challenger.pages,
                current_answer=current.answer,
                challenger_answer=challenger.answer,
                current_score=current_score,
                challenger_score=challenger_score,
                delta=delta,
            )
        )
    return diffs


def _copy_wiki_snapshot(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _serialize_pipeline_report(report) -> dict | None:
    if report is None:
        return None
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
    }


def _config_summary(config: Config) -> dict[str, str]:
    prov = config.effective_provider
    return {
        "fast_model": config.models.fast,
        "heavy_model": config.models.heavy,
        "provider": prov.name,
        "provider_url": prov.url,
    }


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _safe_child(root: Path, *parts: str) -> Path:
    base = root.resolve()
    path = base.joinpath(*parts).resolve()
    path.relative_to(base)
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _collect_raw_notes(raw_dir: Path) -> list[Path]:
    raw_dir = raw_dir.expanduser()
    raw_root = raw_dir.resolve()
    if raw_dir.is_symlink():
        raise ValueError("compare does not support symlinked raw notes")

    notes: list[Path] = []
    pending = [raw_dir]
    while pending:
        current = pending.pop()
        for child in sorted(current.iterdir()):
            if child.is_symlink():
                raise ValueError("compare does not support symlinked raw notes")
            resolved = child.resolve()
            if not _is_within(resolved, raw_root):
                raise ValueError("compare raw notes must stay inside raw/")
            if child.is_dir():
                pending.append(child)
            elif child.is_file() and child.suffix == ".md":
                notes.append(child)
    return sorted(notes)


def _validate_queries_path(queries_path: Path) -> Path:
    path = queries_path.expanduser()
    if not path.exists():
        raise ValueError("--queries must exist")
    if not path.is_file():
        raise ValueError("--queries must be a file")
    if path.is_symlink():
        raise ValueError("--queries must not be a symlink")
    return path.resolve()


def _assert_compare_root_safe(compare_root: Path, active_vault: Path) -> None:
    compare_root = compare_root.resolve()
    active_vault = active_vault.resolve()
    if compare_root == active_vault:
        raise ValueError("compare output root must not be the active vault root")

    raw_root = active_vault / "raw"
    wiki_root = active_vault / "wiki"
    allowed_compare_root = active_vault / ".olw" / "compare"

    if _is_within(compare_root, raw_root) or _is_within(compare_root, wiki_root):
        raise ValueError("compare output root must not be inside active vault raw/ or wiki/")
    if _is_within(compare_root, active_vault) and not _is_within(
        compare_root, allowed_compare_root
    ):
        raise ValueError("compare output root inside the active vault must be under .olw/compare/")


def _safe_rmtree(path: Path, root: Path) -> None:
    path.resolve().relative_to(root.resolve())
    shutil.rmtree(path)


def _make_run_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{hashlib.sha256(ts.encode()).hexdigest()[:6]}"


__all__ = ["run_compare"]
