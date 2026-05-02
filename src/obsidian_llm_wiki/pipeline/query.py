"""
Query pipeline: index-based routing → grounded answer.

Flow:
  1. Read wiki/index.md (no embeddings — index is the routing layer)
  2. Fast model selects relevant pages (PageSelection)
  3. Load page content (up to MAX_PAGES, MAX_CHARS_PER_PAGE each)
  4. Heavy model generates answer (QueryAnswer)
  5. Optionally save to wiki/queries/
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import frontmatter

from ..config import Config
from ..indexer import append_log, generate_index
from ..models import PageSelection, QueryAnswer, WikiArticleRecord
from ..protocols import LLMClientProtocol
from ..state import (
    DuplicateArticlePathError,
    DuplicateSynthesisQuestionHashError,
    StateDB,
    SynthesisInsertConflictError,
)
from ..structured_output import request_structured
from ..telemetry import AppEvent, emit_app_event
from ..vault import (
    atomic_write,
    list_wiki_articles,
    next_available_path,
    parse_note,
    sanitize_filename,
    write_note,
)

MAX_PAGES = 5
MAX_CHARS_PER_PAGE = 8_000

log = logging.getLogger(__name__)


@dataclass
class QuerySaveResult:
    path: Path
    resolution: str = "saved_new"
    duplicate_detected: bool = False
    file_written: bool = True
    error: str | None = None


@dataclass
class QueryRunResult:
    answer: str
    selected_pages: list[str]
    synthesis: QuerySaveResult | None = None
    query_save: QuerySaveResult | None = None

    def __iter__(self):
        yield self.answer
        yield self.selected_pages


class SynthesisSaveError(ValueError):
    resolution = "save_failed"

    def __init__(
        self,
        message: str,
        *,
        path: Path | None = None,
        duplicate_detected: bool = False,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.duplicate_detected = duplicate_detected


class SynthesisChainError(SynthesisSaveError):
    resolution = "rejected_synthesis_chain"


class SynthesisManualEditConflictError(SynthesisSaveError):
    resolution = "manual_edit_conflict"


class SynthesisPathAllocationError(SynthesisSaveError):
    resolution = "save_failed"


# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_index(config: Config) -> str:
    index_path = config.wiki_dir / "index.md"
    if not index_path.exists():
        return ""
    return index_path.read_text(encoding="utf-8")


def _find_page(config: Config, title: str, db: StateDB | None = None) -> Path | None:
    """Resolve a title to a file path with concept > source > synthesis precedence."""

    def priority(path: Path) -> tuple[int, str]:
        rel = path.relative_to(config.vault)
        rel_text = str(rel)
        if "sources" in rel.parts:
            return (1, rel_text.casefold())
        if "synthesis" in rel.parts:
            return (2, rel_text.casefold())
        return (0, rel_text.casefold())

    if title.lower().startswith("sources/"):
        source_title = title.split("/", 1)[1]
        candidate = config.wiki_dir / f"{title}.md"
        if candidate.exists():
            return candidate
        candidate = config.sources_dir / f"{source_title}.md"
        if candidate.exists():
            return candidate
    # Exact filename match (wiki root)
    candidate = config.wiki_dir / f"{title}.md"
    if candidate.exists():
        return candidate
    # Exact filename match (sources/)
    candidate2 = config.sources_dir / f"{title}.md"
    if candidate2.exists():
        return candidate2
    # Frontmatter title scan (case-insensitive fallback)
    matches: list[Path] = []
    for md in config.wiki_dir.rglob("*.md"):
        if ".drafts" in md.parts:
            continue
        try:
            meta, _ = parse_note(md)
            if meta.get("title", "").lower() == title.lower():
                matches.append(md)
        except Exception:
            pass
    if matches:
        return sorted(matches, key=priority)[0]
    # Alias resolution fallback
    if db is not None:
        canonical = db.resolve_alias(title)
        if canonical is not None:
            return _find_page(config, canonical, db=None)
    return None


def _load_pages(config: Config, page_titles: list[str], db: StateDB | None = None) -> str:
    """Return concatenated content of selected pages."""
    parts: list[str] = []
    for title in page_titles[:MAX_PAGES]:
        page = _find_page(config, title, db=db)
        if page is None:
            continue
        try:
            meta, body = parse_note(page)
            page_title = meta.get("title", title)
            parts.append(f"# {page_title}\n\n{body[:MAX_CHARS_PER_PAGE]}")
        except Exception:
            pass
    return "\n\n---\n\n".join(parts)


def _derive_synthesis_title(question: str, model_title: str | None) -> str:
    candidate = (model_title or "").strip()
    if candidate and len(candidate.split()) <= 12 and sanitize_filename(candidate) != "untitled":
        return candidate

    normalized = unicodedata.normalize("NFKC", question).strip()
    normalized = normalized.rstrip("?").strip()
    normalized = re.sub(r"\s+", " ", normalized)
    fallback = " ".join(normalized.split()[:8]).title()
    return fallback or "untitled-synthesis"


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _normalize_question(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized.endswith("?"):
        normalized = normalized[:-1].rstrip()
    return normalized


def _question_hash(question: str) -> str:
    return hashlib.sha256(_normalize_question(question).encode("utf-8")).hexdigest()[:16]


def find_existing_synthesis(db: StateDB, question: str) -> WikiArticleRecord | None:
    return db.find_synthesis_by_question_hash(_question_hash(question))


def _resolve_source_paths(config: Config, source_pages: list[str], db: StateDB) -> list[Path]:
    resolved: list[Path] = []
    for title in source_pages:
        page = _find_page(config, title, db=db)
        if page is not None:
            resolved.append(page)
    return resolved


def _source_hashes(config: Config, source_paths: list[Path]) -> list[dict[str, str]]:
    hashes: list[dict[str, str]] = []
    for path in source_paths:
        _, body = parse_note(path)
        hashes.append(
            {
                "path": str(path.relative_to(config.vault)),
                "hash": _body_hash(body),
            }
        )
    return hashes


def _is_synthesis_source(config: Config, db: StateDB, path: Path) -> bool:
    rel_path = str(path.relative_to(config.vault))
    article = db.get_article(rel_path)
    if article is not None and article.kind == "synthesis":
        return True
    try:
        meta, _ = parse_note(path)
    except Exception:
        return False
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    return any(str(tag).casefold() == "synthesis" for tag in tags)


def _render_synthesis_body(answer: str, source_pages: list[str]) -> str:
    source_lines = ["## Sources", ""]
    if source_pages:
        source_lines.extend(f"- [[{page}]]" for page in source_pages)
    else:
        source_lines.append("- No source pages were selected.")
    return f"{answer.rstrip()}\n\n" + "\n".join(source_lines)


def _build_synthesis_file_text(
    body: str,
    *,
    title: str,
    question: str,
    source_pages: list[str],
    source_page_hashes: list[dict[str, str]],
    question_hash: str,
    content_hash: str,
    created: str,
) -> str:
    meta = {
        "title": title,
        "tags": ["synthesis"],
        "kind": "synthesis",
        "source_question": question,
        "source_pages": source_pages,
        "source_page_hashes": source_page_hashes,
        "question_hash": question_hash,
        "content_hash": content_hash,
        "created": created,
        "status": "published",
    }
    return frontmatter.dumps(frontmatter.Post(body, **meta))


def _reserved_synthesis_names(config: Config, db: StateDB) -> set[str]:
    synthesis_parent = config.synthesis_dir.relative_to(config.vault)
    return {
        Path(article.path).name
        for article in db.list_articles()
        if Path(article.path).parent == synthesis_parent
    }


def _save_synthesis_new(
    config: Config,
    db: StateDB,
    *,
    base_path: Path,
    title: str,
    body: str,
    file_text: str,
    content_hash: str,
    question: str,
    source_pages: list[str],
    question_hash: str | None,
    source_paths: list[str],
    source_page_hashes: list[dict[str, str]],
    duplicate_strategy: str,
    duplicate_detected: bool,
    created_at: datetime,
) -> QuerySaveResult:
    reserved_names = _reserved_synthesis_names(config, db)
    attempts = 0

    while attempts < 16:
        path = next_available_path(base_path, reserved_names=reserved_names)
        relative_path = str(path.relative_to(config.vault))
        record = WikiArticleRecord(
            path=relative_path,
            title=title,
            sources=[],
            content_hash=content_hash,
            created_at=created_at,
            updated_at=datetime.now(),
            is_draft=False,
            kind="synthesis",
            question_hash=question_hash,
            synthesis_sources=source_paths,
            synthesis_source_hashes=[[item["path"], item["hash"]] for item in source_page_hashes],
        )
        try:
            with db._tx():
                db.insert_synthesis_atomic(record)
                atomic_write(path, file_text)
            resolution = (
                "saved_new"
                if not duplicate_detected and question_hash is not None and path == base_path
                else "saved_with_suffix"
            )
            return QuerySaveResult(
                path=path,
                resolution=resolution,
                duplicate_detected=duplicate_detected,
            )
        except DuplicateArticlePathError:
            attempts += 1
            reserved_names.add(path.name)
            continue
        except DuplicateSynthesisQuestionHashError:
            if question_hash is None:
                raise RuntimeError("synthesis question hash conflict without question hash")

            existing = db.find_synthesis_by_question_hash(question_hash)
            if existing is None:
                raise RuntimeError("duplicate synthesis detected without existing row")

            if duplicate_strategy == "keep_existing":
                return QuerySaveResult(
                    path=config.vault / existing.path,
                    resolution="kept_existing",
                    duplicate_detected=True,
                    file_written=False,
                )
            if duplicate_strategy == "update_in_place":
                return _update_existing_synthesis(
                    config,
                    db,
                    existing=existing,
                    title=title,
                    question=question,
                    answer_body=body,
                    source_pages=source_pages,
                    source_paths=source_paths,
                    source_page_hashes=source_page_hashes,
                    duplicate_detected=True,
                )

            duplicate_detected = True
            question_hash = None
            attempts += 1

    raise SynthesisPathAllocationError(
        f"Could not allocate a unique synthesis path for {base_path.name}.",
        path=base_path,
        duplicate_detected=duplicate_detected,
    )


def _update_existing_synthesis(
    config: Config,
    db: StateDB,
    *,
    existing,
    title: str,
    question: str,
    answer_body: str,
    source_pages: list[str],
    source_paths: list[str],
    source_page_hashes: list[dict[str, str]],
    duplicate_detected: bool,
) -> QuerySaveResult:
    path = config.vault / existing.path
    created_text = existing.created_at.strftime("%Y-%m-%d")
    if path.exists():
        try:
            meta, existing_body = parse_note(path)
        except Exception as exc:
            raise SynthesisManualEditConflictError(
                f"Existing synthesis at {path} could not be parsed safely.",
                path=path,
                duplicate_detected=duplicate_detected,
            ) from exc
        if existing.content_hash != _body_hash(existing_body):
            raise SynthesisManualEditConflictError(
                f"Existing synthesis at {path} was manually edited; refusing to overwrite.",
                path=path,
                duplicate_detected=duplicate_detected,
            )
        # Preserve a manually adjusted frontmatter created date on in-place updates.
        created_text = str(meta.get("created") or created_text)

    content_hash = _body_hash(answer_body)
    file_text = _build_synthesis_file_text(
        answer_body,
        title=title,
        question=question,
        source_pages=source_pages,
        source_page_hashes=source_page_hashes,
        question_hash=existing.question_hash or _question_hash(question),
        content_hash=content_hash,
        created=created_text,
    )
    record = WikiArticleRecord(
        path=existing.path,
        title=title,
        sources=[],
        content_hash=content_hash,
        created_at=existing.created_at,
        updated_at=datetime.now(),
        is_draft=False,
        kind="synthesis",
        question_hash=existing.question_hash or _question_hash(question),
        synthesis_sources=source_paths,
        synthesis_source_hashes=[[item["path"], item["hash"]] for item in source_page_hashes],
    )
    with db._tx():
        db._upsert_article_row(record)
        atomic_write(path, file_text)
    return QuerySaveResult(
        path=path,
        resolution="updated_in_place",
        duplicate_detected=duplicate_detected,
    )


def _save_synthesis(
    config: Config,
    db: StateDB,
    question: str,
    answer: str,
    source_pages: list[str],
    title: str,
    duplicate_strategy: str = "keep_existing",
) -> QuerySaveResult:
    config.synthesis_dir.mkdir(parents=True, exist_ok=True)

    question_hash = _question_hash(question)
    existing = find_existing_synthesis(db, question)
    if existing is not None:
        if duplicate_strategy == "keep_existing":
            return QuerySaveResult(
                path=config.vault / existing.path,
                resolution="kept_existing",
                duplicate_detected=True,
                file_written=False,
            )

    resolved_sources = _resolve_source_paths(config, source_pages, db)
    if any(_is_synthesis_source(config, db, path) for path in resolved_sources):
        raise SynthesisChainError("Synthesis sources cannot include another synthesis page")

    source_paths = [str(path.relative_to(config.vault)) for path in resolved_sources]
    source_page_hashes = _source_hashes(config, resolved_sources)
    body = _render_synthesis_body(answer, source_pages)
    content_hash = _body_hash(body)
    base_path = config.synthesis_dir / f"{sanitize_filename(title)}.md"
    duplicate_detected = existing is not None
    if existing is not None and duplicate_strategy == "update_in_place":
        result = _update_existing_synthesis(
            config,
            db,
            existing=existing,
            title=title,
            question=question,
            answer_body=body,
            source_pages=source_pages,
            source_paths=source_paths,
            source_page_hashes=source_page_hashes,
            duplicate_detected=True,
        )
    else:
        created_text = datetime.now().strftime("%Y-%m-%d")
        file_text = _build_synthesis_file_text(
            body,
            title=title,
            question=question,
            source_pages=source_pages,
            source_page_hashes=source_page_hashes,
            question_hash=question_hash,
            content_hash=content_hash,
            created=created_text,
        )
        result = _save_synthesis_new(
            config,
            db,
            base_path=base_path,
            title=title,
            body=body,
            file_text=file_text,
            content_hash=content_hash,
            question=question,
            source_pages=source_pages,
            question_hash=(
                None
                if duplicate_detected and duplicate_strategy == "save_with_suffix"
                else question_hash
            ),
            source_paths=source_paths,
            source_page_hashes=source_page_hashes,
            duplicate_strategy=duplicate_strategy,
            duplicate_detected=duplicate_detected,
            created_at=datetime.now(),
        )

    try:
        generate_index(config, db)
        append_log(config, f"query synthesize | {question[:60]}")
    except Exception as exc:
        log.warning("query synthesize post-save maintenance failed: %s", exc)
    return result


def _emit_synthesis_event(question: str, source_pages: list[str], result: QuerySaveResult) -> None:
    emit_app_event(
        AppEvent(
            name="query_synthesize",
            payload={
                "question_hash": _question_hash(question),
                "resolution": result.resolution,
                "file_written": result.file_written,
                "path": str(result.path) if result.path else None,
                "duplicate_detected": result.duplicate_detected,
                "source_page_count": len(source_pages),
                "error": result.error,
            },
        )
    )


# ── Public API ────────────────────────────────────────────────────────────────


def run_query(
    config: Config,
    client: LLMClientProtocol,
    db: StateDB,
    question: str,
    save: bool = False,
    synthesize: bool = False,
    duplicate_strategy: str = "keep_existing",
) -> QueryRunResult:
    """
    Run a query against the wiki.
    Returns answer, selected pages, and any save metadata.
    """
    index_content = _load_index(config)
    if not index_content:
        return QueryRunResult(
            answer="No wiki index found. Run `olw ingest` and `olw compile` first.",
            selected_pages=[],
        )

    # Step 1: fast model picks pages
    selection_prompt = (
        "You are a routing agent for a personal knowledge wiki.\n\n"
        f"Wiki index:\n{index_content}\n\n"
        f"User question: {question}\n\n"
        "Synthesis pages capture prior answers; consider them when relevant "
        "but never prefer them over a fresh concept page. "
        "Select 1-5 page titles from the index that are most relevant to answer this question. "
        'Return JSON: {"pages": ["Title 1", "Title 2"]}'
    )
    selection = request_structured(
        client=client,
        prompt=selection_prompt,
        model_class=PageSelection,
        model=config.models.fast,
        num_ctx=config.effective_provider.fast_ctx,
        max_retries=2,
        stage="query_select",
    )

    # Step 2: load selected pages
    context = _load_pages(config, selection.pages, db=db)
    if not context:
        context = "(No matching wiki pages found.)"

    # Step 3: heavy model answers
    known_titles = ", ".join(title for title, _ in list_wiki_articles(config.wiki_dir)[:80])
    answer_prompt = (
        "You are answering a question using a personal knowledge wiki.\n\n"
        f"Relevant wiki content:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using the wiki content. Use [[wikilinks]] only for existing wiki pages from "
        f"this list: {known_titles}. Do not create links for terms missing from that list. "
        "Answer in the same language as the user's question. "
        "Also provide a short topic title for the answer subject. "
        'Return JSON: {"answer": "your full markdown answer here", "title": "short title"}'
    )
    result = request_structured(
        client=client,
        prompt=answer_prompt,
        model_class=QueryAnswer,
        model=config.models.heavy,
        num_ctx=config.effective_provider.heavy_ctx,
        max_retries=2,
        stage="query_answer",
    )

    synthesis_title = _derive_synthesis_title(question, result.title)

    query_save = None
    if save:
        query_path = _save_query(config, db, question, result.answer, selection.pages)
        query_save = QuerySaveResult(path=query_path, resolution="saved_new")

    synthesis_save = None
    if synthesize:
        try:
            synthesis_save = _save_synthesis(
                config,
                db,
                question,
                result.answer,
                selection.pages,
                synthesis_title,
                duplicate_strategy,
            )
        except SynthesisSaveError as exc:
            synthesis_save = QuerySaveResult(
                path=exc.path or config.synthesis_dir / f"{sanitize_filename(synthesis_title)}.md",
                resolution=exc.resolution,
                duplicate_detected=exc.duplicate_detected,
                file_written=False,
                error=str(exc),
            )
            _emit_synthesis_event(question, selection.pages, synthesis_save)
            raise
        except SynthesisInsertConflictError as exc:
            synthesis_save = QuerySaveResult(
                path=config.synthesis_dir / f"{sanitize_filename(synthesis_title)}.md",
                resolution="save_failed",
                duplicate_detected=False,
                file_written=False,
                error=str(exc),
            )
            _emit_synthesis_event(question, selection.pages, synthesis_save)
            raise
        except Exception as exc:
            synthesis_save = QuerySaveResult(
                path=config.synthesis_dir / f"{sanitize_filename(synthesis_title)}.md",
                resolution="save_failed",
                duplicate_detected=False,
                file_written=False,
                error=str(exc),
            )
            _emit_synthesis_event(question, selection.pages, synthesis_save)
            raise
        _emit_synthesis_event(question, selection.pages, synthesis_save)

    return QueryRunResult(
        answer=result.answer,
        selected_pages=selection.pages,
        synthesis=synthesis_save,
        query_save=query_save,
    )


def _save_query(
    config: Config,
    db: StateDB,
    question: str,
    answer: str,
    source_pages: list[str],
) -> Path:
    """Write answer to wiki/queries/, update index + log."""
    config.queries_dir.mkdir(parents=True, exist_ok=True)

    slug = sanitize_filename(question[:60])
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"{date_str}-{slug}.md"
    path = config.queries_dir / filename

    meta = {
        "title": question[:80],
        "tags": ["query"],
        "source_pages": source_pages,
        "created": date_str,
        "status": "published",
    }
    write_note(path, meta, answer)
    append_log(config, f"query | {question[:60]}")
    generate_index(config, db)
    return path
