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

from datetime import datetime
from pathlib import Path

from ..config import Config
from ..indexer import append_log, generate_index
from ..models import PageSelection, QueryAnswer
from ..protocols import LLMClientProtocol
from ..state import StateDB
from ..structured_output import request_structured
from ..vault import list_wiki_articles, parse_note, sanitize_filename, write_note

MAX_PAGES = 5
MAX_CHARS_PER_PAGE = 8_000


# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_index(config: Config) -> str:
    index_path = config.wiki_dir / "index.md"
    if not index_path.exists():
        return ""
    return index_path.read_text(encoding="utf-8")


def _find_page(config: Config, title: str, db: StateDB | None = None) -> Path | None:
    """Resolve a title to a file path. Checks wiki/ root then sources/."""
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
    for md in config.wiki_dir.rglob("*.md"):
        if ".drafts" in md.parts:
            continue
        try:
            meta, _ = parse_note(md)
            if meta.get("title", "").lower() == title.lower():
                return md
        except Exception:
            pass
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


# ── Public API ────────────────────────────────────────────────────────────────


def run_query(
    config: Config,
    client: LLMClientProtocol,
    db: StateDB,
    question: str,
    save: bool = False,
) -> tuple[str, list[str]]:
    """
    Run a query against the wiki.
    Returns (answer_markdown, selected_page_titles).
    """
    index_content = _load_index(config)
    if not index_content:
        return (
            "No wiki index found. Run `olw ingest` and `olw compile` first.",
            [],
        )

    # Step 1: fast model picks pages
    selection_prompt = (
        "You are a routing agent for a personal knowledge wiki.\n\n"
        f"Wiki index:\n{index_content}\n\n"
        f"User question: {question}\n\n"
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
        'Return JSON: {"answer": "your full markdown answer here"}'
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

    if save:
        _save_query(config, db, question, result.answer, selection.pages)

    return result.answer, selection.pages


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
