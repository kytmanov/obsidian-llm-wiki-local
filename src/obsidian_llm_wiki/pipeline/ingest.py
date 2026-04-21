"""
Ingest pipeline: raw note → chunk → analyze → embed → update state.

Uses fast model (gemma4:e4b) for analysis.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

from ..config import Config
from ..models import AnalysisResult, Concept, RawNoteRecord
from ..protocols import LLMClientProtocol
from ..state import StateDB
from ..structured_output import request_structured
from ..vault import (
    chunk_text,
    generate_aliases,
    parse_note,
    sanitize_filename,
    sanitize_wikilink_target,
    write_note,
)

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a knowledge analyst. Read the provided note and extract structured information. "
    "Be concise and accurate. Do not invent information not present in the note. "
    "Detect the primary language of the note and return its ISO 639-1 code in the 'language' field "
    "(e.g. 'en', 'fr', 'de'). Use null if uncertain."
)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_analysis_prompt(
    body: str,
    existing_concepts: list[str],
    path_name: str = "",
    chunk_label: str = "",
) -> str:
    concepts_hint = ", ".join(existing_concepts[:30]) if existing_concepts else "none yet"
    label = f" {chunk_label}" if chunk_label else ""
    return (
        f"Analyze this note{label} and extract structured metadata.\n\n"
        f"Existing wiki concepts (reuse these names where applicable): {concepts_hint}\n\n"
        f"For each concept, provide 3-5 short surface forms used in running text "
        f"(abbreviations, short names). Example: name='Program Counter (PC)', "
        f"aliases=['PC', 'program counter']. Use empty list if no natural aliases exist.\n\n"
        f"NOTE CONTENT:\n{body}"
    )


def _merge_chunk_results(results: list[AnalysisResult]) -> AnalysisResult:
    """Merge AnalysisResults from multiple chunks into one.

    Concepts and topics: union (deduplicated, insertion order preserved).
    Aliases for the same concept are merged across chunks.
    Summary: first chunk's (intro is most representative).
    Quality: minimum across chunks (conservative).
    """
    if len(results) == 1:
        return results[0]

    # Dedup concepts by canonical name (case-insensitive), merge aliases
    seen: dict[str, list[str]] = {}  # lower(name) -> accumulated aliases
    order: list[str] = []  # canonical names in insertion order
    canonical_by_lower: dict[str, str] = {}

    for r in results:
        for c in r.concepts:
            key = c.name.lower()
            if key not in seen:
                seen[key] = list(c.aliases)
                order.append(key)
                canonical_by_lower[key] = c.name
            else:
                existing_lower = {a.lower() for a in seen[key]}
                for a in c.aliases:
                    if a.lower() not in existing_lower:
                        seen[key].append(a)
                        existing_lower.add(a.lower())

    all_concepts = [Concept(name=canonical_by_lower[k], aliases=seen[k]) for k in order][:8]

    seen_topics: set[str] = set()
    all_topics: list[str] = []
    for r in results:
        for t in r.suggested_topics:
            if t.lower() not in seen_topics:
                seen_topics.add(t.lower())
                all_topics.append(t)

    quality_rank = {"high": 2, "medium": 1, "low": 0}
    min_result = min(results, key=lambda r: quality_rank.get(r.quality, 1))

    merged_language = next((r.language for r in results if r.language), None)

    return AnalysisResult(
        summary=results[0].summary,
        concepts=all_concepts,
        suggested_topics=all_topics[:5],
        quality=min_result.quality,
        language=merged_language,
    )


def _analyze_body(
    body: str,
    existing_concepts: list[str],
    path_name: str,
    client: LLMClientProtocol,
    config: Config,
) -> AnalysisResult:
    """Analyze note body, splitting into chunks if body exceeds fast_ctx // 2 chars."""
    chunk_size = config.effective_provider.fast_ctx // 2

    if len(body) <= chunk_size:
        prompt = _build_analysis_prompt(body, existing_concepts, path_name)
        return request_structured(
            client=client,
            prompt=prompt,
            model_class=AnalysisResult,
            model=config.models.fast,
            system=_SYSTEM,
            num_ctx=config.effective_provider.fast_ctx,
            stage="ingest",
        )

    # Split into chunks — no overlap needed for concept extraction
    chunks = [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)]
    log.info(
        "Note %s split into %d chunks for analysis (%d chars, chunk_size=%d)",
        path_name or "unknown",
        len(chunks),
        len(body),
        chunk_size,
    )

    def _analyze_chunk(chunk: str, idx: int) -> AnalysisResult:
        import time

        label = f"[part {idx + 1}/{len(chunks)}]"
        log.info("Analyzing %s %s …", path_name or "note", label)
        t0 = time.monotonic()
        prompt = _build_analysis_prompt(chunk, existing_concepts, path_name, chunk_label=label)
        result = request_structured(
            client=client,
            prompt=prompt,
            model_class=AnalysisResult,
            model=config.models.fast,
            system=_SYSTEM,
            num_ctx=config.effective_provider.fast_ctx,
            stage="ingest",
        )
        log.info("Analyzed %s %s (%.1fs)", path_name or "note", label, time.monotonic() - t0)
        return result

    if config.pipeline.ingest_parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        chunk_results: list[AnalysisResult | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = {
                executor.submit(_analyze_chunk, chunk, i): i for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                chunk_results[futures[future]] = future.result()
        results = [r for r in chunk_results if r is not None]
    else:
        results = [_analyze_chunk(chunk, i) for i, chunk in enumerate(chunks)]

    return _merge_chunk_results(results)


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "it",
        "in",
        "on",
        "at",
        "to",
        "by",
        "for",
        "of",
        "as",
        "from",
        "with",
        "this",
        "that",
        "these",
        "those",
        "be",
        "are",
    }
)


def _validate_aliases(canonical: str, raw_aliases: list[str]) -> list[str]:
    """Filter LLM-produced aliases: remove too-short, stopwords, self-matches, duplicates."""
    seen = {canonical.lower()}
    valid: list[str] = []
    for alias in raw_aliases:
        a = alias.strip()
        if not a or a.lower() in seen:
            continue
        if len(a) < 2:
            continue
        if len(a) <= 3 and not a.isupper():
            continue
        if a.lower() in _STOPWORDS:
            continue
        seen.add(a.lower())
        valid.append(a)
    return valid[:5]


def _normalize_concepts(raw_concepts: list[Concept], db: StateDB) -> list[tuple[str, list[str]]]:
    """Case-insensitive dedup against existing canonical concept names.

    Returns (canonical_name, validated_aliases) pairs.
    """
    existing = {n.lower(): n for n in db.list_all_concept_names()}
    seen: set[str] = set()
    result: list[tuple[str, list[str]]] = []
    for concept in raw_concepts:
        name = concept.name.strip()
        if not name:
            continue
        canonical = existing.get(name.lower(), name)
        if canonical in seen:
            continue
        seen.add(canonical)
        aliases = _validate_aliases(canonical, concept.aliases)
        result.append((canonical, aliases))
    return result


_HEADER_SCAN_LINES = 30  # only strip short lines from the opening section

# Media reference patterns for source page preservation
_OBSIDIAN_EMBED_RE = re.compile(
    r"!\[\[([^\]]+\.(?:png|jpg|jpeg|gif|svg|webp|bmp|tiff|avif|pdf|mp4|webm|mov|mp3|wav|ogg))\]\]",
    re.IGNORECASE,
)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _preprocess_web_clip(content: str) -> str:
    """Clean common Obsidian Web Clipper artifacts (nav bars, cookie banners, HTML tags).

    HTML stripping is scoped to the first _HEADER_SCAN_LINES only — body HTML
    (<details>, <kbd>, <sup>, etc.) is intentional and preserved.
    """
    _MD_STARTS = ("#", "-", "*", ">", "[", "!")  # markdown structural chars — always keep
    lines = content.splitlines()

    cleaned = []
    for i, line in enumerate(lines):
        if i < _HEADER_SCAN_LINES:
            # Strip HTML only in header region (nav/banner cleanup)
            line = re.sub(r"<[^>]+>", "", line)
            stripped = line.strip()
            # Skip short non-empty non-markdown lines (nav/banner heuristic)
            if stripped and len(stripped.split()) <= 5 and not stripped.startswith(_MD_STARTS):
                continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _collect_media_refs(body: str) -> list[str]:
    """Extract media references from note body for preservation in source pages."""
    refs: list[str] = []
    for m in _OBSIDIAN_EMBED_RE.finditer(body):
        refs.append(f"- ![[{m.group(1)}]]")
    for m in _MD_IMAGE_RE.finditer(body):
        alt, url = m.group(1), m.group(2)
        refs.append(f"- ![{alt}]({url})")
    return refs


def _create_source_summary_page(
    path: Path,
    src_meta: dict,
    result: AnalysisResult,
    config: Config,
    body: str = "",
) -> Path:
    """
    Generate wiki/sources/{Title}.md from AnalysisResult. No extra LLM call.
    Returns the path written.
    """
    # Derive title from note frontmatter > file stem
    title = src_meta.get("title") or path.stem.replace("-", " ").title()
    safe_name = sanitize_filename(title)
    out_path = config.sources_dir / f"{safe_name}.md"
    config.sources_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d")
    rel_raw = str(path.relative_to(config.vault))
    source_url = src_meta.get("source") or src_meta.get("url") or ""
    aliases = generate_aliases(title, "")  # source pages rarely have abbreviations

    # Build concept list as [[wikilinks]]
    concept_lines = "\n".join(
        f"- [[{sanitize_wikilink_target(c.name)}]]" for c in result.concepts[:8] if c.name.strip()
    )

    out_meta: dict = {
        "title": title,
        "aliases": aliases,
        "tags": ["source"],
        "status": "published",
        "source_file": rel_raw,
        "quality": result.quality,
        "created": now,
    }
    if source_url:
        out_meta["source_url"] = source_url

    body_parts = [
        f"# {title}",
        "",
        "## Summary",
        result.summary,
        "",
        "## Concepts",
        concept_lines,
        "",
        "## Source Info",
        f"- **Quality:** {result.quality}",
        f"- **Raw file:** {rel_raw}",
        f"- **Ingested:** {now}",
    ]
    if source_url:
        body_parts.append(f"- **URL:** {source_url}")

    media_refs = _collect_media_refs(body)
    if media_refs:
        body_parts += ["", "## Media"] + media_refs

    write_note(out_path, out_meta, "\n".join(body_parts))
    log.info("Source summary written: %s", out_path.name)
    return out_path


def ingest_note(
    path: Path,
    config: Config,
    client: LLMClientProtocol,
    db: StateDB,
    rag=None,  # Optional RAGStore, injected in Phase 2
    existing_topics: list[str] | None = None,  # existing concept names for prompt
    force: bool = False,
) -> AnalysisResult | None:
    """
    Ingest a single raw note.

    Returns AnalysisResult or None if skipped (duplicate / already ingested).
    """
    content = path.read_text(encoding="utf-8")
    # Hash body only (strip frontmatter) so copies are detected as duplicates
    # even after ingest has updated the original's frontmatter (olw_status etc.)
    try:
        _, body_for_hash = parse_note(path)
    except Exception:
        body_for_hash = content
    h = _content_hash(body_for_hash)

    # Dedup check
    existing = db.get_raw_by_hash(h)
    if existing and existing.path != str(path.relative_to(config.vault)):
        log.info("Duplicate of %s, skipping %s", existing.path, path.name)
        return None

    rel_path = str(path.relative_to(config.vault))
    record = db.get_raw(rel_path)

    if record and record.status == "ingested" and not force:
        log.info("Already ingested: %s", path.name)
        return None

    # Pre-process web clips
    meta, body = parse_note(path)
    if meta.get("source") or meta.get("url"):  # web clipper adds these
        body = _preprocess_web_clip(body)

    # Chunk + embed only when RAG store is wired in (Phase 2)
    if rag is not None:
        chunks = chunk_text(
            body, chunk_size=config.rag.chunk_size, overlap=config.rag.chunk_overlap
        )
        embeddings = client.embed_batch(chunks, model=config.models.embed)
        rag.add_document(
            doc_id=rel_path,
            chunks=chunks,
            embeddings=embeddings,
            metadata={"source": rel_path, "type": "raw"},
        )

    # LLM analysis — use existing concept names so model can reuse canonical names
    if existing_topics is None:
        existing_topics = db.list_all_concept_names()
    try:
        result: AnalysisResult = _analyze_body(
            body=body,
            existing_concepts=existing_topics,
            path_name=path.name,
            client=client,
            config=config,
        )
    except Exception as e:
        log.error("Analysis failed for %s: %s", path.name, e)
        db.upsert_raw(
            RawNoteRecord(
                path=rel_path,
                content_hash=h,
                status="failed",
                error=str(e),
            )
        )
        return None

    # Update state DB (raw files stay immutable — metadata lives in state.db only)
    db.upsert_raw(
        RawNoteRecord(
            path=rel_path,
            content_hash=h,
            status="ingested",
            summary=result.summary,
            quality=result.quality,
            language=result.language,
            ingested_at=datetime.now(),
        )
    )

    # Normalize concept names against existing canonical names, store linkages
    max_concepts = config.pipeline.max_concepts_per_source
    normalized = _normalize_concepts(result.concepts[:max_concepts], db)
    canonical_names = [name for name, _ in normalized]
    db.upsert_concepts(rel_path, canonical_names)
    for canonical, aliases in normalized:
        if aliases:
            db.upsert_aliases(canonical, aliases)

    # Create source summary page in wiki/sources/ (no extra LLM call)
    try:
        _create_source_summary_page(path, meta, result, config, body=body)
    except Exception as e:
        log.warning("Source summary page failed for %s: %s", path.name, e)

    log.info(
        "Ingested: %s (quality=%s, concepts=%s)",
        path.name,
        result.quality,
        [c.name for c in result.concepts[:3]],
    )
    return result


def ingest_all(
    config: Config,
    client: LLMClientProtocol,
    db: StateDB,
    rag=None,
    force: bool = False,
) -> list[tuple[Path, AnalysisResult | None]]:
    """Ingest all .md files in raw/ (excluding raw/processed/ subfolders)."""
    raw_files = [
        p
        for p in config.raw_dir.rglob("*.md")
        if "processed" not in p.parts and not p.name.startswith(".")
    ]
    # Snapshot concept names once before loop (for consistent prompt context)
    existing_topics = db.list_all_concept_names()
    results = []
    for path in sorted(raw_files):
        result = ingest_note(
            path=path,
            config=config,
            client=client,
            db=db,
            rag=rag,
            existing_topics=existing_topics,
            force=force,
        )
        results.append((path, result))
    return results
