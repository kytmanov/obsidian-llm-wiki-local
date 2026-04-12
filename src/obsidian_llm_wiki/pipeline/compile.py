"""
Compile pipeline: raw notes → wiki articles.

Two compile modes:
  compile_concepts (default, v0.3.0): concept-driven — one article per concept
    extracted during ingest. Incremental: only compiles concepts with new sources.
    Manual-edit protection via content_hash comparison.

  compile_notes (legacy, --legacy flag): two-step LLM planning (CompilePlan →
    SingleArticle). Kept as fallback.

Articles are written to wiki/.drafts/ for human review before publishing.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import frontmatter as fm_lib

from ..config import Config
from ..models import ArticlePlan, CompilePlan, SingleArticle, WikiArticleRecord
from ..ollama_client import OllamaClient
from ..sanitize import sanitize_tags
from ..state import StateDB
from ..structured_output import StructuredOutputError, request_structured
from ..vault import (
    atomic_write,
    build_wiki_frontmatter,
    ensure_wikilinks,
    extract_wikilinks,
    list_wiki_articles,
    parse_note,
    sanitize_filename,
    write_note,
)

log = logging.getLogger(__name__)

# Annotation thresholds — applied to drafts, stripped on approve
_ANNOTATION_CONFIDENCE_THRESHOLD = 0.4
_ANNOTATION_MIN_SOURCES = 2

_STUB_WRITE_SYSTEM = (
    "You are a wiki editor. Write a brief stub article for a wiki concept that was referenced "
    "by other articles but has no source material yet. Keep it under 150 words. Be factual."
)

_PLAN_SYSTEM = (
    "You are a wiki architect. Given source notes, decide what wiki articles to create or update. "
    "Keep article scope atomic (one concept per article). Plan only — no content yet."
)

_WRITE_SYSTEM = (
    "You are a wiki editor. Write a single wiki article from the provided source material. "
    "Be accurate, cite sources via [[wikilinks]] in body text, use ## section headings, "
    "write in evergreen style. Put [[wikilinks]] inline in prose — do not save them for later."
)


def _load_vault_schema(config: Config) -> str:
    """Read vault-schema.md if it exists (injected into write prompts for context)."""
    if config.schema_path.exists():
        try:
            return config.schema_path.read_text(encoding="utf-8")[:1500]
        except Exception:
            pass
    return ""


# Token budget constants
_BUDGET_SYSTEM = 600  # system + schema
_BUDGET_TOPIC_LIST = 400  # existing article titles
_BUDGET_OUTPUT = 1800  # room for generated content
_BUDGET_SAFETY = 200  # buffer

_QUALITY_BONUS = {"high": 0.25, "medium": 0.1, "low": 0.0}


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_olw_annotations(confidence: float, source_paths: list[str], db: StateDB) -> list[str]:
    """Return HTML comment annotations for low-quality drafts. Empty list = no annotations."""
    annotations = []
    if confidence < _ANNOTATION_CONFIDENCE_THRESHOLD:
        annotations.append(
            f"<!-- olw-auto: low-confidence ({confidence:.2f}) — verify before publishing -->"
        )
    if len(source_paths) < _ANNOTATION_MIN_SOURCES:
        annotations.append("<!-- olw-auto: single-source — cross-reference recommended -->")
    if source_paths:
        qualities = []
        for sp in source_paths:
            rec = db.get_raw(sp)
            if rec and rec.quality:
                qualities.append(rec.quality)
        if qualities and all(q == "low" for q in qualities):
            annotations.append("<!-- olw-auto: all sources low-quality — add better sources -->")
    return annotations


def _strip_olw_annotations(body: str) -> str:
    """Remove all olw-auto HTML comment annotations from article body."""
    return re.sub(r"<!--\s*olw-auto:.*?-->\n?", "", body, flags=re.DOTALL)


def _truncate_to_budget(text: str, max_chars: int) -> str:
    """Rough character-based truncation (≈4 chars per token)."""
    limit = max_chars * 4
    if len(text) > limit:
        return text[:limit] + "\n\n[...truncated...]"
    return text


def _gather_sources(
    source_paths: list[str],
    vault: Path,
    max_chars: int = 20000,
) -> tuple[str, list[str]]:
    """
    Read source files, return (combined_text, resolved_paths).
    Truncates if combined content exceeds max_chars.
    """
    parts = []
    resolved = []
    for sp in source_paths:
        # Try path as-is, then prepend raw/ (model often returns bare filenames)
        candidates = [vault / sp, vault / "raw" / sp, vault / "raw" / Path(sp).name]
        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            log.warning("Source not found: %s", sp)
            continue
        try:
            meta, body = parse_note(p)
            parts.append(f"## Source: {p.name}\n{body}")
            resolved.append(sp)
        except Exception as e:
            log.warning("Could not read %s: %s", sp, e)

    combined = "\n\n---\n\n".join(parts)
    return _truncate_to_budget(combined, max_chars), resolved


def _compute_confidence(source_paths: list[str], db: StateDB) -> float:
    """Compute confidence: 0.25 per source + quality bonus from best source."""
    best = "low"
    for sp in source_paths:
        rec = db.get_raw(sp)
        if rec and rec.quality:
            if rec.quality == "high":
                best = "high"
                break
            elif rec.quality == "medium":
                best = "medium"
    return min(1.0, len(source_paths) * 0.25 + _QUALITY_BONUS.get(best, 0.0))


def _inject_body_sections(body: str, source_paths: list[str], config: Config) -> str:
    """
    Append ## Sources and ## See Also sections to article body.

    ## Sources — [[wikilinks]] to source summary pages in wiki/sources/
    ## See Also — [[wikilinks]] derived from wikilinks already in body
    """
    # Strip any existing ## Sources / ## See Also the LLM may have written
    body = re.sub(r"\n## Sources\b.*", "", body, flags=re.DOTALL).rstrip()
    body = re.sub(r"\n## See Also\b.*", "", body, flags=re.DOTALL).rstrip()

    # ## Sources: link to wiki/sources/{title}.md pages
    source_lines = []
    for sp in source_paths:
        # Resolve source file to get its title
        candidates = [
            config.vault / sp,
            config.vault / "raw" / sp,
            config.vault / "raw" / Path(sp).name,
        ]
        raw_path = next((c for c in candidates if c.exists()), None)
        if raw_path:
            try:
                raw_meta, _ = parse_note(raw_path)
                src_title = raw_meta.get("title") or raw_path.stem.replace("-", " ").title()
            except Exception:
                src_title = Path(sp).stem.replace("-", " ").title()
        else:
            src_title = Path(sp).stem.replace("-", " ").title()
        safe_src = sanitize_filename(src_title)
        display = src_title if safe_src != src_title else src_title
        link = f"[[{safe_src}|{display}]]" if safe_src != src_title else f"[[{src_title}]]"
        source_lines.append(f"- {link}")

    # ## See Also: wikilinks already in body (sorted, deduplicated)
    linked = sorted(set(extract_wikilinks(body)))
    see_also_lines = [f"- [[{t}]]" for t in linked if t]

    sections = "\n\n## Sources\n" + "\n".join(source_lines) if source_lines else ""
    if see_also_lines:
        sections += "\n\n## See Also\n" + "\n".join(see_also_lines)

    return body + sections


def _write_draft(
    content_result: SingleArticle,
    config: Config,
    source_paths: list[str],
    db: StateDB,
    confidence: float = 0.5,
    existing_meta: dict | None = None,
    existing_titles: list[str] | None = None,
) -> Path:
    """Write SingleArticle to wiki/.drafts/ and record in state DB."""
    config.drafts_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(content_result.title)
    draft_path = config.drafts_dir / f"{safe_name}.md"

    # Inject wikilinks for known article titles mentioned in body
    body = ensure_wikilinks(content_result.content, existing_titles or [])
    body = _inject_body_sections(body, source_paths, config)

    # Prepend quality annotations (invisible HTML comments, stripped on approve)
    annotations = _build_olw_annotations(confidence, source_paths, db)
    if annotations:
        annotation_block = "\n".join(annotations) + "\n\n"
        # Insert before first ## heading if present, else prepend
        heading_match = re.search(r"^##\s", body, re.MULTILINE)
        if heading_match:
            body = body[: heading_match.start()] + annotation_block + body[heading_match.start() :]
        else:
            body = annotation_block + body

    meta = build_wiki_frontmatter(
        title=content_result.title,
        tags=content_result.tags,
        sources=source_paths,
        confidence=confidence,
        is_draft=True,
        existing_meta=existing_meta,
    )

    post = fm_lib.Post(body, **meta)
    atomic_write(draft_path, fm_lib.dumps(post))

    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title=content_result.title,
            sources=source_paths,
            content_hash=_content_hash(body),
            is_draft=True,
        )
    )

    return draft_path


# ── Concept-driven compile (default) ──────────────────────────────────────────


def _write_concept_prompt(
    concept: str,
    sources: str,
    existing_titles: list[str],
    existing_content: str = "",
    vault_schema: str = "",
    rejection_history: list[str] | None = None,
) -> str:
    titles_str = ", ".join(existing_titles[:50]) if existing_titles else "none yet"
    prompt = f'Write the wiki article: "{concept}"\n'
    if vault_schema:
        prompt += f"\nVAULT CONVENTIONS:\n{vault_schema}\n"
    prompt += (
        f"\nIMPORTANT: Keep the content field under 800 words. Be concise.\n"
        f"Tags must be lowercase, hyphen-separated, no spaces or special characters. "
        f"Good: machine-learning, quantum-computing. Bad: Machine Learning, C++.\n"
        f"Do NOT use inline hashtags (#tag) in the content body — use [[wikilinks]] only.\n"
        f"If source material references images or diagrams, mention their filenames "
        f"so they can be embedded later (e.g. ![[diagram.png]]).\n"
        f"Use [[wikilinks]] inline in prose to link to related concepts.\n\n"
        f"Existing wiki articles to link to: {titles_str}\n\n"
        f"SOURCE MATERIAL:\n{sources}"
    )
    if existing_content:
        prompt += f"\n\nEXISTING ARTICLE (you are updating this):\n{existing_content}"
    if rejection_history:
        # Deduplicate while preserving order (dict.fromkeys trick)
        unique = list(dict.fromkeys(rejection_history))
        prompt += "\n\nPREVIOUS REJECTIONS — address these issues in this version:\n"
        prompt += "\n".join(f"- {fb}" for fb in unique)
    return prompt


def compile_concepts(
    config: Config,
    client: OllamaClient,
    db: StateDB,
    force: bool = False,
    dry_run: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
    concepts: list[str] | None = None,
) -> tuple[list[Path], list[str], dict[str, float]]:
    """
    Concept-driven compile: one article per concept needing compile.

    A concept needs compile if any linked source has status='ingested', or if it
    is a stub (created by olw maintain for broken wikilinks).
    Skips articles whose on-disk content_hash differs from DB (manually edited).
    Pass force=True to recompile even manually-edited articles.

    Pass concepts= to compile only a specific subset (e.g. concepts linked to
    recently changed source files). None = compile all needing compile.
    """
    all_needing = db.concepts_needing_compile()
    if concepts is not None:
        concept_set = set(concepts)
        concept_names = [c for c in all_needing if c in concept_set]
    else:
        concept_names = all_needing

    if not concept_names:
        log.info("No concepts needing compile")
        return [], [], {}

    log.info("Compiling %d concept(s)", len(concept_names))
    existing_titles = [t for t, _ in list_wiki_articles(config.wiki_dir)]
    vault_schema = _load_vault_schema(config)
    total = len(concept_names)

    if dry_run:
        for name in concept_names:
            srcs = db.get_sources_for_concept(name)
            is_stub = db.has_stub(name)
            stub_tag = " [stub]" if is_stub else ""
            print(
                f"  [concept{stub_tag}] {name} — {len(srcs)} source(s): "
                f"{', '.join(Path(s).name for s in srcs)}"
            )
        return [], [], {}

    draft_paths: list[Path] = []
    failed: list[str] = []
    concept_timings: dict[str, float] = {}
    compiled_sources: set[str] = set()

    for idx, name in enumerate(concept_names, 1):
        if on_progress:
            on_progress(idx, total, name)
        _t_concept = time.monotonic()

        source_paths = db.get_sources_for_concept(name)
        is_stub = db.has_stub(name)

        if not source_paths and not is_stub:
            continue

        # Manual edit protection
        safe_name = sanitize_filename(name)
        wiki_path = config.wiki_dir / f"{safe_name}.md"
        existing_meta: dict | None = None

        if wiki_path.exists():
            try:
                existing_meta, existing_body = parse_note(wiki_path)
                if not force:
                    art_rec = db.get_article(str(wiki_path.relative_to(config.vault)))
                    if art_rec and art_rec.content_hash != _content_hash(existing_body):
                        log.info("Skipping '%s' — manually edited (use --force to override)", name)
                        continue
            except Exception:
                pass

        # For stubs: compile with empty sources using a lightweight stub prompt
        if is_stub and not source_paths:
            stub_prompt = (
                f'Write a brief stub wiki article for the concept: "{name}"\n'
                f"This concept is referenced by other articles but has no source material yet.\n"
                f"Keep it under 150 words. Include a note that this is a stub needing sources."
            )
            try:
                result: SingleArticle = request_structured(
                    client=client,
                    prompt=stub_prompt,
                    model_class=SingleArticle,
                    model=config.models.fast,
                    system=_STUB_WRITE_SYSTEM,
                    num_ctx=config.ollama.fast_ctx,
                    num_predict=config.ollama.fast_ctx // 2,
                )
            except StructuredOutputError as e:
                log.error("Failed to write stub '%s': %s", name, e)
                failed.append(name)
                continue
            draft_path = _write_draft(
                content_result=result,
                config=config,
                source_paths=[],
                db=db,
                confidence=0.0,
                existing_meta=existing_meta,
                existing_titles=existing_titles,
            )
            draft_paths.append(draft_path)
            db.delete_stub(name)
            elapsed = time.monotonic() - _t_concept
            concept_timings[name] = elapsed
            log.info("Stub draft written: %s (%.1fs)", draft_path.name, elapsed)
            continue

        # Gather source material within context budget
        sources_text, resolved_paths = _gather_sources(
            source_paths, config.vault, max_chars=config.ollama.heavy_ctx // 2
        )
        if not resolved_paths:
            log.warning("No readable sources for concept '%s', skipping", name)
            failed.append(name)
            continue

        confidence = _compute_confidence(resolved_paths, db)

        # Include snippet of existing article for update prompts
        existing_content = ""
        if existing_meta and wiki_path.exists():
            try:
                _, ex_body = parse_note(wiki_path)
                existing_content = ex_body[:2000]
            except Exception:
                pass

        # Inject rejection history into prompt
        rejection_records = db.get_rejections(name, limit=3)
        rejection_history = (
            [r["feedback"] for r in rejection_records] if rejection_records else None
        )

        write_prompt = _write_concept_prompt(
            name, sources_text, existing_titles, existing_content, vault_schema, rejection_history
        )

        try:
            result = request_structured(
                client=client,
                prompt=write_prompt,
                model_class=SingleArticle,
                model=config.models.heavy,
                system=_WRITE_SYSTEM,
                num_ctx=config.ollama.heavy_ctx,
                num_predict=config.ollama.heavy_ctx // 2,
            )
        except StructuredOutputError as e:
            log.error("Failed to write '%s': %s", name, e)
            failed.append(name)
            continue

        draft_path = _write_draft(
            content_result=result,
            config=config,
            source_paths=resolved_paths,
            db=db,
            confidence=confidence,
            existing_meta=existing_meta,
            existing_titles=existing_titles,
        )
        draft_paths.append(draft_path)
        compiled_sources.update(resolved_paths)
        elapsed = time.monotonic() - _t_concept
        concept_timings[name] = elapsed
        log.info("Draft written: %s (%.1fs)", draft_path.name, elapsed)

    # Mark all sources that fed any compiled concept as 'compiled'
    for sp in compiled_sources:
        db.mark_raw_status(sp, "compiled")

    return draft_paths, failed, concept_timings


# ── Legacy compile (CompilePlan → SingleArticle) ──────────────────────────────


def _plan_prompt(
    source_summary: str,
    existing_titles: list[str],
) -> str:
    titles_str = ", ".join(existing_titles[:50]) if existing_titles else "none yet"
    return (
        f"EXISTING WIKI ARTICLES: {titles_str}\n\n"
        f"SOURCE NOTES TO PROCESS:\n{source_summary}\n\n"
        f"Plan what wiki articles to create or update. Keep scope atomic."
    )


def _write_prompt_legacy(article: ArticlePlan, sources: str, existing_titles: list[str]) -> str:
    titles_str = ", ".join(existing_titles[:50]) if existing_titles else "none yet"
    return (
        f'Write the wiki article: "{article.title}"\n'
        f"Action: {article.action}\n"
        f"Reasoning: {article.reasoning}\n"
        f"IMPORTANT: Keep the content field under 800 words. Be concise.\n"
        f"Tags must be lowercase, hyphen-separated, no spaces or special characters. "
        f"Good: machine-learning, quantum-computing. Bad: Machine Learning, C++.\n"
        f"Do NOT use inline hashtags (#tag) in the content body — use [[wikilinks]] only.\n\n"
        f"Existing wiki articles to link to: {titles_str}\n\n"
        f"SOURCE MATERIAL:\n{sources}"
    )


def compile_notes(
    config: Config,
    client: OllamaClient,
    db: StateDB,
    rag=None,
    source_paths: list[str] | None = None,
    dry_run: bool = False,
) -> tuple[list[Path], list[str]]:
    """
    Legacy compile: LLM plans articles from source summaries, then writes each one.
    Use compile_concepts() instead for incremental, concept-driven compilation.
    """
    # Resolve source files to compile
    if source_paths is None:
        records = db.list_raw(status="ingested")
        paths = [config.vault / r.path for r in records]
    else:
        paths = [config.vault / sp for sp in source_paths]

    if not paths:
        log.info("No ingested notes to compile")
        return [], []

    # Build source summary for planning (use fast model — keep it short)
    summaries = []
    for p in paths:
        try:
            meta, body = parse_note(p)
            short = body[:500].replace("\n", " ")
            summaries.append(f"- {p.name}: {short}")
        except Exception:
            summaries.append(f"- {p.name}: (unreadable)")

    source_summary = "\n".join(summaries)
    existing_titles = [t for t, _ in list_wiki_articles(config.wiki_dir)]

    # ── Step 1: Plan ──────────────────────────────────────────────────────────
    log.info("Planning compilation from %d source notes...", len(paths))
    plan_prompt = _plan_prompt(source_summary, existing_titles)

    try:
        plan: CompilePlan = request_structured(
            client=client,
            prompt=plan_prompt,
            model_class=CompilePlan,
            model=config.models.fast,
            system=_PLAN_SYSTEM,
            num_ctx=config.ollama.fast_ctx,
        )
    except StructuredOutputError as e:
        log.error("Planning failed: %s", e)
        return [], ["__planning_failed__"]

    if not plan.articles:
        log.info("Plan produced no articles")
        return [], []

    log.info(
        "Plan: %d articles to %s",
        len(plan.articles),
        "/".join(set(a.action for a in plan.articles)),
    )

    if dry_run:
        for a in plan.articles:
            print(f"  [{a.action}] {a.path} — {a.title}")
            print(f"    sources: {', '.join(a.source_paths)}")
        return [], []

    # ── Step 2: Write each article ────────────────────────────────────────────
    draft_paths: list[Path] = []
    failed: list[str] = []
    source_rel_paths = [str(p.relative_to(config.vault)) for p in paths]

    for article in plan.articles:
        log.info("Writing: %s", article.title)

        relevant = [sp for sp in article.source_paths if sp] or source_rel_paths
        sources_text, resolved_paths = _gather_sources(
            relevant,
            config.vault,
            max_chars=config.ollama.heavy_ctx // 2,
        )

        write_prompt = _write_prompt_legacy(article, sources_text, existing_titles)

        try:
            result: SingleArticle = request_structured(
                client=client,
                prompt=write_prompt,
                model_class=SingleArticle,
                model=config.models.heavy,
                system=_WRITE_SYSTEM,
                num_ctx=config.ollama.heavy_ctx,
                num_predict=config.ollama.heavy_ctx // 2,
            )
        except StructuredOutputError as e:
            log.error("Failed to write '%s': %s", article.title, e)
            failed.append(article.title)
            continue

        # Preserve existing_meta if updating an existing article
        existing_path = config.wiki_dir / article.path
        existing_meta = None
        if existing_path.exists():
            try:
                existing_meta, _ = parse_note(existing_path)
            except Exception:
                pass

        confidence = _compute_confidence(resolved_paths, db)
        draft_path = _write_draft(
            content_result=result,
            config=config,
            source_paths=resolved_paths,
            db=db,
            confidence=confidence,
            existing_meta=existing_meta,
            existing_titles=existing_titles,
        )
        draft_paths.append(draft_path)
        log.info("Draft written: %s", draft_path.name)

    # Mark source notes as compiled
    if draft_paths:
        for p in paths:
            rel = str(p.relative_to(config.vault))
            db.mark_raw_status(rel, "compiled")

    return draft_paths, failed


# ── Approve / Reject ──────────────────────────────────────────────────────────


def approve_drafts(
    config: Config,
    db: StateDB,
    paths: list[Path] | None = None,
    notes: str = "",
) -> list[Path]:
    """
    Move draft(s) from wiki/.drafts/ to wiki/.
    Returns list of published paths.
    """
    if paths is None:
        # Approve all drafts
        paths = list(config.drafts_dir.rglob("*.md")) if config.drafts_dir.exists() else []

    published = []
    for draft_path in paths:
        if not draft_path.exists():
            log.warning("Draft not found: %s", draft_path)
            continue

        # Target: wiki/ with same relative structure
        rel_to_drafts = draft_path.relative_to(config.drafts_dir)
        target = config.wiki_dir / rel_to_drafts
        target.parent.mkdir(parents=True, exist_ok=True)

        # Write to target with updated status — don't touch draft until target is written
        meta, body = parse_note(draft_path)
        meta["status"] = "published"
        meta["updated"] = datetime.now().strftime("%Y-%m-%d")
        # Sanitize tags defensively — covers old drafts written before sanitization was added
        if isinstance(meta.get("tags"), list):
            meta["tags"] = sanitize_tags([str(t) for t in meta["tags"] if t is not None])
        # Strip olw-auto annotations before publishing
        body = _strip_olw_annotations(body)
        write_note(target, meta, body)  # write to destination first
        draft_path.unlink()  # only remove draft after target is safely written

        # Update state DB: store published content_hash for edit-protection
        draft_rel = str(draft_path.relative_to(config.vault))
        target_rel = str(target.relative_to(config.vault))
        db.publish_article(draft_rel, target_rel)

        # Store content_hash of published body and record approval
        art = db.get_article(target_rel)
        if art:
            try:
                _, pub_body = parse_note(target)
                db.upsert_article(
                    WikiArticleRecord(
                        path=target_rel,
                        title=art.title,
                        sources=art.sources,
                        content_hash=_content_hash(pub_body),
                        created_at=art.created_at,
                        updated_at=art.updated_at,
                        is_draft=False,
                    )
                )
            except Exception:
                pass
            db.approve_article(target_rel, notes=notes)

        published.append(target)
        log.info("Published: %s", target.name)

    return published


def reject_draft(
    draft_path: Path,
    config: Config,
    db: StateDB,
    feedback: str = "",
) -> None:
    """Delete a draft, store rejection feedback and body for future recompiles."""
    # Resolve to canonical path so relative_to(config.vault) works on macOS
    # where /var is a symlink to /private/var but config.vault is always resolved.
    draft_path = draft_path.resolve()
    # Read before deleting — title and body needed for rejection record
    title = draft_path.stem
    draft_body = ""
    if draft_path.exists():
        try:
            meta, draft_body = parse_note(draft_path)
            title = meta.get("title", draft_path.stem)
        except Exception:
            pass

    draft_rel = str(draft_path.relative_to(config.vault))
    db.delete_article(draft_rel)
    if draft_path.exists():
        draft_path.unlink()

    if feedback:
        db.add_rejection(title, feedback, body=draft_body)
        count = db.rejection_count(title)
        if count >= StateDB._REJECTION_CAP:
            log.warning(
                "Concept '%s' blocked after %d rejections — use `olw unblock` to re-enable",
                title,
                count,
            )
        else:
            log.info("Draft rejected with feedback: %s", feedback)
