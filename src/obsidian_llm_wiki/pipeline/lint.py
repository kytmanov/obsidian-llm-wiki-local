"""
Lint pipeline: all structural checks, no LLM required.

Checks:
  orphan           — concept page with no inbound [[wikilinks]] from other pages
  broken_link      — [[Target]] in body that resolves to no file
  missing_frontmatter — required fields (title, status, tags) absent
  stale            — file hash on disk != DB content_hash (manually edited)
  low_confidence   — confidence < LOW_CONFIDENCE_THRESHOLD
  invalid_tag      — tag that is not a valid Obsidian tag name

Fix mode (--fix):
  Auto-fixes missing_frontmatter and invalid_tag fields.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..config import Config
from ..models import LintIssue, LintResult, WikiArticleRecord
from ..sanitize import sanitize_tag, sanitize_tags
from ..state import StateDB
from ..vault import extract_wikilinks, parse_note, write_note

_REQUIRED_FIELDS: frozenset[str] = frozenset({"title", "status", "tags"})
_LOW_CONFIDENCE_THRESHOLD = 0.3

# Pages excluded from orphan + link checks (meta / system pages)
_SYSTEM_STEMS = frozenset({"index", "log"})

# Inline hashtag pattern — Obsidian indexes these as tags
_INLINE_TAG_RE = re.compile(r"(?<![/\w])#([a-zA-Z][^\s#\]]*)")

# Common LLM markdown slip: reference-style link syntax with no URL, e.g.
# [astronomy] or [Zodiac] (text), which Obsidian will not resolve as a link.
_MALFORMED_BRACKET_LINK_RE = re.compile(r"(?<![!\[])\[(?!\[)([^\]\n]+)\](?![\[(])")
_MALFORMED_EMBED_RE = re.compile(r"(?<!\S)!([^\s\[]+\.(?:pdf|png|jpe?g|gif|svg|webp))", re.I)
_OBSIDIAN_EMBED_RE = re.compile(r"!\[\[[^\]]+\.(?:pdf|png|jpe?g|gif|svg|webp)\]\]", re.I)
_PLAIN_CITATION_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\](?!\()")

# Vault-internal directory names that LLMs sometimes write as wikilinks
_VAULT_DIRS = frozenset({"wiki", "raw", "source", "sources", "queries", ".drafts", ".olw"})


# ── Helpers ───────────────────────────────────────────────────────────────────


def _check_tags(
    rel_path: str,
    meta: dict,
    issues: list[LintIssue],
    fix: bool,
    page: Path,
    body: str,
) -> None:
    """Emit invalid_tag issues and optionally fix them. Shared by all page loops."""
    tags = meta.get("tags", [])
    if not isinstance(tags, list):
        issues.append(
            LintIssue(
                path=rel_path,
                issue_type="invalid_tag",
                description=f"tags field is not a list: {tags!r}",
                suggestion="Convert tags to a YAML list.",
                auto_fixable=True,
            )
        )
        if fix:
            meta["tags"] = sanitize_tags([str(tags)])
            write_note(page, meta, body)
    else:
        non_str = [t for t in tags if not isinstance(t, str)]
        str_tags = [t for t in tags if isinstance(t, str)]
        # also catch empty strings — sanitize_tags drops them but t == sanitize_tag(t) == ""
        invalid = non_str + [t for t in str_tags if not sanitize_tag(t) or t != sanitize_tag(t)]
        if invalid:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="invalid_tag",
                    description=f"Invalid tags: {', '.join(str(t) for t in invalid)}",
                    suggestion=f"Sanitized: {', '.join(sanitize_tag(str(t)) for t in invalid)}",
                    auto_fixable=True,
                )
            )
            if fix:
                meta["tags"] = sanitize_tags([str(t) for t in tags])
                write_note(page, meta, body)


def _body_hash(body: str) -> str:
    """Hash page body only (matches compile._content_hash — excludes frontmatter)."""
    return hashlib.sha256(body.encode()).hexdigest()


def _check_malformed_links(rel_path: str, body: str, issues: list[LintIssue]) -> None:
    seen: set[str] = set()
    for match in _MALFORMED_BRACKET_LINK_RE.finditer(body):
        text = match.group(1).strip()
        if not text or text in seen:
            continue
        if text.startswith("!"):
            continue
        if text.startswith("S") and re.fullmatch(r"S\d+(?:\s*,\s*S\d+)*", text):
            continue
        seen.add(text)
        issues.append(
            LintIssue(
                path=rel_path,
                issue_type="malformed_link",
                description=f"[{text}] is not a valid Markdown or Obsidian link",
                suggestion=f"Use [[{text}]] for an Obsidian link or remove the brackets.",
                auto_fixable=False,
            )
        )

    for line in body.splitlines():
        stripped = line.rstrip()
        if not stripped.endswith("[") or stripped.endswith(("[[", "![", "![[")):
            continue
        if "dangling_open_bracket" in seen:
            continue
        seen.add("dangling_open_bracket")
        issues.append(
            LintIssue(
                path=rel_path,
                issue_type="malformed_link",
                description="Dangling '[' at end of line is not a valid Markdown or Obsidian link",
                suggestion="Complete the link target or remove the trailing bracket.",
                auto_fixable=False,
            )
        )


def _check_broken_wikilinks(
    rel_path: str,
    body: str,
    title_index: dict[str, Path],
    issues: list[LintIssue],
) -> None:
    seen_broken: set[str] = set()
    for link in extract_wikilinks(body):
        if link.lower() in title_index or link.lower() in seen_broken:
            continue
        # Skip bare URLs and vault path fragments accidentally wrapped in [[...]]
        is_url = link.startswith(("http://", "https://")) or (
            "/" in link and "." in link.split("/")[0]
        )
        stripped_link = link.rstrip("/")
        is_path_fragment = stripped_link in _VAULT_DIRS or (
            not link.lower().startswith("sources/")
            and link.startswith(tuple(d + "/" for d in _VAULT_DIRS))
        )
        if is_url or is_path_fragment:
            continue
        seen_broken.add(link.lower())
        issues.append(
            LintIssue(
                path=rel_path,
                issue_type="broken_link",
                description=f"[[{link}]] has no matching wiki page",
                suggestion=f"Create a page for '{link}' or remove the link.",
                auto_fixable=False,
            )
        )


def _check_malformed_embeds(rel_path: str, body: str, issues: list[LintIssue]) -> None:
    seen: set[str] = set()
    for match in _MALFORMED_EMBED_RE.finditer(body):
        target = match.group(1).strip()
        if not target or target in seen:
            continue
        seen.add(target)
        issues.append(
            LintIssue(
                path=rel_path,
                issue_type="malformed_embed",
                description=f"!{target} is not valid Obsidian embed syntax",
                suggestion=f"Use ![[{target}]] or remove the media reference.",
                auto_fixable=True,
            )
        )


def _repair_malformed_embeds(body: str) -> str:
    return _MALFORMED_EMBED_RE.sub(lambda m: f"![[{m.group(1).strip()}]]", body)


def _repair_plain_citations(body: str) -> str:
    if "## Sources" not in body:
        return body
    before_sources, sources = body.split("## Sources", 1)
    before_sources = _PLAIN_CITATION_RE.sub(
        lambda match: f"[{match.group(1)}](#Sources)", before_sources
    )
    sources = re.sub(r"\[(S\d+(?:\s*,\s*S\d+)*)\]\(#Sources\)", r"[\1]", sources)
    return before_sources + "## Sources" + sources


def _mask_markdown_links(body: str) -> tuple[str, list[tuple[str, str]]]:
    replacements: list[tuple[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        token = f"@@OLW_LINK_{len(replacements)}@@"
        replacements.append((token, match.group(0)))
        return token

    return re.sub(r"\[[^\]\n]+\]\([^)]*\)", repl, body), replacements


def _restore_markdown_links(body: str, replacements: list[tuple[str, str]]) -> str:
    for token, original in replacements:
        body = body.replace(token, original)
    return body


def _update_article_hash(db: StateDB, rel_path: str, meta: dict, body: str) -> None:
    art = db.get_article(rel_path)
    if art is None:
        return
    db.upsert_article(
        WikiArticleRecord(
            path=art.path,
            title=art.title or str(meta.get("title", Path(rel_path).stem)),
            sources=art.sources,
            content_hash=_body_hash(body),
            is_draft=art.is_draft,
            created_at=art.created_at,
            updated_at=art.updated_at,
            approved_at=art.approved_at,
            approval_notes=art.approval_notes,
        )
    )


def _write_fixed_note(page: Path, rel_path: str, meta: dict, body: str, db: StateDB) -> None:
    write_note(page, meta, body)
    _update_article_hash(db, rel_path, meta, body)


def _title_from_file(path: Path) -> str:
    try:
        meta, _ = parse_note(path)
        return str(meta.get("title", path.stem))
    except Exception:
        return path.stem


def _normalized_graph_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.replace("-", " ").strip()).casefold()


def _add_graph_quality_issues(
    config: Config,
    title_index: dict[str, Path],
    issues: list[LintIssue],
) -> None:
    welcome = config.vault / "Welcome.md"
    if welcome.exists():
        issues.append(
            LintIssue(
                path="Welcome.md",
                issue_type="graph_noise",
                description="Obsidian starter Welcome.md appears in graph view",
                suggestion="Delete Welcome.md or filter it from graph view with -file:Welcome.",
                auto_fixable=False,
            )
        )

    if config.drafts_dir.exists() and config.pipeline.draft_media != "embed":
        for draft in sorted(config.drafts_dir.rglob("*.md")):
            try:
                _, body = parse_note(draft)
            except Exception:
                continue
            if _OBSIDIAN_EMBED_RE.search(body):
                issues.append(
                    LintIssue(
                        path=str(draft.relative_to(config.vault)),
                        issue_type="graph_noise",
                        description=(
                            "Draft contains media embeds that create attachment nodes in graph view"
                        ),
                        suggestion=(
                            'Use draft_media = "reference" or move media embeds to source pages.'
                        ),
                        auto_fixable=False,
                    )
                )

    duplicate_examples: list[tuple[Path, Path]] = []
    if config.raw_dir.exists() and config.sources_dir.exists():
        raw_titles = {
            _normalized_graph_title(_title_from_file(path)): path
            for path in config.raw_dir.rglob("*.md")
        }
        for source in sorted(config.sources_dir.glob("*.md")):
            source_title = _title_from_file(source)
            key = _normalized_graph_title(source_title)
            raw_path = raw_titles.get(key)
            if raw_path is None:
                continue
            duplicate_examples.append((source, raw_path))

    if duplicate_examples:
        example_source, example_raw = duplicate_examples[0]
        suffix = "" if len(duplicate_examples) == 1 else f" and {len(duplicate_examples) - 1} more"
        issues.append(
            LintIssue(
                path=str(example_source.relative_to(config.vault)),
                issue_type="graph_noise",
                description=(
                    "Source summary titles closely duplicate raw note titles, e.g. "
                    f"{example_raw.relative_to(config.vault)}{suffix}"
                ),
                suggestion="Filter raw/ or wiki/sources/ from Obsidian graph view.",
                auto_fixable=False,
            )
        )

    concept_targets = {
        title.lower() for title, path in title_index.items() if ".drafts" in path.parts
    }
    disconnected: list[Path] = []
    if len(concept_targets) >= 2 and config.drafts_dir.exists():
        for draft in sorted(config.drafts_dir.rglob("*.md")):
            try:
                meta, body = parse_note(draft)
            except Exception:
                continue
            own_title = str(meta.get("title", draft.stem)).lower()
            concept_links = {
                link.lower()
                for link in extract_wikilinks(body)
                if link.lower() in concept_targets and link.lower() != own_title
            }
            if not concept_links:
                disconnected.append(draft)

    if disconnected:
        example = disconnected[0]
        suffix = "" if len(disconnected) == 1 else f" and {len(disconnected) - 1} more"
        issues.append(
            LintIssue(
                path=str(example.relative_to(config.vault)),
                issue_type="graph_connectivity",
                description=(
                    "Generated drafts have no links to other concept drafts, e.g. "
                    f"{example.name}{suffix}"
                ),
                suggestion="Add related concept links or recompile after related concepts exist.",
                auto_fixable=False,
            )
        )


def _build_title_index(config: Config, db: StateDB | None = None) -> dict[str, Path]:
    """Map lowercase title/stem → path for every wiki page, including drafts.

    Also indexes frontmatter aliases and (when db provided) DB alias map.
    Ambiguous aliases (same alias → multiple pages) are excluded so they stay broken.
    """
    index: dict[str, Path] = {}
    alias_targets: dict[str, list[Path]] = {}  # alias_lower → candidate paths

    for md in config.wiki_dir.rglob("*.md"):
        index[md.stem.lower()] = md
        try:
            rel_no_suffix = str(md.relative_to(config.wiki_dir).with_suffix(""))
            index[rel_no_suffix.lower()] = md
        except ValueError:
            pass
        try:
            meta, _ = parse_note(md)
            title = meta.get("title", "")
            if title:
                index[title.lower()] = md
                base_title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
                if base_title and base_title != title:
                    alias_targets.setdefault(base_title.lower(), []).append(md)
            aliases = meta.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            elif not isinstance(aliases, list):
                aliases = []
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    alias_targets.setdefault(alias.strip().lower(), []).append(md)
        except Exception:
            pass

    # Add DB alias map: alias → canonical title → path (via title index)
    if db is not None:
        for alias_lower, canonical in db.list_alias_map().items():
            target = index.get(canonical.lower())
            if target is not None:
                alias_targets.setdefault(alias_lower, []).append(target)

    # Commit unambiguous aliases to index (don't overwrite canonical title/stem entries)
    for alias_lower, targets in alias_targets.items():
        unique = list({id(t): t for t in targets}.values())
        if len(unique) == 1 and alias_lower not in index:
            index[alias_lower] = unique[0]

    return index


def _build_inbound_index(config: Config) -> dict[str, set[str]]:
    """Map target title (lower) → set of page stems that link to it."""
    inbound: dict[str, set[str]] = {}
    for md in config.wiki_dir.rglob("*.md"):
        if ".drafts" in md.parts:
            continue
        try:
            _, body = parse_note(md)
        except Exception:
            continue
        for link in extract_wikilinks(body):
            key = link.lower()
            inbound.setdefault(key, set()).add(md.stem)
    return inbound


def _concept_pages(config: Config) -> list[Path]:
    """Root-level wiki pages that are concept articles (not system files)."""
    if not config.wiki_dir.exists():
        return []
    pages = []
    for md in sorted(config.wiki_dir.glob("*.md")):
        if md.stem.lower() in _SYSTEM_STEMS:
            continue
        pages.append(md)
    return pages


def _all_wiki_pages(config: Config) -> list[Path]:
    """All wiki pages including drafts, sources/ and queries/ (excluded: system stems)."""
    if not config.wiki_dir.exists():
        return []
    pages = []
    for md in sorted(config.wiki_dir.rglob("*.md")):
        if md.parent == config.wiki_dir and md.stem.lower() in _SYSTEM_STEMS:
            continue
        pages.append(md)
    return pages


# ── Public API ────────────────────────────────────────────────────────────────


def run_lint(config: Config, db: StateDB, fix: bool = False) -> LintResult:
    issues: list[LintIssue] = []

    # ── Config sanity checks ──────────────────────────────────────────────────
    # The default article_max_tokens was raised from 4096 to 16384 to avoid
    # silent truncation on long-form articles. Existing wiki.toml files written
    # by older `olw setup` runs still pin 4096 and won't pick up the new default.
    if config.pipeline.article_max_tokens == 4096:
        issues.append(
            LintIssue(
                path="wiki.toml",
                issue_type="config_outdated",
                description=(
                    f"pipeline.article_max_tokens is {config.pipeline.article_max_tokens} "
                    "(matches the legacy default 4096). Long articles may truncate "
                    "silently on local LLM providers."
                ),
                suggestion=(
                    "Raise to 16384 in wiki.toml [pipeline] section, or delete the line "
                    "to pick up the new default."
                ),
                auto_fixable=False,
            )
        )

    title_index = _build_title_index(config, db=db)
    inbound_index = _build_inbound_index(config)

    # DB records keyed by vault-relative path
    db_articles = {a.path: a for a in db.list_articles(drafts_only=False) if not a.is_draft}

    pages = _concept_pages(config)
    all_pages = _all_wiki_pages(config)

    for page in pages:
        rel_path = str(page.relative_to(config.vault))

        try:
            meta, body = parse_note(page)
        except Exception as exc:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="missing_frontmatter",
                    description=f"Failed to parse frontmatter: {exc}",
                    suggestion="Fix or recreate the file.",
                    auto_fixable=False,
                )
            )
            continue

        title = meta.get("title", page.stem)

        # ── Missing frontmatter ───────────────────────────────────────────────
        missing = _REQUIRED_FIELDS - set(meta.keys())
        if missing:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="missing_frontmatter",
                    description=f"Missing fields: {', '.join(sorted(missing))}",
                    suggestion=f"Add: {', '.join(f'{f}: ...' for f in sorted(missing))}",
                    auto_fixable=True,
                )
            )
            if fix:
                for field in sorted(missing):
                    if field == "title":
                        meta["title"] = page.stem
                    elif field == "status":
                        meta["status"] = "published"
                    elif field == "tags":
                        meta["tags"] = []
                write_note(page, meta, body)

        # ── Invalid tags ──────────────────────────────────────────────────────
        _check_tags(rel_path, meta, issues, fix, page, body)

        # ── Low confidence ────────────────────────────────────────────────────
        confidence = meta.get("confidence")
        if confidence is not None:
            try:
                conf_val = float(confidence)
                if conf_val < _LOW_CONFIDENCE_THRESHOLD:
                    issues.append(
                        LintIssue(
                            path=rel_path,
                            issue_type="low_confidence",
                            description=(
                                f"Confidence {conf_val:.2f} below "
                                f"threshold {_LOW_CONFIDENCE_THRESHOLD}"
                            ),
                            suggestion="Add more source notes covering this concept.",
                            auto_fixable=False,
                        )
                    )
            except (ValueError, TypeError):
                pass

        # ── Manually edited (stale hash) ──────────────────────────────────────
        db_rec = db_articles.get(rel_path)
        if db_rec:
            if _body_hash(body) != db_rec.content_hash:
                issues.append(
                    LintIssue(
                        path=rel_path,
                        issue_type="stale",
                        description="File modified manually since last compile.",
                        suggestion=(
                            "Run `olw compile --force` to recompile, "
                            "or keep edits (page is protected)."
                        ),
                        auto_fixable=False,
                    )
                )

        # ── Broken wikilinks ──────────────────────────────────────────────────
        _check_broken_wikilinks(rel_path, body, title_index, issues)

        # ── Malformed markdown links ─────────────────────────────────────────
        _check_malformed_links(rel_path, body, issues)
        _check_malformed_embeds(rel_path, body, issues)
        if fix:
            fixed_body = _repair_plain_citations(_repair_malformed_embeds(body))
            if fixed_body != body:
                body = fixed_body
                _write_fixed_note(page, rel_path, meta, body, db)

        # ── Inline hashtags ───────────────────────────────────────────────────
        masked_body, markdown_links = _mask_markdown_links(body)
        inline_tags = _INLINE_TAG_RE.findall(masked_body)
        body = _restore_markdown_links(masked_body, markdown_links)
        if inline_tags:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="inline_tag",
                    description=f"Inline #tags in body: {', '.join(f'#{t}' for t in inline_tags)}",
                    suggestion="Replace inline #tags with [[wikilinks]] or frontmatter tags.",
                    auto_fixable=False,
                )
            )

        # ── Orphan ───────────────────────────────────────────────────────────
        # Linked-by: pages that contain [[title]] or [[stem]] in their body
        linked_by = inbound_index.get(title.lower(), set()) | inbound_index.get(
            page.stem.lower(), set()
        )
        # Exclude self-links and the index page
        linked_by -= {page.stem, "index", "log"}
        if not linked_by:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="orphan",
                    description="No other wiki page links to this page.",
                    suggestion="Reference this concept from related pages or run `olw compile`.",
                    auto_fixable=False,
                )
            )

    # ── Tag + frontmatter checks for sources/ and queries/ ────────────────────
    concept_page_paths = {p for p in pages}
    for page in all_pages:
        if page in concept_page_paths:
            continue  # already checked above
        rel_path = str(page.relative_to(config.vault))
        try:
            meta, body = parse_note(page)
        except Exception as exc:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="missing_frontmatter",
                    description=f"Failed to parse frontmatter: {exc}",
                    suggestion="Fix YAML syntax in frontmatter.",
                    auto_fixable=False,
                )
            )
            continue

        # Invalid tags
        _check_tags(rel_path, meta, issues, fix, page, body)

        # Malformed links in sources/queries are useful to surface too.
        _check_malformed_links(rel_path, body, issues)
        _check_malformed_embeds(rel_path, body, issues)
        if fix:
            fixed_body = _repair_plain_citations(_repair_malformed_embeds(body))
            if fixed_body != body:
                body = fixed_body
                _write_fixed_note(page, rel_path, meta, body, db)

        # Draft/source/query links should be valid too; otherwise review sees a
        # healthy vault while pending drafts contain invented pages.
        _check_broken_wikilinks(rel_path, body, title_index, issues)

        # Missing required frontmatter
        missing = _REQUIRED_FIELDS - set(meta.keys())
        if missing:
            issues.append(
                LintIssue(
                    path=rel_path,
                    issue_type="missing_frontmatter",
                    description=f"Missing fields: {', '.join(sorted(missing))}",
                    suggestion=f"Add: {', '.join(f'{f}: ...' for f in sorted(missing))}",
                    auto_fixable=True,
                )
            )
            if fix:
                for field in sorted(missing):
                    if field == "title":
                        meta["title"] = page.stem
                    elif field == "status":
                        meta["status"] = "published"
                    elif field == "tags":
                        meta["tags"] = []
                write_note(page, meta, body)

    if config.pipeline.graph_quality_checks:
        _add_graph_quality_issues(config, title_index, issues)

    # ── Health score ──────────────────────────────────────────────────────────
    # Score based on structural wiki health. Graph-quality findings are advisory:
    # they should be visible in lint output without turning a structurally healthy
    # vault into a failing one or driving the score negative.
    total = max(len(all_pages), 1)
    advisory_issue_types = {"graph_noise", "graph_connectivity"}
    pages_with_issues = len(
        {iss.path for iss in issues if iss.issue_type not in advisory_issue_types}
    )
    score = round(100.0 * (1 - pages_with_issues / total), 1)
    score = max(0.0, min(100.0, score))

    # Summary
    if not issues:
        summary = f"Wiki healthy. {len(all_pages)} pages checked, no issues."
    else:
        counts: dict[str, int] = {}
        for iss in issues:
            counts[iss.issue_type] = counts.get(iss.issue_type, 0) + 1
        parts = [f"{v} {k}" for k, v in sorted(counts.items())]
        summary = f"{len(issues)} issue(s): {', '.join(parts)}. {len(all_pages)} pages checked."

    return LintResult(issues=issues, health_score=round(score, 1), summary=summary)
