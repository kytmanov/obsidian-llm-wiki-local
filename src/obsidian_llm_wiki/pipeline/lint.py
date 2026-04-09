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
from ..models import LintIssue, LintResult
from ..sanitize import sanitize_tag, sanitize_tags
from ..state import StateDB
from ..vault import extract_wikilinks, parse_note, write_note

_REQUIRED_FIELDS: frozenset[str] = frozenset({"title", "status", "tags"})
_LOW_CONFIDENCE_THRESHOLD = 0.3

# Pages excluded from orphan + link checks (meta / system pages)
_SYSTEM_STEMS = frozenset({"index", "log"})

# Inline hashtag pattern — Obsidian indexes these as tags
_INLINE_TAG_RE = re.compile(r"(?<![/\w])#([a-zA-Z][^\s#\]]*)")

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


def _build_title_index(config: Config) -> dict[str, Path]:
    """Map lowercase title/stem → path for every published wiki page."""
    index: dict[str, Path] = {}
    for md in config.wiki_dir.rglob("*.md"):
        if ".drafts" in md.parts:
            continue
        index[md.stem.lower()] = md
        try:
            meta, _ = parse_note(md)
            title = meta.get("title", "")
            if title:
                index[title.lower()] = md
        except Exception:
            pass
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
    """All wiki pages including sources/ and queries/ (excluded: drafts, system stems)."""
    if not config.wiki_dir.exists():
        return []
    pages = []
    for md in sorted(config.wiki_dir.rglob("*.md")):
        if ".drafts" in md.parts:
            continue
        if md.parent == config.wiki_dir and md.stem.lower() in _SYSTEM_STEMS:
            continue
        pages.append(md)
    return pages


# ── Public API ────────────────────────────────────────────────────────────────


def run_lint(config: Config, db: StateDB, fix: bool = False) -> LintResult:
    issues: list[LintIssue] = []

    title_index = _build_title_index(config)
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
        seen_broken: set[str] = set()
        for link in extract_wikilinks(body):
            if link.lower() in title_index or link.lower() in seen_broken:
                continue
            # Skip bare URLs and vault path fragments accidentally wrapped in [[...]]
            is_url = link.startswith(("http://", "https://")) or (
                "/" in link and "." in link.split("/")[0]
            )
            is_path_fragment = link.rstrip("/") in _VAULT_DIRS or link.startswith(
                tuple(d + "/" for d in _VAULT_DIRS)
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

        # ── Inline hashtags ───────────────────────────────────────────────────
        inline_tags = _INLINE_TAG_RE.findall(body)
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

    # ── Health score ──────────────────────────────────────────────────────────
    # Score based on % of clean pages across all checked pages
    total = max(len(all_pages), 1)
    pages_with_issues = len({iss.path for iss in issues})
    score = round(100.0 * (1 - pages_with_issues / total), 1)

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
