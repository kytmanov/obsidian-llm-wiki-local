"""
Wiki maintenance — self-initiated health operations.

Used by `olw maintain` to:
  - Create stub drafts for broken wikilinks
  - Suggest orphan link fixes
  - Suggest concept merges for near-duplicates
  - Report source quality distribution
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import frontmatter as fm_lib

from ..config import Config
from ..models import LintIssue
from ..state import StateDB
from ..vault import (
    _mask_code_blocks,
    _restore_code_blocks,
    atomic_write,
    list_wiki_articles,
    normalize_wikilinks,
    parse_note,
    sanitize_filename,
    write_note,
)

log = logging.getLogger(__name__)

_STUB_BODY = """\
> [!info] This is a stub article — referenced by other pages but no source material yet.

Add raw notes about this topic to `raw/` and run `olw compile` to generate a full article.
"""

_CONCEPT_MERGE_THRESHOLD = 0.7

_WIKILINK_REPAIR_RE = re.compile(r"\[\[([^\]|#]+?)(?:#([^\]|]*))?(?:\|([^\]]*))?\]\]")


@dataclass
class FixReport:
    repaired: int = 0
    repaired_links: list[tuple[str, str, str]] = field(default_factory=list)
    still_broken: list[LintIssue] = field(default_factory=list)
    skipped_files: list[Path] = field(default_factory=list)


def fix_broken_links(
    config: Config,
    db: StateDB,
    broken_link_issues: list[LintIssue],
    dry_run: bool = False,
) -> FixReport:
    """Rewrite broken [[Alias]] links to [[Canonical|Alias]] using the alias map.

    Only unambiguous matches are rewritten. Ambiguous or unknown links fall through
    to still_broken for stub creation.
    """
    report = FixReport()

    if not broken_link_issues:
        return report

    alias_map = db.list_alias_map()
    if not alias_map:
        report.still_broken = list(broken_link_issues)
        return report

    # Build set of known published article titles (lowercase) so we don't rewrite good links
    known_lower = {t.lower() for t, _ in list_wiki_articles(config.wiki_dir)}

    # Group issues by source file for one write per file
    issues_by_file: dict[str, list[LintIssue]] = {}
    for issue in broken_link_issues:
        issues_by_file.setdefault(issue.path, []).append(issue)

    for rel_path, file_issues in issues_by_file.items():
        page = config.vault / rel_path
        try:
            meta, body = parse_note(page)
        except Exception as exc:
            log.warning("fix_broken_links: skipping %s — parse error: %s", rel_path, exc)
            report.skipped_files.append(page)
            report.still_broken.extend(file_issues)
            continue

        original_body = body
        # Mask code fences so repair doesn't rewrite [[...]] inside ```code``` or `inline`.
        masked_body, spans = _mask_code_blocks(body)
        repaired_in_file: list[tuple[str, str]] = []
        still_broken_in_file: list[LintIssue] = []

        for issue in file_issues:
            target = _extract_link_target(issue.description)
            if target is None:
                still_broken_in_file.append(issue)
                continue

            # Skip if target is already a known title (shouldn't be in broken list, but guard)
            if target.lower() in known_lower:
                still_broken_in_file.append(issue)
                continue

            canonical = alias_map.get(target.lower())
            if canonical is None or canonical.lower() not in known_lower:
                still_broken_in_file.append(issue)
                continue

            # Rewrite all occurrences of [[target ...]] → [[canonical|target]]
            def _make_rewriter(t: str, canon: str):
                def _rewrite(m: re.Match) -> str:
                    if m.group(1).strip().lower() != t.lower():
                        return m.group(0)
                    fragment = m.group(2)
                    display = m.group(3) if m.group(3) is not None else m.group(1).strip()
                    frag_part = f"#{fragment}" if fragment else ""
                    return f"[[{canon}{frag_part}|{display}]]"

                return _rewrite

            new_masked = _WIKILINK_REPAIR_RE.sub(_make_rewriter(target, canonical), masked_body)
            if new_masked != masked_body:
                repaired_in_file.append((f"[[{target}]]", f"[[{canonical}|{target}]]"))
                masked_body = new_masked
            else:
                still_broken_in_file.append(issue)

        body = _restore_code_blocks(masked_body, spans)

        if body != original_body:
            if dry_run:
                for old, new in repaired_in_file:
                    log.info("dry-run: would rewrite %s → %s in %s", old, new, rel_path)
            else:
                write_note(page, meta, body)
            for old, new in repaired_in_file:
                report.repaired += 1
                report.repaired_links.append((rel_path, old, new))

        report.still_broken.extend(still_broken_in_file)

    return report


def normalize_published_alias_links(
    config: Config,
    db: StateDB,
    dry_run: bool = False,
) -> int:
    """Rewrite alias-form [[Alias]] links in published articles to [[Canonical|Alias]].

    Complements compile's normalize_wikilinks pass: articles published before an alias
    was registered never got that normalization. This pass runs on all published articles.

    Only unambiguous alias rewrites are applied. Returns number of files modified.
    """
    alias_map = db.list_alias_map()
    if not alias_map:
        return 0

    known_titles = {t.lower() for t, _ in list_wiki_articles(config.wiki_dir)}
    modified = 0

    for _title, path in list_wiki_articles(config.wiki_dir):
        try:
            meta, body = parse_note(path)
        except Exception as exc:
            log.warning("normalize_published_alias_links: skipping %s — %s", path.name, exc)
            continue

        new_body = normalize_wikilinks(body, alias_map, known_titles)
        if new_body == body:
            continue

        if dry_run:
            log.info("dry-run: would normalize alias links in %s", path.name)
        else:
            write_note(path, meta, new_body)
            log.info("Normalized alias links in %s", path.name)
        modified += 1

    return modified


def create_stubs(
    config: Config,
    db: StateDB,
    broken_link_issues: list[LintIssue] | None = None,
    max_stubs: int = 5,
) -> list[Path]:
    """
    Create stub drafts for broken wikilinks.

    Finds [[Target]] references that have no matching article, creates placeholder
    drafts and registers them in the stubs table so compile can pick them up.

    Pass broken_link_issues to avoid re-running lint. If None, lint runs internally.
    """
    if broken_link_issues is None:
        from .lint import run_lint

        result = run_lint(config, db)
        broken_link_issues = [i for i in result.issues if i.issue_type == "broken_link"]

    # Extract target concept names from broken link descriptions
    # LintIssue description format: "[[Target]] not found" or similar
    created: list[Path] = []
    seen: set[str] = set()

    for issue in broken_link_issues:
        if len(created) >= max_stubs:
            log.info("Stub cap (%d) reached — stopping", max_stubs)
            break

        # Extract target from description (e.g. "[[Quantum Computing]] not found")
        target = _extract_link_target(issue.description)
        if not target:
            # Fall back to path stem
            target = Path(issue.path).stem
        # Strip trailing .md — model sometimes generates [[raw-note.md]] wikilinks
        if target.lower().endswith(".md"):
            target = target[:-3]
        if not target:
            continue

        if target in seen:
            continue
        seen.add(target)

        # Skip if already has a stub, draft, or published article
        if db.has_stub(target):
            continue
        safe_name = sanitize_filename(target)
        draft_path = config.drafts_dir / f"{safe_name}.md"
        wiki_path = config.wiki_dir / f"{safe_name}.md"
        if draft_path.exists() or wiki_path.exists():
            continue

        # Register in stubs table
        db.add_stub(target, source="auto")

        # Write placeholder draft
        config.drafts_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "title": target,
            "status": "stub",
            "tags": ["stub"],
            "sources": [],
            "confidence": 0.0,
            "created": datetime.now().strftime("%Y-%m-%d"),
            "updated": datetime.now().strftime("%Y-%m-%d"),
        }
        post = fm_lib.Post(_STUB_BODY, **meta)
        atomic_write(draft_path, fm_lib.dumps(post))
        created.append(draft_path)
        log.info("Stub created: %s", draft_path.name)

    return created


def _extract_link_target(description: str) -> str | None:
    """Extract [[Target]] from a lint issue description string."""
    match = re.search(r"\[\[([^\]]+)\]\]", description)
    return match.group(1) if match else None


def suggest_orphan_links(config: Config, db: StateDB) -> list[tuple[str, list[str]]]:
    """
    For each orphan article, find other articles that mention its title unlinked.

    Returns list of (orphan_title, [paths_that_mention_it]).
    """
    from .lint import run_lint

    result = run_lint(config, db)
    orphan_issues = [i for i in result.issues if i.issue_type == "orphan"]
    if not orphan_issues:
        return []

    # Load all published article bodies
    wiki_pages: dict[str, str] = {}
    if config.wiki_dir.exists():
        for p in config.wiki_dir.rglob("*.md"):
            if ".drafts" in p.parts:
                continue
            try:
                meta, body = parse_note(p)
                wiki_pages[str(p.relative_to(config.vault))] = body
            except Exception:
                pass

    suggestions = []
    for issue in orphan_issues:
        orphan_path = config.vault / issue.path
        try:
            meta, _ = parse_note(orphan_path)
            orphan_title = meta.get("title", orphan_path.stem)
        except Exception:
            orphan_title = orphan_path.stem

        # Find pages that mention the orphan title in plain text (not as wikilink)
        mentions = []
        title_pattern = re.compile(
            r"(?<!\[\[)\b" + re.escape(orphan_title) + r"\b(?!\]\])",
            re.IGNORECASE,
        )
        for page_path, body in wiki_pages.items():
            if page_path == issue.path:
                continue
            if title_pattern.search(body):
                mentions.append(page_path)

        if mentions:
            suggestions.append((orphan_title, mentions))

    return suggestions


def suggest_concept_merges(config: Config, db: StateDB) -> list[tuple[str, str, float]]:
    """
    Find near-duplicate concept pairs using Jaccard similarity on title tokens.

    Returns list of (concept_a, concept_b, similarity_score) for pairs > threshold.
    No LLM required — purely token-overlap based.
    """
    concepts = db.list_all_concept_names()
    if len(concepts) < 2:
        return []

    def tokenize(name: str) -> frozenset[str]:
        # Lowercase, split on spaces/hyphens/underscores, filter short tokens
        tokens = re.split(r"[\s\-_]+", name.lower())
        return frozenset(t for t in tokens if len(t) > 1)

    tokenized = [(c, tokenize(c)) for c in concepts]
    suggestions = []

    for i, (a, tokens_a) in enumerate(tokenized):
        for b, tokens_b in tokenized[i + 1 :]:
            if not tokens_a or not tokens_b:
                continue
            intersection = len(tokens_a & tokens_b)
            union = len(tokens_a | tokens_b)
            if union == 0:
                continue
            score = intersection / union
            if score >= _CONCEPT_MERGE_THRESHOLD:
                suggestions.append((a, b, round(score, 2)))

    suggestions.sort(key=lambda x: -x[2])
    return suggestions
