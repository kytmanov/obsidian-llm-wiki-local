"""
Obsidian vault file operations.

Uses python-frontmatter (not regex split) so --- inside note bodies
doesn't corrupt parsing.
"""

from __future__ import annotations

import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import frontmatter

from .sanitize import sanitize_tags

# ── Frontmatter ───────────────────────────────────────────────────────────────


def parse_note(path: Path) -> tuple[dict[str, Any], str]:
    """Return (frontmatter_dict, body_text). Safe against --- in body."""
    post = frontmatter.load(str(path))
    return dict(post.metadata), post.content


def write_note(path: Path, metadata: dict[str, Any], body: str) -> None:
    """Write markdown file with YAML frontmatter atomically (crash-safe)."""
    post = frontmatter.Post(body, **metadata)
    atomic_write(path, frontmatter.dumps(post))


def update_frontmatter(path: Path, updates: dict[str, Any]) -> None:
    """Merge updates into existing frontmatter, preserve body."""
    meta, body = parse_note(path)
    meta.update(updates)
    write_note(path, meta, body)


# ── Wikilinks ─────────────────────────────────────────────────────────────────

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
# Images/media embedded via ![[file.ext]] — filter these from link extraction
_MEDIA_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".bmp",
        ".tiff",
        ".avif",
        ".mp4",
        ".webm",
        ".ogv",
        ".mov",
        ".mkv",
        ".avi",
        ".mp3",
        ".wav",
        ".ogg",
        ".flac",
        ".m4a",
        ".pdf",
        ".csv",
        ".xlsx",
        ".docx",
    }
)


def extract_wikilinks(content: str) -> list[str]:
    """Return all [[target]] titles, excluding targets with media file extensions.

    Note: filters by extension regardless of embed syntax (![[...]]) vs normal link ([[...]]).
    This prevents media filenames from appearing as broken wikilinks in lint checks.
    """
    content = re.sub(r"```[\s\S]*?```|`[^`]+`", " ", content)
    raw = _WIKILINK_RE.findall(content)
    return [t for t in raw if not any(t.lower().endswith(ext) for ext in _MEDIA_EXTENSIONS)]


def _mask_code_blocks(content: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace code blocks and image/embed syntax with placeholders.

    Protects: ```...```, `...`, ![[embed]], ![alt](url) from wikilink insertion.
    """
    # Combine: code blocks + embed/image patterns
    combined_re = re.compile(r"```[\s\S]*?```|`[^`]+`|!\[\[[^\]]+\]\]|!\[[^\]]*\]\([^)]*\)")
    replacements: list[tuple[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        token = f"__OBSIDIAN_LLM_WIKI_MASK_{len(replacements)}__"
        replacements.append((token, match.group(0)))
        return token

    return combined_re.sub(replace, content), replacements


def _restore_code_blocks(content: str, replacements: list[tuple[str, str]]) -> str:
    for token, original in replacements:
        content = content.replace(token, original)
    return content


def ensure_wikilinks(content: str, targets: list[str]) -> str:
    """
    Wrap exact whole-word title matches in [[wikilinks]].
    Skips: code blocks, already-linked titles, partial-word matches.
    Case-sensitive to avoid false positives.
    """
    if not targets:
        return content

    masked, spans = _mask_code_blocks(content)

    for target in targets:
        # Already linked? Skip.
        if f"[[{target}]]" in masked or f"[[{target}|" in masked:
            continue
        # Whole-word boundary match, case-sensitive, first occurrence only
        pattern = re.compile(r"(?<!\[)(?<!\|)\b" + re.escape(target) + r"\b(?!\])")
        masked = pattern.sub(f"[[{target}]]", masked, count=1)

    return _restore_code_blocks(masked, spans)


# ── Article utilities ─────────────────────────────────────────────────────────


def list_wiki_articles(wiki_dir: Path) -> list[tuple[str, Path]]:
    """Return [(title, path)] for all non-draft wiki articles."""
    articles = []
    for md in wiki_dir.rglob("*.md"):
        if ".drafts" in md.parts:
            continue
        try:
            meta, _ = parse_note(md)
            title = meta.get("title", md.stem)
        except Exception:
            title = md.stem
        articles.append((title, md))
    return articles


def list_draft_articles(drafts_dir: Path) -> list[tuple[str, Path]]:
    """Return [(title, path)] for all draft articles."""
    if not drafts_dir.exists():
        return []
    articles = []
    for md in drafts_dir.rglob("*.md"):
        try:
            meta, _ = parse_note(md)
            title = meta.get("title", md.stem)
            sources = meta.get("sources", [])
        except Exception:
            title = md.stem
            sources = []
        articles.append((title, md, sources))
    return articles


# ── Wikilink target safety ────────────────────────────────────────────────────

_WIKILINK_UNSAFE = re.compile(r"[\[\]|#^]")


def sanitize_wikilink_target(name: str) -> str:
    """Remove characters that break [[wikilink]] syntax: [ ] | # ^"""
    return _WIKILINK_UNSAFE.sub("", name).strip()


# ── Filename safety ───────────────────────────────────────────────────────────

_FORBIDDEN_CHARS = re.compile(r'[*"\\/<>:|?#^\[\]]')


def sanitize_filename(title: str, max_len: int = 100) -> str:
    """Strip Obsidian-forbidden chars and truncate to max_len. Never returns empty string."""
    name = _FORBIDDEN_CHARS.sub("", title).strip()
    if len(name) > max_len:
        # Truncate at word boundary
        name = name[:max_len].rsplit(" ", 1)[0]
    return name or "untitled"


def atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write content to path atomically: write .tmp then rename (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w", encoding=encoding) as f:
            f.write(content)
        Path(tmp).replace(path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def generate_aliases(title: str, source_text: str) -> list[str]:
    """
    Generate aliases from title: always add lowercase variant.
    Detect 'Full Name (ABBR)' patterns in source text and add abbreviations.
    """
    aliases: set[str] = set()
    lower = title.lower()
    if lower != title:
        aliases.add(lower)
    # Match: <title> (TWO_PLUS_CAPS)
    pattern = re.compile(re.escape(title) + r"\s*\(([A-Z]{2,})\)")
    for m in pattern.finditer(source_text):
        aliases.add(m.group(1))
    return sorted(aliases)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """
    Split text into chunks. Primary split on ## headings; fallback to
    word-count windows with overlap.
    """
    # Try heading-based split first
    sections = re.split(r"\n(?=## )", text)
    chunks: list[str] = []

    for section in sections:
        words = section.split()
        if len(words) <= chunk_size:
            if section.strip():
                chunks.append(section.strip())
        else:
            # Sliding window
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i : i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk.strip())

    return chunks if chunks else [text.strip()]


def build_wiki_frontmatter(
    title: str,
    tags: list[str],
    sources: list[str],
    confidence: float,
    is_draft: bool = True,
    existing_meta: dict | None = None,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    now = datetime.now().strftime("%Y-%m-%d")
    meta: dict[str, Any] = {
        "title": title,
        "tags": sanitize_tags(tags),
        "sources": sources,
        "confidence": round(confidence, 2),
        "status": "draft" if is_draft else "published",
        "updated": now,
    }
    if aliases:
        meta["aliases"] = aliases
    if existing_meta and "created" in existing_meta:
        meta["created"] = existing_meta["created"]
    else:
        meta["created"] = now
    return meta


_WIKILINK_FULL_RE = re.compile(r"\[\[([^\]|#]+?)(?:#([^\]|]*))?(?:\|([^\]]*))?\]\]")


def normalize_wikilinks(body: str, alias_map: dict[str, str], known_titles: set[str]) -> str:
    """Rewrite [[Alias]] → [[Canonical|Alias]] for unambiguous aliases.

    - If target is already a canonical title: leave unchanged.
    - If target matches an unambiguous alias: rewrite target to canonical, preserve display.
    - If ambiguous or unknown: leave unchanged (lint will flag).
    - Fragments (#) and display text (|) are preserved.
    - Code blocks are protected from rewrites.
    """
    masked, spans = _mask_code_blocks(body)
    known_lower = {t.lower() for t in known_titles}

    def _rewrite(m: re.Match) -> str:
        target = m.group(1).strip()
        fragment = m.group(2)  # may be None
        display = m.group(3)  # may be None

        # Already a known canonical title → leave unchanged
        if target.lower() in known_lower:
            return m.group(0)

        # Try alias map
        canonical = alias_map.get(target.lower())
        if canonical is None:
            return m.group(0)  # unknown / ambiguous

        # Build rewritten link preserving display and fragment
        effective_display = display if display is not None else target
        frag_part = f"#{fragment}" if fragment else ""
        return f"[[{canonical}{frag_part}|{effective_display}]]"

    rewritten = _WIKILINK_FULL_RE.sub(_rewrite, masked)
    return _restore_code_blocks(rewritten, spans)
