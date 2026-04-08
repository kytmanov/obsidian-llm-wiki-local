"""Tests for vault.py — pure functions, no LLM required."""

from __future__ import annotations

from obsidian_llm_wiki.vault import (
    atomic_write,
    build_wiki_frontmatter,
    chunk_text,
    ensure_wikilinks,
    extract_wikilinks,
    generate_aliases,
    parse_note,
    sanitize_filename,
    sanitize_wikilink_target,
    write_note,
)

# ── parse_note ────────────────────────────────────────────────────────────────


def test_parse_note_with_frontmatter(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("---\ntitle: Test\ntags: [a, b]\n---\n\nBody text here.")
    meta, body = parse_note(p)
    assert meta["title"] == "Test"
    assert meta["tags"] == ["a", "b"]
    assert "Body text here" in body


def test_parse_note_no_frontmatter(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("Just body text, no frontmatter.")
    meta, body = parse_note(p)
    assert meta == {}
    assert "Just body text" in body


def test_parse_note_dashes_in_body(tmp_path):
    """python-frontmatter must not get confused by --- in body."""
    p = tmp_path / "note.md"
    p.write_text("---\ntitle: Test\n---\n\nHeader\n---\nSeparator above.")
    meta, body = parse_note(p)
    assert meta["title"] == "Test"
    assert "Separator above" in body


def test_write_note_roundtrip(tmp_path):
    p = tmp_path / "out.md"
    write_note(p, {"title": "Hello", "tags": ["x"]}, "Body content.")
    meta, body = parse_note(p)
    assert meta["title"] == "Hello"
    assert "Body content" in body


# ── wikilinks ─────────────────────────────────────────────────────────────────


def test_extract_wikilinks():
    content = "See [[Quantum Entanglement]] and [[Bell States|Bell's theorem]]."
    links = extract_wikilinks(content)
    assert "Quantum Entanglement" in links
    assert "Bell States" in links


def test_extract_wikilinks_excludes_image_embeds():
    content = "![[photo.png]] and [[Real Link]]"
    assert extract_wikilinks(content) == ["Real Link"]


def test_extract_wikilinks_excludes_pdf():
    assert extract_wikilinks("![[doc.pdf]]") == []


def test_extract_wikilinks_keeps_note_transclusion():
    """![[other-note]] (no media extension) = note transclusion, keep it."""
    assert extract_wikilinks("![[other-note]]") == ["other-note"]


def test_extract_wikilinks_excludes_jpg():
    assert extract_wikilinks("![[image.jpg]]") == []


def test_ensure_wikilinks_no_mangle_image_alt():
    """![Quantum Computing](img.png) must not become ![[[ Quantum Computing]]](img.png)."""
    content = "See ![Quantum Computing](img.png) for details."
    result = ensure_wikilinks(content, ["Quantum Computing"])
    assert "![Quantum Computing](img.png)" in result
    assert "![[[" not in result


def test_ensure_wikilinks_no_mangle_obsidian_embed():
    """![[Quantum Computing]] must not become ![[[[Quantum Computing]]]]."""
    content = "See ![[Quantum Computing]] for details."
    result = ensure_wikilinks(content, ["Quantum Computing"])
    assert "![[Quantum Computing]]" in result
    assert "![[[[" not in result


def test_ensure_wikilinks_basic():
    content = "Quantum Entanglement is a physical phenomenon."
    result = ensure_wikilinks(content, ["Quantum Entanglement"])
    assert "[[Quantum Entanglement]]" in result


def test_ensure_wikilinks_no_double_wrap():
    content = "See [[Quantum Entanglement]] already."
    result = ensure_wikilinks(content, ["Quantum Entanglement"])
    assert result.count("[[Quantum Entanglement]]") == 1


def test_ensure_wikilinks_word_boundary():
    """Should not wrap partial matches."""
    content = "Python scripting is used here."
    result = ensure_wikilinks(content, ["Python"])
    # "Python" is a standalone word here — should link
    assert "[[Python]]" in result


def test_ensure_wikilinks_no_substring_in_word():
    """Should NOT wrap 'Python' inside 'CPython'."""
    content = "CPython is the reference implementation."
    result = ensure_wikilinks(content, ["Python"])
    assert "[[Python]]" not in result
    assert "CPython" in result


def test_ensure_wikilinks_skip_code_blocks():
    content = "Use `Python` in code. Python is great."
    result = ensure_wikilinks(content, ["Python"])
    # Should only link the second "Python", not the one in backticks
    assert "`Python`" in result or "`[[Python]]`" not in result


def test_ensure_wikilinks_empty_targets():
    content = "Some text here."
    assert ensure_wikilinks(content, []) == content


# ── chunk_text ────────────────────────────────────────────────────────────────

# ── sanitize_filename ─────────────────────────────────────────────────────────


def test_sanitize_filename_strips_forbidden():
    assert sanitize_filename('A*B"C/D') == "ABCD"


def test_sanitize_filename_max_len():
    long_title = "word " * 30  # 150 chars
    result = sanitize_filename(long_title.strip(), max_len=20)
    assert len(result) <= 20


def test_sanitize_filename_empty_becomes_untitled():
    assert sanitize_filename("***///") == "untitled"


def test_sanitize_filename_normal():
    assert sanitize_filename("Quantum Computing") == "Quantum Computing"


# ── atomic_write ──────────────────────────────────────────────────────────────


def test_atomic_write_creates_file(tmp_path):
    p = tmp_path / "out.md"
    atomic_write(p, "hello world")
    assert p.read_text() == "hello world"


def test_atomic_write_overwrites(tmp_path):
    p = tmp_path / "out.md"
    p.write_text("old")
    atomic_write(p, "new")
    assert p.read_text() == "new"


def test_atomic_write_no_tmp_left(tmp_path):
    p = tmp_path / "out.md"
    atomic_write(p, "content")
    tmps = list(tmp_path.glob("*.tmp"))
    assert tmps == []


# ── generate_aliases ──────────────────────────────────────────────────────────


def test_generate_aliases_lowercase():
    aliases = generate_aliases("Quantum Computing", "some text")
    assert "quantum computing" in aliases


def test_generate_aliases_same_case_no_duplicate():
    aliases = generate_aliases("quantum computing", "some text")
    assert "quantum computing" not in aliases  # title == lower, skip


def test_generate_aliases_abbreviation():
    text = "Quantum Computing (QC) is fascinating."
    aliases = generate_aliases("Quantum Computing", text)
    assert "QC" in aliases


def test_generate_aliases_multiple_abbreviations():
    text = "Machine Learning (ML) and Deep Learning (DL) are related."
    aliases = generate_aliases("Machine Learning", text)
    assert "ML" in aliases
    assert "DL" not in aliases  # only matches "Machine Learning (..."


# ── sanitize_wikilink_target ──────────────────────────────────────────────────


def test_sanitize_wikilink_target_strips_closing_bracket():
    assert sanitize_wikilink_target("foo]bar") == "foobar"


def test_sanitize_wikilink_target_strips_opening_bracket():
    assert sanitize_wikilink_target("foo[bar") == "foobar"


def test_sanitize_wikilink_target_strips_pipe():
    assert sanitize_wikilink_target("A|B") == "AB"


def test_sanitize_wikilink_target_strips_hash():
    assert sanitize_wikilink_target("title#section") == "titlesection"


def test_sanitize_wikilink_target_passthrough():
    assert sanitize_wikilink_target("Normal Title") == "Normal Title"


def test_sanitize_wikilink_target_preserves_colon():
    # Colons are fine inside wikilinks
    assert sanitize_wikilink_target("Python: Guide") == "Python: Guide"


# ── build_wiki_frontmatter ────────────────────────────────────────────────────


def test_build_wiki_frontmatter_sanitizes_tags():
    meta = build_wiki_frontmatter(
        title="Test",
        tags=["quantum computing", "C++ stuff"],
        sources=[],
        confidence=0.8,
    )
    assert meta["tags"] == ["quantum-computing", "c-stuff"]


def test_build_wiki_frontmatter_preserves_valid_tags():
    meta = build_wiki_frontmatter(
        title="Test",
        tags=["physics", "ai"],
        sources=[],
        confidence=0.8,
    )
    assert meta["tags"] == ["physics", "ai"]


def test_build_wiki_frontmatter_deduplicates_tags():
    meta = build_wiki_frontmatter(
        title="Test",
        tags=["AI", "ai", "machine learning", "machine-learning"],
        sources=[],
        confidence=0.5,
    )
    assert meta["tags"] == ["ai", "machine-learning"]


def test_chunk_text_heading_split():
    text = (
        "# Title\n\nIntro paragraph.\n\n## Section 1\n\n"
        "Content one.\n\n## Section 2\n\nContent two."
    )
    chunks = chunk_text(text, chunk_size=500)
    assert len(chunks) >= 1


def test_chunk_text_sliding_window():
    # Generate text longer than chunk_size words
    words = ["word"] * 1000
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # All chunks should be non-empty
    assert all(c.strip() for c in chunks)


def test_chunk_text_short_note():
    text = "Short note."
    chunks = chunk_text(text, chunk_size=512)
    assert chunks == ["Short note."]
