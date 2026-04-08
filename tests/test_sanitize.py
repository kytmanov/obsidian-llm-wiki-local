"""Tests for sanitize.py — pure functions, no I/O."""

from __future__ import annotations

from obsidian_llm_wiki.sanitize import sanitize_tag, sanitize_tags

# ── sanitize_tag ──────────────────────────────────────────────────────────────


def test_spaces_to_hyphens():
    assert sanitize_tag("quantum computing") == "quantum-computing"


def test_multiple_spaces():
    assert sanitize_tag("machine learning basics") == "machine-learning-basics"


def test_cpp_special_chars_removed():
    assert sanitize_tag("C++ programming") == "c-programming"


def test_leading_hash_stripped():
    assert sanitize_tag("#my-tag") == "my-tag"


def test_valid_tag_passthrough():
    assert sanitize_tag("physics") == "physics"


def test_garbage_returns_empty():
    assert sanitize_tag("!!!") == ""


def test_slashes_preserved():
    assert sanitize_tag("science/physics") == "science/physics"


def test_pure_numbers_valid():
    assert sanitize_tag("2024") == "2024"


def test_underscores_preserved():
    assert sanitize_tag("my_tag") == "my_tag"


def test_hyphens_preserved():
    assert sanitize_tag("already-hyphenated") == "already-hyphenated"


def test_mixed_case_lowercased():
    assert sanitize_tag("MachineLearning") == "machinelearning"


def test_uppercase_tag_lowercased():
    assert sanitize_tag("AI") == "ai"


def test_leading_hyphen_stripped():
    assert sanitize_tag("-bad-start") == "bad-start"


def test_leading_underscore_stripped():
    assert sanitize_tag("_bad-start") == "bad-start"


def test_whitespace_only_returns_empty():
    assert sanitize_tag("   ") == ""


def test_empty_string_returns_empty():
    assert sanitize_tag("") == ""


def test_unicode_special_chars_removed():
    assert sanitize_tag("café") == "caf"


def test_at_symbol_removed():
    assert sanitize_tag("tag@user") == "taguser"


# ── sanitize_tags ─────────────────────────────────────────────────────────────


def test_dedup_case_insensitive():
    result = sanitize_tags(["AI", "ai"])
    assert result == ["ai"]


def test_filters_empty_results():
    result = sanitize_tags(["!!!", "valid"])
    assert result == ["valid"]


def test_preserves_order():
    result = sanitize_tags(["beta", "alpha", "gamma"])
    assert result == ["beta", "alpha", "gamma"]


def test_filters_all_garbage():
    result = sanitize_tags(["!!!", "@@@", "###"])
    assert result == []


def test_empty_list():
    assert sanitize_tags([]) == []


def test_mixed_valid_and_invalid():
    result = sanitize_tags(["quantum computing", "!!!", "physics"])
    assert result == ["quantum-computing", "physics"]


def test_no_duplicate_after_sanitize():
    # Both sanitize to "machine-learning"
    result = sanitize_tags(["machine learning", "machine-learning"])
    assert result == ["machine-learning"]
