"""
Tests for structured_output.py — the most critical module.
All tests use mocked OllamaClient; no Ollama required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.models import AnalysisResult, CompilePlan, SingleArticle
from obsidian_llm_wiki.ollama_client import OllamaClient
from obsidian_llm_wiki.structured_output import (
    StructuredOutputError,
    _extract_json,
    _unwrap,
    request_structured,
)


def _client(response: str) -> OllamaClient:
    c = MagicMock(spec=OllamaClient)
    c.generate.return_value = response
    return c


def _load_fixture(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / name).read_text()


# ── Tier 1: direct JSON parse ──────────────────────────────────────────────────


def test_valid_analysis_json(fixtures_dir):
    raw = (fixtures_dir / "analysis_valid.json").read_text()
    result = request_structured(
        client=_client(raw),
        prompt="analyze",
        model_class=AnalysisResult,
        model="gemma4:e4b",
    )
    assert result.quality == "high"
    assert "quantum entanglement" in result.key_concepts
    assert len(result.suggested_topics) > 0


def test_valid_compile_plan(fixtures_dir):
    raw = (fixtures_dir / "compile_plan_valid.json").read_text()
    result = request_structured(
        client=_client(raw),
        prompt="plan",
        model_class=CompilePlan,
        model="gemma4:e4b",
    )
    assert len(result.articles) == 1
    assert result.articles[0].action == "create"


def test_valid_single_article(fixtures_dir):
    raw = (fixtures_dir / "single_article_valid.json").read_text()
    result = request_structured(
        client=_client(raw),
        prompt="write",
        model_class=SingleArticle,
        model="qwen2.5:14b",
    )
    assert result.title == "Quantum Entanglement"
    assert "quantum-physics" in result.tags
    assert "## Overview" in result.content


# ── Tier 2: extract from fenced blocks ────────────────────────────────────────


def test_fenced_json_extraction(fixtures_dir):
    inner = (fixtures_dir / "analysis_valid.json").read_text()
    wrapped = f"Here is the analysis:\n\n```json\n{inner}\n```\n\nDone."
    result = request_structured(
        client=_client(wrapped),
        prompt="analyze",
        model_class=AnalysisResult,
        model="gemma4:e4b",
    )
    assert result.quality == "high"


def test_bare_json_in_prose(fixtures_dir):
    inner = (fixtures_dir / "analysis_valid.json").read_text()
    wrapped = f"Sure, here you go:\n{inner}\nHope that helps!"
    result = request_structured(
        client=_client(wrapped),
        prompt="analyze",
        model_class=AnalysisResult,
        model="gemma4:e4b",
    )
    assert result.quality == "high"


# ── Tier 3: retry on failure ───────────────────────────────────────────────────


def test_retry_on_invalid_json(fixtures_dir):
    valid = (fixtures_dir / "analysis_valid.json").read_text()
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "not json at all"
        return valid

    c = MagicMock(spec=OllamaClient)
    c.generate.side_effect = lambda **kwargs: side_effect(**kwargs)

    result = request_structured(
        client=c,
        prompt="analyze",
        model_class=AnalysisResult,
        model="gemma4:e4b",
        max_retries=2,
    )
    assert result.quality == "high"
    assert call_count == 2  # failed once, succeeded on retry


def test_exhausted_retries_raises():
    c = _client("this is never valid json !!!!")
    with pytest.raises(StructuredOutputError):
        request_structured(
            client=c,
            prompt="analyze",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            max_retries=1,
        )


def test_schema_validation_failure():
    # Valid JSON but wrong schema (missing required fields)
    bad = json.dumps({"wrong_field": "value"})
    c = _client(bad)
    with pytest.raises(StructuredOutputError):
        request_structured(
            client=c,
            prompt="analyze",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            max_retries=0,
        )


def test_single_article_missing_required_field():
    # Missing required 'content' field should fail validation
    bad = json.dumps(
        {
            "title": "Test",
            "tags": [],
            # content missing
        }
    )
    c = _client(bad)
    with pytest.raises(StructuredOutputError):
        request_structured(
            client=c,
            prompt="write",
            model_class=SingleArticle,
            model="qwen2.5:14b",
            max_retries=0,
        )


# ── Tier 2b: bare ``` fenced block extraction ─────────────────────────────


def test_bare_fenced_block_extraction(fixtures_dir):
    """Bare ``` block (no json tag) triggers line 61."""
    inner = (fixtures_dir / "analysis_valid.json").read_text()
    wrapped = f"Here is the result:\n\n```\n{inner}\n```\n\nDone."
    result = request_structured(
        client=_client(wrapped),
        prompt="analyze",
        model_class=AnalysisResult,
        model="gemma4:e4b",
    )
    assert result.quality == "high"
    assert "quantum entanglement" in result.key_concepts


# ── _unwrap direct tests ──────────────────────────────────────────────────


def test_unwrap_non_dict_returns_as_is():
    """Line 79: non-dict input is returned unchanged."""
    assert _unwrap("hello", AnalysisResult) == "hello"
    assert _unwrap(42, AnalysisResult) == 42
    assert _unwrap([1, 2], AnalysisResult) == [1, 2]


def test_unwrap_single_key_wrapper_with_dict_value():
    """Line 86: single-key wrapper whose value is a dict."""
    inner = {"summary": "s", "quality": "high"}
    wrapped = {"AnalysisResult": inner}
    assert _unwrap(wrapped, AnalysisResult) == inner


def test_unwrap_json_schema_echo():
    """Lines 91-94: JSON Schema echo with flat property values."""
    data = {
        "description": "Analysis result",
        "properties": {
            "summary": "A short summary",
            "quality": "high",
        },
    }
    result = _unwrap(data, AnalysisResult)
    assert result == {"summary": "A short summary", "quality": "high"}


def test_unwrap_json_schema_echo_skipped_for_real_schema():
    """Lines 91-94: real schema dicts with 'type' are NOT unwrapped."""
    data = {
        "description": "Analysis result",
        "properties": {
            "summary": {"type": "string", "description": "s"},
            "quality": {"type": "string", "description": "q"},
        },
    }
    result = _unwrap(data, AnalysisResult)
    assert result is data  # unchanged
