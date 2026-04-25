"""
Tests for structured_output.py — the most critical module.
All tests use mocked OllamaClient; no Ollama required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.models import AnalysisResult, CompilePlan, LintResult, SingleArticle
from obsidian_llm_wiki.ollama_client import OllamaClient
from obsidian_llm_wiki.structured_output import (
    StructuredOutputError,
    _make_template,
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
    assert any(c.name == "quantum entanglement" for c in result.concepts)
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


def test_single_article_sanitizes_tags():
    """SingleArticle validator cleans tags at parse time."""
    article = SingleArticle(title="X", content="Y", tags=["bad tag", "C++", "physics"])
    assert "bad-tag" in article.tags
    assert "c" in article.tags
    assert "physics" in article.tags
    # Original dirty values gone
    assert "bad tag" not in article.tags
    assert "C++" not in article.tags


def test_num_predict_passed_to_generate():
    """num_predict forwarded to client.generate so output isn't truncated mid-JSON."""
    raw = json.dumps({"title": "T", "content": "body", "tags": ["t"]})
    c = _client(raw)
    request_structured(
        client=c,
        prompt="write",
        model_class=SingleArticle,
        model="qwen2.5:14b",
        num_ctx=16384,
        num_predict=8192,
    )
    _, kwargs = c.generate.call_args
    assert kwargs.get("num_predict") == 8192


def test_num_predict_default_is_minus_one():
    """Default num_predict=-1 means unlimited — Ollama generates until stop token."""
    raw = json.dumps({"title": "T", "content": "body", "tags": ["t"]})
    c = _client(raw)
    request_structured(
        client=c,
        prompt="write",
        model_class=SingleArticle,
        model="qwen2.5:14b",
    )
    _, kwargs = c.generate.call_args
    assert kwargs.get("num_predict") == -1


def test_truncated_json_fails_all_retries():
    """Truncated JSON (output cut off mid-string) exhausts retries and raises."""
    truncated = '{"title": "T", "content": "body that got cut off'
    c = _client(truncated)
    with pytest.raises(StructuredOutputError, match="Invalid JSON"):
        request_structured(
            client=c,
            prompt="write",
            model_class=SingleArticle,
            model="qwen2.5:14b",
            max_retries=1,
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


def test_invalid_backslash_escape_in_content_is_repaired():
    raw = r'{"title":"T","content":"Windows path C:\Projects\Vault\File.md","tags":["t"]}'
    c = _client(raw)

    result = request_structured(
        client=c,
        prompt="write",
        model_class=SingleArticle,
        model="qwen2.5:14b",
        max_retries=0,
    )

    assert result.content == r"Windows path C:\Projects\Vault\File.md"


# ── _make_template: nested object rendering ──────────────────────────────────


def test_template_expands_nested_object_array():
    """AnalysisResult.concepts is list[Concept] — template must show the object shape,
    not the array's description string."""
    template = json.loads(_make_template(AnalysisResult))
    concepts = template["concepts"]
    assert isinstance(concepts, list) and len(concepts) == 1
    assert isinstance(concepts[0], dict)
    assert set(concepts[0].keys()) == {"name", "aliases"}
    assert isinstance(concepts[0]["aliases"], list)


def test_template_expands_compile_plan_articles():
    template = json.loads(_make_template(CompilePlan))
    articles = template["articles"]
    assert isinstance(articles[0], dict)
    assert {"title", "action", "path", "reasoning", "source_paths"} <= set(articles[0].keys())
    assert set(articles[0]["action"].split(" | ")) == {"create", "update"}


def test_template_expands_lint_issues_with_enum():
    template = json.loads(_make_template(LintResult))
    issue = template["issues"][0]
    assert isinstance(issue, dict)
    assert "orphan" in issue["issue_type"]
    assert issue["auto_fixable"] is True


def test_template_primitive_array_keeps_description_hint():
    """list[str] still rendered as legacy single-string hint (not object)."""
    template = json.loads(_make_template(AnalysisResult))
    assert template["suggested_topics"] == [
        "Titles of wiki articles this note should feed into (max 5)"
    ]


def test_template_optional_field_keeps_outer_description():
    """Optional[str] (anyOf[str, null]) must still carry the parent field description."""
    template = json.loads(_make_template(AnalysisResult))
    assert "ISO 639-1" in template["language"]
