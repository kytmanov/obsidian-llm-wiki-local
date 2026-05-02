"""
Tests for telemetry.LLMCallEvent emission from request_structured.

Covers: sink no-op, tier 1/2/3 paths, final-failure event, stage tagging,
_last_stats aggregation across retries, context isolation.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.models import AnalysisResult, SingleArticle
from obsidian_llm_wiki.ollama_client import OllamaClient
from obsidian_llm_wiki.structured_output import StructuredOutputError, request_structured
from obsidian_llm_wiki.telemetry import (
    AppEvent,
    app_event_sink,
    current_app_sink,
    current_sink,
    emit_app_event,
    telemetry_sink,
)


def _valid_analysis() -> str:
    return json.dumps(
        {
            "summary": "test summary",
            "quality": "high",
            "concepts": [{"name": "x", "aliases": []}],
            "suggested_topics": ["y"],
            "language": "en",
        }
    )


def _mock_client(response, last_stats: dict | None = None) -> OllamaClient:
    c = MagicMock(spec=OllamaClient)
    if callable(response):
        c.generate.side_effect = response
    else:
        c.generate.return_value = response
    c._last_stats = last_stats or {}
    return c


# ── No-sink baseline ──────────────────────────────────────────────────────────


def test_no_sink_no_emission():
    """Outside telemetry_sink(), request_structured must not crash and emit nothing."""
    assert current_sink() is None
    c = _mock_client(_valid_analysis())
    result = request_structured(
        client=c,
        prompt="p",
        model_class=AnalysisResult,
        model="gemma4:e4b",
        stage="ingest",
    )
    assert result.quality == "high"
    # Still no sink after call
    assert current_sink() is None


# ── Tier 1: direct JSON parse ─────────────────────────────────────────────────


def test_tier1_success_emits_one_event():
    stats = {"latency_ms": 42, "prompt_tokens": 100, "completion_tokens": 30}
    c = _mock_client(_valid_analysis(), last_stats=stats)

    with telemetry_sink() as events:
        request_structured(
            client=c,
            prompt="p",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            stage="ingest",
        )

    assert len(events) == 1
    ev = events[0]
    assert ev.stage == "ingest"
    assert ev.model == "gemma4:e4b"
    assert ev.tier == 1
    assert ev.retries == 0
    assert ev.latency_ms == 42
    assert ev.prompt_tokens == 100
    assert ev.completion_tokens == 30
    assert ev.error is None


# ── Tier 2: extracted from fenced block ───────────────────────────────────────


def test_tier2_extracted_event():
    wrapped = f"Here you go:\n\n```json\n{_valid_analysis()}\n```\n"
    c = _mock_client(wrapped, last_stats={"latency_ms": 50})

    with telemetry_sink() as events:
        request_structured(
            client=c,
            prompt="p",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            stage="ingest",
        )

    assert len(events) == 1
    assert events[0].tier == 2
    assert events[0].retries == 0


# ── Tier 3: retry-success ─────────────────────────────────────────────────────


def test_tier3_retry_success_event():
    call_count = 0

    def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "totally not json"
        return _valid_analysis()

    c = MagicMock(spec=OllamaClient)
    c.generate.side_effect = side_effect
    c._last_stats = {"latency_ms": 25, "prompt_tokens": 10, "completion_tokens": 5}

    with telemetry_sink() as events:
        request_structured(
            client=c,
            prompt="p",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            max_retries=2,
            stage="ingest",
        )

    assert len(events) == 1
    ev = events[0]
    assert ev.tier == 3
    assert ev.retries == 1
    # Latency accumulated across both attempts
    assert ev.latency_ms == 50
    # Tokens accumulated across both attempts
    assert ev.prompt_tokens == 20
    assert ev.completion_tokens == 10


# ── Tier -1: final failure ────────────────────────────────────────────────────


def test_final_failure_emits_tier_minus_one():
    c = _mock_client("never valid !!!", last_stats={"latency_ms": 10})

    with telemetry_sink() as events, pytest.raises(StructuredOutputError):
        request_structured(
            client=c,
            prompt="p",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            max_retries=1,
            stage="compile_article",
        )

    assert len(events) == 1
    ev = events[0]
    assert ev.tier == -1
    assert ev.retries == 1
    assert ev.error is not None
    assert ev.stage == "compile_article"
    # Cumulative across 2 attempts
    assert ev.latency_ms == 20


# ── Missing _last_stats keys are tolerated ────────────────────────────────────


def test_missing_tokens_reported_as_none():
    """Client that didn't populate token keys → event.prompt_tokens is None."""
    c = _mock_client(_valid_analysis(), last_stats={"latency_ms": 10})

    with telemetry_sink() as events:
        request_structured(
            client=c,
            prompt="p",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            stage="ingest",
        )

    ev = events[0]
    assert ev.prompt_tokens is None
    assert ev.completion_tokens is None
    assert ev.latency_ms == 10


def test_partial_token_fields_preserve_none_for_missing_side():
    c = _mock_client(_valid_analysis(), last_stats={"latency_ms": 10, "completion_tokens": 7})

    with telemetry_sink() as events:
        request_structured(
            client=c,
            prompt="p",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            stage="ingest",
        )

    ev = events[0]
    assert ev.prompt_tokens is None
    assert ev.completion_tokens == 7


def test_no_last_stats_attr_tolerated():
    """Client without _last_stats attribute → event still emitted cleanly."""
    c = MagicMock(spec=["generate"])
    c.generate.return_value = _valid_analysis()

    with telemetry_sink() as events:
        request_structured(
            client=c,
            prompt="p",
            model_class=AnalysisResult,
            model="gemma4:e4b",
            stage="ingest",
        )

    assert len(events) == 1
    assert events[0].latency_ms == 0


# ── Multiple calls share sink ─────────────────────────────────────────────────


def test_multiple_calls_accumulate_in_sink():
    c = _mock_client(_valid_analysis(), last_stats={"latency_ms": 5})

    with telemetry_sink() as events:
        for _ in range(3):
            request_structured(
                client=c,
                prompt="p",
                model_class=AnalysisResult,
                model="gemma4:e4b",
                stage="ingest",
            )

    assert len(events) == 3
    assert all(e.tier == 1 for e in events)


# ── Stage tagging distinguishes call sites ────────────────────────────────────


def test_stage_is_passthrough():
    raw = json.dumps({"title": "T", "content": "body", "tags": ["t"]})
    c = _mock_client(raw, last_stats={"latency_ms": 3})

    with telemetry_sink() as events:
        request_structured(
            client=c,
            prompt="w",
            model_class=SingleArticle,
            model="qwen2.5:14b",
            stage="query_answer",
        )

    assert events[0].stage == "query_answer"
    assert events[0].model == "qwen2.5:14b"


def test_app_event_sink_collects_events():
    assert current_app_sink() is None

    with app_event_sink() as events:
        emit_app_event(AppEvent(name="query_synthesize", payload={"ok": True}))

    assert len(events) == 1
    assert events[0].name == "query_synthesize"
    assert events[0].payload == {"ok": True}


def test_app_event_sink_is_context_scoped():
    assert current_app_sink() is None
    with app_event_sink() as events:
        assert current_app_sink() is events
    assert current_app_sink() is None
