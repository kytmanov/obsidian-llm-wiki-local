"""Tests for CLI --fast-model/--heavy-model/--provider/--provider-url wiring."""

from __future__ import annotations

from obsidian_llm_wiki.cli import _model_override_kwargs


def test_no_flags_empty_kwargs():
    assert _model_override_kwargs(None, None, None, None) == {}


def test_fast_model_only():
    kw = _model_override_kwargs("gemma4:e4b", None, None, None)
    assert kw == {"models": {"fast": "gemma4:e4b"}}


def test_heavy_model_only():
    kw = _model_override_kwargs(None, "qwen2.5:14b", None, None)
    assert kw == {"models": {"heavy": "qwen2.5:14b"}}


def test_both_models():
    kw = _model_override_kwargs("gemma4:e4b", "qwen2.5:14b", None, None)
    assert kw == {"models": {"fast": "gemma4:e4b", "heavy": "qwen2.5:14b"}}


def test_provider_only_without_url():
    kw = _model_override_kwargs(None, None, "groq", None)
    assert kw == {"provider": {"name": "groq"}}


def test_provider_url_only_without_name():
    kw = _model_override_kwargs(None, None, None, "http://localhost:1234/v1")
    assert kw == {"provider": {"url": "http://localhost:1234/v1"}}


def test_all_flags_together():
    kw = _model_override_kwargs(
        "gemma4:e4b",
        "qwen2.5:14b",
        "groq",
        "https://api.groq.com/openai/v1",
    )
    assert kw == {
        "models": {"fast": "gemma4:e4b", "heavy": "qwen2.5:14b"},
        "provider": {"name": "groq", "url": "https://api.groq.com/openai/v1"},
    }
