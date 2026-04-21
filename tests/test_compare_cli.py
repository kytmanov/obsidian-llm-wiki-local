"""Tests for `olw compare` CLI wiring — spec parsing + privacy gate."""

from __future__ import annotations

import pytest

from obsidian_llm_wiki.cli import _any_cloud_contestant, _parse_contestant_spec


def test_parse_basic():
    s = _parse_contestant_spec("baseline:fast=gemma4:e4b,heavy=gemma4:e4b")
    assert s.name == "baseline"
    assert s.fast_model == "gemma4:e4b"
    assert s.heavy_model == "gemma4:e4b"
    assert s.provider_name is None
    assert s.provider_url is None


def test_parse_with_provider():
    s = _parse_contestant_spec(
        "groq:fast=llama3:70b,heavy=llama3:70b,provider=groq,url=https://api.groq.com/openai/v1"
    )
    assert s.provider_name == "groq"
    assert s.provider_url == "https://api.groq.com/openai/v1"


def test_parse_missing_colon_rejected():
    with pytest.raises(Exception, match="NAME:"):
        _parse_contestant_spec("fast=x,heavy=y")


def test_parse_missing_models_rejected():
    with pytest.raises(Exception, match="fast=.*heavy="):
        _parse_contestant_spec("x:fast=y")


def test_parse_bad_token_rejected():
    with pytest.raises(Exception, match="missing '='"):
        _parse_contestant_spec("x:fast=y,noequals")


def test_parse_empty_name_rejected():
    with pytest.raises(Exception, match="name is empty"):
        _parse_contestant_spec(":fast=m,heavy=m")


def test_parse_spaces_tolerated():
    s = _parse_contestant_spec("  spaced : fast=m , heavy=m ")
    assert s.name == "spaced"
    assert s.fast_model == "m"


def test_any_cloud_ollama_default():
    s = _parse_contestant_spec("x:fast=m,heavy=m")
    assert _any_cloud_contestant([s]) is False


def test_any_cloud_explicit_local():
    s = _parse_contestant_spec("x:fast=m,heavy=m,provider=lm_studio")
    assert _any_cloud_contestant([s]) is False


def test_any_cloud_detected_for_groq():
    s = _parse_contestant_spec("x:fast=m,heavy=m,provider=groq")
    assert _any_cloud_contestant([s]) is True


def test_any_cloud_mixed_local_and_cloud():
    a = _parse_contestant_spec("a:fast=m,heavy=m")
    b = _parse_contestant_spec("b:fast=m,heavy=m,provider=openrouter")
    assert _any_cloud_contestant([a, b]) is True
