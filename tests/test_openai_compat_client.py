"""Tests for OpenAI-compat client truncation detection + auto-downgrade."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from obsidian_llm_wiki.openai_compat_client import (
    LLMBadRequestError,
    LLMTruncatedError,
    OpenAICompatClient,
)


def _make_client() -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="https://api.example.com/v1",
        provider_name="test",
        api_key="sk-test",
    )


def _ok_response(content: str, finish_reason: str | None = "stop") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    resp.text = ""
    return resp


def _bad_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 400
    resp.text = text
    resp.json.return_value = {"error": {"message": text}}
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400", request=MagicMock(), response=resp
    )
    return resp


def test_generate_returns_content_on_finish_stop():
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("hello", finish_reason="stop"))
    assert client.generate(prompt="hi", model="m", num_predict=2048) == "hello"


def test_generate_raises_truncated_on_finish_length():
    """finish_reason='length' raises with the cap surfaced for actionable error."""
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("partial...", finish_reason="length"))
    with pytest.raises(LLMTruncatedError) as exc_info:
        client.generate(prompt="hi", model="m", num_predict=4096)
    err = exc_info.value
    assert err.max_tokens == 4096
    assert err.finish_reason == "length"
    assert "article_max_tokens" in str(err)


def test_generate_raises_truncated_on_finish_max_tokens():
    """Anthropic-via-OpenAI-compat uses 'max_tokens' as the truncation signal."""
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("partial", finish_reason="max_tokens"))
    with pytest.raises(LLMTruncatedError):
        client.generate(prompt="hi", model="m", num_predict=8192)


def test_generate_raises_truncated_on_empty_content():
    """Empty content with no length signal — defensive raise to surface silent
    failure modes from providers that don't set finish_reason properly."""
    client = _make_client()
    client._post_chat = MagicMock(return_value=_ok_response("", finish_reason="stop"))
    with pytest.raises(LLMTruncatedError):
        client.generate(prompt="hi", model="m", num_predict=4096)


def test_cloud_auto_downgrade_halves_max_tokens_on_exceed_error():
    """When a cloud provider rejects max_tokens as too large, we should halve
    and retry once, not bubble the 400 to the user."""
    client = _make_client()
    bad = _bad_response(
        '{"error": {"message": "max_tokens exceeds the maximum allowed for this model"}}'
    )
    good = _ok_response("response after halving")
    client._post_chat = MagicMock(side_effect=[bad, good])

    result = client.generate(prompt="hi", model="m", num_predict=16384)

    assert result == "response after halving"
    # Second call should send half the original cap
    second_call = client._post_chat.call_args_list[1]
    assert second_call.args[0]["max_tokens"] == 8192


def test_cloud_auto_downgrade_does_not_fire_on_unrelated_400():
    """Ensure we don't strip max_tokens on 400s that aren't about cap exceeding."""
    client = _make_client()
    bad = _bad_response('{"error": {"message": "model not found"}}')
    client._post_chat = MagicMock(return_value=bad)

    with pytest.raises(LLMBadRequestError):
        client.generate(prompt="hi", model="m", num_predict=4096)
    # Only one call — no downgrade attempted
    assert client._post_chat.call_count == 1


def test_cloud_auto_downgrade_skips_when_max_tokens_already_below_floor():
    """Auto-downgrade must never increase the requested cap on a provider-limit 400."""
    client = _make_client()
    bad = _bad_response(
        '{"error": {"message": "max_tokens exceeds the maximum allowed for this model"}}'
    )
    client._post_chat = MagicMock(return_value=bad)

    with pytest.raises(LLMBadRequestError):
        client.generate(prompt="hi", model="m", num_predict=256)

    assert client._post_chat.call_count == 1


def test_lm_studio_auto_downgrade_strips_max_tokens_on_n_keep_error():
    """Existing auto-downgrade #2 (n_keep > n_ctx) still works after our changes."""
    client = _make_client()
    bad = _bad_response(
        '{"error": {"message": "tokens to keep from initial prompt exceeds n_ctx"}}'
    )
    good = _ok_response("response without max_tokens")
    client._post_chat = MagicMock(side_effect=[bad, good])

    result = client.generate(prompt="hi", model="m", num_predict=4096)

    assert result == "response without max_tokens"
    # Second call should not have max_tokens at all
    second_call = client._post_chat.call_args_list[1]
    assert "max_tokens" not in second_call.args[0]


def test_truncated_error_message_handles_no_cap_sent():
    """When num_predict was -1 (no cap), the error should reflect that —
    user has a model/context issue, not an article_max_tokens issue."""
    err = LLMTruncatedError(
        provider="lmstudio",
        max_tokens=0,
        finish_reason="length",
    )
    assert "context limit" in str(err) or "no max_tokens sent" in str(err)


def test_truncated_error_message_suggests_double():
    """Error message should suggest a higher value than current cap."""
    err = LLMTruncatedError(
        provider="ollama",
        max_tokens=4096,
        finish_reason="length",
    )
    msg = str(err)
    assert "article_max_tokens" in msg
    # suggested = max(cap*2, 32768) → 32768 here
    assert "32768" in msg
