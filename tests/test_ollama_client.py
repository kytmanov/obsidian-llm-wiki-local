"""Tests for OllamaClient._last_stats capture."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from obsidian_llm_wiki.ollama_client import OllamaClient, OllamaError


def _make_client() -> OllamaClient:
    return OllamaClient(base_url="http://localhost:11434", timeout=5.0)


def test_generate_captures_last_stats_on_success():
    """`_last_stats` populated with latency + prompt_eval_count/eval_count."""
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "response": "hello",
        "prompt_eval_count": 77,
        "eval_count": 22,
    }
    with patch.object(client._client, "post", return_value=mock_resp):
        result = client.generate(prompt="hi", model="gemma4:e4b")
    assert result == "hello"
    assert client._last_stats["prompt_tokens"] == 77
    assert client._last_stats["completion_tokens"] == 22
    assert client._last_stats["latency_ms"] >= 0


def test_generate_last_stats_on_connect_error():
    client = _make_client()
    with patch.object(client._client, "post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(OllamaError):
            client.generate(prompt="hi", model="gemma4:e4b")
    assert "latency_ms" in client._last_stats
    assert "prompt_tokens" not in client._last_stats


def test_generate_last_stats_missing_token_fields():
    """Ollama response missing eval counts → tokens = None."""
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"response": "x"}
    with patch.object(client._client, "post", return_value=mock_resp):
        client.generate(prompt="hi", model="gemma4:e4b")
    assert client._last_stats["prompt_tokens"] is None
    assert client._last_stats["completion_tokens"] is None
