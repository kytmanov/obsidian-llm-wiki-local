"""Tests for ollama_client — HTTP wrapper around Ollama API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from obsidian_llm_wiki.ollama_client import OllamaClient, OllamaError


@pytest.fixture
def client() -> OllamaClient:
    return OllamaClient(base_url="http://localhost:11434", timeout=10.0)


# ── Construction ──────────────────────────────────────────────────────────────


def test_base_url_strips_trailing_slash():
    c = OllamaClient(base_url="http://localhost:11434/")
    assert c.base_url == "http://localhost:11434"


def test_context_manager():
    with OllamaClient() as c:
        assert c.base_url == "http://localhost:11434"


# ── healthcheck ───────────────────────────────────────────────────────────────


def test_healthcheck_returns_true(client: OllamaClient):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch.object(client._client, "get", return_value=mock_resp):
        assert client.healthcheck() is True


def test_healthcheck_returns_false_on_connect_error(client: OllamaClient):
    with patch.object(client._client, "get", side_effect=httpx.ConnectError("refused")):
        assert client.healthcheck() is False


def test_healthcheck_returns_false_on_non_200(client: OllamaClient):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    with patch.object(client._client, "get", return_value=mock_resp):
        assert client.healthcheck() is False


# ── require_healthy ───────────────────────────────────────────────────────────


def test_require_healthy_passes(client: OllamaClient):
    with patch.object(client, "healthcheck", return_value=True):
        client.require_healthy()  # should not raise


def test_require_healthy_raises(client: OllamaClient):
    with patch.object(client, "healthcheck", return_value=False):
        with pytest.raises(OllamaError, match="Ollama not running"):
            client.require_healthy()


# ── list_models ───────────────────────────────────────────────────────────────


def test_list_models(client: OllamaClient):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "models": [
            {"name": "llama3.2:3b"},
            {"name": "gemma4:e4b"},
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client._client, "get", return_value=mock_resp):
        models = client.list_models()
        assert models == ["llama3.2:3b", "gemma4:e4b"]


def test_list_models_empty(client: OllamaClient):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"models": []}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client._client, "get", return_value=mock_resp):
        assert client.list_models() == []


# ── generate ──────────────────────────────────────────────────────────────────


def test_generate_returns_response(client: OllamaClient):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "Hello world"}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client._client, "post", return_value=mock_resp):
        result = client.generate("say hello", model="test:7b")
        assert result == "Hello world"


def test_generate_sends_correct_payload(client: OllamaClient):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "ok"}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
        client.generate("prompt", model="m", system="sys", format="json", num_ctx=4096)
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "m"
        assert payload["prompt"] == "prompt"
        assert payload["system"] == "sys"
        assert payload["format"] == "json"
        assert payload["options"]["num_ctx"] == 4096
        assert payload["stream"] is False


def test_generate_raises_on_connect_error(client: OllamaClient):
    with patch.object(client._client, "post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(OllamaError, match="Ollama not running"):
            client.generate("hello", model="test")


def test_generate_raises_on_http_error(client: OllamaClient):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "internal error"
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=mock_resp
    )
    with patch.object(client._client, "post", return_value=mock_resp):
        with pytest.raises(OllamaError, match="HTTP error"):
            client.generate("hello", model="test")


def test_generate_no_format(client: OllamaClient):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": "ok"}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client._client, "post", return_value=mock_resp) as mock_post:
        client.generate("prompt", model="m")
        payload = mock_post.call_args[1]["json"]
        assert "format" not in payload


# ── embed ─────────────────────────────────────────────────────────────────────


def test_embed_batch_returns_embeddings(client: OllamaClient):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client._client, "post", return_value=mock_resp):
        result = client.embed_batch(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_batch_empty_returns_empty(client: OllamaClient):
    result = client.embed_batch([])
    assert result == []


def test_embed_returns_single_vector(client: OllamaClient):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client._client, "post", return_value=mock_resp):
        result = client.embed("hello")
        assert result == [0.1, 0.2, 0.3]


def test_embed_batch_raises_on_connect_error(client: OllamaClient):
    with patch.object(client._client, "post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(OllamaError, match="Ollama not running"):
            client.embed_batch(["hello"])


# ── close ─────────────────────────────────────────────────────────────────────


def test_close_calls_client_close(client: OllamaClient):
    with patch.object(client._client, "close") as mock_close:
        client.close()
        mock_close.assert_called_once()
