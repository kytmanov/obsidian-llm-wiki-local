"""
Tests for providers registry, OpenAICompatClient, and client_factory.
All tests are offline — no real provider required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from obsidian_llm_wiki.openai_compat_client import LLMBadRequestError, LLMError, OpenAICompatClient
from obsidian_llm_wiki.providers import (
    get_provider,
    list_all_providers,
    list_cloud_providers,
    list_local_providers,
)


@pytest.fixture
def cfg_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect global config to a temp dir so tests don't touch ~/.config."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    return tmp_path


# ── providers registry ────────────────────────────────────────────────────────


def test_get_provider_known():
    p = get_provider("groq")
    assert p is not None
    assert p.name == "groq"
    assert p.requires_auth is True
    assert p.default_url.startswith("https://")


def test_get_provider_ollama():
    p = get_provider("ollama")
    assert p is not None
    assert p.is_local is True
    assert p.requires_auth is False


def test_get_provider_unknown():
    assert get_provider("nonexistent_provider_xyz") is None


def test_list_local_excludes_ollama():
    local = list_local_providers()
    names = [p.name for p in local]
    assert "ollama" not in names
    assert all(p.is_local for p in local)
    assert "lm_studio" in names


def test_list_cloud_providers():
    cloud = list_cloud_providers()
    assert all(not p.is_local for p in cloud)
    assert all(p.name != "custom" for p in cloud)
    assert any(p.name == "groq" for p in cloud)


def test_list_all_providers_order():
    all_p = list_all_providers()
    names = [p.name for p in all_p]
    # Ollama first, custom last
    assert names[0] == "ollama"
    assert names[-1] == "custom"
    # all providers present
    assert "groq" in names
    assert "lm_studio" in names


# ── OpenAICompatClient ────────────────────────────────────────────────────────


def _make_client(**kwargs) -> OpenAICompatClient:
    defaults = dict(base_url="http://localhost:1234/v1", provider_name="test")
    defaults.update(kwargs)
    return OpenAICompatClient(**defaults)


def test_generate_success():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "hello world"}}]}
    with patch.object(client._client, "post", return_value=mock_resp):
        result = client.generate("say hi", model="test-model")
    assert result == "hello world"


def test_generate_with_system():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    captured = {}

    def fake_post(url, json=None, **kw):
        captured["payload"] = json
        return mock_resp

    with patch.object(client._client, "post", side_effect=fake_post):
        client.generate("prompt", model="m", system="sys")
    msgs = captured["payload"]["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": "prompt"}


def test_generate_json_mode_injected():
    client = _make_client(supports_json_mode=True)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    captured = {}

    def fake_post(url, json=None, **kw):
        captured["payload"] = json
        return mock_resp

    with patch.object(client._client, "post", side_effect=fake_post):
        client.generate("p", model="m", format="json")
    assert captured["payload"].get("response_format") == {"type": "json_object"}


def test_generate_json_mode_400_retry():
    """On 400 with json mode, client retries without response_format."""
    client = _make_client(supports_json_mode=True)
    bad_resp = MagicMock()
    bad_resp.status_code = 400
    bad_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400", request=MagicMock(), response=bad_resp
    )
    bad_resp.json.return_value = {}

    good_resp = MagicMock()
    good_resp.status_code = 200
    good_resp.raise_for_status.return_value = None
    good_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    call_count = {"n": 0}

    def fake_post(url, json=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return bad_resp
        return good_resp

    with patch.object(client._client, "post", side_effect=fake_post):
        result = client.generate("p", model="m", format="json")
    assert result == "ok"
    assert call_count["n"] == 2


def test_generate_max_tokens_set_when_positive():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "x"}}]}
    captured = {}

    def fake_post(url, json=None, **kw):
        captured["payload"] = json
        return mock_resp

    with patch.object(client._client, "post", side_effect=fake_post):
        client.generate("p", model="m", num_predict=512)
    assert captured["payload"]["max_tokens"] == 512


def test_generate_max_tokens_omitted_when_minus_one():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "x"}}]}
    captured = {}

    def fake_post(url, json=None, **kw):
        captured["payload"] = json
        return mock_resp

    with patch.object(client._client, "post", side_effect=fake_post):
        client.generate("p", model="m", num_predict=-1)
    assert "max_tokens" not in captured["payload"]


def test_generate_connect_error_raises_llmerror():
    client = _make_client()
    with patch.object(client._client, "post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(LLMError, match="Cannot connect"):
            client.generate("p", model="m")


def test_generate_400_raises_llm_bad_request_error():
    """HTTP 400 is wrapped as LLMBadRequestError, not the generic LLMError."""
    client = _make_client(supports_json_mode=False)
    bad_resp = MagicMock()
    bad_resp.status_code = 400
    bad_resp.text = '{"error":"bad input"}'
    bad_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400", request=MagicMock(), response=bad_resp
    )
    with patch.object(client._client, "post", return_value=bad_resp):
        with pytest.raises(LLMBadRequestError):
            client.generate("p", model="m")


def test_generate_nkeep_400_retry_strips_max_tokens():
    """On 400 'tokens to keep' error, client retries without max_tokens."""
    client = _make_client(supports_json_mode=False)

    nkeep_resp = MagicMock()
    nkeep_resp.status_code = 400
    nkeep_resp.text = '{"error":"The number of tokens to keep from the initial prompt is greater"}'
    nkeep_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400", request=MagicMock(), response=nkeep_resp
    )

    good_resp = MagicMock()
    good_resp.status_code = 200
    good_resp.raise_for_status.return_value = None
    good_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    payloads = []

    def fake_post(url, json=None, **kw):
        payloads.append(json)
        return nkeep_resp if len(payloads) == 1 else good_resp

    with patch.object(client._client, "post", side_effect=fake_post):
        result = client.generate("p", model="m", num_predict=4096)

    assert result == "ok"
    assert len(payloads) == 2
    assert "max_tokens" in payloads[0]
    assert "max_tokens" not in payloads[1]


def test_generate_chained_response_format_then_nkeep():
    """Chained: first 400 strips response_format, second 400 strips max_tokens.
    The final retry must not reintroduce response_format."""
    client = _make_client(supports_json_mode=True)

    rf_resp = MagicMock()
    rf_resp.status_code = 400
    rf_resp.text = '{"error":"response_format.type must be json_schema or text"}'
    rf_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400", request=MagicMock(), response=rf_resp
    )

    nkeep_resp = MagicMock()
    nkeep_resp.status_code = 400
    nkeep_resp.text = '{"error":"tokens to keep exceeds context length"}'
    nkeep_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "400", request=MagicMock(), response=nkeep_resp
    )

    good_resp = MagicMock()
    good_resp.status_code = 200
    good_resp.raise_for_status.return_value = None
    good_resp.json.return_value = {"choices": [{"message": {"content": "done"}}]}

    call_count = {"n": 0}

    def fake_post(url, json=None, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return rf_resp  # has response_format + max_tokens → 400 rf error
        if call_count["n"] == 2:
            return nkeep_resp  # has max_tokens, no response_format → 400 nkeep
        return good_resp  # no response_format, no max_tokens → 200

    payloads = []

    def capturing_post(url, json=None, **kw):
        payloads.append(json)
        return fake_post(url, json=json, **kw)

    with patch.object(client._client, "post", side_effect=capturing_post):
        result = client.generate("p", model="m", format="json", num_predict=4096)

    assert result == "done"
    assert call_count["n"] == 3
    # First request: has both fields
    assert "response_format" in payloads[0]
    assert "max_tokens" in payloads[0]
    # Second request (after rf downgrade): no response_format, still has max_tokens
    assert "response_format" not in payloads[1]
    assert "max_tokens" in payloads[1]
    # Third request (after nkeep downgrade): neither field
    assert "response_format" not in payloads[2]
    assert "max_tokens" not in payloads[2]


def test_generate_timeout_raises_llmerror():
    client = _make_client()
    with patch.object(
        client._client, "post", side_effect=httpx.ReadTimeout("timeout", request=MagicMock())
    ):
        with pytest.raises(LLMError, match="timed out"):
            client.generate("p", model="m")


def test_generate_captures_last_stats_on_success():
    """`_last_stats` populated with latency + token usage from response body."""
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45},
    }
    with patch.object(client._client, "post", return_value=mock_resp):
        client.generate("p", model="m")
    assert client._last_stats["prompt_tokens"] == 123
    assert client._last_stats["completion_tokens"] == 45
    assert client._last_stats["latency_ms"] >= 0


def test_generate_last_stats_latency_only_on_error():
    """On error paths only latency recorded, token fields absent."""
    client = _make_client()
    with patch.object(client._client, "post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(LLMError):
            client.generate("p", model="m")
    assert "latency_ms" in client._last_stats
    assert "prompt_tokens" not in client._last_stats


def test_generate_last_stats_missing_usage_is_none():
    """Provider without `usage` in response → tokens = None (not zero)."""
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"choices": [{"message": {"content": "x"}}]}
    with patch.object(client._client, "post", return_value=mock_resp):
        client.generate("p", model="m")
    assert client._last_stats["prompt_tokens"] is None
    assert client._last_stats["completion_tokens"] is None


def test_generate_429_backoff_retries(monkeypatch):
    """429 triggers exponential backoff; client eventually succeeds."""
    client = _make_client()

    slept: list[float] = []
    monkeypatch.setattr(
        "obsidian_llm_wiki.openai_compat_client.time.sleep",
        lambda s: slept.append(s),
    )

    rate_resp = MagicMock()
    rate_resp.status_code = 429
    rate_resp.headers = {}

    good_resp = MagicMock()
    good_resp.status_code = 200
    good_resp.raise_for_status.return_value = None
    good_resp.json.return_value = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    calls = {"n": 0}

    def fake_post(url, json=None, **kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            return rate_resp
        return good_resp

    with patch.object(client._client, "post", side_effect=fake_post):
        result = client.generate("p", model="m")
    assert result == "ok"
    assert calls["n"] == 3
    assert len(slept) == 2  # backed off twice before success


def test_generate_429_honors_retry_after_header(monkeypatch):
    """Retry-After header value used as sleep delay."""
    client = _make_client()
    slept: list[float] = []
    monkeypatch.setattr(
        "obsidian_llm_wiki.openai_compat_client.time.sleep",
        lambda s: slept.append(s),
    )

    rate_resp = MagicMock()
    rate_resp.status_code = 429
    rate_resp.headers = {"Retry-After": "2"}

    good_resp = MagicMock()
    good_resp.status_code = 200
    good_resp.raise_for_status.return_value = None
    good_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    call_seq = [rate_resp, good_resp]

    def fake_post(url, json=None, **kw):
        return call_seq.pop(0)

    with patch.object(client._client, "post", side_effect=fake_post):
        client.generate("p", model="m")
    assert slept == [2.0]


def test_generate_429_gives_up_after_60s_budget(monkeypatch):
    """After cumulative sleep > 60s budget, client returns 429 and raises LLMError."""
    client = _make_client()

    monkeypatch.setattr(
        "obsidian_llm_wiki.openai_compat_client.time.sleep",
        lambda s: None,  # instant
    )

    rate_resp = MagicMock()
    rate_resp.status_code = 429
    rate_resp.headers = {"Retry-After": "70"}  # single wait exceeds 60s → give up immediately
    rate_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429", request=MagicMock(), response=rate_resp
    )
    rate_resp.text = "rate limit"

    with patch.object(client._client, "post", return_value=rate_resp):
        with pytest.raises(LLMError, match="429"):
            client.generate("p", model="m")


def test_generate_401_raises_llmerror():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    err = httpx.HTTPStatusError("401", request=MagicMock(), response=mock_resp)
    with patch.object(client._client, "post", side_effect=err):
        with pytest.raises(LLMError, match="401"):
            client.generate("p", model="m")


def test_healthcheck_true_on_200():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch.object(client._client, "get", return_value=mock_resp):
        assert client.healthcheck() is True


def test_healthcheck_true_on_401():
    """401 means server running but wrong key — still 'healthy' for connectivity."""
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    with patch.object(client._client, "get", return_value=mock_resp):
        assert client.healthcheck() is True


def test_healthcheck_false_on_connect_error():
    client = _make_client()
    with patch.object(client._client, "get", side_effect=httpx.ConnectError("refused")):
        assert client.healthcheck() is False


def test_list_models():
    client = _make_client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
    with patch.object(client._client, "get", return_value=mock_resp):
        models = client.list_models()
    assert models == ["model-a", "model-b"]


def test_list_models_detailed():
    client = _make_client(provider_name="groq")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"data": [{"id": "llama3"}]}
    with patch.object(client._client, "get", return_value=mock_resp):
        detailed = client.list_models_detailed()
    assert detailed == [{"name": "llama3", "size_gb": "(cloud)"}]


def test_embed_batch_no_embeddings_support():
    client = _make_client(supports_embeddings=False)
    with pytest.raises(LLMError, match="does not support embeddings"):
        client.embed_batch(["text"], model="embed-model")


def test_embed_batch_success():
    client = _make_client(supports_embeddings=True)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.2, 0.3]},
            {"index": 0, "embedding": [0.0, 0.1]},
        ]
    }
    with patch.object(client._client, "post", return_value=mock_resp):
        result = client.embed_batch(["a", "b"], model="embed")
    # sorted by index
    assert result == [[0.0, 0.1], [0.2, 0.3]]


def test_azure_auth_header():
    client = OpenAICompatClient(
        base_url="https://my.openai.azure.com/openai/deployments/gpt4",
        api_key="azure-key",
        azure=True,
    )
    assert client._build_headers() == {"api-key": "azure-key"}


def test_bearer_auth_header():
    client = _make_client(api_key="sk-abc")
    assert client._build_headers() == {"Authorization": "Bearer sk-abc"}


def test_no_auth_header():
    client = _make_client(api_key=None)
    assert client._build_headers() == {}


def test_azure_chat_url_includes_api_version():
    client = OpenAICompatClient(
        base_url="https://res.openai.azure.com/openai/deployments/gpt4",
        azure=True,
        azure_api_version="2024-02-15-preview",
    )
    url = client._chat_url()
    assert "api-version=2024-02-15-preview" in url


# ── client_factory ────────────────────────────────────────────────────────────


def test_build_client_ollama(tmp_path):
    from obsidian_llm_wiki.client_factory import build_client
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.ollama_client import OllamaClient

    config = Config(vault=tmp_path)
    client = build_client(config)
    assert isinstance(client, OllamaClient)


def test_build_client_openai_compat(tmp_path):
    from obsidian_llm_wiki.client_factory import build_client
    from obsidian_llm_wiki.config import Config, ProviderConfig

    config = Config(
        vault=tmp_path,
        provider=ProviderConfig(name="groq", url="https://api.groq.com/openai/v1"),
    )
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        client = build_client(config)
    assert isinstance(client, OpenAICompatClient)
    assert client.provider_name == "groq"


def test_build_client_resolves_env_api_key(tmp_path):
    from obsidian_llm_wiki.client_factory import build_client
    from obsidian_llm_wiki.config import Config, ProviderConfig

    config = Config(
        vault=tmp_path,
        provider=ProviderConfig(name="groq", url="https://api.groq.com/openai/v1"),
    )
    with patch.dict("os.environ", {"GROQ_API_KEY": "from-env"}):
        client = build_client(config)
    assert client._api_key == "from-env"


def test_build_client_resolves_generic_olw_api_key(tmp_path):
    from obsidian_llm_wiki.client_factory import build_client
    from obsidian_llm_wiki.config import Config, ProviderConfig

    config = Config(
        vault=tmp_path,
        provider=ProviderConfig(name="custom", url="http://localhost:9999/v1"),
    )
    with patch.dict("os.environ", {"OLW_API_KEY": "generic-key"}, clear=False):
        client = build_client(config)
    assert client._api_key == "generic-key"


# ── embed_batch timeout ────────────────────────────────────────────────────────


def test_embed_batch_timeout_raises_llmerror():
    """Timeout during embeddings must raise LLMError, not propagate raw TimeoutException."""
    client = _make_client(supports_embeddings=True)
    with patch.object(
        client._client,
        "post",
        side_effect=httpx.ReadTimeout("timeout", request=MagicMock()),
    ):
        with pytest.raises(LLMError, match="timed out"):
            client.embed_batch(["text"], model="embed")


# ── effective_provider backward compat ────────────────────────────────────────


def test_effective_provider_legacy_ollama_section(tmp_path):
    """Config with only [ollama] section (no [provider]) should migrate transparently."""
    from obsidian_llm_wiki.config import Config, OllamaConfig

    config = Config(vault=tmp_path, ollama=OllamaConfig(url="http://myhost:11434", timeout=300.0))
    prov = config.effective_provider
    assert prov.name == "ollama"
    assert prov.url == "http://myhost:11434"
    assert prov.timeout == 300.0


def test_effective_provider_new_section_takes_priority(tmp_path):
    """[provider] section overrides [ollama] when both present."""
    from obsidian_llm_wiki.config import Config, OllamaConfig, ProviderConfig

    config = Config(
        vault=tmp_path,
        ollama=OllamaConfig(url="http://old:11434"),
        provider=ProviderConfig(name="groq", url="https://api.groq.com/openai/v1"),
    )
    prov = config.effective_provider
    assert prov.name == "groq"
    assert "groq" in prov.url


def test_build_client_legacy_ollama_config(tmp_path):
    """build_client() with default (legacy) Config must return OllamaClient."""
    from obsidian_llm_wiki.client_factory import build_client
    from obsidian_llm_wiki.config import Config
    from obsidian_llm_wiki.ollama_client import OllamaClient

    config = Config(vault=tmp_path)  # no [provider] — uses [ollama] defaults
    client = build_client(config)
    assert isinstance(client, OllamaClient)


# ── azure_api_version in wiki.toml ────────────────────────────────────────────


def test_default_wiki_toml_azure_includes_api_version():
    from obsidian_llm_wiki.config import default_wiki_toml

    toml = default_wiki_toml(
        provider_name="azure",
        provider_url="https://myres.openai.azure.com/openai/deployments/gpt4",
        azure_api_version="2024-05-01-preview",
    )
    assert "azure_api_version" in toml
    assert "2024-05-01-preview" in toml


def test_default_wiki_toml_non_azure_no_api_version():
    from obsidian_llm_wiki.config import default_wiki_toml

    toml = default_wiki_toml(provider_name="groq", provider_url="https://api.groq.com/openai/v1")
    assert "azure_api_version" not in toml


# ── setup wizard edge cases ────────────────────────────────────────────────────


def test_setup_wizard_cloud_provider_empty_api_key(tmp_path, cfg_dir):
    """Cloud provider with empty API key: config saved with api_key=None."""
    from click.testing import CliRunner

    from obsidian_llm_wiki.cli import cli
    from obsidian_llm_wiki.global_config import load_global_config
    from obsidian_llm_wiki.providers import list_all_providers

    runner = CliRunner()
    groq_number = next(str(i) for i, p in enumerate(list_all_providers(), 1) if p.name == "groq")

    with patch("obsidian_llm_wiki.openai_compat_client.OpenAICompatClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        # provider=groq, URL=default, api_key=empty(Enter), fast, heavy, no vault
        result = runner.invoke(
            cli,
            ["setup"],
            input=f"{groq_number}\n\n\nllama3\nllama3\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.provider_name == "groq"
    assert cfg.api_key is None


def test_setup_wizard_azure_saves_provider_name(tmp_path, cfg_dir):
    """Azure setup wizard should save provider_name='azure' and azure_api_version."""
    from click.testing import CliRunner

    from obsidian_llm_wiki.cli import cli
    from obsidian_llm_wiki.global_config import load_global_config
    from obsidian_llm_wiki.providers import list_all_providers

    runner = CliRunner()
    # Find Azure's menu number
    azure_number = next(str(i) for i, p in enumerate(list_all_providers(), 1) if p.name == "azure")

    with patch("obsidian_llm_wiki.openai_compat_client.OpenAICompatClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            # provider=azure, URL, api key, fast model, heavy model, no vault
            input=f"{azure_number}\nhttps://myres.openai.azure.com/openai/deployments/gpt4\nmy-key\nmodel-a\nmodel-b\n\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.provider_name == "azure"
    assert cfg.azure_api_version == "2024-02-15-preview"
    assert cfg.api_key == "my-key"
