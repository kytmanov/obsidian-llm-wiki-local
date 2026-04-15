"""
Provider registry for OpenAI-compatible LLM services.

Each entry describes a known provider's default URL, auth requirements,
and capability flags. The registry is used by the setup wizard and
client factory — it is pure data with no runtime dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderInfo:
    name: str
    display_name: str
    default_url: str
    requires_auth: bool
    supports_json_mode: bool  # response_format: {type: json_object}
    supports_embeddings: bool  # /v1/embeddings endpoint
    is_local: bool
    default_timeout: float  # seconds; local=600, cloud=120
    env_var: str | None  # conventional env var for API key, e.g. "GROQ_API_KEY"
    azure: bool = False  # uses api-key header + api-version query param


# fmt: off
PROVIDER_REGISTRY: dict[str, ProviderInfo] = {
    # ── Local (no auth required, slow inference → long timeout) ──────────────
    "ollama":     ProviderInfo("ollama",     "Ollama",     "http://localhost:11434",           False, True,  True,  True,  600.0, None),
    "lm_studio":  ProviderInfo("lm_studio",  "LM Studio",  "http://localhost:1234/v1",         False, True,  True,  True,  600.0, None),
    "vllm":       ProviderInfo("vllm",       "vLLM",       "http://localhost:8000/v1",         False, True,  True,  True,  600.0, None),
    "llama_cpp":  ProviderInfo("llama_cpp",  "llama.cpp",  "http://localhost:8080/v1",         False, True,  False, True,  600.0, None),
    "localai":    ProviderInfo("localai",    "LocalAI",    "http://localhost:8080/v1",         False, True,  True,  True,  600.0, None),
    "tgi":        ProviderInfo("tgi",        "TGI",        "http://localhost:3000/v1",         False, True,  False, True,  600.0, None),
    "sglang":     ProviderInfo("sglang",     "SGLang",     "http://localhost:30000/v1",        False, True,  True,  True,  600.0, None),
    "llamafile":  ProviderInfo("llamafile",  "Llamafile",  "http://localhost:8080/v1",         False, True,  False, True,  600.0, None),
    "lemonade":   ProviderInfo("lemonade",   "Lemonade",   "http://localhost:8000/v1",         False, True,  False, True,  600.0, None),
    # ── Cloud (API key required, fast inference → short timeout) ─────────────
    "groq":        ProviderInfo("groq",        "Groq",        "https://api.groq.com/openai/v1",          True, True,  False, False, 120.0, "GROQ_API_KEY"),
    "together":    ProviderInfo("together",    "Together AI", "https://api.together.xyz/v1",             True, True,  True,  False, 120.0, "TOGETHER_API_KEY"),
    "fireworks":   ProviderInfo("fireworks",   "Fireworks AI","https://api.fireworks.ai/inference/v1",   True, True,  True,  False, 120.0, "FIREWORKS_API_KEY"),
    "deepinfra":   ProviderInfo("deepinfra",   "DeepInfra",   "https://api.deepinfra.com/v1/openai",     True, True,  True,  False, 120.0, "DEEPINFRA_API_KEY"),
    "openrouter":  ProviderInfo("openrouter",  "OpenRouter",  "https://openrouter.ai/api/v1",            True, True,  False, False, 120.0, "OPENROUTER_API_KEY"),
    "mistral":     ProviderInfo("mistral",     "Mistral AI",  "https://api.mistral.ai/v1",               True, True,  True,  False, 120.0, "MISTRAL_API_KEY"),
    "deepseek":    ProviderInfo("deepseek",    "DeepSeek",    "https://api.deepseek.com/v1",             True, True,  False, False, 120.0, "DEEPSEEK_API_KEY"),
    "siliconflow": ProviderInfo("siliconflow", "SiliconFlow", "https://api.siliconflow.cn/v1",           True, True,  True,  False, 120.0, "SILICONFLOW_API_KEY"),
    "perplexity":  ProviderInfo("perplexity",  "Perplexity",  "https://api.perplexity.ai",               True, False, False, False, 120.0, "PERPLEXITY_API_KEY"),
    "xai":         ProviderInfo("xai",         "xAI (Grok)",  "https://api.x.ai/v1",                    True, True,  False, False, 120.0, "XAI_API_KEY"),
    "azure":       ProviderInfo("azure",       "Azure OpenAI","",                                        True, True,  True,  False, 120.0, "AZURE_OPENAI_API_KEY", azure=True),
    # ── Custom / unknown provider ─────────────────────────────────────────────
    "custom":      ProviderInfo("custom",      "Custom",      "",                                        False,True,  False, False, 300.0, None),
}
# fmt: on


def get_provider(name: str) -> ProviderInfo | None:
    """Return ProviderInfo for a known provider name, or None if unknown."""
    return PROVIDER_REGISTRY.get(name)


def list_local_providers() -> list[ProviderInfo]:
    """Return all local (no-auth) providers, excluding 'ollama' (handled natively)."""
    return [p for p in PROVIDER_REGISTRY.values() if p.is_local and p.name != "ollama"]


def list_cloud_providers() -> list[ProviderInfo]:
    """Return all cloud (auth-required) providers."""
    return [p for p in PROVIDER_REGISTRY.values() if not p.is_local and p.name != "custom"]


def list_all_providers() -> list[ProviderInfo]:
    """Return all providers in display order: Ollama first, then local, then cloud, then custom."""
    ollama = [PROVIDER_REGISTRY["ollama"]]
    local = list_local_providers()
    cloud = list_cloud_providers()
    custom = [PROVIDER_REGISTRY["custom"]]
    return ollama + local + cloud + custom
