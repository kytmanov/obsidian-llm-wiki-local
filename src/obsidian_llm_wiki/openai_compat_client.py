"""
OpenAI-compatible LLM client.

Covers all providers that implement the /v1/chat/completions spec:
  Local:  LM Studio, vLLM, llama.cpp, LocalAI, TGI, SGLang, Llamafile, Lemonade
  Cloud:  Groq, Together AI, Fireworks, DeepInfra, OpenRouter, Mistral, DeepSeek,
          SiliconFlow, Perplexity, xAI, Azure OpenAI

URL construction: endpoints are appended directly to base_url, which must
already include any path prefix (e.g. "https://api.groq.com/openai/v1").
Azure base_url ends at the deployment level, so /chat/completions appends
correctly without an extra /v1 segment.

Auth:
  - Standard providers: Authorization: Bearer {api_key}
  - Azure:              api-key: {api_key}  +  ?api-version= query param
  - Local no-auth:      no header

JSON mode: if supports_json_mode=True, format="json" injects
  response_format: {"type": "json_object"}.
  If the provider returns HTTP 400, the request is retried once without it
  (transparent auto-downgrade for models that reject the field).
"""

from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Base error for all LLM client failures (OllamaError inherits from this)."""


class LLMBadRequestError(LLMError):
    """HTTP 400 from the provider — usually bad input (prompt/context too long, etc.).

    Unlike transient connection or rate-limit errors this is per-request and non-retryable
    at the pipeline level, so compile_concepts catches it per-concept rather than aborting
    the whole run.
    """


class LLMTruncatedError(LLMError):
    """Model stopped at the max_tokens cap (finish_reason="length"/"max_tokens") and
    either returned no usable content or content known to be truncated.

    Carries enough context for the pipeline to render an actionable error message
    that points the user at the exact config knob to adjust.
    """

    def __init__(
        self,
        provider: str,
        max_tokens: int,
        completion_tokens: int | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.provider = provider
        self.max_tokens = max_tokens
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason

        if max_tokens > 0:
            suggested = max(max_tokens * 2, 32768)
            detail = (
                f"output truncated at max_tokens={max_tokens} "
                f"(finish_reason={finish_reason or 'unknown'}). "
                f"Raise pipeline.article_max_tokens in your wiki.toml "
                f"(suggested: {suggested}) or reduce source size."
            )
        elif finish_reason in ("length", "max_tokens"):
            detail = (
                f"output hit provider/model context limit "
                f"(finish_reason={finish_reason}; no max_tokens sent). "
                "Check that your loaded model n_ctx matches heavy_ctx in wiki.toml, "
                "or reduce source size."
            )
        else:
            detail = (
                f"model returned empty response (finish_reason={finish_reason or 'unknown'}). "
                "Likely causes: model context exhausted, response_format rejected silently, "
                "or model/prompt incompatibility. Check model logs."
            )
        super().__init__(f"{provider}: {detail}")


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        provider_name: str = "custom",
        api_key: str | None = None,
        timeout: float = 300.0,
        supports_json_mode: bool = True,
        supports_embeddings: bool = False,
        azure: bool = False,
        azure_api_version: str = "2024-02-15-preview",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider_name = provider_name
        self._api_key = api_key
        self._timeout = timeout
        self.supports_json_mode = supports_json_mode
        self.supports_embeddings = supports_embeddings
        self._azure = azure
        self._azure_api_version = azure_api_version
        self._client = httpx.Client(
            headers=self._build_headers(),
            timeout=timeout,
        )
        self._last_stats: dict = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        if not self._api_key:
            return {}
        if self._azure:
            return {"api-key": self._api_key}
        return {"Authorization": f"Bearer {self._api_key}"}

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _api_url(self, path: str) -> str:
        """Like _url() but appends ?api-version= for Azure endpoints."""
        url = self._url(path)
        if self._azure:
            url = f"{url}?api-version={self._azure_api_version}"
        return url

    def _chat_url(self) -> str:
        return self._api_url("chat/completions")

    def _models_url(self) -> str:
        """Return the correct models/health endpoint URL.

        Azure base_url ends at the deployment level, so /models appended there
        gives an invalid path. Derive the resource-level URL by stripping
        everything from /openai/ onwards, then append /openai/models.
        """
        if self._azure:
            idx = self.base_url.find("/openai/")
            resource = self.base_url[:idx] if idx >= 0 else self.base_url
            return f"{resource}/openai/models?api-version={self._azure_api_version}"
        return self._api_url("models")

    def _wrap_error(self, exc: Exception, context: str = "") -> LLMError:
        prefix = f"{self.provider_name}: " if self.provider_name else ""
        if isinstance(exc, httpx.ConnectError):
            if self._is_local():
                return LLMError(
                    f"{prefix}Cannot connect to {self.base_url}. Make sure the service is running."
                )
            return LLMError(f"{prefix}Cannot reach {self.base_url}. Check your network connection.")
        if isinstance(exc, httpx.TimeoutException):
            return LLMError(f"{prefix}Request timed out ({self._timeout}s). {context}")
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 400:
                return LLMBadRequestError(f"{prefix}HTTP {code}: {exc.response.text[:200]}")
            if code == 401:
                return LLMError(f"{prefix}HTTP 401 Unauthorized. Check your API key.")
            if code == 429:
                return LLMError(f"{prefix}HTTP 429 Rate limit exceeded. Wait and retry.")
            return LLMError(f"{prefix}HTTP {code}: {exc.response.text[:200]}")
        return LLMError(f"{prefix}{exc}")

    def _is_local(self) -> bool:
        return self.base_url.startswith("http://localhost") or self.base_url.startswith(
            "http://127.0.0.1"
        )

    def _post_chat(self, payload: dict) -> httpx.Response:
        """POST to chat endpoint with 429 exponential backoff (max ~60s cumulative)."""
        delay = 1.0
        waited = 0.0
        while True:
            resp = self._client.post(self._chat_url(), json=payload)
            if resp.status_code != 429:
                return resp
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else delay
            except ValueError:
                wait = delay
            if waited + wait > 60.0:
                return resp
            log.debug("%s: HTTP 429, backing off %.1fs", self.provider_name, wait)
            time.sleep(wait)
            waited += wait
            delay = min(delay * 2, 16.0)

    # ── Health ────────────────────────────────────────────────────────────────

    def healthcheck(self) -> bool:
        try:
            resp = self._client.get(self._models_url(), timeout=5)
            # 200 = healthy + auth ok; 401 = service running but wrong key
            return resp.status_code in (200, 401)
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
        except Exception:
            return False

    def require_healthy(self) -> None:
        if not self.healthcheck():
            if self._is_local():
                raise LLMError(
                    f"Cannot reach {self.provider_name} at {self.base_url}. "
                    f"Make sure the service is running."
                )
            raise LLMError(
                f"Cannot reach {self.provider_name} at {self.base_url}. "
                f"Check your network and API key."
            )

    def list_models(self) -> list[str]:
        try:
            resp = self._client.get(self._models_url())
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except (httpx.HTTPError, KeyError, ValueError):
            return []

    def list_models_detailed(self) -> list[dict]:
        """Return list of {'name': str, 'size_gb': str} — matches OllamaClient shape."""
        models = self.list_models()
        return [{"name": m, "size_gb": "(cloud)"} for m in models]

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        model: str,
        system: str = "",
        format: str | None = None,
        num_ctx: int = 8192,
        num_predict: int = -1,
    ) -> str:
        """
        Call /v1/chat/completions. Signature is identical to OllamaClient.generate().

        num_ctx is silently ignored (server-managed for cloud providers).
        num_predict > 0 maps to max_tokens; -1 omits the field (provider default).
        format="json" injects response_format when supports_json_mode=True.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {"model": model, "messages": messages, "stream": False}

        use_json_mode = format == "json" and self.supports_json_mode
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        if num_predict > 0:
            payload["max_tokens"] = num_predict

        t0 = time.monotonic()
        try:
            resp = self._post_chat(payload)
            # Each auto-downgrade strips one unsupported field and retries.
            # Use current_payload so retries chain (each builds on the previous
            # stripped payload rather than the original).
            current_payload = payload

            # Auto-downgrade 1: provider rejects response_format → retry without it
            if resp.status_code == 400 and use_json_mode:
                log.debug(
                    "%s: HTTP 400 with response_format, retrying without json mode",
                    self.provider_name,
                )
                current_payload = {
                    k: v for k, v in current_payload.items() if k != "response_format"
                }
                resp = self._post_chat(current_payload)

            # Auto-downgrade 2: n_keep > context (LM Studio / llama.cpp) → retry without max_tokens
            if resp.status_code == 400 and "max_tokens" in current_payload:
                err_text = resp.text.lower()
                if "tokens to keep" in err_text or "n_keep" in err_text:
                    log.warning(
                        "%s: HTTP 400 n_keep error, retrying without max_tokens "
                        "(model n_ctx may be smaller than configured heavy_ctx; "
                        "output is now uncapped for this request)",
                        self.provider_name,
                    )
                    current_payload = {
                        k: v for k, v in current_payload.items() if k != "max_tokens"
                    }
                    resp = self._post_chat(current_payload)

            # Auto-downgrade 3: cloud provider rejects max_tokens as too large → halve+retry
            if resp.status_code == 400 and "max_tokens" in current_payload:
                err_text = resp.text.lower()
                cloud_cap_signals = (
                    "max_tokens",
                    "max tokens",
                    "completion_tokens",
                    "completion tokens",
                    "output tokens",
                )
                exceed_signals = ("exceed", "too large", "maximum", "greater than", "is too high")
                if any(s in err_text for s in cloud_cap_signals) and any(
                    s in err_text for s in exceed_signals
                ):
                    current_max_tokens = int(current_payload["max_tokens"])
                    if current_max_tokens > 512:
                        halved = max(512, current_max_tokens // 2)
                        log.warning(
                            "%s: HTTP 400 max_tokens exceeds provider limit, halving %d → %d",
                            self.provider_name,
                            current_max_tokens,
                            halved,
                        )
                        current_payload = {**current_payload, "max_tokens": halved}
                        resp = self._post_chat(current_payload)
                    else:
                        log.warning(
                            "%s: HTTP 400 max_tokens exceeds provider limit, but skipping "
                            "auto-downgrade because max_tokens=%d is already at or below "
                            "the 512 retry floor",
                            self.provider_name,
                            current_max_tokens,
                        )

            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._last_stats = {"latency_ms": int((time.monotonic() - t0) * 1000)}
            raise self._wrap_error(e) from e
        except httpx.TimeoutException as e:
            self._last_stats = {"latency_ms": int((time.monotonic() - t0) * 1000)}
            raise self._wrap_error(e) from e
        except httpx.RequestError as e:
            self._last_stats = {"latency_ms": int((time.monotonic() - t0) * 1000)}
            raise self._wrap_error(e) from e

        try:
            body = resp.json()
            choice = body["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, ValueError) as e:
            self._last_stats = {"latency_ms": int((time.monotonic() - t0) * 1000)}
            raise LLMError(
                f"{self.provider_name}: unexpected response format: {resp.text[:200]}"
            ) from e

        usage = body.get("usage") or {}
        self._last_stats = {
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

        # Detect truncation: explicit length signal OR empty content (covers
        # providers that omit finish_reason but emit empty body when capped).
        is_length_signal = finish_reason in ("length", "max_tokens")
        is_empty_content = not (content or "").strip()
        if is_length_signal or is_empty_content:
            cap = int(current_payload.get("max_tokens", 0)) if current_payload else 0
            raise LLMTruncatedError(
                provider=self.provider_name,
                max_tokens=cap,
                completion_tokens=usage.get("completion_tokens"),
                finish_reason=finish_reason or ("empty_content" if is_empty_content else None),
            )

        return content

    # ── Embeddings ────────────────────────────────────────────────────────────

    def embed_batch(self, texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
        if not texts:
            return []
        if not self.supports_embeddings:
            raise LLMError(
                f"{self.provider_name} does not support embeddings. "
                f"Disable RAG or use a provider that supports it "
                f"(Ollama, Together AI, Mistral AI, Fireworks AI, SiliconFlow)."
            )
        try:
            resp = self._client.post(
                self._api_url("embeddings"),
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise self._wrap_error(e) from e
        except httpx.TimeoutException as e:
            raise self._wrap_error(e) from e
        except httpx.RequestError as e:
            raise self._wrap_error(e) from e

        # OpenAI API may return embeddings out of order — sort by index
        try:
            data = resp.json().get("data", [])
            data.sort(key=lambda x: x.get("index", 0))
            return [item["embedding"] for item in data]
        except (ValueError, KeyError) as e:
            raise LLMError(
                f"{self.provider_name}: unexpected embeddings response: {resp.text[:200]}"
            ) from e

    def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        return self.embed_batch([text], model=model)[0]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()
