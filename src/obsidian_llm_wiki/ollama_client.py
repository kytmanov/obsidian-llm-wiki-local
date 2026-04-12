"""
Thin httpx wrapper around Ollama's HTTP API.
Replaces the entire langchain/langchain-ollama dependency tree.
"""

from __future__ import annotations

import logging

import httpx

log = logging.getLogger(__name__)

_STARTUP_HINT = (
    "Ollama not running. Start it with:\n"
    "  ollama serve\n"
    "Then pull required models:\n"
    "  ollama pull gemma4:e4b\n"
    "  ollama pull qwen2.5:14b\n"
    "  ollama pull nomic-embed-text"
)


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    # ── Health ────────────────────────────────────────────────────────────────

    def healthcheck(self) -> bool:
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def require_healthy(self) -> None:
        if not self.healthcheck():
            raise OllamaError(_STARTUP_HINT)

    def list_models(self) -> list[str]:
        resp = self._client.get(f"{self.base_url}/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def list_models_detailed(self) -> list[dict]:
        """Return list of {'name': str, 'size_gb': str} for the setup wizard table."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [
                {
                    "name": m["name"],
                    "size_gb": f"{m.get('size', 0) / 1e9:.1f} GB",
                }
                for m in models
            ]
        except (httpx.HTTPError, KeyError, ValueError):
            return []

    def pull_model(self, model: str) -> None:
        """Pull model if not present. Streams progress to stderr."""
        import sys

        with self._client.stream(
            "POST",
            f"{self.base_url}/api/pull",
            json={"name": model},
            timeout=600,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    print(f"\r{line}", end="", file=sys.stderr)
        print(file=sys.stderr)

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
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_ctx": num_ctx, "num_predict": num_predict},
        }
        if format:
            payload["format"] = format
        try:
            resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
        except httpx.ConnectError:
            raise OllamaError(_STARTUP_HINT)
        except httpx.TimeoutException as e:
            raise OllamaError(f"Ollama request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            raise OllamaError(
                f"Ollama HTTP error: {e.response.status_code} {e.response.text}"
            ) from e
        return resp.json()["response"]

    # ── Embeddings ────────────────────────────────────────────────────────────

    def embed_batch(self, texts: list[str], model: str = "nomic-embed-text") -> list[list[float]]:
        """Single HTTP call for multiple texts. Returns list of embedding vectors."""
        if not texts:
            return []
        try:
            resp = self._client.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
        except httpx.ConnectError:
            raise OllamaError(_STARTUP_HINT)
        return resp.json()["embeddings"]

    def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        return self.embed_batch([text], model=model)[0]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()
