"""
Shared protocol (structural interface) for LLM clients.

Both OllamaClient and OpenAICompatClient implement this interface via
duck typing. Using a Protocol keeps the code loosely coupled — no
inheritance required, existing tests using MagicMock continue to work.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClientProtocol(Protocol):
    def generate(
        self,
        prompt: str,
        model: str,
        system: str = ...,
        format: str | None = ...,
        num_ctx: int = ...,
        num_predict: int = ...,
    ) -> str: ...

    def embed_batch(self, texts: list[str], model: str = ...) -> list[list[float]]: ...

    def embed(self, text: str, model: str = ...) -> list[float]: ...

    def healthcheck(self) -> bool: ...

    def require_healthy(self) -> None: ...

    def list_models(self) -> list[str]: ...

    def list_models_detailed(self) -> list[dict]: ...

    def close(self) -> None: ...

    def __enter__(self) -> LLMClientProtocol: ...

    def __exit__(self, *args) -> None: ...
