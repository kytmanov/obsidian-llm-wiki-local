"""
Per-call LLM telemetry sink.

`request_structured` emits one `LLMCallEvent` per call into a ContextVar-scoped
sink when one is active. When no sink is set (normal pipeline runs), emission
is a cheap no-op.

The `olw compare` runner sets a sink around each contestant-seed pipeline run
to collect reliability/efficiency signals without touching pipeline code.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class LLMCallEvent:
    stage: str
    model: str
    tier: int  # 1=native-json parse, 2=extracted, 3=retry-success, -1=final-failure
    retries: int
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    num_ctx: int
    error: str | None = None
    contestant: str | None = None
    seed: int | None = None
    model_role: str | None = None
    extra: dict = field(default_factory=dict)


_sink: ContextVar[list[LLMCallEvent] | None] = ContextVar("olw_llm_telemetry", default=None)


def emit(event: LLMCallEvent) -> None:
    sink = _sink.get()
    if sink is not None:
        sink.append(event)


def current_sink() -> list[LLMCallEvent] | None:
    return _sink.get()


@contextmanager
def telemetry_sink() -> Iterator[list[LLMCallEvent]]:
    events: list[LLMCallEvent] = []
    token = _sink.set(events)
    try:
        yield events
    finally:
        _sink.reset(token)
