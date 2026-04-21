"""
Static pricing table for advisory cost estimates.

USD per million tokens. Local providers are $0 by fiat (true compute
cost is unreported — GPU/electricity out of scope). Cloud prices are
snapshotted on the `LAST_UPDATED` date and reported in the compare
report with that caveat.

Cost is an ADVISORY column only — not weight-bearing. See plan
§"Advisory columns" for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

LAST_UPDATED = "2026-04-19"


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    model: str
    input_per_mtok: float  # USD per 1M prompt tokens
    output_per_mtok: float  # USD per 1M completion tokens


# fmt: off
_PRICES: tuple[ModelPrice, ...] = (
    # Local — always $0
    ModelPrice("ollama",     "*",                    0.0,  0.0),
    ModelPrice("lm_studio",  "*",                    0.0,  0.0),
    ModelPrice("vllm",       "*",                    0.0,  0.0),
    ModelPrice("llama_cpp",  "*",                    0.0,  0.0),
    ModelPrice("localai",    "*",                    0.0,  0.0),
    ModelPrice("tgi",        "*",                    0.0,  0.0),
    ModelPrice("sglang",     "*",                    0.0,  0.0),
    ModelPrice("llamafile",  "*",                    0.0,  0.0),
    ModelPrice("lemonade",   "*",                    0.0,  0.0),
    # Cloud — snapshot prices (USD/Mtok, input/output)
    ModelPrice("groq",       "llama-3.1-70b-versatile", 0.59, 0.79),
    ModelPrice("groq",       "llama-3.1-8b-instant",    0.05, 0.08),
    ModelPrice("openrouter", "*",                       1.00, 2.00),  # generic midpoint
    ModelPrice("together",   "*",                       0.60, 0.60),  # generic midpoint
    ModelPrice("fireworks",  "*",                       0.50, 0.50),
    ModelPrice("deepinfra",  "*",                       0.50, 0.50),
    ModelPrice("mistral",    "mistral-large-latest",    2.00, 6.00),
    ModelPrice("mistral",    "*",                       0.25, 0.75),
    ModelPrice("deepseek",   "*",                       0.27, 1.10),
    ModelPrice("xai",        "*",                       2.00, 10.00),
    ModelPrice("perplexity", "*",                       1.00, 1.00),
)
# fmt: on


def lookup(provider: str, model: str) -> ModelPrice | None:
    """Exact provider+model match first, then provider+wildcard fallback."""
    p = provider.lower()
    m = (model or "").lower()
    exact = [x for x in _PRICES if x.provider == p and x.model.lower() == m]
    if exact:
        return exact[0]
    wildcard = [x for x in _PRICES if x.provider == p and x.model == "*"]
    if wildcard:
        return wildcard[0]
    return None


def estimate_cost_usd(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """None when the provider/model is unpriced — surfaced to the report."""
    price = lookup(provider, model)
    if price is None:
        return None
    return (
        prompt_tokens / 1_000_000 * price.input_per_mtok
        + completion_tokens / 1_000_000 * price.output_per_mtok
    )
