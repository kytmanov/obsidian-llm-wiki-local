"""
Structured output extraction from local LLMs.

Strategy: three-tier fallback.
  Tier 1 — Ollama format=json + schema in system prompt (ideal path)
  Tier 2 — Extract from ```json fenced block in response
  Tier 3 — Retry with error feedback (max_retries times)

Every LLM-facing model uses a small, flat Pydantic schema to maximise
reliability on small (4B) models. Never ask a small model to produce
a nested list of complex objects in one shot.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from .ollama_client import OllamaClient

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)

# Template-based instruction: a concrete fill-in example is less likely to
# confuse small models than a full JSON Schema object (which they may echo back).
_SCHEMA_INSTRUCTION = """\
You MUST respond with ONLY valid JSON. No prose before or after.
Return the JSON object directly. Do NOT wrap it or add extra keys.

Fill in this exact JSON structure with real content:

{template}

Replace each placeholder string with actual content. Keep the same keys and types.
Respond with nothing but the completed JSON object."""


class StructuredOutputError(Exception):
    pass


def _make_template(model_class: type[T]) -> str:
    """Build a fill-in JSON example from model fields (simpler than raw JSON Schema)."""
    schema = model_class.model_json_schema()
    props = schema.get("properties", {})

    template: dict = {}
    for field_name, field_schema in props.items():
        ftype = field_schema.get("type", "string")
        desc = field_schema.get("description", "")
        enum = field_schema.get("enum")

        if enum:
            template[field_name] = " | ".join(str(e) for e in enum)
        elif ftype == "array":
            template[field_name] = [desc[:60] or f"<{field_name} item>"]
        elif ftype in ("integer", "number"):
            template[field_name] = 0
        elif ftype == "boolean":
            template[field_name] = True
        else:
            template[field_name] = desc[:80] or f"<{field_name}>"

    return json.dumps(template, indent=2)


def _schema_system(model_class: type[T]) -> str:
    template = _make_template(model_class)
    return _SCHEMA_INSTRUCTION.format(template=template)


def _extract_json(text: str) -> str | None:
    """Try to find JSON in raw LLM response text."""
    # Tier 2a: ```json fenced block
    m = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Tier 2b: bare ``` block
    m = re.search(r"```\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Tier 2c: first {...} in response
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0).strip()
    return None


def _unwrap(data: dict, model_class: type[T]) -> dict:
    """
    Unwrap containers that models sometimes produce instead of flat objects.

    Handles three patterns:
    1.  Single-key dict wrapper:    {"AnalysisResult": {...}} → {...}
    1b. Single-key string wrapper:  {"result": "{...json...}"} → parsed inner dict
    2.  JSON-Schema echo:           {"description": "...", "properties": {"title": ..., ...}}
                                    → extract leaf values from "properties"
    """
    if not isinstance(data, dict):
        return data

    # Pattern 1 / 1b: single-key wrapper
    if len(data) == 1:
        key = next(iter(data))
        value = data[key]
        if isinstance(value, dict):
            return value
        # 1b: value is a JSON-encoded string — model put the whole object in one field
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

    # Pattern 2: model echoes JSON Schema format with "properties" dict
    # {"description": "...", "properties": {"field": "value", ...}}
    if "properties" in data and isinstance(data["properties"], dict):
        props = data["properties"]
        # Only use if values look like actual data (not nested schema dicts)
        if all(not isinstance(v, dict) or "type" not in v for v in props.values()):
            return props

    return data


def _try_parse(raw: str, model_class: type[T]) -> tuple[T | None, str]:
    """Try direct JSON parse + Pydantic validation. Returns (result, error_str)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    # Try direct validation first
    last_err = ""
    try:
        return model_class.model_validate(data), ""
    except ValidationError as e:
        last_err = str(e)
    # Try unwrapping single-key container ({"ClassName": {...}})
    try:
        return model_class.model_validate(_unwrap(data, model_class)), ""
    except ValidationError as e:
        last_err = str(e)
    except Exception:
        pass
    return None, last_err


def request_structured(
    client: OllamaClient,
    prompt: str,
    model_class: type[T],
    model: str,
    system: str = "",
    num_ctx: int = 8192,
    num_predict: int = -1,
    max_retries: int = 2,
) -> T:
    """
    Request structured output from Ollama, parse into Pydantic model.

    Args:
        client:      OllamaClient instance
        prompt:      User-facing prompt
        model_class: Pydantic model to parse response into
        model:       Ollama model name
        system:      Optional domain context (prepended before schema instruction)
        num_ctx:     Context window size
        max_retries: How many retry attempts after initial failure

    Raises:
        StructuredOutputError: if all attempts exhausted
    """
    schema_system = _schema_system(model_class)
    full_system = f"{system}\n\n{schema_system}" if system.strip() else schema_system

    last_error: str = ""
    current_prompt = prompt

    for attempt in range(max_retries + 1):
        log.debug("structured_output attempt %d/%d model=%s", attempt + 1, max_retries + 1, model)

        # Tier 1: JSON mode
        raw = client.generate(
            prompt=current_prompt,
            model=model,
            system=full_system,
            format="json",
            num_ctx=num_ctx,
            num_predict=num_predict,
        )

        # Try direct parse
        result, parse_err = _try_parse(raw, model_class)
        if result is not None:
            return result
        last_error = parse_err
        log.debug("Tier 1 parse failed, trying extraction")

        # Tier 2: extract from text
        extracted = _extract_json(raw)
        if extracted:
            result, parse_err = _try_parse(extracted, model_class)
            if result is not None:
                return result
            if parse_err:
                last_error = parse_err

        log.debug(
            "structured_output attempt %d failed: %s. Raw (first 300): %s",
            attempt + 1,
            last_error,
            raw[:300],
        )

        if attempt < max_retries:
            current_prompt = (
                f"Your previous response was invalid.\n"
                f"Error: {last_error}\n\n"
                f"Original request:\n{prompt}\n\n"
                f"Respond with ONLY valid JSON matching the schema. Nothing else."
            )

    raise StructuredOutputError(
        f"Failed to get valid {model_class.__name__} after {max_retries + 1} attempts. "
        f"Last error: {last_error}"
    )
