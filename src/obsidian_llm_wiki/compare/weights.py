"""
Weight management for olw compare.

Two layers:
  • top-level weights (reliability, structure, ...) split equally across
    their sub-dims unless overridden.
  • nested sub-dim weights (structure.orphan_rate = 0.15) take precedence
    over the top-level split.

Loading order: built-in defaults → mode-specific defaults (baseline adds
`agreement.*`) → user TOML override → normalization.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from .corpus import CorpusMode

# Sub-dims grouped by top-level key. Order matters only for presentation.
SUB_DIMS: dict[str, list[str]] = {
    "reliability": [
        "final_success_rate",
        "retry_rate",
        "pipeline_failure_rate",
    ],
    "efficiency": [
        "wall_time",
        "chars_per_concept",
    ],
    "structure": [
        "concept_coverage",
        "concept_precision",
        "tag_validity",
        "wikilink_density",
        "orphan_rate",
        "broken_link_rate",
        "lint_health",
    ],
    "fidelity": ["source_overlap"],
    "query": [
        "page_selection_f1",
        "answer_containment",
        "refusal_correctness",
    ],
    "consistency": ["seed_variance"],
}

DEFAULT_TOP_WEIGHTS: dict[str, float] = {
    "reliability": 0.25,
    "structure": 0.25,
    "fidelity": 0.15,
    "query": 0.15,
    "efficiency": 0.10,
    "consistency": 0.10,
}

BASELINE_TOP_WEIGHTS: dict[str, float] = {
    "reliability": 0.22,
    "structure": 0.22,
    "fidelity": 0.13,
    "query": 0.13,
    "agreement": 0.15,
    "efficiency": 0.08,
    "consistency": 0.07,
}


def default_top_weights(mode: CorpusMode) -> dict[str, float]:
    if mode is CorpusMode.BASELINE:
        return dict(BASELINE_TOP_WEIGHTS)
    return dict(DEFAULT_TOP_WEIGHTS)


def build_flat_weights(
    mode: CorpusMode,
    override_path: Path | None = None,
) -> dict[str, float]:
    """Return a flat sub-dim → weight map summing to 1.0.

    If `override_path` is given, it is parsed as TOML with optional
    `[weights]` (top-level) and/or `[weights.<dim>]` nested tables.
    """
    top = default_top_weights(mode)
    nested_overrides: dict[str, dict[str, float]] = {}

    if override_path is not None:
        data = tomllib.loads(override_path.read_text())
        w = data.get("weights", {})
        for key, val in w.items():
            if isinstance(val, (int, float)):
                top[key] = float(val)
            elif isinstance(val, dict):
                nested_overrides[key] = {k: float(v) for k, v in val.items()}

    flat: dict[str, float] = {}
    for top_key, top_w in top.items():
        subs = SUB_DIMS.get(top_key)
        if not subs:
            # Unknown top-level key (e.g. "agreement" not implemented yet):
            # allocate to a single synthetic sub-dim so weight is preserved.
            flat[f"{top_key}.total"] = top_w
            continue
        explicit = nested_overrides.get(top_key, {})
        implicit_subs = [s for s in subs if s not in explicit]
        explicit_sum = sum(explicit.values())
        remaining = max(0.0, top_w - explicit_sum)
        for s in subs:
            if s in explicit:
                flat[f"{top_key}.{s}"] = explicit[s]
            else:
                flat[f"{top_key}.{s}"] = (
                    remaining / len(implicit_subs) if implicit_subs else 0.0
                )

    return _normalize(flat)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {k: v / total for k, v in weights.items()}


def compute_overall(
    scores: dict,
    weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Weighted sum across `scores` (dotted-path → DimScore) using `weights`.

    Returns (overall, active_weights) where active_weights is `weights`
    re-normalized over dims that actually contributed a value (n > 0).
    This implements the plan's "redistribute absent dims" rule: if a dim
    is n/a for a contestant, the remaining dims' weights scale up to fill.
    """
    active = {k: w for k, w in weights.items() if scores.get(k) and scores[k].n > 0}
    total = sum(active.values())
    if total <= 0:
        return 0.0, {}
    norm = {k: w / total for k, w in active.items()}
    overall = sum(scores[k].mean * w for k, w in norm.items())
    return overall, norm
