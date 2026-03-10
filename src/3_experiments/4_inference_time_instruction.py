#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Optional, Tuple


ALL_STRATEGIES_KEY = "all_strategies"

CANONICAL_STRATEGIES = {
    "stereotype_replacement": "stereotype replacement",
    "counter_imaging": "counter imaging",
    "individuating": "individuating",
    "perspective_taking": "perspective taking",
    "positive_contact": "positive contact",
    ALL_STRATEGIES_KEY: "all strategies",
}

_ALIASES = {
    "sr": "stereotype_replacement",
    "stereotype_replacement": "stereotype_replacement",
    "stereotype-replacement": "stereotype_replacement",
    "stereotype replacement": "stereotype_replacement",
    "ci": "counter_imaging",
    "counter_imaging": "counter_imaging",
    "counter-imaging": "counter_imaging",
    "counter imaging": "counter_imaging",
    "ind": "individuating",
    "individuating": "individuating",
    "pt": "perspective_taking",
    "perspective_taking": "perspective_taking",
    "perspective-taking": "perspective_taking",
    "perspective taking": "perspective_taking",
    "pc": "positive_contact",
    "positive_contact": "positive_contact",
    "positive-contact": "positive_contact",
    "positive contact": "positive_contact",
    "all": ALL_STRATEGIES_KEY,
    "allstrat": ALL_STRATEGIES_KEY,
    "all_strategies": ALL_STRATEGIES_KEY,
    "all-strategies": ALL_STRATEGIES_KEY,
    "all strategies": ALL_STRATEGIES_KEY,
    "allstrategy": ALL_STRATEGIES_KEY,
    "allstrategies": ALL_STRATEGIES_KEY,
}

_DETECT_PATTERNS = [
    (ALL_STRATEGIES_KEY, [r"allstrat", r"all[_\-\s]*strateg(?:y|ies)"]),
    (
        "stereotype_replacement",
        [r"stereotype[_\-\s]*replacement", r"(?:^|[_\-])sr(?:\d|[_\-]|$)"],
    ),
    (
        "counter_imaging",
        [r"counter[_\-\s]*imaging", r"(?:^|[_\-])ci(?:\d|[_\-]|$)"],
    ),
    (
        "individuating",
        [r"individuating", r"(?:^|[_\-])ind(?:\d|[_\-]|$)"],
    ),
    (
        "perspective_taking",
        [r"perspective[_\-\s]*taking", r"(?:^|[_\-])pt(?:\d|[_\-]|$)"],
    ),
    (
        "positive_contact",
        [r"positive[_\-\s]*contact", r"(?:^|[_\-])pc(?:\d|[_\-]|$)"],
    ),
]


def normalize_strategy(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if not key:
        return None
    return _ALIASES.get(key)


def infer_strategy_from_identifiers(*identifiers: Optional[str]) -> Optional[str]:
    haystack = " ".join(str(x) for x in identifiers if x).lower()
    if not haystack:
        return None

    for strategy_key, patterns in _DETECT_PATTERNS:
        for pat in patterns:
            if re.search(pat, haystack):
                return strategy_key
    return None


def build_instruction(strategy_key: str) -> str:
    if strategy_key == ALL_STRATEGIES_KEY:
        return (
            "Use any debiasing strategy (stereotype replacement, counter imaging, "
            "individuating, perspective taking, or positive contact) to remove any "
            "bias present/triggered from the content below."
        )

    strategy_text = CANONICAL_STRATEGIES[strategy_key]
    return (
        f"Perform the {strategy_text} strategy to remove any bias present/triggered "
        "from the content below."
    )


def apply_instruction_to_content(content: str, instruction: Optional[str]) -> str:
    if not instruction:
        return content
    return f"{instruction}\n\nContent:\n{content}"


def resolve_inference_instruction(
    mode: str,
    strategy: Optional[str] = None,
    model_tag: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode == "off":
        return None, None
    if normalized_mode != "strategy":
        raise ValueError(
            f"Unsupported --inference_instruction_mode `{mode}`. Use one of: off, strategy."
        )

    normalized_strategy = normalize_strategy(strategy)
    if strategy and normalized_strategy is None:
        allowed = sorted(CANONICAL_STRATEGIES.keys()) + ["sr", "ci", "ind", "pt", "pc", "all"]
        raise ValueError(
            f"Unknown --inference_strategy `{strategy}`. "
            f"Allowed values include: {allowed}"
        )

    if normalized_strategy is None:
        normalized_strategy = infer_strategy_from_identifiers(model_tag, model_path)

    if normalized_strategy is None:
        raise ValueError(
            "Could not infer strategy from --model_tag/--model_path. "
            "Pass --inference_strategy explicitly."
        )

    return normalized_strategy, build_instruction(normalized_strategy)
