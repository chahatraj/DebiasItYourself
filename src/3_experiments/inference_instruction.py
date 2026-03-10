#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent / "4_inference_time_instruction.py"
_spec = importlib.util.spec_from_file_location("inference_time_instruction_shared", _MODULE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not import {_MODULE_PATH}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ALL_STRATEGIES_KEY = _mod.ALL_STRATEGIES_KEY
CANONICAL_STRATEGIES = _mod.CANONICAL_STRATEGIES

normalize_strategy = _mod.normalize_strategy
infer_strategy_from_identifiers = _mod.infer_strategy_from_identifiers
build_instruction = _mod.build_instruction
apply_instruction_to_content = _mod.apply_instruction_to_content
resolve_inference_instruction = _mod.resolve_inference_instruction
