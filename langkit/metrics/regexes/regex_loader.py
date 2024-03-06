from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, List, Optional, TypedDict


class Pattern(TypedDict, total=True):
    name: str
    expressions: List[str]
    substitutions: Optional[List[str]]


class PatternGroups(TypedDict, total=True):
    patterns: List[Pattern]


class CompiledPattern(TypedDict, total=True):
    name: str
    expressions: List[re.Pattern[str]]
    substitutions: Optional[List[str]]


class CompiledPatternGroups(TypedDict):
    patterns: List[CompiledPattern]


def load_pattern_string(pattern_file_content: str) -> CompiledPatternGroups:
    unvalidated_patterns = json.loads(pattern_file_content)

    if not _is_pattern_group(unvalidated_patterns):
        raise ValueError(f"Invalid pattern group found: {unvalidated_patterns}")

    try:
        return _compile_pattern_groups(PatternGroups(patterns=unvalidated_patterns))
    except Exception as e:
        raise ValueError(f"Invalid pattern group found: {unvalidated_patterns}") from e  # TODO make nicer looking


@lru_cache
def load_patterns_file(file_path: str) -> CompiledPatternGroups:
    with open(file_path, "r", encoding="utf-8") as f:
        return load_pattern_string(f.read())


def _is_pattern(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and "name" in obj
        and isinstance(obj["name"], str)
        and "expressions" in obj
        and isinstance(obj["expressions"], list)
        and all(isinstance(expr, str) for expr in obj["expressions"])  # type: ignore[reportUnknownVariableType]
    )


def _is_pattern_group(obj: Any) -> bool:
    return isinstance(obj, list) and all(_is_pattern(item) for item in obj)  # type: ignore[reportUnknownVariableType]


def _compile_pattern_groups(pattern_groups: PatternGroups) -> CompiledPatternGroups:
    return {
        "patterns": [
            {
                "name": pattern["name"],
                "expressions": [re.compile(expr) for expr in pattern["expressions"]],
                "substitutions": pattern["substitutions"] if "substitutions" in pattern else None,
            }
            for pattern in pattern_groups["patterns"]
        ]
    }
