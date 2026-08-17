from __future__ import annotations

import json
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

BASE_DIR = Path(__file__).resolve().parents[2]
EXCLUDED_TAGS_PATH = BASE_DIR / "data" / "excluded_tags.json"
DEFAULT_EXCLUDED_TAGS: tuple[str, ...] = (
    "simple_background",
    "halftone",
    "*_background",
    "censored",
    "*_censor",
    "*_censoring",
)


def format_exclusion_rules(rules: Iterable[str]) -> str:
    return ", ".join(rules)


DEFAULT_EXCLUSION_TEXT = format_exclusion_rules(DEFAULT_EXCLUDED_TAGS)


def parse_exclusion_rules(value: str) -> list[str]:
    return list(
        dict.fromkeys(
            rule.strip().lower().replace(" ", "_")
            for rule in (value or "").replace("\n", ",").split(",")
            if rule.strip()
        )
    )


def is_excluded(tag: str, rules: Iterable[str]) -> bool:
    name = tag.strip().lower()
    return any(fnmatchcase(name, rule) for rule in rules)


def split_excluded(
    tags: Iterable[T],
    rules: Iterable[str],
    *,
    key: Callable[[T], str] = str,
) -> tuple[list[T], list[T]]:
    rules = list(rules)
    kept: list[T] = []
    excluded: list[T] = []
    for tag in tags:
        (excluded if is_excluded(key(tag), rules) else kept).append(tag)
    return kept, excluded


def load_exclusion_text(path: Path = EXCLUDED_TAGS_PATH) -> str:
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_EXCLUSION_TEXT

    rules = stored.get("excluded_tags") if isinstance(stored, dict) else stored
    if not isinstance(rules, list):
        return DEFAULT_EXCLUSION_TEXT
    return format_exclusion_rules(
        parse_exclusion_rules(", ".join(str(rule) for rule in rules))
    )


def save_exclusion_text(value: str, path: Path = EXCLUDED_TAGS_PATH) -> str:
    rules = parse_exclusion_rules(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"excluded_tags": rules}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return format_exclusion_rules(rules)
