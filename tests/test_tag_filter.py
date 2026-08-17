from __future__ import annotations

import json

from danbooru_prompt_compiler.tag_filter import (
    DEFAULT_EXCLUSION_TEXT,
    is_excluded,
    load_exclusion_text,
    parse_exclusion_rules,
    save_exclusion_text,
    split_excluded,
)


def test_default_exclusions_cover_censor_tags() -> None:
    rules = parse_exclusion_rules(DEFAULT_EXCLUSION_TEXT)

    assert is_excluded("censored", rules)
    assert is_excluded("bar_censor", rules)
    assert is_excluded("mosaic_censoring", rules)
    assert is_excluded("simple_background", rules)
    assert is_excluded("green_background", rules)
    assert not is_excluded("censor", rules)
    assert not is_excluded("1girl", rules)


def test_rules_are_normalized_and_deduplicated() -> None:
    rules = parse_exclusion_rules("Bar Censor, censored,\n censored , *_censor,")

    assert rules == ["bar_censor", "censored", "*_censor"]


def test_split_excluded_keeps_order_and_reports_removals() -> None:
    kept, excluded = split_excluded(
        ["1girl", "censored", "rain", "bar_censor"],
        parse_exclusion_rules("censored, *_censor"),
    )

    assert kept == ["1girl", "rain"]
    assert excluded == ["censored", "bar_censor"]


def test_saved_exclusions_round_trip(tmp_path) -> None:
    path = tmp_path / "excluded_tags.json"

    saved = save_exclusion_text("Censored,  bar_censor , censored", path)

    assert saved == "censored, bar_censor"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "excluded_tags": ["censored", "bar_censor"]
    }
    assert load_exclusion_text(path) == "censored, bar_censor"


def test_missing_or_broken_store_falls_back_to_defaults(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    assert load_exclusion_text(missing) == DEFAULT_EXCLUSION_TEXT
    assert load_exclusion_text(broken) == DEFAULT_EXCLUSION_TEXT
