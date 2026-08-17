from __future__ import annotations

import json
from pathlib import Path

import pytest

from danbooru_prompt_compiler.tag_filter import (
    DEFAULT_EXCLUSION_TEXT,
    is_excluded,
    load_exclusion_text,
    parse_exclusion_rules,
    save_exclusion_text,
    split_excluded,
)

TAG_DICTIONARY_PATH = Path(__file__).resolve().parents[1] / "data" / "tags.json"


def _dictionary_tags() -> list[str]:
    if not TAG_DICTIONARY_PATH.is_file():
        pytest.skip("tag dictionary is not available")
    stored = json.loads(TAG_DICTIONARY_PATH.read_text(encoding="utf-8"))
    tags = stored if isinstance(stored, list) else stored.get("tags", [])
    return [tag["name"] if isinstance(tag, dict) else str(tag) for tag in tags]


def test_default_exclusions_cover_censor_tags() -> None:
    rules = parse_exclusion_rules(DEFAULT_EXCLUSION_TEXT)

    assert is_excluded("censored", rules)
    assert is_excluded("bar_censor", rules)
    assert is_excluded("mosaic_censoring", rules)
    assert is_excluded("censored_nipples", rules)
    assert is_excluded("censor", rules)
    assert is_excluded("simple_background", rules)
    assert is_excluded("green_background", rules)
    assert not is_excluded("1girl", rules)


def test_no_censor_tag_in_the_dictionary_survives_the_defaults() -> None:
    rules = parse_exclusion_rules(DEFAULT_EXCLUSION_TEXT)
    censor_tags = [tag for tag in _dictionary_tags() if "censor" in tag]

    assert censor_tags, "the dictionary should contain censorship tags"
    assert [tag for tag in censor_tags if not is_excluded(tag, rules)] == []


def test_default_exclusions_cover_rendered_text_tags() -> None:
    rules = parse_exclusion_rules(DEFAULT_EXCLUSION_TEXT)
    text_tags = [
        tag
        for tag in _dictionary_tags()
        if tag.endswith("_text") or tag in {"text_focus", "watermark", "signature", "logo"}
    ]

    assert "english_text" in text_tags
    assert [tag for tag in text_tags if not is_excluded(tag, rules)] == []


@pytest.mark.parametrize(
    "tag",
    [
        "1girl",
        "texture",
        "paper_texture",
        "speech_bubble",
        "text_messaging",
        "analogous_colors",
        "long_hair",
    ],
)
def test_default_exclusions_keep_ordinary_tags(tag: str) -> None:
    # Wildcards are easy to over-broaden; these are the near misses.
    assert not is_excluded(tag, parse_exclusion_rules(DEFAULT_EXCLUSION_TEXT))


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
