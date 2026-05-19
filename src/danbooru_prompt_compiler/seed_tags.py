from __future__ import annotations

from .normalizer import normalize_tags, parse_tag_text

DEFAULT_SEED_TAGS = ("1girl", "solo")
EDIT_KEYWORD_SEEDS = {
    "神社": ("shrine",),
    "雨": ("rain",),
    "魔法少女": ("mahou_shoujo",),
    "悪落ち": ("dark_persona",),
    "悪堕ち": ("dark_persona",),
    "evil": ("dark_persona",),
    "dark": ("dark_persona",),
}
LOW_VALUE_SEED_TAGS = {
    "highres",
    "absurdres",
    "commentary",
    "commentary_request",
    "artist_name",
    "character_name",
}
SUBJECT_SEED_TAGS = {"1girl", "1boy", "2girls", "2boys", "solo"}
SCENE_KEYWORD_SEEDS = {"shrine", "rain", "city", "night", "forest", "school"}


def infer_subset_seed_tags(
    scene_description: str,
    *,
    edit_instruction: str | None = None,
    max_seed_tags: int = 2,
) -> list[str]:
    prompt_tags = _parse_prompt_seed_tags(scene_description)
    seeds: list[str] = []

    if edit_instruction:
        for keyword, keyword_seeds in EDIT_KEYWORD_SEEDS.items():
            if keyword in edit_instruction:
                seeds.extend(keyword_seeds)

    seeds.extend(tag for tag in prompt_tags if tag in SCENE_KEYWORD_SEEDS)
    seeds.extend(tag for tag in prompt_tags if tag in SUBJECT_SEED_TAGS)

    for tag in prompt_tags:
        if tag in LOW_VALUE_SEED_TAGS:
            continue
        seeds.append(tag)

    if not seeds:
        seeds.extend(DEFAULT_SEED_TAGS)

    return _dedupe(seeds)[:max_seed_tags]


def _parse_prompt_seed_tags(raw_text: str) -> list[str]:
    tags: list[str] = []
    for parsed in parse_tag_text(raw_text):
        tags.extend(parsed.split())
    return normalize_tags(tags)


def _dedupe(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped
