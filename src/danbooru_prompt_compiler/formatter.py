from __future__ import annotations

from collections.abc import Iterable
from enum import Enum


class OutputFormat(str, Enum):
    grouped = "grouped"
    flat = "flat"


CATEGORY_ORDER = ["subject", "appearance", "clothing", "pose", "scene", "style", "composition", "other"]

SUBJECT_TAGS = {
    "1girl",
    "1boy",
    "2girls",
    "2boys",
    "solo",
    "girl",
    "boy",
    "multiple_girls",
    "multiple_boys",
}

POSE_KEYWORDS = (
    "standing",
    "sitting",
    "lying",
    "walking",
    "running",
    "kneeling",
    "pose",
    "looking",
    "facing",
    "holding",
    "smile",
    "blush",
    "open_mouth",
    "closed_mouth",
)

APPEARANCE_KEYWORDS = (
    "hair",
    "eyes",
    "breasts",
    "skin",
    "tail",
    "ears",
    "horns",
    "fang",
)

CLOTHING_KEYWORDS = (
    "shirt",
    "skirt",
    "dress",
    "sleeves",
    "uniform",
    "jacket",
    "coat",
    "kimono",
    "clothes",
    "shoes",
    "boots",
    "hat",
    "ribbon",
)

SCENE_TAGS = {
    "shrine",
    "city",
    "highway",
    "street",
    "indoors",
    "outdoors",
    "indoor",
    "outdoor",
    "garden",
    "forest",
    "sky",
    "clouds",
    "water",
    "rain",
    "snow",
    "night",
    "day",
    "evening",
    "sunset",
    "umbrella",
}

SCENE_KEYWORDS = (
    "background",
    "city",
    "street",
    "road",
    "room",
    "window",
    "building",
    "temple",
    "shrine",
    "rain",
    "snow",
    "cloud",
    "sky",
    "umbrella",
)

STYLE_KEYWORDS = (
    "lighting",
    "light",
    "shadow",
    "coloring",
    "pixel",
    "retro",
    "masterpiece",
    "highres",
    "detailed",
)

COMPOSITION_KEYWORDS = (
    "view",
    "angle",
    "portrait",
    "close-up",
    "closeup",
    "depth_of_field",
    "from_",
)


def format_variant(tags: list[str], output_format: OutputFormat = OutputFormat.grouped) -> str:
    if output_format == OutputFormat.flat:
        return ", ".join(tags)

    grouped = group_tags(tags)
    copy_lines = _copy_lines(grouped)
    lines: list[str] = []
    if copy_lines:
        lines.extend(["===", *copy_lines, "==="])
    lines.append("")
    for category in CATEGORY_ORDER:
        category_tags = grouped.get(category, [])
        if category_tags:
            lines.append(f"{category}: {', '.join(category_tags)}")
    return "\n".join(lines)


def format_clipboard_text(tags: list[str], output_format: OutputFormat = OutputFormat.grouped) -> str:
    if output_format == OutputFormat.flat:
        return ", ".join(tags)
    return "\n".join(_copy_lines(group_tags(tags)))


def group_tags(tags: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {category: [] for category in CATEGORY_ORDER}
    for tag in tags:
        grouped[_category_for_tag(tag)].append(tag)
    return {category: values for category, values in grouped.items() if values}


def _category_for_tag(tag: str) -> str:
    if tag in SUBJECT_TAGS:
        return "subject"
    if tag in SCENE_TAGS or _contains_any(tag, SCENE_KEYWORDS):
        return "scene"
    if _contains_any(tag, POSE_KEYWORDS):
        return "pose"
    if _contains_any(tag, APPEARANCE_KEYWORDS):
        return "appearance"
    if _contains_any(tag, CLOTHING_KEYWORDS):
        return "clothing"
    if _contains_any(tag, STYLE_KEYWORDS):
        return "style"
    if _contains_any(tag, COMPOSITION_KEYWORDS):
        return "composition"
    return "other"


def _contains_any(tag: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in tag for keyword in keywords)


def _copy_lines(grouped: dict[str, list[str]]) -> list[str]:
    return [
        ", ".join(grouped[category])
        for category in CATEGORY_ORDER
        if grouped.get(category)
    ]
