from __future__ import annotations

import re

TAG_PATTERN = re.compile(r"^[a-z0-9_()]+$")
TAG_ALIASES = {
    "single_girl": "1girl",
    "one_girl": "1girl",
    "girl_alone": "1girl",
    "1girl_alone": "1girl",
}
BLOCKED_PROSE_TAGS = {
    "add",
    "added",
    "change",
    "changed",
    "edit",
    "edited",
    "instruction",
    "subtle",
    "remix",
    "composition",
    "operation_words",
}
BLOCKED_PROSE_PREFIXES = (
    "note_",
    "here_",
    "these_",
    "the_",
    "this_",
    "if_",
    "based_",
    "output_",
    "prose",
    "explanations",
    "bullets",
    "quotes",
)


def normalize_tags(raw_tags: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []

    for tag in raw_tags:
        cleaned = tag.strip().lower().replace(" ", "_")
        cleaned = TAG_ALIASES.get(cleaned, cleaned)
        if (
            not cleaned
            or cleaned in seen
            or cleaned in BLOCKED_PROSE_TAGS
            or cleaned.startswith(BLOCKED_PROSE_PREFIXES)
            or not TAG_PATTERN.fullmatch(cleaned)
        ):
            continue
        seen.add(cleaned)
        normalized.append(cleaned)

    return normalized


def parse_tag_text(raw_text: str) -> list[str]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return []

    if len(lines) == 1 and "," in lines[0]:
        parts = lines[0].split(",")
    else:
        parts = []
        for line in lines:
            line = line.removeprefix("- ").removeprefix("* ")
            if "," in line:
                parts.extend(line.split(","))
            else:
                parts.append(line)

    return [p.strip() for p in parts]
