from __future__ import annotations

import re

TAG_PATTERN = re.compile(r"^[a-z0-9_()]+$")
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
}


def normalize_tags(raw_tags: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []

    for tag in raw_tags:
        cleaned = tag.strip().lower().replace(" ", "_")
        if (
            not cleaned
            or cleaned in seen
            or cleaned in BLOCKED_PROSE_TAGS
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
