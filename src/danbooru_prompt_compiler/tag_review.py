"""Judging inferred tags against the picture they came from.

The tagger scores every tag on its own, so it reports plausible neighbours it
could not rule out and misses whatever fell under the threshold. A vision model
is the wrong tool for producing tags - asked for them directly it answered
``red_circle`` and ``kappa-7391``, neither of them Danbooru - but it can judge a
list it is handed against the image.

So the model only ever votes on names. Its removals have to name a tag that is
already on the list, its additions have to exist in the loaded dictionary, and
anything else it says is dropped and reported rather than acted on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REMOVE_PATTERN = re.compile(r"^\s*[-*]?\s*\**remove\**\s*[:：]\s*(.*)$", re.IGNORECASE)
ADD_PATTERN = re.compile(r"^\s*[-*]?\s*\**add\**\s*[:：]\s*(.*)$", re.IGNORECASE)
NONE_WORDS = {"", "none", "-", "なし", "n/a"}


@dataclass(frozen=True)
class TagReview:
    tags: list[str]
    removed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.removed or self.added)


def build_tag_review_request(tags: list[str], *, description: str = "") -> str:
    """Ask the vision model to judge a list, never to write one."""
    listed = ", ".join(tags)
    parts = [
        "These Danbooru tags were inferred from the attached image by a separate "
        "tagger. Judge them against the image.",
        "",
        f"Tags: {listed}",
        _labelled("Description of the image", description),
        "",
        "Answer with exactly two lines and nothing else:",
        "Remove: the listed tags that are not true of this image",
        "Add: Danbooru tags that are true of this image but missing from the list",
        "",
        "Write `Remove: none` or `Add: none` when there is nothing to report. "
        "Use canonical Danbooru spelling with underscores, and only name tags you "
        "can point to in the image.",
    ]
    return "\n".join(part for part in parts if part is not None)


def apply_tag_review(
    raw_output: str,
    *,
    tags: list[str],
    known_tags: set[str],
    protected: list[str] | None = None,
) -> TagReview:
    """The reviewed list, plus what was refused and why it was refusable."""
    protected_set = {_normalize(tag) for tag in (protected or [])}
    current = list(dict.fromkeys(tags))
    current_set = set(current)

    proposed_removals = _parse_line(raw_output, REMOVE_PATTERN)
    proposed_additions = _parse_line(raw_output, ADD_PATTERN)

    # A removal only means something for a tag that is on the list, and a tag the
    # user typed by hand is a statement about the image, not a guess to overrule.
    removed = [
        tag
        for tag in proposed_removals
        if tag in current_set and tag not in protected_set
    ]
    added: list[str] = []
    rejected: list[str] = []
    for tag in proposed_additions:
        if tag in current_set or tag in added:
            continue
        if tag in known_tags:
            added.append(tag)
        else:
            rejected.append(tag)

    removed_set = set(removed)
    reviewed = [tag for tag in current if tag not in removed_set] + added
    return TagReview(tags=reviewed, removed=removed, added=added, rejected=rejected)


def _parse_line(raw_output: str, pattern: re.Pattern[str]) -> list[str]:
    for line in (raw_output or "").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        body = match.group(1).strip()
        if body.lower() in NONE_WORDS:
            return []
        return [
            normalized
            for normalized in (_normalize(part) for part in body.split(","))
            if normalized and normalized.lower() not in NONE_WORDS
        ]
    return []


def _normalize(tag: str) -> str:
    return tag.strip().strip("`\"'").replace(" ", "_").lower()


def _labelled(label: str, value: str) -> str | None:
    value = (value or "").strip()
    return f"{label}: {value}" if value else None
