"""Proposing the moment just after the panel in front of you.

Rewriting a tag list with a text model does not produce a next panel; asked for
one it hands back the list it was given. On the sample portrait every change
setting returned `standing, looking_at_viewer, closed_mouth,
holding_bow_(weapon)` unchanged, and raising the change amount only dropped
appearance tags - moving the panel less rather than more, which is backwards.

So the model that can see the picture is asked instead, and only for a verdict
on names: which of the current tags stop being true a moment later, and which
dictionary tags become true. Its answer is then checked. A panel whose pose,
gaze and framing match the one it came from is not a next panel, and saying so
is more use than handing back a copy.
"""

from __future__ import annotations

import re

from .formatter import group_tags

NEXT_PATTERN = re.compile(r"^\s*[-*]?\s*\**next\**\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)
LABELLED_PATTERN = re.compile(
    r"^\s*[-*]?\s*\**(next|remove|add)\**\s*[:：]", re.IGNORECASE
)
# A line that is a tag list rather than a sentence: no sentence punctuation.
TAG_LINE_PATTERN = re.compile(r"^[A-Za-z0-9_()\-, ]+$")

# The change slider names aspects; the formatter names categories. This is the
# one place the two vocabularies meet.
PRESERVE_CATEGORIES: dict[str, str] = {
    "character": "subject",
    "appearance": "appearance",
    "clothing": "clothing",
}
# What has to differ for the result to be a different moment at all. Appearance
# and clothing are deliberately absent: a girl who has not moved is not a next
# panel just because her hair tag changed.
MOVING_CATEGORIES: tuple[str, ...] = ("pose", "composition")


def protected_tags(tags: list[str], preserve: list[str]) -> list[str]:
    """The current tags the proposal is not allowed to drop."""
    grouped = group_tags(tags)
    keep: list[str] = []
    for aspect in preserve:
        for tag in grouped.get(PRESERVE_CATEGORIES.get(aspect, aspect), []):
            if tag not in keep:
                keep.append(tag)
    return keep


def build_next_panel_request(
    tags: list[str],
    *,
    description: str,
    movement: str,
    latitude: str,
    protected: list[str],
) -> str:
    """Ask what changes in the next instant, in names rather than prose."""
    parts = [
        "The attached image is the current panel of a sequence. Decide what the "
        "very next instant looks like: the same character, a moment later.",
        "",
        f"Tags of the current panel: {', '.join(tags)}",
        _labelled("Description of the current panel", description),
        f"How much time passes: {movement}",
        f"What may be different: {latitude}",
        _labelled(
            "These stay true and must not be removed", ", ".join(protected)
        ),
        "",
        "Answer with exactly three lines and nothing else:",
        "Next: one sentence saying what the character is doing in that instant",
        "Remove: the current tags that stop being true in it",
        "Add: Danbooru tags that become true in it",
        "",
        "Write the sentence first and let it decide the tags. If the character "
        "is in the middle of an action - drawing a bow, reaching, running, "
        "turning - the next instant is the next stage of that action, so name "
        "that stage. A change of gaze or expression on its own is not a next "
        "panel: say what the hands and the body do. Choose something the body "
        "in the image could physically reach from where it is now. Use "
        "canonical Danbooru spelling with underscores, and never name the "
        "passage of time itself.",
    ]
    return "\n".join(part for part in parts if part is not None)


def normalize_panel_answer(raw_output: str) -> str:
    """Put the labels back when the model answered in the right shape without them.

    Asked for three labelled lines it sometimes returns three bare ones - the
    sentence, the removals, the additions - and the content is exactly right.
    Discarding that over a missing prefix threw away the best answers measured,
    so an unlabelled answer in the expected shape is relabelled instead.
    """
    lines = [line.strip() for line in (raw_output or "").splitlines() if line.strip()]
    if len(lines) != 3 or any(LABELLED_PATTERN.match(line) for line in lines):
        return raw_output
    if not all(TAG_LINE_PATTERN.match(line) for line in lines[1:]):
        return raw_output
    return f"Next: {lines[0]}\nRemove: {lines[1]}\nAdd: {lines[2]}"


def described_moment(raw_output: str) -> str:
    """The sentence the model wrote before its tags, if it wrote one.

    Worth surfacing: it says what the panel is meant to be, which the tag list
    alone leaves the reader to infer.
    """
    for line in (raw_output or "").splitlines():
        match = NEXT_PATTERN.match(line)
        if match:
            return match.group(1).strip()
    return ""


def panel_moved(current: list[str], proposed: list[str]) -> bool:
    """Whether the proposal is a different moment rather than a re-sort."""
    return _moving(current) != _moving(proposed)


def _moving(tags: list[str]) -> set[str]:
    grouped = group_tags(tags)
    return {tag for category in MOVING_CATEGORIES for tag in grouped.get(category, [])}


def _labelled(label: str, value: str) -> str | None:
    value = (value or "").strip()
    return f"{label}: {value}" if value else None
