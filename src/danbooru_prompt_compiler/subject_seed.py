"""A subject invented on the spot, so a run needs nothing typed into it.

Picking a subject and a handful of situations by hand is the part of a sweep
that is work rather than result. This composes a subject from a small curated
vocabulary instead - a few hundred thousand combinations out of five slots -
and it does so locally: no model call to wait for, no call to fail, and real
variety rather than whatever sentence a small model reaches for first.

Both shapes come out of the same pick, because the sweep asks for one or the
other: `1girl, solo, elf, grey_hair, long_hair, leather_armor` for the tag
compiler, "an elf with long silver hair, in worn leather armour, carrying a
longbow" for the prose one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
VOCABULARY_PATH = BASE_DIR / "subjects" / "vocabulary.yaml"


@dataclass(frozen=True)
class SubjectOption:
    prose: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SubjectSlot:
    name: str
    label: str
    options: list[SubjectOption]
    # The word that joins this slot's fragment to the ones before it: "with"
    # for hair, "in" for an outfit. Empty for the opening fragment.
    lead: str = ""


def load_subject_vocabulary(path: Path = VOCABULARY_PATH) -> list[SubjectSlot]:
    """The slots on disk, in the order they are written in.

    Order is the sentence order, so it belongs to the file rather than to the
    code that reads it.
    """
    try:
        stored = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(stored, dict):
        return []
    slots = []
    for entry in stored.get("slots") or []:
        if not isinstance(entry, dict):
            continue
        options = [
            SubjectOption(
                prose=str(option.get("prose") or "").strip(),
                tags=[str(tag) for tag in option.get("tags") or []],
            )
            for option in entry.get("options") or []
            if isinstance(option, dict) and str(option.get("prose") or "").strip()
        ]
        if not options:
            continue
        slots.append(
            SubjectSlot(
                name=str(entry.get("name") or ""),
                label=str(entry.get("label") or entry.get("name") or ""),
                options=options,
                lead=str(entry.get("lead") or ""),
            )
        )
    return slots


def random_subject(
    slots: list[SubjectSlot],
    *,
    as_prose: bool = False,
    rng: random.Random | None = None,
) -> str:
    """One subject, as a sentence or as a tag line.

    An option with no tags at all - "nothing at all" for a prop - is a way of
    leaving a slot out, so it contributes to the sentence and nothing to the
    tags.
    """
    if not slots:
        return ""
    chooser = rng or random
    picked = [(slot, chooser.choice(slot.options)) for slot in slots]
    if as_prose:
        fragments = [
            f"{slot.lead} {option.prose}".strip() if slot.lead else option.prose
            for slot, option in picked
        ]
        return ", ".join(fragment for fragment in fragments if fragment)
    tags: list[str] = []
    for _slot, option in picked:
        for tag in option.tags:
            if tag not in tags:
                tags.append(tag)
    return ", ".join(tags)


def random_situation_names(
    names: list[str], count: int, *, rng: random.Random | None = None
) -> list[str]:
    """A sample of situations, kept in the order they were given in.

    The order is the page's order, so a random pick still reads down the groups
    the way the page does rather than in whatever order the sample came out.
    """
    if count < 1 or not names:
        return []
    chooser = rng or random
    chosen = set(chooser.sample(names, min(count, len(names))))
    return [name for name in names if name in chosen]
