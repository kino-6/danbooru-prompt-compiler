"""What is going on in the picture, as a direction rather than a description.

The presets in ``presets/`` say how an image should be rendered - SDXL, a JRPG
look, clean line art. They say nothing about what is happening in it, and that
is the other half of a prompt: the same character eating breakfast, mid-fight,
or halfway through a sentence is three different pictures.

A situation is one of those directions. It carries words for the model and a
handful of Danbooru tags that typify it, and it works two ways: on its own it is
enough to generate from, and alongside an instruction or an image it steers what
is already there. The tags are candidates, not output - they go in as a hint the
compiler may use, and the dictionary has the final say as everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
SITUATION_DIR = BASE_DIR / "situations"
NO_SITUATION = ""


@dataclass(frozen=True)
class Situation:
    name: str
    label: str
    guidance: str
    tags: list[str] = field(default_factory=list)
    order: int = 100


def load_situations(directory: Path = SITUATION_DIR) -> list[Situation]:
    """Every situation on disk, ordered for display. Bad files are skipped."""
    situations: list[Situation] = []
    for path in sorted(directory.glob("*.yaml")):
        situation = _read_situation(path)
        if situation is not None:
            situations.append(situation)
    return sorted(situations, key=lambda item: (item.order, item.label))


def find_situation(name: str, situations: list[Situation]) -> Situation | None:
    """The named situation, or None for the no-situation case.

    Unlike a template, having none is a legitimate answer: most runs are not
    aimed at any particular kind of moment.
    """
    if not name:
        return None
    for situation in situations:
        if situation.name == name:
            return situation
    return None


def situation_choices(situations: list[Situation]) -> list[tuple[str, str]]:
    """Dropdown entries, with the no-situation case named rather than blank."""
    return [("（指定なし）", NO_SITUATION)] + [
        (situation.label, situation.name) for situation in situations
    ]


def situation_direction(situation: Situation | None) -> str:
    """The words a situation adds to whatever the run was already asked to do."""
    if situation is None:
        return ""
    # "Make it a scene of this situation" is an instruction to replace, and it
    # was taken as one: an elf with a bow in a battle came back as a battle with
    # no elf. What the situation governs is named before the situation itself.
    lines = [
        "状況の指定。人物・外見・服装・場所は変えず、"
        "「何をしているか」だけをこれに合わせる: " + situation.guidance.strip()
    ]
    if situation.tags:
        # Candidates rather than requirements: the scene decides which of them
        # are true, and the dictionary decides whether they may be written.
        lines.append(
            "参考タグ（当てはまるものだけ使い、無理に全部入れない）: "
            + ", ".join(situation.tags)
        )
    return "\n".join(lines)


def _read_situation(path: Path) -> Situation | None:
    try:
        stored = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(stored, dict):
        return None
    guidance = str(stored.get("guidance") or "").strip()
    if not guidance:
        return None
    tags = stored.get("tags")
    return Situation(
        name=path.stem,
        label=str(stored.get("label") or path.stem),
        guidance=guidance,
        tags=[str(tag) for tag in tags] if isinstance(tags, list) else [],
        order=int(stored.get("order") or 100),
    )
