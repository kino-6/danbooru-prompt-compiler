"""Natural-language image prompts built from a slot-filled template.

The Danbooru pipeline emits comma-separated tags. Newer image models take prose
instead, and the reliable way to write that prose is an atomic schema: one task
line, an ordered set of named sections, a delivery line, and an explicit list of
things to avoid. The templates in ``templates/*.yaml`` define those sections per
category; the model only fills them in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = BASE_DIR / "templates"
AVOID_SECTION = "Avoid"
DELIVERY_SECTION = "Delivery"
SECTION_PATTERN = re.compile(r"^\s*[-*]?\s*\**([A-Za-z][A-Za-z /]*?)\**\s*[:：]\s*(.+?)\s*$")
FENCE_PATTERN = re.compile(r"^\s*```.*$")


@dataclass(frozen=True)
class SceneTemplate:
    name: str
    label: str
    task: str
    sections: list[tuple[str, str]]
    delivery: str
    order: int = 100

    @property
    def section_names(self) -> list[str]:
        return [name for name, _guidance in self.sections]


def load_templates(directory: Path = TEMPLATE_DIR) -> list[SceneTemplate]:
    """Every template on disk, ordered for display. Unreadable files are skipped."""
    templates: list[SceneTemplate] = []
    for path in sorted(directory.glob("*.yaml")):
        template = _read_template(path)
        if template is not None:
            templates.append(template)
    return sorted(templates, key=lambda template: (template.order, template.label))


def find_template(name: str, templates: list[SceneTemplate]) -> SceneTemplate:
    for template in templates:
        if template.name == name:
            return template
    if not templates:
        raise ValueError("自然文プロンプトのテンプレートが見つかりません。")
    return templates[0]


def build_scene_prompt(
    template: SceneTemplate,
    *,
    image_tags: list[str],
    image_description: str,
    instruction: str,
    base_prompt: str,
    avoid_terms: list[str],
) -> str:
    """The request handed to the text model, one section per template slot."""
    section_lines = [
        f"{name}: {guidance}" for name, guidance in template.sections
    ]
    known = [
        _labelled("Danbooru tags of the reference image", ", ".join(image_tags)),
        _labelled("Description of the reference image", image_description),
        _labelled("Existing prompt", base_prompt),
        _labelled("User request", instruction),
    ]
    return "\n".join(
        part
        for part in (
            "/no_think",
            "You write prompts for a natural-language image model.",
            template.task,
            "",
            "Fill in every section below on its own line, in this exact order, "
            "using the format `Section: content`.",
            "Write plain English sentence fragments, not tag lists, and describe "
            "only what should be visible in the image.",
            "Base every section on the reference material; do not invent a "
            "different character, setting, or outfit.",
            "Do not add sections, headings, numbering, markdown, or commentary.",
            "",
            *section_lines,
            "",
            *[part for part in known if part],
            "",
            _avoid_request(avoid_terms),
        )
        if part is not None
    )


def render_scene_prompt(
    raw_output: str,
    template: SceneTemplate,
    *,
    avoid_terms: list[str],
) -> str:
    """Force the model's answer back into the template's shape.

    A small model drops sections, reorders them, or wraps the answer in a code
    fence, so the sections it did fill are kept and everything else is rebuilt.
    """
    filled = _parse_sections(raw_output)
    lines = [template.task, ""]
    for name in template.section_names:
        value = filled.get(name.lower())
        if value:
            lines.append(f"{name}: {value}")
    if len(lines) == 2:
        # Nothing parsed: keep the model's own words rather than an empty shell.
        body = _strip_fences(raw_output).strip()
        if body:
            lines.append(body)
    lines.append(f"{DELIVERY_SECTION}: {filled.get('delivery') or template.delivery}")
    if avoid_terms:
        lines.append(f"{AVOID_SECTION}: {', '.join(avoid_terms)}")
    return "\n".join(lines).strip()


def humanize_avoid_terms(terms: list[str]) -> list[str]:
    """Tag-shaped exclusion words as words a prose model understands."""
    humanized: list[str] = []
    for term in terms:
        cleaned = term.replace("_", " ").strip()
        if cleaned and cleaned not in humanized:
            humanized.append(cleaned)
    return humanized


def _read_template(path: Path) -> SceneTemplate | None:
    try:
        stored = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(stored, dict):
        return None

    sections = stored.get("sections")
    if not isinstance(sections, dict) or not sections:
        return None
    return SceneTemplate(
        name=path.stem,
        label=str(stored.get("label") or path.stem),
        task=str(stored.get("task") or ""),
        sections=[(str(name), str(guidance)) for name, guidance in sections.items()],
        delivery=str(stored.get("delivery") or ""),
        order=int(stored.get("order") or 100),
    )


def _labelled(label: str, value: str) -> str:
    value = (value or "").strip()
    return f"{label}: {value}" if value else ""


def _avoid_request(avoid_terms: list[str]) -> str:
    if not avoid_terms:
        return ""
    return (
        "Never describe any of these, and never mention them in your answer: "
        f"{', '.join(avoid_terms)}."
    )


def _parse_sections(raw_output: str) -> dict[str, str]:
    filled: dict[str, str] = {}
    for line in _strip_fences(raw_output).splitlines():
        match = SECTION_PATTERN.match(line)
        if not match:
            continue
        name, value = match.group(1).strip().lower(), match.group(2).strip()
        if value and name not in filled:
            filled[name] = value
    return filled


def _strip_fences(raw_output: str) -> str:
    return "\n".join(
        line for line in (raw_output or "").splitlines() if not FENCE_PATTERN.match(line)
    )
