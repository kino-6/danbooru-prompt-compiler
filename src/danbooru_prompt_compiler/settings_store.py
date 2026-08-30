"""Remembering how the workbench was left.

Only the settings are kept, never the work: an image, an instruction or a
half-edited prompt belongs to the session that made it, and finding yesterday's
instruction waiting in the box is worse than finding it empty. What is kept is
the part nobody wants to set twice - which models to use, where Ollama is, how
many outputs, how far the next panel may move.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[2]
SETTINGS_PATH = BASE_DIR / "data" / "webui_settings.json"

# Everything else in the request is the work itself, or is resolved per run.
REMEMBERED_FIELDS: tuple[str, ...] = (
    "router_model",
    "compiler_model",
    "ollama_url",
    "general_threshold",
    "character_threshold",
    "max_image_tags",
    "variants",
    "generate_next_panel",
    "next_panel_change",
    "next_panel_time",
    "scene_template",
    "scene_model",
    "scene_sees_image",
    "also_prose",
    "action_override",
    "use_vision",
    "vision_model",
    "allow_private_image_urls",
    "apply_tag_exclusions",
)


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    """What was saved last time, or nothing at all.

    A file that has gone missing, unreadable or strange is not worth an error on
    startup; the built-in defaults are always a working answer.
    """
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    return {name: stored[name] for name in REMEMBERED_FIELDS if name in stored}


def save_settings(values: dict[str, Any], path: Path = SETTINGS_PATH) -> None:
    """Keep the remembered fields, quietly. A run must not fail over this."""
    kept = {name: values[name] for name in REMEMBERED_FIELDS if name in values}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(kept, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def remembered(stored: dict[str, Any], name: str, fallback: Any) -> Any:
    """The saved value for a control, or the built-in default.

    A saved value of the wrong type - a hand-edited file, or a field that has
    changed shape since it was written - falls back rather than reaching Gradio,
    where it would break the page rather than one control.
    """
    if name not in stored:
        return fallback
    value = stored[name]
    if isinstance(fallback, bool):
        return value if isinstance(value, bool) else fallback
    if isinstance(fallback, (int, float)):
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback
    if isinstance(fallback, str):
        return value if isinstance(value, str) else fallback
    return value
