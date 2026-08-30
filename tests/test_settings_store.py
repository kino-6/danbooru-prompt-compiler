from __future__ import annotations

import json
from pathlib import Path

from danbooru_prompt_compiler.settings_store import (
    REMEMBERED_FIELDS,
    load_settings,
    remembered,
    save_settings,
)
from danbooru_prompt_compiler.web_service import WebRunRequest


def test_the_work_is_never_remembered_only_the_settings() -> None:
    # Yesterday's instruction waiting in the box is worse than an empty one.
    for field in ("image_path", "image_url", "instruction", "base_prompt",
                  "edited_tags", "edited_description", "excluded_tags"):
        assert field not in REMEMBERED_FIELDS
    assert set(REMEMBERED_FIELDS) <= set(WebRunRequest.model_fields)


def test_a_round_trip_keeps_the_settings_and_drops_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "webui_settings.json"
    request = WebRunRequest(
        instruction="夜にして",
        vision_model="unseen-gemma4:26b",
        variants=2,
        next_panel_time=0.9,
    )

    save_settings(request.model_dump(), path)
    loaded = load_settings(path)

    assert loaded["vision_model"] == "unseen-gemma4:26b"
    assert loaded["variants"] == 2
    assert loaded["next_panel_time"] == 0.9
    assert "instruction" not in loaded


def test_a_missing_or_broken_file_is_not_worth_an_error_on_startup(tmp_path: Path) -> None:
    assert load_settings(tmp_path / "absent.json") == {}

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert load_settings(broken) == {}

    wrong_shape = tmp_path / "list.json"
    wrong_shape.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_settings(wrong_shape) == {}


def test_saving_where_it_cannot_write_does_not_take_the_run_down(tmp_path: Path) -> None:
    # The directory stands where the file should go, so the write must fail.
    blocked = tmp_path / "settings.json"
    blocked.mkdir()

    save_settings({"variants": 3}, blocked)


def test_a_value_of_the_wrong_type_falls_back_instead_of_reaching_gradio() -> None:
    stored = {
        "variants": "four",
        "use_vision": "yes",
        "general_threshold": None,
        "vision_model": 7,
        "next_panel_time": 0.9,
    }

    # A hand-edited file should cost one control its memory, not the whole page.
    assert remembered(stored, "variants", 4) == 4
    assert remembered(stored, "use_vision", True) is True
    assert remembered(stored, "general_threshold", 0.35) == 0.35
    assert remembered(stored, "vision_model", "qwen3-vl:8b") == "qwen3-vl:8b"
    assert remembered(stored, "next_panel_time", 0.3) == 0.9
    assert remembered({}, "variants", 4) == 4


def test_a_boolean_is_not_accepted_where_a_number_belongs() -> None:
    # bool is an int in Python, and True as 出力数 would reach Gradio as 1.
    assert remembered({"variants": True}, "variants", 4) == 4


def test_the_saved_file_is_readable_by_a_person(tmp_path: Path) -> None:
    path = tmp_path / "webui_settings.json"
    save_settings({"vision_model": "unseen-gemma4:26b", "variants": 2}, path)

    text = path.read_text(encoding="utf-8")
    assert json.loads(text)["vision_model"] == "unseen-gemma4:26b"
    assert text.endswith("\n")
    assert "\n  " in text
