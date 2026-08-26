from __future__ import annotations

import textwrap
from pathlib import Path

from danbooru_prompt_compiler.scene_prompt import (
    SceneTemplate,
    build_scene_prompt,
    find_template,
    humanize_avoid_terms,
    load_templates,
    render_scene_prompt,
)

TEMPLATE = SceneTemplate(
    name="demo",
    label="デモ",
    task="Produce a demo image.",
    sections=[("Subject", "who is in it"), ("Lighting", "how it is lit")],
    delivery="One finished image.",
)


def test_builtin_templates_load_in_display_order() -> None:
    templates = load_templates()
    names = [template.name for template in templates]

    assert names[:2] == ["character_sheet", "storyboard_panel"]
    assert all(template.sections for template in templates)
    assert all(template.task and template.delivery for template in templates)


def test_unreadable_templates_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "broken.yaml").write_text("sections: [not, a, mapping]", encoding="utf-8")
    (tmp_path / "good.yaml").write_text(
        textwrap.dedent(
            """
            label: よい
            task: Do the thing.
            sections:
              Subject: who
            delivery: One image.
            """
        ),
        encoding="utf-8",
    )

    templates = load_templates(tmp_path)

    assert [template.name for template in templates] == ["good"]


def test_find_template_falls_back_to_the_first_entry() -> None:
    other = SceneTemplate(name="other", label="他", task="t", sections=[("A", "a")], delivery="d")

    assert find_template("other", [TEMPLATE, other]) is other
    assert find_template("missing", [TEMPLATE, other]) is TEMPLATE


def test_request_carries_the_slots_and_the_reference_material() -> None:
    request = build_scene_prompt(
        TEMPLATE,
        image_tags=["1girl", "rain"],
        image_description="石段に立つ少女",
        instruction="夜にして",
        base_prompt="",
        avoid_terms=["censored", "watermark"],
    )

    assert "Subject: who is in it" in request
    assert "Lighting: how it is lit" in request
    assert "1girl, rain" in request
    assert "石段に立つ少女" in request
    assert "夜にして" in request
    assert "censored, watermark" in request
    # An empty field must not leave a dangling label behind.
    assert "Existing prompt:" not in request


def test_render_rebuilds_the_template_shape_from_a_sloppy_answer() -> None:
    raw = textwrap.dedent(
        """
        ```
        **Lighting**: overcast rain light
        Subject: a young woman on stone steps
        Notes: this line is not part of the template
        ```
        """
    )

    rendered = render_scene_prompt(raw, TEMPLATE, avoid_terms=["censored"])

    assert rendered.splitlines() == [
        "Produce a demo image.",
        "",
        "Subject: a young woman on stone steps",
        "Lighting: overcast rain light",
        "Delivery: One finished image.",
        "Avoid: censored",
    ]


def test_render_keeps_prose_the_model_wrote_outside_the_sections() -> None:
    rendered = render_scene_prompt("A girl stands in the rain.", TEMPLATE, avoid_terms=[])

    assert "A girl stands in the rain." in rendered
    assert rendered.endswith("Delivery: One finished image.")


def test_avoid_terms_read_as_words_not_tags() -> None:
    assert humanize_avoid_terms(["simple_background", "bar_censor", "bar_censor"]) == [
        "simple background",
        "bar censor",
    ]
