from __future__ import annotations

import textwrap
from pathlib import Path

from danbooru_prompt_compiler.scene_prompt import (
    SceneTemplate,
    build_scene_prompt,
    find_template,
    flatten_scene_prompt,
    humanize_avoid_terms,
    humanize_tags,
    load_templates,
    render_scene_prompt,
    scene_avoid_line,
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


def test_humanize_tags_drops_qualifiers_and_collapses_the_duplicates_it_makes() -> None:
    # Two tags can share a word once the qualifier goes. The reference image is
    # what tells them apart afterwards, so the duplicate is not worth keeping.
    assert humanize_tags(["bow_(weapon)", "bow_(ornament)", "long_hair"]) == [
        "bow",
        "long hair",
    ]


def test_request_names_the_tags_as_observed_facts_without_their_qualifiers() -> None:
    request = build_scene_prompt(
        TEMPLATE,
        image_tags=["bow_(weapon)", "arrow_(projectile)", "pointy_ears"],
        image_description="",
        instruction="",
        base_prompt="",
        avoid_terms=humanize_avoid_terms(["bar_censor", "*censor*"]),
    )

    assert "Observed in the reference image: bow, arrow, pointy ears" in request
    assert "(weapon)" not in request
    assert "Every term listed as observed was detected in the reference image" in request
    # Exclusion words are the user's own wording, wildcards included, so they are
    # humanized on their own terms rather than run through the tag rules.
    assert "bar censor, *censor*" in request


def test_the_attached_image_is_announced_only_when_it_travels_with_the_request() -> None:
    options = dict(
        image_tags=["1girl"],
        image_description="",
        instruction="",
        base_prompt="",
        avoid_terms=[],
    )
    attached = "The reference image is attached."

    assert attached in build_scene_prompt(TEMPLATE, sees_image=True, **options)
    assert attached not in build_scene_prompt(TEMPLATE, **options)


def test_a_request_without_tags_makes_no_claim_about_observations() -> None:
    request = build_scene_prompt(
        TEMPLATE,
        image_tags=[],
        image_description="",
        instruction="雨の神社",
        base_prompt="",
        avoid_terms=[],
    )

    assert "Observed in the reference image" not in request
    assert "was detected in the reference image" not in request


RENDERED = (
    "Produce a character design sheet for a single character.\n"
    "\n"
    "Subject: a miko, long black hair\n"
    "Clothing: white kosode and red hakama.\n"
    "Pose: standing before a shrine\n"
    "Delivery: One full-body main view on a plain neutral background.\n"
    "Avoid: simple background, halftone, signature"
)


def test_the_pasteable_prose_drops_the_scaffolding_that_wrote_it() -> None:
    plain = flatten_scene_prompt(RENDERED)

    # `Subject:` and the rest are how the prompt was written; an image model
    # reads them as words.
    assert "Subject:" not in plain and "Clothing:" not in plain
    assert plain.startswith("a miko, long black hair. white kosode and red hakama.")
    assert "standing before a shrine." in plain
    # The task line addresses the model about the deliverable, not the picture.
    assert "character design sheet" not in plain
    assert "One full-body main view" not in plain
    # Every image model takes the negative separately, so it is not in the body.
    assert "simple background" not in plain


def test_the_avoid_terms_come_back_on_their_own() -> None:
    assert scene_avoid_line(RENDERED) == "simple background, halftone, signature"
    assert scene_avoid_line("Subject: a girl") == ""


def test_prose_with_nothing_parseable_flattens_to_nothing() -> None:
    assert flatten_scene_prompt("") == ""
    assert flatten_scene_prompt("just a sentence with no sections") == ""
