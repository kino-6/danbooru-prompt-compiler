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

    # Never in the answer's own shape: a `Section: guidance` line is a valid-
    # looking answer, and copying one back is the easiest thing to do.
    assert "- Subject -> cover who is in it" in request
    assert "Subject: who is in it" not in request
    assert "- Lighting -> cover how it is lit" in request
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
    # One section per line: run together they cannot be told apart afterwards.
    assert plain.splitlines() == [
        "a miko, long black hair.",
        "white kosode and red hakama.",
        "standing before a shrine.",
    ]
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


def test_the_sub_headings_a_model_writes_inside_a_section_go_too() -> None:
    # The template's guidance is a list of what to cover, and the model answers
    # by repeating each word as a label. Stripping the section name alone left
    # exactly the noise the stripping was for.
    plain = flatten_scene_prompt(
        "Subject: a miko, apparent age: young adult, hair: long black\n"
        "Clothing: Outfit: white kosode and red hakama, accessories: none\n"
        "Lighting: key light direction: from above, colour temperature: cool"
    )

    for label in ("Outfit:", "accessories:", "colour temperature:", "apparent age:"):
        assert label not in plain
    assert plain.splitlines() == [
        "a miko, young adult, long black.",
        "white kosode and red hakama, none.",
        "from above, cool.",
    ]


def test_stripping_labels_leaves_the_separators_readable() -> None:
    plain = flatten_scene_prompt("Subject: a miko, hair: long black, eyes: dark")

    # The match reaches back over the separator's space to take the label.
    assert ",young" not in plain
    assert plain == "a miko, long black, dark."


def test_a_colon_that_is_not_a_label_survives() -> None:
    plain = flatten_scene_prompt("Layout: full body shot, aspect ratio 16:9, centered")

    assert "16:9" in plain


def test_prose_without_sub_headings_is_left_alone() -> None:
    plain = flatten_scene_prompt("Subject: a young elf girl with a bow, standing in profile")

    assert plain == "a young elf girl with a bow, standing in profile."


def test_a_section_that_repeats_another_is_not_printed_twice() -> None:
    # A small model fills every section with the whole picture rather than its
    # own aspect, so seven sections came back as the same sentence seven times.
    plain = flatten_scene_prompt(
        "Subject: an elf girl, holding a bow, standing, looking at the viewer\n"
        "Clothing: a simple tunic, holding a bow, standing\n"
        "Pose: standing, looking at the viewer, one foot forward"
    )

    assert plain.splitlines() == [
        "an elf girl, holding a bow, standing, looking at the viewer.",
        "a simple tunic.",
        "one foot forward.",
    ]


def test_a_section_with_nothing_new_to_say_is_dropped_entirely() -> None:
    plain = flatten_scene_prompt(
        "Subject: an elf girl, standing\n"
        "Clothing: an elf girl, standing\n"
        "Pose: kneeling"
    )

    assert plain.splitlines() == ["an elf girl, standing.", "kneeling."]


def test_the_request_asks_each_section_to_stay_in_its_lane() -> None:
    request = build_scene_prompt(
        TEMPLATE,
        image_tags=["1girl"],
        image_description="",
        instruction="",
        base_prompt="",
        avoid_terms=[],
    )

    assert "Each section covers its own aspect and nothing else" in request
    assert "Never repeat in one section what another section has already said" in request


def test_a_section_answered_with_its_own_guidance_is_not_an_answer() -> None:
    # Presented as `Section: guidance`, the request was a list of valid-looking
    # answers, and this one came back verbatim run after run.
    rendered = render_scene_prompt(
        "Subject: who is in it\nLighting: warm evening light",
        TEMPLATE,
        avoid_terms=[],
    )

    assert "who is in it" not in rendered
    assert "Lighting: warm evening light" in rendered


def test_an_echo_is_recognised_through_punctuation_and_case() -> None:
    rendered = render_scene_prompt(
        "Subject: Who is in it.\nLighting: warm evening light",
        TEMPLATE,
        avoid_terms=[],
    )

    assert "Who is in it" not in rendered


def test_the_request_says_the_guidance_is_not_an_answer() -> None:
    request = build_scene_prompt(
        TEMPLATE,
        image_tags=["1girl"],
        image_description="",
        instruction="",
        base_prompt="",
        avoid_terms=[],
    )

    assert "they are not an answer and must never be written back" in request


def test_the_situation_is_given_as_a_direction_not_as_words_to_reuse() -> None:
    """A resting scene came back as the direction itself, options and all.

    "off duty and unguarded, sitting or lying down, weight let go, gear set
    aside, a drink or a book to hand" is the instruction, not a picture: the
    model has to choose one version of it, the way it does for the sections.
    """
    request = build_scene_prompt(
        TEMPLATE,
        image_tags=[],
        image_description="",
        instruction="",
        base_prompt="",
        avoid_terms=[],
        situation_guidance="Off duty and unguarded: a drink or a book to hand.",
    )

    assert "Off duty and unguarded: a drink or a book to hand." in request
    assert "Never write these words back." in request
    assert "choose one concrete version" in request


def test_no_situation_adds_nothing_to_the_request() -> None:
    request = build_scene_prompt(
        TEMPLATE,
        image_tags=["1girl"],
        image_description="",
        instruction="",
        base_prompt="",
        avoid_terms=[],
    )

    assert "Situation to depict" not in request


REST_GUIDANCE = (
    "Off duty and unguarded: sitting or lying down, weight let go, "
    "gear set aside, a drink or a book to hand."
)


def test_a_section_that_copied_the_direction_keeps_only_its_own_words() -> None:
    """Asking the model not to copy works most of the time, which is not enough.

    The same sweep answered "A silver-haired elf with a bow." on one run and
    that subject followed by the whole direction on the next, so the direction
    is taken back out rather than left to the dice.
    """
    rendered = render_scene_prompt(
        "Subject: A silver-haired elf with a bow, off duty and unguarded, "
        "sitting or lying down, weight let go, gear set aside, a drink or a "
        "book to hand.",
        TEMPLATE,
        avoid_terms=[],
        situation_guidance=REST_GUIDANCE,
    )

    assert "Subject: A silver-haired elf with a bow." in rendered
    assert "weight let go" not in rendered
    assert "gear set aside" not in rendered


def test_the_model_s_own_realisation_of_the_direction_survives() -> None:
    rendered = render_scene_prompt(
        "Subject: The elf is lying down on a weathered bench with a book open.",
        TEMPLATE,
        avoid_terms=[],
        situation_guidance=REST_GUIDANCE,
    )

    # "lying down on a weathered bench" is the picture the direction asked for,
    # not the direction: scrubbing it would be the bug, not the fix.
    assert "lying down on a weathered bench with a book open" in rendered


def test_a_section_left_with_nothing_keeps_what_the_model_wrote() -> None:
    rendered = render_scene_prompt(
        "Subject: Off duty and unguarded.",
        TEMPLATE,
        avoid_terms=[],
        situation_guidance=REST_GUIDANCE,
    )

    # An empty section says less than an echoed one; a fragment says least.
    assert "Subject: Off duty and unguarded." in rendered
