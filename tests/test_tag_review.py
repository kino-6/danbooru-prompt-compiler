from __future__ import annotations

from danbooru_prompt_compiler.tag_review import (
    apply_tag_review,
    build_tag_review_request,
)

KNOWN = {"1girl", "elf", "pointy_ears", "quiver", "bow_(weapon)", "smile"}


def test_the_request_asks_for_a_verdict_not_for_a_tag_list() -> None:
    request = build_tag_review_request(["1girl", "elf"], description="弓を持つ少女")

    assert "Tags: 1girl, elf" in request
    assert "弓を持つ少女" in request
    assert "Remove:" in request and "Add:" in request
    # The model votes on names it is given; it is never asked to write the list.
    assert "true of this image but missing from the list" in request


def test_a_proposal_outside_the_dictionary_is_dropped_and_reported() -> None:
    review = apply_tag_review(
        "Remove: none\nAdd: pointy_ears, red_circle, kappa-7391",
        tags=["1girl"],
        known_tags=KNOWN,
    )

    assert review.tags == ["1girl", "pointy_ears"]
    assert review.added == ["pointy_ears"]
    assert review.rejected == ["red_circle", "kappa-7391"]


def test_a_hand_typed_tag_survives_a_proposed_removal() -> None:
    review = apply_tag_review(
        "Remove: elf, smile\nAdd: none",
        tags=["1girl", "elf", "smile"],
        known_tags=KNOWN,
        protected=["elf"],
    )

    assert review.tags == ["1girl", "elf"]
    assert review.removed == ["smile"]


def test_a_removal_of_something_not_on_the_list_changes_nothing() -> None:
    review = apply_tag_review(
        "Remove: quiver\nAdd: none",
        tags=["1girl", "elf"],
        known_tags=KNOWN,
    )

    assert review.tags == ["1girl", "elf"]
    assert review.removed == []
    assert review.changed is False


def test_an_answer_in_no_recognizable_shape_leaves_the_list_alone() -> None:
    review = apply_tag_review(
        "I think the picture looks quite nice, honestly.",
        tags=["1girl", "elf"],
        known_tags=KNOWN,
    )

    assert review.tags == ["1girl", "elf"]
    assert review.changed is False


def test_sloppy_spelling_is_normalized_before_it_is_judged() -> None:
    review = apply_tag_review(
        "Remove: `Elf`\nAdd: Pointy Ears",
        tags=["1girl", "elf"],
        known_tags=KNOWN,
    )

    assert review.removed == ["elf"]
    assert review.added == ["pointy_ears"]
    assert review.tags == ["1girl", "pointy_ears"]
