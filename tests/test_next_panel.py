from __future__ import annotations

from danbooru_prompt_compiler.next_panel import (
    build_next_panel_request,
    described_moment,
    normalize_panel_answer,
    panel_moved,
    protected_tags,
)

CURRENT = [
    "1girl",
    "solo",
    "long_hair",
    "school_uniform",
    "standing",
    "looking_at_viewer",
    "holding_bow_(weapon)",
]


def test_protected_tags_follow_the_aspects_the_slider_holds_fixed() -> None:
    keep = protected_tags(CURRENT, ["character", "appearance", "clothing"])

    assert "1girl" in keep and "solo" in keep
    assert "long_hair" in keep
    assert "school_uniform" in keep
    # Pose is what the next panel is for; holding it fixed would defeat the point.
    assert "standing" not in keep
    assert "looking_at_viewer" not in keep


def test_dropping_an_aspect_from_the_slider_releases_its_tags() -> None:
    assert "school_uniform" not in protected_tags(CURRENT, ["character", "appearance"])
    assert "long_hair" not in protected_tags(CURRENT, ["character"])


def test_a_panel_that_only_changes_appearance_has_not_moved() -> None:
    restyled = [tag if tag != "long_hair" else "short_hair" for tag in CURRENT]

    # A girl who has not moved is not a next panel because her hair tag changed.
    assert panel_moved(CURRENT, restyled) is False
    assert panel_moved(CURRENT, CURRENT) is False


def test_a_panel_that_changes_the_pose_has_moved() -> None:
    drawing = [tag for tag in CURRENT if tag != "holding_bow_(weapon)"] + ["drawing_bow"]
    turned = [tag for tag in CURRENT if tag != "looking_at_viewer"] + ["looking_back"]

    assert panel_moved(CURRENT, drawing) is True
    assert panel_moved(CURRENT, turned) is True


def test_the_request_asks_for_the_sentence_before_the_tags() -> None:
    request = build_next_panel_request(
        CURRENT,
        description="弓を構える少女",
        movement="a second or two - the action reaches its next stage",
        latitude="the pose and the framing only",
        protected=["1girl", "long_hair"],
    )

    assert "Tags of the current panel: 1girl, solo" in request
    assert "弓を構える少女" in request
    assert "How much time passes: a second or two" in request
    assert "What may be different: the pose and the framing only" in request
    assert "must not be removed: 1girl, long_hair" in request
    # The sentence comes first so that it drags the tags along behind it.
    assert request.index("Next:") < request.index("Remove:") < request.index("Add:")
    assert "A change of gaze or expression on its own is not a next panel" in request


def test_the_moment_sentence_is_read_back_and_a_missing_one_is_survivable() -> None:
    answer = (
        "Next: the character pulls the bowstring further back towards her face.\n"
        "Remove: holding_bow_(weapon)\n"
        "Add: drawing_bow"
    )

    assert described_moment(answer) == (
        "the character pulls the bowstring further back towards her face."
    )
    assert described_moment("Remove: none\nAdd: none") == ""
    assert described_moment("") == ""


def test_an_answer_that_drops_the_labels_is_relabelled_rather_than_discarded() -> None:
    # Measured: the model answers in the right shape without the prefixes, and
    # the content is exactly right. Throwing it away lost the best proposals.
    bare = (
        "The character pulls the bowstring back further to prepare for the shot.\n"
        "from_side, holding_bow_(weapon)\n"
        "aiming, drawing_bow, looking_at_viewer"
    )

    normalized = normalize_panel_answer(bare)

    assert "Next: The character pulls the bowstring back further" in normalized
    assert "Remove: from_side, holding_bow_(weapon)" in normalized
    assert "Add: aiming, drawing_bow, looking_at_viewer" in normalized
    assert described_moment(normalized).startswith("The character pulls")


def test_a_labelled_answer_is_left_exactly_as_it_is() -> None:
    labelled = "Next: she draws the bow.\nRemove: holding_bow_(weapon)\nAdd: drawing_bow"

    assert normalize_panel_answer(labelled) == labelled


def test_prose_is_not_mistaken_for_the_three_line_shape() -> None:
    # Three lines, but the last two are sentences rather than tag lists.
    prose = (
        "She stands with the bow.\n"
        "Then she draws it back, slowly.\n"
        "The arrow points at the ground."
    )

    assert normalize_panel_answer(prose) == prose
    assert normalize_panel_answer("just one line") == "just one line"
