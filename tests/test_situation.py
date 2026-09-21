from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

import pytest

from danbooru_prompt_compiler.situation import (
    NO_SITUATION,
    Situation,
    find_situation,
    group_situations,
    load_situations,
    sampled_tags,
    situation_choices,
    situation_direction,
)

BATTLE = Situation(
    name="battle",
    label="戦闘",
    guidance="Mid-fight: a weapon already in motion.",
    tags=["fighting_stance", "motion_blur"],
)


def test_the_shipped_situations_load_in_display_order() -> None:
    situations = load_situations()

    names = [situation.name for situation in situations]
    assert "everyday" in names and "battle" in names
    assert names.index("everyday") < names.index("battle")
    assert all(situation.guidance for situation in situations)


def test_a_file_without_a_direction_is_not_a_situation(tmp_path: Path) -> None:
    (tmp_path / "empty.yaml").write_text("label: 空\n", encoding="utf-8")
    (tmp_path / "broken.yaml").write_text("[not, a, mapping]", encoding="utf-8")
    (tmp_path / "good.yaml").write_text(
        textwrap.dedent(
            """
            label: よい
            guidance: Something is happening.
            tags: [running]
            """
        ),
        encoding="utf-8",
    )

    assert [item.name for item in load_situations(tmp_path)] == ["good"]


def test_having_no_situation_is_a_named_choice_rather_than_a_blank() -> None:
    choices = situation_choices([BATTLE])

    # Most runs are not aimed at any particular kind of moment, so the absence
    # needs a name in the list rather than an empty row to guess at.
    assert choices[0] == ("（指定なし）", NO_SITUATION)
    # The category rides in the label: a dropdown cannot show headings, and a
    # flat list of forty-odd is one nobody reads to the end.
    assert choices[1] == ("その他 / 戦闘", "battle")
    assert find_situation(NO_SITUATION, [BATTLE]) is None
    assert find_situation("nonesuch", [BATTLE]) is None
    assert find_situation("battle", [BATTLE]) is BATTLE


def test_the_direction_says_what_it_governs_before_it_says_anything_else() -> None:
    direction = situation_direction(BATTLE)

    # Led with the situation, a direction this definite reads as the whole brief
    # and the subject disappears - an elf with a bow came back as a battle.
    assert direction.index("何をしているか") < direction.index("Mid-fight")
    # Identity is what must survive; the setting, the light and the framing are
    # the situation's to move, or several situations come back as one picture
    # with a different verb.
    assert "「誰か」" in direction
    assert "変えない" in direction
    assert "場所・光・構図" in direction
    # Tags are candidates, not a shopping list to empty into the output.
    assert "無理に全部入れない" in direction
    assert "fighting_stance, motion_blur" in direction


def test_no_situation_adds_nothing_at_all() -> None:
    assert situation_direction(None) == ""


def test_prose_gets_the_direction_without_the_reference_tags() -> None:
    """A prose model writes a tag list down rather than weighing it.

    The reference tags are addressed to the tag compiler, which checks them
    against the dictionary. Handed to the prose model they came out verbatim:
    "off duty and unguarded, sitting, closed eyes, smile, indoors".
    """
    situation = Situation(
        name="rest",
        label="休息",
        guidance="Off duty and unguarded.",
        tags=["sitting", "closed_eyes"],
    )

    with_tags = situation_direction(situation)
    without = situation_direction(situation, with_tags=False)

    assert "sitting" in with_tags
    assert "Off duty and unguarded." in without
    assert "sitting" not in without
    assert "参考タグ" not in without


def test_every_situation_on_disk_is_complete_and_grouped() -> None:
    situations = load_situations()

    assert len(situations) >= 40
    for situation in situations:
        assert situation.guidance.strip()
        assert situation.category.strip()
        assert situation.label.strip()


def test_every_situation_tag_is_a_tag_the_dictionary_knows() -> None:
    """A situation's tags are candidates for the compiler, not free text.

    A tag the dictionary has never heard of is dropped silently downstream, so
    a typo in a YAML file costs the situation part of its steer and says
    nothing about it.
    """
    dictionary_path = Path(__file__).resolve().parents[1] / "data" / "tags.json"
    if not dictionary_path.exists():  # pragma: no cover - dictionary is fetched
        pytest.skip("tag dictionary has not been fetched yet")
    known = set(json.loads(dictionary_path.read_text(encoding="utf-8")))

    unknown = {
        situation.name: [tag for tag in situation.tags if tag not in known]
        for situation in load_situations()
    }

    assert not {name: bad for name, bad in unknown.items() if bad}


def test_categories_come_out_in_the_order_their_members_ask_for() -> None:
    situations = load_situations()

    grouped = group_situations(situations)

    assert [category for category, _members in grouped] == sorted(
        {situation.category for situation in situations},
        key=lambda category: min(
            situation.order
            for situation in situations
            if situation.category == category
        ),
    )
    # Every situation lands in exactly one group.
    assert sum(len(members) for _category, members in grouped) == len(situations)


BIG = Situation(
    name="battle",
    label="戦闘",
    guidance="Mid-fight.",
    tags=["a", "b", "c", "d", "e", "f", "g", "h"],
)


def test_only_some_of_the_candidates_are_offered_on_any_one_run() -> None:
    """The whole pool handed over came back verbatim, every run.

    Five plausible tags read as an answer, and copying an answer is the easiest
    thing a model can do - so the same nine words arrived in the same order
    however many times the same situation was asked for.
    """
    drawn = sampled_tags(BIG, 3, rng=random.Random(0))

    assert len(drawn) == 3
    assert set(drawn) <= set(BIG.tags)


def test_the_draw_keeps_the_order_the_file_lists_them_in() -> None:
    drawn = sampled_tags(BIG, 4, rng=random.Random(5))

    # Which ones were drawn is the variation. Shuffling on top of that only
    # makes two identical draws look different.
    assert drawn == [tag for tag in BIG.tags if tag in drawn]


def test_asking_for_all_of_them_or_none_gives_the_whole_pool() -> None:
    assert sampled_tags(BIG, 0) == BIG.tags
    assert sampled_tags(BIG, len(BIG.tags)) == BIG.tags
    assert sampled_tags(BIG, 99) == BIG.tags


def test_a_direction_that_samples_offers_fewer_than_the_pool_holds() -> None:
    direction = situation_direction(BIG, sample=3)

    listed = direction.split("参考タグ")[1]
    assert sum(tag in listed for tag in BIG.tags) == 3


def test_every_situation_has_enough_candidates_to_draw_from() -> None:
    """Sampling five of five is not sampling; it is the old fixed list."""
    for situation in load_situations():
        assert len(situation.tags) >= 8, f"{situation.name} has {len(situation.tags)}"
