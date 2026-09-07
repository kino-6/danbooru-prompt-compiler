from __future__ import annotations

import textwrap
from pathlib import Path

from danbooru_prompt_compiler.situation import (
    NO_SITUATION,
    Situation,
    find_situation,
    load_situations,
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
    assert choices[1] == ("戦闘", "battle")
    assert find_situation(NO_SITUATION, [BATTLE]) is None
    assert find_situation("nonesuch", [BATTLE]) is None
    assert find_situation("battle", [BATTLE]) is BATTLE


def test_the_direction_says_what_it_governs_before_it_says_anything_else() -> None:
    direction = situation_direction(BATTLE)

    # Led with the situation, a direction this definite reads as the whole brief
    # and the subject disappears - an elf with a bow came back as a battle.
    assert direction.index("何をしているか") < direction.index("Mid-fight")
    assert "人物・外見・服装・場所は変えず" in direction
    # Tags are candidates, not a shopping list to empty into the output.
    assert "無理に全部入れない" in direction
    assert "fighting_stance, motion_blur" in direction


def test_no_situation_adds_nothing_at_all() -> None:
    assert situation_direction(None) == ""
