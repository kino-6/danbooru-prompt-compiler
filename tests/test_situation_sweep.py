from __future__ import annotations

import pytest

from danbooru_prompt_compiler.situation_sweep import (
    SituationRun,
    compare_situation_runs,
    merge_situation_runs,
    run_situation_sweep,
    split_situation_blocks,
)
from danbooru_prompt_compiler.web_service import WebRunResult
from danbooru_prompt_compiler.webui import PROGRESS_LABELS


class SweepService:
    """A service that answers each situation with its own name."""

    def __init__(self, fails: set[str] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.progress_callbacks: list[object] = []
        self.fails = fails or set()

    def run(self, *, image_path, on_progress=None, **options) -> WebRunResult:
        self.calls.append({"image_path": image_path, **options})
        self.progress_callbacks.append(on_progress)
        situation = str(options.get("situation", ""))
        if situation in self.fails:
            raise RuntimeError(f"{situation} は失敗しました")
        return WebRunResult(
            action_plan={"action": "compile"},
            inferred_tags="",
            output=f"tags for {situation}",
            status="ok",
            candidates=[f"tags for {situation}"],
            prose_plain=f"prose for {situation}",
            prose_avoid="simple background",
        )


SWEEP_LABELS = {"battle": "戦闘", "rest": "休息", "travel": "移動"}


def test_a_sweep_runs_every_chosen_situation_and_keeps_them_apart() -> None:
    service = SweepService()

    runs = run_situation_sweep(
        service, ["battle", "rest", "travel"], SWEEP_LABELS
    )

    assert [run.name for run in runs] == ["battle", "rest", "travel"]
    assert [run.prompt for run in runs] == [
        "tags for battle",
        "tags for rest",
        "tags for travel",
    ]
    assert [call["situation"] for call in service.calls] == [
        "battle",
        "rest",
        "travel",
    ]
    # Nothing but the situation differs, or the boxes would not be comparable.
    assert {call["instruction"] for call in service.calls} == {""}
    assert {call["variants"] for call in service.calls} == {1}


def test_a_sweep_carries_the_shared_subject_into_every_situation() -> None:
    service = SweepService()

    run_situation_sweep(
        service, ["battle", "rest"], SWEEP_LABELS, instruction="弓を持ったエルフ"
    )

    assert {call["instruction"] for call in service.calls} == {"弓を持ったエルフ"}


def test_a_sweep_asks_for_prose_when_prose_was_chosen() -> None:
    service = SweepService()

    runs = run_situation_sweep(
        service, ["battle"], SWEEP_LABELS, as_prose=True
    )

    assert service.calls[0]["action_override"] == "scene_prompt"
    assert runs[0].prompt == "prose for battle"


def test_one_failed_situation_does_not_lose_the_others() -> None:
    service = SweepService(fails={"rest"})

    runs = run_situation_sweep(
        service, ["battle", "rest", "travel"], SWEEP_LABELS
    )

    assert [bool(run.error) for run in runs] == [False, True, False]
    assert runs[0].prompt and runs[2].prompt
    assert "失敗" in runs[1].error


def test_a_sweep_says_which_situation_it_is_on_and_for_how_long() -> None:
    service = SweepService()
    stages: list[str] = []

    run_situation_sweep(
        service,
        ["battle", "rest"],
        SWEEP_LABELS,
        on_progress=lambda stage, _fraction: stages.append(stage),
    )

    # The name, not a number: eight runs in a row is long enough that "which
    # one is it on" is a real question. And the clock, because a run can sit on
    # one stage for two minutes waiting for the card.
    assert "戦闘" in stages[0] and "1/2" in stages[0]
    assert "秒経過" in stages[0]
    assert any("休息" in stage for stage in stages)


def test_a_sweep_passes_each_run_its_own_progress_reporter() -> None:
    """Reported only between situations, the line sat still for the whole run.

    With another program on the card that is up to two minutes of waiting
    before a token is asked for, and it read as hung.
    """
    service = SweepService()
    stages: list[str] = []

    run_situation_sweep(
        service,
        ["battle"],
        SWEEP_LABELS,
        on_progress=lambda stage, _fraction: stages.append(stage),
        stage_labels=PROGRESS_LABELS,
    )

    assert service.progress_callbacks and all(
        callback is not None for callback in service.progress_callbacks
    )
    # The service's own stage names arrive in the page's words, under the
    # situation they belong to.
    service.progress_callbacks[0]("gpu_wait", 0.03)
    assert "戦闘" in stages[-1]
    assert PROGRESS_LABELS["gpu_wait"] in stages[-1]


def test_without_the_page_s_wording_a_stage_is_named_by_its_key() -> None:
    service = SweepService()
    stages: list[str] = []

    run_situation_sweep(
        service,
        ["battle"],
        SWEEP_LABELS,
        on_progress=lambda stage, _fraction: stages.append(stage),
    )
    service.progress_callbacks[0]("gpu_wait", 0.03)

    # The sweep does not own the page's words, so without them it still says
    # something rather than nothing.
    assert "gpu_wait" in stages[-1]


def test_the_sweep_bar_moves_inside_a_situation_as_well_as_between_them() -> None:
    service = SweepService()
    seen: list[float] = []

    run_situation_sweep(
        service,
        ["battle", "rest"],
        SWEEP_LABELS,
        on_progress=lambda _stage, fraction: seen.append(fraction),
    )
    service.progress_callbacks[0]("compilation", 0.7)

    # Two situations, so the first one's own 70% is 35% of the sweep.
    assert seen[-1] == pytest.approx(0.35)


def test_a_sweep_ignores_situations_it_has_no_label_for() -> None:
    service = SweepService()

    runs = run_situation_sweep(service, ["battle", "unknown"], SWEEP_LABELS)

    assert [run.name for run in runs] == ["battle"]


TAG_RUNS = [
    SituationRun(
        "battle",
        "戦闘",
        "1girl, solo\nsilver_hair\nholding_weapon, dynamic_pose\nelf, bow, fighting_stance",
    ),
    SituationRun(
        "rest",
        "休息",
        "1girl, solo\nsilver_hair\nsitting, holding_book\nelf, bow, relaxed",
    ),
]


def test_a_comparison_says_the_shared_part_once() -> None:
    """Side by side the answers looked identical, and mostly they were.

    The subject is deliberately the same in every run, so what differs is a few
    tags inside a dozen - and reading a dozen to find them is not comparing.
    """
    comparison = compare_situation_runs(TAG_RUNS)

    assert comparison.shared == "1girl, solo\nsilver_hair\nelf, bow"
    assert comparison.distinct["battle"] == "holding_weapon, dynamic_pose\nfighting_stance"
    assert comparison.distinct["rest"] == "sitting, holding_book\nrelaxed"


def test_a_comparison_of_prose_keeps_whole_sentences() -> None:
    runs = [
        SituationRun("battle", "戦闘", "An elf with a bow.\nMid-fight, weight forward."),
        SituationRun("rest", "休息", "An elf with a bow.\nSlouched on a bench."),
    ]

    comparison = compare_situation_runs(runs, as_prose=True)

    # Half a sentence is not something anyone can read or paste, so prose
    # compares line by line rather than word by word.
    assert comparison.shared == "An elf with a bow."
    assert comparison.distinct["battle"] == "Mid-fight, weight forward."


def test_one_run_alone_has_nothing_to_compare_against() -> None:
    comparison = compare_situation_runs(TAG_RUNS[:1])

    assert comparison.shared == ""
    assert comparison.distinct == {}


def test_a_failed_run_is_left_out_of_the_shared_part() -> None:
    runs = [*TAG_RUNS, SituationRun("magic", "魔法", error="落ちました")]

    comparison = compare_situation_runs(runs)

    # Otherwise one failure empties the shared set and every box goes back to
    # being the whole prompt.
    assert comparison.shared == "1girl, solo\nsilver_hair\nelf, bow"
    assert "magic" not in comparison.distinct


def test_the_merged_text_can_be_split_back_into_the_blocks_it_holds() -> None:
    """Glued together carelessly this would be worse than the boxes it replaced.

    So the joining is a stated format, and the function that undoes it is held
    against the one that makes it rather than assumed to match.
    """
    _shared, merged = merge_situation_runs(TAG_RUNS, view="full")

    assert split_situation_blocks(merged) == [
        ("1 戦闘", TAG_RUNS[0].prompt),
        ("2 休息", TAG_RUNS[1].prompt),
    ]
    # A blank line cannot occur inside a prompt, whose own lines are single
    # newlines apart, so it is unambiguous as the boundary.
    assert merged.count("\n\n") == len(TAG_RUNS) - 1
    assert merged.startswith("# 1 戦闘\n")


def test_the_merged_text_says_the_shared_part_once_at_the_top() -> None:
    """Written whole it repeated `1girl, solo` and `indoors` in every block.

    A set of prompts wants the base once and then the parts, so that is the
    shape the merged text has by default.
    """
    shared, merged = merge_situation_runs(TAG_RUNS)
    blocks = dict(split_situation_blocks(merged))

    assert merged.startswith("# 共通\n")
    assert blocks["共通"] == shared == "1girl, solo\nsilver_hair\nelf, bow"
    assert blocks["1 戦闘"] == "holding_weapon, dynamic_pose\nfighting_stance"
    assert blocks["2 休息"] == "sitting, holding_book\nrelaxed"
    # Said once, not once per block.
    assert merged.count("1girl") == 1


def test_the_shared_prompt_is_reported_whichever_shape_was_asked_for() -> None:
    shared_first, _ = merge_situation_runs(TAG_RUNS)
    whole, _ = merge_situation_runs(TAG_RUNS, view="full")

    # The box is for lifting out the base on its own, so it is filled either
    # way; only the merged text changes shape.
    assert shared_first == whole == "1girl, solo\nsilver_hair\nelf, bow"


def test_a_failed_situation_keeps_its_block_and_says_why() -> None:
    runs = [*TAG_RUNS, SituationRun("magic", "魔法", error="落ちました")]

    _shared, merged = merge_situation_runs(runs)

    # Which one failed is the whole of the answer; a missing block would not
    # say, and the blocks have to stay in step with what was asked for.
    assert dict(split_situation_blocks(merged))["3 魔法"] == "落ちました"


def test_nothing_shared_means_no_shared_block_is_written() -> None:
    runs = [
        SituationRun("battle", "戦闘", "holding_weapon"),
        SituationRun("rest", "休息", "sitting"),
    ]

    shared, merged = merge_situation_runs(runs)

    # An empty 共通 block would be a heading over nothing, and the blocks have
    # had nothing taken out of them.
    assert shared == ""
    assert "# 共通" not in merged
    assert dict(split_situation_blocks(merged))["1 戦闘"] == "holding_weapon"
