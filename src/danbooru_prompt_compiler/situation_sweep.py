"""One subject through several situations, and the text that holds the answers.

The situation tab generates a prompt per situation and hands them back as a
shared prompt and one merged text. Everything here is independent of the page:
running the sweep, finding what the answers share, and joining them into - and
splitting them back out of - the merged text. The page only lays it out.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .ollama_diagnostics import format_ollama_error
from .web_service import ProgressCallback, WebPromptService


@dataclass(frozen=True)
class SituationRun:
    """One situation's answer in a sweep, kept whole or with its failure.

    A sweep is several model calls in a row, so a situation that fails must not
    take the ones beside it with it: the failure is carried here and shown in
    that situation's own box.
    """

    name: str
    label: str
    prompt: str = ""
    avoid: str = ""
    error: str = ""
    # What happened about the GPU. A sweep keeps only the prompts, so without
    # this a run that quietly went to the CPU - and took four times as long
    # because of it - would never say why.
    gpu_note: str = ""


def run_situation_sweep(
    service: WebPromptService,
    situations: Sequence[str],
    labels: Mapping[str, str],
    *,
    instruction: str = "",
    as_prose: bool = False,
    options: Mapping[str, object] | None = None,
    on_progress: ProgressCallback | None = None,
    stage_labels: Mapping[str, str] | None = None,
) -> list[SituationRun]:
    """One prompt per situation: the same subject through several moments.

    The runs are identical but for the situation, so what differs between the
    answers is what the situation did. ``stage_labels`` is the page's wording
    for each stage of a run; it is passed in because the words belong to the
    page, not to the sweep.
    """
    chosen = [name for name in situations if name in labels]
    shared = dict(options or {})
    runs: list[SituationRun] = []
    total = len(chosen)
    started = time.monotonic()
    for index, name in enumerate(chosen):
        label = labels[name]

        def report(stage: str = "", fraction: float = 0.0, *, at=index, of=label):
            """Where the sweep is, what that run is doing, and for how long.

            Reporting only between situations left the line reading
            "（1/8） - 0.0%" for as long as the first one took - which, with
            another program holding the card, is up to two minutes of the GPU
            wait before a single token is asked for. It looked hung. So each
            run's own stages come through here, named after the situation they
            belong to, and the elapsed time says the thing is still moving even
            when the stage does not change.
            """
            if on_progress is None:
                return
            detail = (stage_labels or {}).get(stage, stage)
            waited = time.monotonic() - started
            # Named rather than numbered: eight runs in a row is long enough
            # that "which one is it on" is a real question.
            head = f"「{of}」（{at + 1}/{total}）"
            body = f"{head}{detail}" if detail else head
            on_progress(
                f"{body} — {waited:.0f}秒経過",
                (at + min(max(fraction, 0.0), 1.0)) / total,
            )

        report()
        try:
            result = service.run(
                image_path=None,
                situation=name,
                instruction=instruction,
                action_override="scene_prompt" if as_prose else "compile",
                variants=1,
                use_vision=False,
                on_progress=report,
                **shared,
            )
        except Exception as exc:
            runs.append(
                SituationRun(
                    name,
                    label,
                    error=format_ollama_error(
                        exc,
                        [
                            str(shared.get("compiler_model", "")),
                            str(shared.get("scene_model", "")),
                        ],
                    ),
                )
            )
            continue
        prompt = result.prose_plain if as_prose else ""
        if not prompt:
            prompt = result.candidates[0] if result.candidates else ""
        runs.append(
            SituationRun(
                name,
                label,
                prompt=prompt,
                avoid=result.prose_avoid or "",
                gpu_note=result.gpu_note,
            )
        )
    return runs


@dataclass(frozen=True)
class SituationComparison:
    """What every situation said, and what each one said on its own."""

    shared: str
    distinct: dict[str, str]


def _comparable_lines(prompt: str, as_prose: bool) -> list[list[str]]:
    """Each line of a prompt as the units worth comparing it by.

    A tag line compares tag by tag; a prose line compares as the whole
    sentence, because half a sentence is not a thing anyone can read or paste.
    """
    lines = [line.strip() for line in (prompt or "").splitlines() if line.strip()]
    if as_prose:
        return [[line] for line in lines]
    return [
        [unit.strip() for unit in line.split(",") if unit.strip()] for line in lines
    ]


def _rendered_lines(grid: list[list[str]], keep) -> str:
    """The grid back as text, with the units that fail `keep` taken out."""
    lines = []
    for line in grid:
        kept = [unit for unit in line if keep(unit)]
        if kept:
            lines.append(", ".join(kept))
    return "\n".join(lines)


def compare_situation_runs(
    runs: Sequence[SituationRun], *, as_prose: bool = False
) -> SituationComparison:
    """Split a sweep into the part they all share and the part each one adds.

    This backs the 違いだけ view, which is an aid for reading a set of
    generated prompts rather than the thing the tab produces. The subject is
    deliberately the same in every run, so most of every answer is the same and
    the lines that differ are easy to miss among the ones that do not.
    """
    grids = {
        run.name: _comparable_lines(run.prompt, as_prose)
        for run in runs
        if run.prompt and not run.error
    }
    if len(grids) < 2:
        return SituationComparison(shared="", distinct={})
    shared = set.intersection(
        *({unit for line in grid for unit in line} for grid in grids.values())
    )
    first = next(iter(grids.values()))
    return SituationComparison(
        shared=_rendered_lines(first, lambda unit: unit in shared),
        distinct={
            name: _rendered_lines(grid, lambda unit: unit not in shared)
            for name, grid in grids.items()
        },
    )


NOTHING_OF_ITS_OWN = "共通部分と同じで、このシチュエーション固有の要素はありませんでした。"
# One blank line between situations, and a marked name line opening each. A
# prompt's own lines are separated by single newlines, so a blank line cannot
# occur inside one and is unambiguous as the boundary: `text.split("\n\n")`
# gives the blocks back, and dropping each block's first line gives the prompt.
SITUATION_BLOCK_SEPARATOR = "\n\n"
SITUATION_BLOCK_PREFIX = "# "


def situation_block(label: str, body: str) -> str:
    """One situation's part of the merged text, named and separable."""
    return f"{SITUATION_BLOCK_PREFIX}{label}\n{(body or '').strip()}"


def split_situation_blocks(merged: str) -> list[tuple[str, str]]:
    """The merged text back into (label, prompt) pairs.

    Here so the format is not just an assumption made twice: whatever the page
    joins, this takes apart, and the tests hold the two against each other.
    """
    blocks = []
    for chunk in (merged or "").split(SITUATION_BLOCK_SEPARATOR):
        lines = chunk.strip().splitlines()
        if not lines:
            continue
        head = lines[0]
        if not head.startswith(SITUATION_BLOCK_PREFIX):
            continue
        blocks.append(
            (head[len(SITUATION_BLOCK_PREFIX) :].strip(), "\n".join(lines[1:]).strip())
        )
    return blocks


SHARED_BLOCK_LABEL = "共通"
# 共通 first and then the rest, deduplicated against it, rather than every
# block carrying the same opening lines over again.
SHARED_FIRST_VIEW = "shared_first"


def merge_situation_runs(
    runs: Sequence[SituationRun],
    *,
    as_prose: bool = False,
    view: str = SHARED_FIRST_VIEW,
) -> tuple[str, str]:
    """The shared prompt, and every situation merged into one splittable text.

    A box per situation meant hunting through up to forty-six of them for the
    one wanted, so the answers are one text - and written whole, that text
    repeated `1girl, solo` and `indoors` in every block. It leads with the
    shared block instead and each situation keeps only what is its own, which
    is the structure a set of prompts wants: the base once, then the parts.

    Glued together carelessly this would be worse than the boxes it replaced,
    so the joining is a stated format with a function to undo it.
    """
    comparison = compare_situation_runs(runs, as_prose=as_prose)
    # Nothing shared means there is nothing to lift out, so the blocks stay
    # whole and no 共通 block is written for an empty one.
    lift_shared = view == SHARED_FIRST_VIEW and bool(comparison.shared)
    blocks = []
    if lift_shared:
        blocks.append(situation_block(SHARED_BLOCK_LABEL, comparison.shared))
    for number, run in enumerate(runs, start=1):
        if run.error:
            body = run.error
        elif lift_shared:
            body = comparison.distinct.get(run.name, "") or NOTHING_OF_ITS_OWN
        else:
            body = run.prompt
        # Numbered as well as named: the number says which prompt of the set
        # this is, the name says which one to reach for.
        blocks.append(situation_block(f"{number} {run.label}", body))
    return comparison.shared, SITUATION_BLOCK_SEPARATOR.join(blocks)
