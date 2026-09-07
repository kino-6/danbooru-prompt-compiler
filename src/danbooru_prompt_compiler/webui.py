from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import typer

from .formatter import group_tags
from .image_source import load_image_url_preview, resolve_image_source
from .normalizer import normalize_tags, parse_tag_text
from .ollama_diagnostics import check_ollama, format_ollama_error, restart_ollama_model
from .scene_prompt import load_templates
from .settings_store import load_settings, remembered, save_settings
from .situation import (
    NO_SITUATION,
    group_situations,
    load_situations,
    situation_choices,
)
from .subject_seed import (
    load_subject_vocabulary,
    random_situation_names,
    random_subject,
)
from .tag_filter import (
    DEFAULT_EXCLUSION_TEXT,
    EXCLUDED_TAGS_PATH,
    load_exclusion_text,
    save_exclusion_text,
)
from .web_service import (
    DEFAULT_COMPILER_MODEL,
    DEFAULT_GPU_WAIT_GB,
    DEFAULT_NEXT_PANEL_CHANGE,
    DEFAULT_NEXT_PANEL_TIME,
    DEFAULT_OLLAMA_URL,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_SCENE_MODEL,
    DEFAULT_SCENE_TEMPLATE,
    DEFAULT_VISION_MODEL,
    SCENE_MODEL_CHOICES,
    TEXT_MODEL_CHOICES,
    VISION_MODEL_CHOICES,
    WEB_RUN_FIELDS,
    ProgressCallback,
    WebPromptService,
    WebRunRequest,
    WebRunResult,
)


web_app = typer.Typer(help="Launch the local Danbooru Prompt Workbench web UI.")


@dataclass(frozen=True)
class Task:
    """One entry in the task selector, and everything that follows from it."""

    action: str
    label: str
    fields: frozenset[str]


# What the user is trying to do, asked first. Everything the answer cannot reach
# is hidden, because a control that does nothing for the selected task is worse
# than a missing one: it invites a setting that will be silently ignored. One
# definition per task, so a task cannot be offered without saying what it shows.
TASKS: tuple[Task, ...] = (
    Task(
        "auto",
        "おまかせ（指示から判断）",
        frozenset(
            {"vision", "instruction", "base_prompt", "follow_up", "panel_change",
             "variants", "run", "next_panel", "situation"}
        ),
    ),
    # Tagging is pure ONNX: no instruction to give, no model to describe with.
    Task("tag_image", "画像からタグを抽出", frozenset({"run"})),
    Task(
        "compile",
        "テキストからプロンプト",
        frozenset({"instruction", "follow_up", "variants", "run", "situation"}),
    ),
    Task(
        "edit",
        "既存プロンプトを編集",
        frozenset(
            {"vision", "instruction", "base_prompt", "follow_up", "variants", "run",
             "situation"}
        ),
    ),
    # base_prompt earns its place here: a next panel can be asked for from a
    # prompt alone, with no picture at all.
    Task(
        "next_panel",
        "次のコマ",
        frozenset(
            {"vision", "instruction", "base_prompt", "panel_change", "variants",
             "next_panel", "situation"}
        ),
    ),
    Task(
        "scene_prompt",
        "自然文プロンプト",
        frozenset(
            {"vision", "instruction", "base_prompt", "variants", "scene_template",
             "scene_settings", "scene_prompt", "situation"}
        ),
    ),
    # The review reads the description as context but takes no instruction, and
    # it answers with one list rather than a number of variants.
    Task("verify_tags", "タグをVLMで確認", frozenset({"vision", "run"})),
)
TASK_CHOICES: tuple[tuple[str, str], ...] = tuple(
    (task.label, task.action) for task in TASKS
)
TASK_FIELDS: dict[str, frozenset[str]] = {task.action: task.fields for task in TASKS}
# The order the visibility updates are returned in, so the wiring and the
# outputs list cannot drift apart.
TASK_FIELD_ORDER: tuple[str, ...] = (
    "vision",
    "instruction",
    "base_prompt",
    "follow_up",
    "panel_change",
    "variants",
    "scene_template",
    "situation",
    "scene_settings",
    "run",
    "next_panel",
    "scene_prompt",
)


def task_field_visibility(task: str) -> list[bool]:
    """Which groups the chosen task shows, in `TASK_FIELD_ORDER`."""
    shown = TASK_FIELDS.get(task, TASK_FIELDS["auto"])
    return [name in shown for name in TASK_FIELD_ORDER]


PROGRESS_LABELS = {
    "preparing": "準備しています",
    "gpu_wait": "他タスクのGPU使用が収まるのを待っています",
    "routing": "指示を解釈しています",
    "tagging": "画像タグを推測しています",
    "vision": "VLMで構図を確認しています",
    "compilation": "プロンプトを生成しています",
    "complete": "完了",
}
MAX_HISTORY_ITEMS = 20
# One output box per situation on disk, so dropping a YAML file into
# situations/ adds a box without anyone editing the page. The floor only keeps
# a small set from collapsing the layout.
MIN_SITUATION_SLOTS = 8
# What the comparison tab writes prose with unless told otherwise.
SWEEP_SCENE_TEMPLATE = "scene_illustration"
MAX_OUTPUT_VARIANTS = 4
# Prompt boxes 2-4 hold the next-panel proposals that follow box 1.
NEXT_PANEL_SLOTS = MAX_OUTPUT_VARIANTS - 1
IMAGE_INPUT_JS = r"""
() => {
  if (document.documentElement.dataset.imageUrlDropReady === "true") return;
  document.documentElement.dataset.imageUrlDropReady = "true";

  const isImageWorkspace = (event) =>
    event.target instanceof Element && event.target.closest("#image-workspace");

  const loadImageUrl = (url) => {
    const field = document.querySelector(
      "#dropped-image-url-input textarea, #dropped-image-url-input input"
    );
    const button = document.querySelector(
      "button#dropped-image-url-button, #dropped-image-url-button button"
    );
    if (!field || !button) return false;
    field.value = url;
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
    setTimeout(() => button.click(), 100);
    return true;
  };

  const loadImageFile = (file) => {
    const input = document.querySelector('#image-workspace input[type="file"]');
    if (!input || !file) return false;
    const transfer = new DataTransfer();
    transfer.items.add(
      new File([file], file.name || "clipboard.png", {
        type: file.type || "image/png",
      })
    );
    input.files = transfer.files;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  };

  // Gradio's own dropzone exists only while the workspace is empty, and it
  // handled that case correctly long before this file did. Taking it over
  // gained nothing and put the first load at risk, so the handler below steps
  // in only once there is an image in the way.
  const workspaceHasImage = () =>
    !!document.querySelector("#image-workspace img");

  const isTextEntry = (element) =>
    element instanceof Element &&
    (element.isContentEditable ||
      element.tagName === "TEXTAREA" ||
      element.tagName === "INPUT");

  document.addEventListener("dragover", (event) => {
    if (!isImageWorkspace(event)) return;
    const types = Array.from(event.dataTransfer?.types || []);
    // A dragover nobody cancels tells the browser this is not a drop target and
    // the drop never fires. Gradio cancels it for files while it still owns the
    // empty workspace; once an image is loaded nobody does but us.
    if (!types.includes("Files") || workspaceHasImage()) event.preventDefault();
  }, true);
  document.addEventListener("drop", (event) => {
    if (!isImageWorkspace(event)) return;
    const dropped = Array.from(event.dataTransfer?.files || [])
      .find((file) => (file.type || "").startsWith("image/"));
    if (dropped) {
      if (!workspaceHasImage()) return;
      // Feed the file in the same way a paste does, rather than leaving it to a
      // dropzone that is not there when the workspace already holds an image.
      event.preventDefault();
      event.stopPropagation();
      loadImageFile(dropped);
      return;
    }
    if (event.dataTransfer?.files?.length) return;
    const uriList = event.dataTransfer?.getData("text/uri-list") || "";
    const plainText = event.dataTransfer?.getData("text/plain") || "";
    const html = event.dataTransfer?.getData("text/html") || "";
    const htmlUrl = html.match(/<img[^>]+src=["']([^"']+)/i)?.[1] || "";
    const listedUrl = uriList
      .split(/\r?\n/)
      .find((line) => line && !line.startsWith("#"));
    const url = (listedUrl || htmlUrl || plainText).trim();
    if (!/^https?:\/\//i.test(url)) return;

    event.preventDefault();
    event.stopPropagation();
    loadImageUrl(url);
  }, true);
  document.addEventListener("paste", (event) => {
    const items = Array.from(event.clipboardData?.items || []);
    const pastedText = (event.clipboardData?.getData("text/plain") || "").trim();
    const imageItem = items.find(
      (item) => item.kind === "file" && (item.type || "").startsWith("image/")
    );
    if (imageItem) {
      if (pastedText && isTextEntry(document.activeElement)) return;
      if (!loadImageFile(imageItem.getAsFile())) return;
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (!isImageWorkspace(event)) return;
    if (!/^https?:\/\//i.test(pastedText)) return;
    if (!loadImageUrl(pastedText)) return;
    event.preventDefault();
    event.stopPropagation();
  }, true);
}
"""


# The output already groups tags; these are the same groups as copyable parts,
# so a prompt can be reused piecewise - the character without the scene, the
# clothing without the pose.
PART_LABELS: tuple[tuple[str, str], ...] = (
    ("subject", "人物"),
    ("appearance", "外見"),
    ("clothing", "服装"),
    ("pose", "ポーズ"),
    ("scene", "情景"),
    ("style", "画風"),
    ("composition", "構図"),
    ("other", "その他"),
)


def prompt_parts(prompt: str) -> dict[str, str]:
    """The grouped tag lines of a prompt, by category.

    Read back from the prompt box rather than kept from the run, so an edited
    box and an adopted candidate both give the parts you can see.
    """
    tags = normalize_tags(parse_tag_text(_prompt_body(prompt)))
    return {
        category: ", ".join(values) for category, values in group_tags(tags).items()
    }


def _prompt_body(prompt: str) -> str:
    """The tags out of a grouped block, without its `category:` labels."""
    lines: list[str] = []
    for line in (prompt or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("===") or stripped.startswith("["):
            continue
        _label, sep, rest = stripped.partition(":")
        lines.append(rest if sep else stripped)
    return ", ".join(lines)


def prepend_history(
    history: list[dict[str, str]] | None,
    *,
    action: str,
    instruction: str,
    output: str,
    limit: int = MAX_HISTORY_ITEMS,
) -> list[dict[str, str]]:
    entry = {
        "action": action,
        "instruction": instruction,
        "output": output,
    }
    return [entry, *(history or [])][:limit]


def adopt_candidate(candidate: str | None) -> str:
    return candidate or ""


def blank_prompt_boxes() -> list[str]:
    return ["" for _ in range(MAX_OUTPUT_VARIANTS)]


def prompt_box_values(candidates: list[str]) -> list[str]:
    values = candidates[:MAX_OUTPUT_VARIANTS]
    return [values[index] if index < len(values) else "" for index in range(MAX_OUTPUT_VARIANTS)]


def accept_dropped_image(image_path: str | None):
    return image_path or None, ""


def store_excluded_tags(
    excluded_tags: str,
    path: Path = EXCLUDED_TAGS_PATH,
) -> tuple[str, str]:
    saved = save_exclusion_text(excluded_tags, path)
    return saved, f"除外ワードを保存しました（{len(saved.split(', ')) if saved else 0}件）。"


def restore_default_excluded_tags(path: Path = EXCLUDED_TAGS_PATH) -> tuple[str, str]:
    saved = save_exclusion_text(DEFAULT_EXCLUSION_TEXT, path)
    return saved, "既定の除外ワードに戻して保存しました。"


def preview_image_url(image_url: str, allow_private_hosts: bool):
    return load_image_url_preview(
        (image_url or "").strip(),
        allow_private_hosts=allow_private_hosts,
    )


def diagnose_ollama(
    ollama_url: str,
    router_model: str,
    compiler_model: str,
    vision_model: str,
    scene_model: str,
    use_vision: bool,
) -> str:
    # A larger prose model is the one most likely not to be pulled yet.
    required = [router_model, compiler_model, scene_model]
    if use_vision:
        required.append(vision_model)
    return check_ollama(ollama_url, required).message


def recover_vision_model(
    service: WebPromptService,
    ollama_url: str,
    vision_model: str,
) -> str:
    """Drop the cached description and reload the model that stopped answering."""
    clear_cache = getattr(service, "clear_description_cache", None)
    if callable(clear_cache):
        clear_cache()
    return restart_ollama_model(ollama_url, vision_model)


@dataclass(frozen=True)
class WorkbenchOutputs:
    """Everything one run produces, independent of the Gradio component types."""

    action_plan: dict[str, object]
    inferred_tags: str
    image_description: str
    prompts: list[str]
    status: str
    candidates: list[str]
    history: list[dict[str, str]]
    # Empty on the error paths, which have no result to write prose from.
    prose_prompt: str = ""
    prose_plain: str = ""
    prose_avoid: str = ""


def run_workbench(
    service: WebPromptService,
    request: WebRunRequest,
    history: list[dict[str, str]] | None,
    *,
    on_progress: ProgressCallback | None = None,
) -> WorkbenchOutputs:
    """Resolve the image source, run the request, and fold errors into the status."""
    follow_up: WebRunResult | None = None
    follow_up_error = ""
    try:
        with resolve_image_source(
            request.image_path,
            request.image_url,
            allow_private_hosts=request.allow_private_image_urls,
        ) as resolved_image_path:
            result = service.run(
                image_path=resolved_image_path,
                on_progress=on_progress,
                **request.service_options(),
            )
            if _wants_next_panel_follow_up(request, result):
                try:
                    follow_up = service.run(
                        image_path=resolved_image_path,
                        on_progress=on_progress,
                        **{
                            **request.service_options(),
                            "action_override": "next_panel",
                            "variants": NEXT_PANEL_SLOTS,
                            # Reuse what the first run already resolved so the
                            # panels continue exactly the prompt in box 1.
                            "edited_tags": result.inferred_tags,
                            "edited_description": result.image_description,
                        },
                    )
                except Exception as exc:
                    # The primary result is still worth showing on its own.
                    follow_up_error = str(exc)
    except Exception as exc:
        return WorkbenchOutputs(
            action_plan={},
            inferred_tags="",
            image_description=request.edited_description,
            prompts=blank_prompt_boxes(),
            status="Error: "
            + format_ollama_error(
                exc,
                [
                    request.router_model,
                    request.compiler_model,
                    request.scene_model,
                    request.vision_model if request.use_vision else "",
                ],
            ),
            candidates=[],
            history=history or [],
        )

    candidates = _merge_candidates(result, follow_up)
    status = result.status
    if follow_up is not None:
        status += f"\n\n次のコマ: {len(follow_up.candidates)}件"
        # Only what the follow-up learned about the panels is worth carrying up;
        # the rest of its status just repeats what is already above. Without
        # this the sentence the model wrote about each panel was thrown away,
        # and the run looked like it had explained nothing.
        if follow_up.panel_note:
            status += f"\n\n{follow_up.panel_note}"
    elif follow_up_error:
        status += f"\n\n次のコマの生成に失敗: {follow_up_error}"
    updated_history = prepend_history(
        history,
        action=str(result.action_plan.get("action", "")),
        instruction=request.instruction or "",
        output=result.output,
    )
    return WorkbenchOutputs(
        action_plan=result.action_plan,
        inferred_tags=result.inferred_tags,
        image_description=result.image_description,
        prompts=prompt_box_values(candidates),
        prose_prompt=result.prose_prompt,
        prose_plain=result.prose_plain,
        prose_avoid=result.prose_avoid,
        status=status,
        candidates=candidates,
        history=updated_history,
    )


def _wants_next_panel_follow_up(request: WebRunRequest, result: WebRunResult) -> bool:
    """Whether boxes 2-4 should hold next-panel proposals for this result."""
    if not request.generate_next_panel:
        return False
    # A next-panel run already fills every box with panels, and a prose prompt
    # would be followed by tag prompts in a different format. The request is
    # checked as well as the result, so an explicit choice always wins.
    followed_by_panels = {"next_panel", "scene_prompt"}
    if request.action_override in followed_by_panels:
        return False
    if result.action_plan.get("action") in followed_by_panels:
        return False
    # The follow-up continues an existing prompt, so it needs one to continue.
    return bool(result.inferred_tags or request.base_prompt.strip())


def _merge_candidates(
    result: WebRunResult,
    follow_up: WebRunResult | None,
) -> list[str]:
    if follow_up is None:
        return result.candidates
    # Box 1 keeps the current prompt; the panels take the boxes after it.
    return [*result.candidates[:1], *follow_up.candidates[:NEXT_PANEL_SLOTS]]


def _prompt_updates(gr, prompts: list[str]):
    """Show a prompt box only once it has something in it.

    Four empty boxes are the tallest thing on an untouched page and they teach
    nothing; a run that answers with one list should leave one box behind.
    """
    return [gr.update(value=value, visible=bool(value)) for value in prompts]


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


def run_situation_sweep(
    service: WebPromptService,
    situations: Sequence[str],
    labels: Mapping[str, str],
    *,
    instruction: str = "",
    as_prose: bool = False,
    options: Mapping[str, object] | None = None,
    on_progress: ProgressCallback | None = None,
) -> list[SituationRun]:
    """One prompt per situation: the same subject through several moments.

    The point of asking for more than one is comparison - the same character
    eating breakfast, mid-fight, and halfway through a sentence - so the runs
    are identical but for the situation, and each keeps its own box.
    """
    chosen = [name for name in situations if name in labels]
    shared = dict(options or {})
    runs: list[SituationRun] = []
    for index, name in enumerate(chosen):
        label = labels[name]
        # Named rather than numbered: six runs in a row is long enough that
        # "which one is it on" is a real question.
        if on_progress is not None:
            on_progress(
                f"「{label}」を生成しています（{index + 1}/{len(chosen)}）",
                index / len(chosen),
            )
        try:
            result = service.run(
                image_path=None,
                situation=name,
                instruction=instruction,
                action_override="scene_prompt" if as_prose else "compile",
                variants=1,
                use_vision=False,
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
            SituationRun(name, label, prompt=prompt, avoid=result.prose_avoid or "")
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


def _situation_outputs(gr, runs: list[SituationRun], *, as_prose: bool, view: str):
    """The shared prompt and the merged one, shown only once they hold something."""
    shared, merged = merge_situation_runs(runs, as_prose=as_prose, view=view)
    return [
        gr.update(value=shared, visible=bool(shared)),
        gr.update(value=merged, visible=bool(merged)),
    ]


def _situation_summary(gr, runs: list[SituationRun]):
    """The shared avoid list, and what the sweep managed."""
    avoid = next((run.avoid for run in runs if run.avoid), "")
    failed = [run.label for run in runs if run.error]
    summary = f"{len(runs) - len(failed)}件を生成しました。"
    if failed:
        summary += "失敗: " + "、".join(failed)
    return [gr.update(value=avoid, visible=bool(avoid)), summary]


def build_app(*, service: WebPromptService | None = None):
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("Web UI dependencies are missing; run 'uv sync --extra web'.") from exc

    prompt_service = service or WebPromptService()
    stored = load_settings()

    def dispatch(values, progress, *, action_override: str | None = None):
        """Gradio adapter: ordered values in, Gradio component updates out."""
        *run_values, history = values
        request = WebRunRequest.from_values(run_values)
        # Saved from the request rather than from a button, so the settings that
        # come back are the ones that were actually last run with.
        save_settings(request.model_dump())
        if action_override is not None:
            request = request.model_copy(update={"action_override": action_override})

        def report_progress(stage: str, fraction: float) -> None:
            progress(fraction, desc=PROGRESS_LABELS.get(stage, stage))

        outputs = run_workbench(
            prompt_service,
            request,
            history,
            on_progress=report_progress,
        )
        return (
            outputs.action_plan,
            outputs.inferred_tags,
            outputs.image_description,
            *_prompt_updates(gr, outputs.prompts),
            gr.update(value=outputs.prose_plain, visible=bool(outputs.prose_plain)),
            gr.update(value=outputs.prose_avoid, visible=bool(outputs.prose_avoid)),
            outputs.prose_prompt,
            outputs.status,
            gr.Radio(
                choices=outputs.candidates,
                value=outputs.candidates[0] if outputs.candidates else None,
            ),
            outputs.history,
            outputs.history,
        )

    # gradio.helpers.special_args reads the signature from the left and stops at
    # the first parameter that is not positional, so a trailing keyword-only
    # progress is never recognised: the handler then gets an unwired Progress
    # whose calls go nowhere, and the browser shows only "processing | 47.2s".
    # Leading it is what gets the reports onto the queue.
    def handle_request(progress=gr.Progress(), *values):
        return dispatch(values, progress)

    def handle_scene_prompt(progress=gr.Progress(), *values):
        return dispatch(values, progress, action_override="scene_prompt")

    def handle_next_panel(progress=gr.Progress(), *values):
        # An image alone is enough here; the router would otherwise read a
        # missing instruction as a request for plain tag extraction.
        return dispatch(values, progress, action_override="next_panel")

    situations = load_situations()
    vocabulary = load_subject_vocabulary()

    with gr.Blocks(title="Danbooru Prompt Workbench") as demo:
        gr.HTML("<style>.url-drop-bridge { display: none !important; }</style>")
        with gr.Tabs():
            # The workbench is first and opens by default: the situation sweep
            # is one thing you might want, not the way in.
            with gr.Tab("ワークベンチ", elem_id="workbench-tab"):
                task = _build_task_selector(gr, stored)
                with gr.Row():
                    image = _build_image_column(gr, stored)
                    controls = _build_instruction_column(gr, stored)
                results = _build_result_section(gr)
                with gr.Row():
                    settings = _build_advanced_settings(gr, stored)
                    results = SimpleNamespace(
                        **vars(results), **vars(_build_run_details(gr))
                    )
            sweep = _build_situation_tab(gr, situations)

        # A field name can own more than one component - a button and the hint
        # that explains it have to appear and disappear together.
        task_components = {
            "vision": [image.vision_box],
            "instruction": [controls.instruction],
            "base_prompt": [controls.base_prompt_box],
            "follow_up": [controls.follow_up_box],
            "panel_change": [controls.next_panel_box],
            "variants": [controls.variants_box],
            "scene_template": [controls.scene_template],
            "situation": [controls.situation],
            "scene_settings": [settings.scene_settings_box],
            "run": [controls.run_button],
            "next_panel": [controls.next_panel_button],
            "scene_prompt": [controls.scene_prompt_button],
        }
        assert tuple(task_components) == TASK_FIELD_ORDER
        # The page opens on the remembered task, so the built-in visibility
        # would otherwise be a layout for a task nobody selected.
        initial = dict(
            zip(
                TASK_FIELD_ORDER,
                task_field_visibility(task.action_override.value),
            )
        )
        for name, components in task_components.items():
            for component in components:
                component.visible = initial[name]

        # These two sit beside 実行 under おまかせ, but stand alone under their own
        # task, and the only action on the page should not look like a secondary one.
        promotable = {
            "next_panel": controls.next_panel_button,
            "scene_prompt": controls.scene_prompt_button,
        }

        def show_task_fields(selected: str):
            shown = dict(zip(TASK_FIELD_ORDER, task_field_visibility(selected)))
            updates = []
            for name in TASK_FIELD_ORDER:
                for component in task_components[name]:
                    if promotable.get(name) is component:
                        updates.append(
                            gr.update(
                                visible=shown[name],
                                variant="secondary" if shown["run"] else "primary",
                            )
                        )
                    else:
                        updates.append(gr.update(visible=shown[name]))
            return updates

        task.action_override.change(
            show_task_fields,
            inputs=task.action_override,
            outputs=[
                component
                for name in TASK_FIELD_ORDER
                for component in task_components[name]
            ],
            queue=False,
        )

        def show_parts(prompt: str):
            found = prompt_parts(prompt)
            return [
                gr.update(visible=bool(found)),
                *(
                    gr.update(
                        value=found.get(category, ""),
                        visible=bool(found.get(category)),
                    )
                    for category, _label in PART_LABELS
                ),
            ]

        # Driven by the box rather than by the run, so an edited prompt and an
        # adopted candidate both split into the parts you can actually see.
        results.prompts[0].change(
            show_parts,
            inputs=results.prompts[0],
            outputs=[results.parts_box, *results.parts],
            queue=False,
        )

        inputs = [
            *_run_inputs(
                task=task,
                image=image,
                controls=controls,
                settings=settings,
                results=results,
            ),
            results.history_state,
        ]
        outputs = [
            results.action_plan,
            results.inferred_tags,
            image.description,
            *results.prompts,
            results.prose_plain,
            results.prose_avoid,
            results.prose_prompt,
            results.status,
            results.candidate_selector,
            results.history_state,
            results.history_output,
        ]
        run_event = controls.run_button.click(
            handle_request,
            inputs=inputs,
            outputs=outputs,
            api_name="run_prompt_workbench",
            concurrency_limit=1,
        )
        next_panel_event = controls.next_panel_button.click(
            handle_next_panel,
            inputs=inputs,
            outputs=outputs,
            api_name="run_next_panel",
            concurrency_limit=1,
        )

        def cleared_prompt_state(status: str):
            """Tags, description, base prompt, outputs, candidates, plan, and status."""
            return (
                "",
                "",
                "",
                *_prompt_updates(gr, blank_prompt_boxes()),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
                "",
                gr.Radio(choices=[], value=None),
                {},
                status,
            )

        def handle_image_upload(image_path):
            return (
                *accept_dropped_image(image_path),
                *cleared_prompt_state(""),
            )

        def handle_image_url(image_url, allow_private_hosts):
            return (
                preview_image_url(image_url, allow_private_hosts),
                None,
                (image_url or "").strip(),
                *cleared_prompt_state("URL画像を読み込みました。"),
            )

        cleared_outputs = [
            results.inferred_tags,
            image.description,
            controls.base_prompt,
            *results.prompts,
            results.prose_plain,
            results.prose_avoid,
            results.prose_prompt,
            results.candidate_selector,
            results.action_plan,
            results.status,
        ]
        image.workspace.input(
            handle_image_upload,
            inputs=image.workspace,
            outputs=[image.active_file, settings.url_input, *cleared_outputs],
            queue=False,
        )
        image_url_outputs = [
            image.workspace,
            image.active_file,
            settings.url_input,
            *cleared_outputs,
        ]
        settings.url_button.click(
            handle_image_url,
            inputs=[settings.url_input, settings.allow_private_image_urls],
            outputs=image_url_outputs,
            queue=False,
        )
        image.dropped_url_button.click(
            handle_image_url,
            inputs=[image.dropped_url, settings.allow_private_image_urls],
            outputs=image_url_outputs,
            queue=False,
            api_name=False,
        )
        settings.diagnostic_button.click(
            diagnose_ollama,
            inputs=[
                settings.ollama_url,
                settings.router_model,
                settings.compiler_model,
                settings.vision_model,
                settings.scene_model,
                image.use_vision,
            ],
            outputs=settings.diagnostic_output,
            queue=False,
        )
        image.recover_vision_button.click(
            lambda ollama_url, vision_model: recover_vision_model(
                prompt_service, ollama_url, vision_model
            ),
            inputs=[settings.ollama_url, settings.vision_model],
            outputs=image.recover_vision_status,
            queue=False,
            api_name="recover_vision_model",
        )
        settings.save_excluded_tags_button.click(
            store_excluded_tags,
            inputs=settings.excluded_tags,
            outputs=[settings.excluded_tags, settings.excluded_tags_status],
            queue=False,
        )
        settings.reset_excluded_tags_button.click(
            restore_default_excluded_tags,
            inputs=None,
            outputs=[settings.excluded_tags, settings.excluded_tags_status],
            queue=False,
        )
        results.adopt_button.click(
            adopt_candidate,
            inputs=results.candidate_selector,
            outputs=controls.base_prompt,
            queue=False,
        )
        scene_prompt_event = controls.scene_prompt_button.click(
            handle_scene_prompt,
            inputs=inputs,
            outputs=outputs,
            api_name="run_scene_prompt",
            concurrency_limit=1,
        )
        submit_event = controls.instruction.submit(
            handle_request,
            inputs=inputs,
            outputs=outputs,
            api_name=False,
            concurrency_limit=1,
        )
        controls.cancel_button.click(
            fn=None,
            cancels=[run_event, next_panel_event, scene_prompt_event, submit_event],
            queue=False,
        )

        def run_sweep(progress, chosen, subject, style, view, settings_values):
            """The sweep itself, once the situations and subject are settled."""
            as_prose = style == "prose"
            if not chosen:
                return [
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    "シチュエーションを1つ以上選んでください。",
                    [],
                ]
            (
                template,
                ollama_url,
                compiler_model,
                scene_model,
                gpu_wait_gb,
                apply_tag_exclusions,
                excluded_tags,
            ) = settings_values
            runs = run_situation_sweep(
                prompt_service,
                chosen,
                {item.name: item.label for item in situations},
                instruction=(subject or "").strip(),
                as_prose=as_prose,
                options={
                    "ollama_url": ollama_url,
                    "compiler_model": compiler_model,
                    "scene_model": scene_model or compiler_model,
                    "scene_template": template,
                    "gpu_wait_gb": gpu_wait_gb,
                    "apply_tag_exclusions": apply_tag_exclusions,
                    "excluded_tags": excluded_tags,
                },
                on_progress=lambda stage, fraction: progress(
                    fraction, desc=PROGRESS_LABELS.get(stage, stage)
                ),
            )
            return [
                *_situation_outputs(gr, runs, as_prose=as_prose, view=view),
                *_situation_summary(gr, runs),
                # Kept so switching between the two shapes re-reads the same
                # answers instead of asking the models for them again.
                (runs, as_prose),
            ]

        def _chosen_and_rest(values):
            """The ticked situations, in page order, and everything after them."""
            picks = set()
            for group in values[: len(sweep.pickers)]:
                picks.update(group or [])
            return (
                [item.name for item in situations if item.name in picks],
                values[len(sweep.pickers) :],
            )

        def handle_situation_sweep(progress=gr.Progress(), *values):
            # The picks arrive one list per category, so they are gathered back
            # into the order the situations are defined in - the blocks should
            # read the way the groups do, not in the order they were ticked.
            chosen, rest = _chosen_and_rest(values)
            subject, style, view, *settings_values = rest
            return run_sweep(progress, chosen, subject, style, view, settings_values)

        def handle_random_sweep(progress=gr.Progress(), *values):
            """Invent a subject, pick the situations, and run - in one call.

            Chained as two events this raced: the second press ran on the
            values the first press had left behind, because the randomised ones
            had not reached the browser and come back yet. It finished in a
            second with the previous answer still on screen. Deciding and
            running in the same call cannot get that wrong; the picks are
            returned alongside the answers so they still show in the controls,
            which is what lets a random run be adjusted and repeated.
            """
            count, *rest_values = values
            _ignored, rest = _chosen_and_rest(rest_values)
            _typed_subject, style, view, *settings_values = rest
            chosen = random_situation_names(
                [item.name for item in situations], int(count or 1)
            )
            subject = random_subject(vocabulary, as_prose=style == "prose")
            picked = set(chosen)
            return [
                subject,
                *(
                    gr.update(
                        value=[item.name for item in members if item.name in picked]
                    )
                    for _category, members in group_situations(situations)
                ),
                *run_sweep(progress, chosen, subject, style, view, settings_values),
            ]

        # Named once: the random path runs on the same controls, and two lists
        # kept in step by hand would drift the first time a setting was added
        # to one of them. The random path prepends its own count.
        situation_inputs = [
            *sweep.pickers,
            sweep.subject,
            sweep.output_style,
            sweep.view,
            sweep.template,
            settings.ollama_url,
            settings.compiler_model,
            settings.scene_model,
            settings.gpu_wait_gb,
            settings.apply_tag_exclusions,
            settings.excluded_tags,
        ]
        situation_outputs = [
            sweep.shared,
            sweep.merged,
            sweep.avoid,
            sweep.status,
            sweep.runs_state,
        ]
        situation_event = sweep.run_button.click(
            handle_situation_sweep,
            inputs=situation_inputs,
            outputs=situation_outputs,
            # Drawn on the status line under the button. Left to Gradio's own
            # choice it went onto the output boxes, which are all hidden on the
            # first run - so the run that most needs reporting reported nothing.
            show_progress_on=sweep.status,
            api_name="run_situation_sweep",
            concurrency_limit=1,
        )

        def handle_situation_view(view: str, state):
            runs, as_prose = state if state else ([], False)
            return _situation_outputs(gr, runs, as_prose=as_prose, view=view)

        sweep.view.change(
            handle_situation_view,
            inputs=[sweep.view, sweep.runs_state],
            outputs=[sweep.shared, sweep.merged],
            queue=False,
        )
        grouped_situations = group_situations(situations)
        random_event = sweep.random_button.click(
            handle_random_sweep,
            inputs=[sweep.random_count, *situation_inputs],
            outputs=[sweep.subject, *sweep.pickers, *situation_outputs],
            show_progress_on=sweep.status,
            api_name="run_random_situations",
            concurrency_limit=1,
        )
        sweep.cancel_button.click(
            fn=None, cancels=[situation_event, random_event], queue=False
        )
        sweep.select_all_button.click(
            lambda: [
                gr.update(value=[item.name for item in members])
                for _category, members in grouped_situations
            ],
            outputs=sweep.pickers,
            queue=False,
        )
        # Forty-six ticks are quicker to undo than to undo one at a time.
        sweep.clear_button.click(
            lambda: [gr.update(value=[]) for _ in sweep.pickers],
            outputs=sweep.pickers,
            queue=False,
        )
        # The template only shapes prose, so it is only asked for when prose is
        # what was chosen.
        sweep.output_style.change(
            lambda style: gr.update(visible=style == "prose"),
            inputs=sweep.output_style,
            outputs=sweep.template,
            queue=False,
        )
        demo.load(
            fn=None,
            js=IMAGE_INPUT_JS,
            queue=False,
            api_name=False,
        )

    return demo


def _run_inputs(*, task, image, controls, settings, results) -> list:
    """One component per WebRunRequest field, ordered by that single definition."""
    run_components = {
        "image_path": image.active_file,
        "image_url": settings.url_input,
        "instruction": controls.instruction,
        "base_prompt": controls.base_prompt,
        "router_model": settings.router_model,
        "compiler_model": settings.compiler_model,
        "ollama_url": settings.ollama_url,
        "general_threshold": settings.general_threshold,
        "character_threshold": settings.character_threshold,
        "max_image_tags": settings.max_image_tags,
        "variants": controls.variants,
        "generate_next_panel": controls.generate_next_panel,
        "next_panel_change": controls.next_panel_change,
        "next_panel_time": controls.next_panel_time,
        "next_panel_chain": controls.next_panel_chain,
        "scene_template": controls.scene_template,
        "situation": controls.situation,
        "scene_model": settings.scene_model,
        "scene_sees_image": settings.scene_sees_image,
        "also_prose": controls.also_prose,
        "edited_tags": results.inferred_tags,
        "edited_description": image.description,
        "action_override": task.action_override,
        "use_vision": image.use_vision,
        "vision_model": settings.vision_model,
        "allow_private_image_urls": settings.allow_private_image_urls,
        "gpu_wait_gb": settings.gpu_wait_gb,
        "apply_tag_exclusions": settings.apply_tag_exclusions,
        "excluded_tags": settings.excluded_tags,
    }
    missing = set(WEB_RUN_FIELDS) - set(run_components)
    if missing:  # pragma: no cover - guards a wiring mistake at build time
        raise RuntimeError(f"Web run fields without a component: {sorted(missing)}")
    return [run_components[name] for name in WEB_RUN_FIELDS]


def _build_task_selector(gr, stored: dict) -> SimpleNamespace:
    """The first question, and the one that decides what the rest of the page shows."""
    action_override = gr.Radio(
        choices=list(TASK_CHOICES),
        value=remembered(stored, "action_override", "auto"),
        label="やりたいこと",
        elem_id="task-selector",
    )
    return SimpleNamespace(action_override=action_override)


def _build_image_column(gr, stored: dict) -> SimpleNamespace:
    """Unified image workspace plus the hidden bridge that URL drops write into."""
    with gr.Column():
        workspace = gr.Image(
            type="filepath",
            sources=["upload"],
            label="画像",
            placeholder="ここへ画像をドロップ、クリックして選択、または Ctrl+V で貼り付け",
            height=150,
            interactive=True,
            elem_id="image-workspace",
            buttons=["fullscreen"],
        )
        active_file = gr.File(
            file_count="single",
            file_types=["image"],
            type="filepath",
            visible="hidden",
        )
        dropped_url = gr.Textbox(
            elem_id="dropped-image-url-input",
            elem_classes="url-drop-bridge",
            container=False,
        )
        dropped_url_button = gr.Button(
            "ドロップURLを読み込む",
            elem_id="dropped-image-url-button",
            elem_classes="url-drop-bridge",
        )
        with gr.Group() as vision_box:
            # The switch and its recovery button share one line: the button is
            # pressed rarely, and a row of its own cost 40px on every page.
            with gr.Row():
                use_vision = gr.Checkbox(
                    value=remembered(stored, "use_vision", True),
                    label="VLMで画像を説明する",
                    info="ポーズや位置関係の解析にも使います。",
                    scale=3,
                )
                recover_vision_button = gr.Button(
                    "VLMを復旧",
                    elem_id="recover-vision-button",
                    size="sm",
                    scale=1,
                )
            # The hint only matters once the button has been pressed, so it
            # takes no room on the page before then.
            recover_vision_status = gr.Markdown()
            description = gr.Textbox(
                label="画像の説明（VLM）",
                lines=2,
                buttons=["copy"],
                interactive=True,
                elem_id="image-description-editor",
                placeholder="VLMを有効にして実行すると、画像の内容がここに入ります。",
            )
    return SimpleNamespace(
        workspace=workspace,
        vision_box=vision_box,
        active_file=active_file,
        dropped_url=dropped_url,
        dropped_url_button=dropped_url_button,
        use_vision=use_vision,
        description=description,
        recover_vision_button=recover_vision_button,
        recover_vision_status=recover_vision_status,
    )


def _build_instruction_column(gr, stored: dict) -> SimpleNamespace:
    """Instruction, optional base prompt, output count, and the run controls."""
    with gr.Column():
        instruction = gr.Textbox(
            label="どうしたい？",
            placeholder="例: タグを推測して / 次のコマで振り返らせて / 夜に変更して",
            lines=2,
        )
        with gr.Accordion("既存プロンプトから編集（任意）", open=False) as base_prompt_box:
            base_prompt = gr.Textbox(
                label="既存プロンプト",
                placeholder="画像の代わりに既存タグを編集するときに入力",
                lines=4,
                elem_id="base-prompt-input",
            )
        with gr.Row() as follow_up_box:
            generate_next_panel = gr.Checkbox(
                value=remembered(stored, "generate_next_panel", True),
                label="次のコマも生成する（出力2〜4）",
            )
            also_prose = gr.Checkbox(
                value=remembered(stored, "also_prose", True),
                label="英文プロンプトも出す",
                elem_id="also-prose-input",
            )
        # Two axes of the same question, so they sit on one line and toggle as one.
        with gr.Row() as next_panel_box:
            next_panel_time = gr.Slider(
                0.0,
                1.0,
                value=remembered(stored, "next_panel_time", DEFAULT_NEXT_PANEL_TIME),
                step=0.1,
                label="経過する時間",
                elem_id="next-panel-time",
            )
            next_panel_chain = gr.Checkbox(
                value=remembered(stored, "next_panel_chain", False),
                label="1コマずつ進める",
                elem_id="next-panel-chain",
            )
            next_panel_change = gr.Slider(
                0.0,
                1.0,
                value=remembered(stored, "next_panel_change", DEFAULT_NEXT_PANEL_CHANGE),
                step=0.1,
                label="変わってよい範囲",
                elem_id="next-panel-change",
            )
        scene_template = gr.Dropdown(
            choices=[(template.label, template.name) for template in load_templates()],
            value=remembered(stored, "scene_template", DEFAULT_SCENE_TEMPLATE),
            label="自然文プロンプトのテンプレート",
            elem_id="scene-template",
            visible=False,
            info="「自然文プロンプト」で使う骨組みです。templates/ にYAMLを足せば増やせます。",
        )
        with gr.Row() as variants_box:
            # container=False drops Gradio's label with its padding, so the
            # label is written beside the pills on the same line instead.
            gr.Markdown("出力数", container=False, scale=0)
            variants = gr.Radio(
                choices=[1, 2, 3, 4],
                value=remembered(stored, "variants", 4),
                label="出力数",
                container=False,
                scale=4,
            )
        # Its own line, with its own label. Sharing the output count's row saved
        # 43px and cost the control its name - container=False takes the label
        # with it - so it read as an unexplained box belonging to 出力数, whose
        # own choices it pushed onto a second line.
        situation = gr.Dropdown(
            choices=situation_choices(load_situations()),
            value=remembered(stored, "situation", NO_SITUATION),
            label="シチュエーション",
            elem_id="situation-input",
            info="日常・戦闘などの方向づけ。これだけでも生成できます。",
        )
        with gr.Row():
            run_button = gr.Button("実行", variant="primary")
            next_panel_button = gr.Button(
                "次のコマ",
                elem_id="next-panel-button",
            )
            scene_prompt_button = gr.Button(
                "自然文プロンプト",
                elem_id="scene-prompt-button",
                visible=False,
            )
            cancel_button = gr.Button("停止", variant="stop")
    return SimpleNamespace(
        instruction=instruction,
        base_prompt=base_prompt,
        base_prompt_box=base_prompt_box,
        variants=variants,
        variants_box=variants_box,
        generate_next_panel=generate_next_panel,
        follow_up_box=follow_up_box,
        also_prose=also_prose,
        next_panel_change=next_panel_change,
        next_panel_time=next_panel_time,
        next_panel_chain=next_panel_chain,
        next_panel_box=next_panel_box,
        scene_template=scene_template,
        situation=situation,
        scene_prompt_button=scene_prompt_button,
        run_button=run_button,
        next_panel_button=next_panel_button,
        cancel_button=cancel_button,
    )


def _build_advanced_settings(gr, stored: dict) -> SimpleNamespace:
    """Folded models, diagnostics, tagging thresholds, and exclusion words."""
    with gr.Accordion("詳細設定", open=False):
        with gr.Row():
            router_model = gr.Dropdown(
                choices=list(TEXT_MODEL_CHOICES),
                value=remembered(stored, "router_model", DEFAULT_ROUTER_MODEL),
                allow_custom_value=True,
                label="指示ルーターモデル",
            )
            compiler_model = gr.Dropdown(
                choices=list(TEXT_MODEL_CHOICES),
                value=remembered(stored, "compiler_model", DEFAULT_COMPILER_MODEL),
                allow_custom_value=True,
                label="プロンプト生成モデル",
            )
            ollama_url = gr.Textbox(
                value=remembered(stored, "ollama_url", DEFAULT_OLLAMA_URL),
                label="Ollama URL",
            )
            vision_model = gr.Dropdown(
                choices=list(VISION_MODEL_CHOICES),
                value=remembered(stored, "vision_model", DEFAULT_VISION_MODEL),
                allow_custom_value=True,
                label="VLMモデル",
                info=(
                    "一覧にないモデルは直接入力できます。"
                    "既定のモデルが説明を拒否・省略する画像では無検閲のものを選んでください。"
                ),
            )
        with gr.Group(visible=False) as scene_settings_box:
            scene_model = gr.Dropdown(
                choices=list(SCENE_MODEL_CHOICES),
                value=remembered(stored, "scene_model", DEFAULT_SCENE_MODEL),
                allow_custom_value=True,
                label="自然文プロンプト用モデル",
                elem_id="scene-model-input",
                info="英文の作文はタグ生成より重いので、ここだけ大きめにできます。",
            )
            scene_sees_image = gr.Checkbox(
                value=False,
                label="自然文プロンプトに画像を渡す",
                elem_id="scene-sees-image-input",
                info=(
                    "自然文プロンプト用モデルがVLMのときだけ有効です。"
                    "画像を直接見て書くぶん描写は濃くなりますが、"
                    "タグは事実として併せて渡すので特徴は落ちません。"
                ),
            )
        gpu_wait_gb = gr.Slider(
            0.0,
            12.0,
            value=remembered(stored, "gpu_wait_gb", DEFAULT_GPU_WAIT_GB),
            step=0.5,
            label="他タスクのGPU使用で待つ閾値（GB）",
            elem_id="gpu-wait-input",
            info="他のプログラムがこれ以上VRAMを使っていたら、空くまで少し待ちます。0で無効。",
        )
        allow_private_image_urls = gr.Checkbox(
            value=remembered(stored, "allow_private_image_urls", False),
            label="プライベート画像URLを許可",
        )
        with gr.Accordion(
            "URLから読み込む（補助）",
            open=False,
            elem_id="image-url-accordion",
        ):
            url_input = gr.Textbox(
                label="画像URL",
                placeholder="https://example.com/image.png",
                elem_id="image-url-input",
            )
            url_button = gr.Button(
                "URLを読み込む",
                elem_id="image-url-load-button",
            )
            gr.Markdown(
                "Webページ上の画像や画像URLは、画像欄へ直接ドロップすることもできます。"
            )
        diagnostic_button = gr.Button("Ollama接続確認")
        diagnostic_output = gr.Markdown(label="Ollama診断")
        with gr.Row():
            general_threshold = gr.Slider(
                0.0,
                1.0,
                value=remembered(stored, "general_threshold", 0.35),
                step=0.01,
                label="一般タグ閾値",
            )
            character_threshold = gr.Slider(
                0.0,
                1.0,
                value=remembered(stored, "character_threshold", 0.85),
                step=0.01,
                label="キャラクター閾値",
            )
            max_image_tags = gr.Slider(
                1,
                100,
                value=remembered(stored, "max_image_tags", 50),
                step=1,
                label="画像タグ上限",
            )
        with gr.Accordion("除外ワード", open=True):
            apply_tag_exclusions = gr.Checkbox(
                value=remembered(stored, "apply_tag_exclusions", True),
                label="除外ワードを適用",
            )
            excluded_tags = gr.Textbox(
                value=load_exclusion_text(),
                label="除外ワード（カンマ区切り、*使用可）",
                lines=3,
                elem_id="excluded-tags-input",
                info=(
                    "画像タグと出力プロンプトの両方から取り除きます。"
                    "例: *censor*, *_text, watermark, *_background"
                ),
            )
            with gr.Row():
                save_excluded_tags_button = gr.Button("除外ワードを保存")
                reset_excluded_tags_button = gr.Button("既定に戻す")
            excluded_tags_status = gr.Markdown(
                "保存すると次回起動時もこの除外ワードを使います。"
                "保存済みの内容は既定より優先されるので、"
                "更新された既定を取り込むときは「既定に戻す」を押してください。"
            )
    return SimpleNamespace(
        router_model=router_model,
        compiler_model=compiler_model,
        ollama_url=ollama_url,
        vision_model=vision_model,
        scene_model=scene_model,
        scene_sees_image=scene_sees_image,
        scene_settings_box=scene_settings_box,
        allow_private_image_urls=allow_private_image_urls,
        gpu_wait_gb=gpu_wait_gb,
        url_input=url_input,
        url_button=url_button,
        diagnostic_button=diagnostic_button,
        diagnostic_output=diagnostic_output,
        general_threshold=general_threshold,
        character_threshold=character_threshold,
        max_image_tags=max_image_tags,
        apply_tag_exclusions=apply_tag_exclusions,
        excluded_tags=excluded_tags,
        save_excluded_tags_button=save_excluded_tags_button,
        reset_excluded_tags_button=reset_excluded_tags_button,
        excluded_tags_status=excluded_tags_status,
    )


def _build_situation_tab(gr, situations: list) -> SimpleNamespace:
    """One prompt per situation, from one subject, in one run.

    This generates: the output is a prompt for each situation picked, ready to
    paste. Reading them against each other is something you may then want to do
    - 出力の見せかた is there for it - but it is not what the tab is for, and
    calling the tab a comparison described the smaller half of it.

    It needs neither an image nor a router, so it gets its own tab rather than
    another mode of the workbench, and none of the controls that do not apply.
    """
    with gr.Tab("シチュエーション一括生成", elem_id="situation-tab"):
        gr.Markdown(
            "シチュエーションごとにプロンプトを1件ずつ作り、1つにまとめて返します。"
            "画像は使いません。"
            "「おまかせ生成」は主題とシチュエーションをその場でランダムに決めて、"
            "そのまま最後まで実行します。何も入力・選択しなくて構いません。"
            "1件につきモデルを1回呼ぶので、件数を増やすとその分だけ時間がかかります。"
        )
        # What to make and what came out, side by side at the top; the picker
        # gets its own full-width row underneath. Stacked, the answers began
        # below the fold however few there were. Moved into this column with
        # the picker, the picker doubled in height at half the width and took
        # the run button off the screen instead - it is five short rows wide
        # and ten tall, so width is what it wants.
        with gr.Row():
            # The controls need less room than the answers do, and prose runs
            # to long lines.
            with gr.Column(scale=2):
                subject = gr.Textbox(
                    label="共通の主題（任意）",
                    placeholder="例: 弓を持った銀髪のエルフ",
                    lines=2,
                    elem_id="situation-subject",
                    info="すべてのシチュエーションで共通の人物・場面。"
                    "空欄ならシチュエーションだけで生成します。",
                )
                with gr.Row():
                    output_style = gr.Radio(
                        choices=[("タグ", "tags"), ("自然文", "prose")],
                        value="tags",
                        label="出力形式",
                        elem_id="situation-style",
                    )
                    # A character sheet is the same character rendered
                    # neutrally, so its Lighting and Layout slots ("key light
                    # direction", "framing, camera distance") have nothing to
                    # do with the moment and came back word for word in every
                    # situation. A scene illustration asks about place, time of
                    # day and mood, which is what a situation actually changes.
                    template = gr.Dropdown(
                        choices=[(item.label, item.name) for item in load_templates()],
                        value=SWEEP_SCENE_TEMPLATE,
                        label="自然文プロンプトのテンプレート",
                        visible=False,
                        elem_id="situation-template",
                    )
                # The hands-off path first, and it is the primary one:
                # choosing a subject and a handful of situations by hand is the
                # part of a sweep that is work rather than result.
                random_count = gr.Slider(
                    1,
                    8,
                    value=3,
                    step=1,
                    label="おまかせで選ぶ件数",
                    elem_id="situation-random-count",
                )
                with gr.Row():
                    random_button = gr.Button(
                        "おまかせ生成",
                        variant="primary",
                        scale=3,
                        elem_id="situation-random",
                    )
                    # A sweep is one model call per situation, so it runs long
                    # enough that leaving without a way to stop it would be its
                    # own bug.
                    cancel_button = gr.Button(
                        "停止", variant="stop", scale=1, elem_id="situation-cancel"
                    )
                # Two rows rather than four buttons across half the page: at
                # that width the fourth wrapped onto a line of its own anyway,
                # and wrapped it was 停止 that got the whole line.
                with gr.Row():
                    select_all_button = gr.Button("すべて選択")
                    clear_button = gr.Button("選択解除")
                run_button = gr.Button("選んだ分を生成", elem_id="situation-run")
                # Under the buttons, because the progress is drawn on it and
                # this is the column the buttons are in. Put with the results
                # instead it sat 308px away and above the button that starts
                # the run, so pressing it changed nothing anywhere near where
                # it was pressed - which reads as a button that does nothing.
                # It carries a line from the start: an empty Markdown is zero
                # pixels tall, and the progress is drawn inside it.
                status = gr.Markdown(
                    "シチュエーションを選んで「おまかせ生成」を押してください。",
                    elem_id="situation-status",
                )
            with gr.Column(scale=3):
                # 全文 first and by default: what this tab produces is prompts
                # to paste, and a text holding only the lines that differ is
                # not one. 違いだけ is for reading the set, which is a second
                # thing you may want to do with it rather than what it is for.
                view = gr.Radio(
                    choices=[
                        ("共通をまとめる", SHARED_FIRST_VIEW),
                        ("ブロックごとに完結", "full"),
                    ],
                    value=SHARED_FIRST_VIEW,
                    label="出力の見せかた",
                    elem_id="situation-view",
                    info="「共通をまとめる」は先頭に共通ブロックを置き、"
                    "以降の各ブロックからその分を除きます。"
                    "「ブロックごとに完結」は各ブロック単体で完成形になります。",
                )
                shared = gr.Textbox(
                    label="共通プロンプト",
                    lines=3,
                    buttons=["copy"],
                    interactive=True,
                    visible=False,
                    elem_id="situation-shared",
                    info="どのシチュエーションでも同じだった部分です。"
                    "「共通をまとめる」では統合プロンプトの先頭ブロックと同じもので、"
                    "土台だけを取り出したいとき用です。",
                )
                # One text rather than a box per situation: up to forty-six
                # boxes is not a result anyone reads, it is a haystack. Joined
                # carelessly it would be worse than the boxes, so the format is
                # stated and `split_situation_blocks` undoes it.
                merged = gr.Textbox(
                    label="統合プロンプト",
                    lines=16,
                    max_lines=28,
                    buttons=["copy"],
                    interactive=True,
                    visible=False,
                    elem_id="situation-merged",
                    info="空行区切り、各ブロックの先頭が「# 番号 シチュエーション名」です。"
                    "空行で分割し、先頭行を外せば各ブロックの中身になります。"
                    "「共通をまとめる」では、先頭の「# 共通」と各ブロックを"
                    "合わせて1件分です。",
                )
                # Shared rather than one per block: the avoid list comes from
                # the exclusion rules, so it is the same for every situation.
                avoid = gr.Textbox(
                    label="除外（ネガティブプロンプト）",
                    lines=2,
                    buttons=["copy"],
                    interactive=True,
                    visible=False,
                    elem_id="situation-avoid",
                )
        # Full width, which is what keeps it to five short rows. One group per
        # category rather than one list of forty-odd, and the groups come from
        # the files: a new situation joins its category and a new category
        # appears on its own, neither needing this page changed.
        pickers = []
        for category, members in group_situations(situations):
            pickers.append(
                gr.CheckboxGroup(
                    choices=[(item.label, item.name) for item in members],
                    value=[],
                    label=category,
                    elem_id=f"situation-picker-{len(pickers) + 1}",
                )
            )
    return SimpleNamespace(
        subject=subject,
        output_style=output_style,
        template=template,
        pickers=pickers,
        select_all_button=select_all_button,
        clear_button=clear_button,
        run_button=run_button,
        random_button=random_button,
        random_count=random_count,
        cancel_button=cancel_button,
        view=view,
        shared=shared,
        merged=merged,
        avoid=avoid,
        status=status,
        runs_state=gr.State([]),
    )


def _build_result_section(gr) -> SimpleNamespace:
    """Editable image tags, prompt outputs, candidate history, and run details."""
    with gr.Accordion("画像タグの確認・修正", open=False):
        inferred_tags = gr.Textbox(
            label="画像タグ",
            lines=4,
            buttons=["copy"],
            interactive=True,
            elem_id="inferred-tags-editor",
            info="必要な場合だけ修正して、もう一度実行してください。",
        )
    def prompt_box(number: int):
        return gr.Textbox(
            label=f"出力プロンプト {number}",
            lines=4,
            buttons=["copy"],
            interactive=True,
            visible=False,
            elem_id=f"prompt-output-{number}",
        )

    # Boxes 2-4 come and go together: a task that answers with one list has no
    # use for three empty boxes the height of the answer.
    prompts = []
    with gr.Row():
        prompts.append(prompt_box(1))
        prompts.append(prompt_box(2))
    with gr.Row():
        prompts.append(prompt_box(3))
        prompts.append(prompt_box(4))
    # The same groups the output is already organized into, one box each, so a
    # prompt can be reused piecewise. Hidden until a run fills them, like the
    # prompt boxes above.
    parts: list = []
    # The rows keep their gap even with every child hidden, so the whole block
    # comes and goes rather than each box on its own.
    with gr.Group(visible=False) as parts_box:
        for row_start in range(0, len(PART_LABELS), 4):
            with gr.Row():
                for category, label in PART_LABELS[row_start : row_start + 4]:
                    parts.append(
                        gr.Textbox(
                            label=label,
                            lines=2,
                            buttons=["copy"],
                            interactive=True,
                            visible=False,
                            elem_id=f"prompt-part-{category}",
                        )
                    )
    # The pasteable pair comes first; the templated form is how the prose was
    # written, which is reference material rather than something to paste.
    prose_plain = gr.Textbox(
        label="英文プロンプト（貼り付け用）",
        lines=4,
        buttons=["copy"],
        interactive=True,
        visible=False,
        elem_id="prose-plain-output",
        info="ラベルを外した本文です。そのまま画像モデルに貼り付けられます。",
    )
    prose_avoid = gr.Textbox(
        label="除外（ネガティブプロンプト）",
        lines=2,
        buttons=["copy"],
        interactive=True,
        visible=False,
        elem_id="prose-avoid-output",
    )
    # Errors land here, so it must not be hidden inside a collapsed section.
    status = gr.Markdown(label="状態", elem_id="run-status")
    history_state = gr.State([])
    return SimpleNamespace(
        inferred_tags=inferred_tags,
        prompts=prompts,
        parts=parts,
        prose_plain=prose_plain,
        prose_avoid=prose_avoid,
        parts_box=parts_box,
        history_state=history_state,
        status=status,
    )


def _build_run_details(gr) -> SimpleNamespace:
    """Candidates, history, and the plan: everything about the run just made."""
    with gr.Accordion("実行の詳細", open=False):
        with gr.Row():
            candidate_selector = gr.Radio(
                choices=[],
                label="生成候補",
            )
            adopt_button = gr.Button("選択候補を採用")
        prose_prompt = gr.Textbox(
            label="英文プロンプト（テンプレート形式）",
            lines=6,
            buttons=["copy"],
            interactive=True,
            elem_id="prose-prompt-output",
            info="`Subject:` などのラベル付き。書き上がりの確認用です。",
        )
        history_output = gr.JSON(label="実行履歴（新しい順・最大20件）")
        action_plan = gr.JSON(label="実行計画")
    return SimpleNamespace(
        prose_prompt=prose_prompt,
        candidate_selector=candidate_selector,
        adopt_button=adopt_button,
        history_output=history_output,
        action_plan=action_plan,
    )


@web_app.command()
def main(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7860, "--port", min=1, max=65535),
    inbrowser: bool = typer.Option(True, "--inbrowser/--no-inbrowser"),
) -> None:
    demo = build_app()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        inbrowser=inbrowser,
        share=False,
    )


if __name__ == "__main__":
    web_app()
