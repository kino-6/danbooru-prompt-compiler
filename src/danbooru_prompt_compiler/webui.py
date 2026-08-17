from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import typer

from .image_source import load_image_url_preview, resolve_image_source
from .ollama_diagnostics import check_ollama, format_ollama_error
from .tag_filter import (
    DEFAULT_EXCLUSION_TEXT,
    EXCLUDED_TAGS_PATH,
    load_exclusion_text,
    save_exclusion_text,
)
from .web_service import (
    DEFAULT_COMPILER_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_VISION_MODEL,
    WEB_RUN_FIELDS,
    ProgressCallback,
    WebPromptService,
    WebRunRequest,
)


web_app = typer.Typer(help="Launch the local Danbooru Prompt Workbench web UI.")
PROGRESS_LABELS = {
    "routing": "指示を解釈しています",
    "tagging": "画像タグを推測しています",
    "vision": "VLMで構図を確認しています",
    "compilation": "プロンプトを生成しています",
    "complete": "完了",
}
MAX_HISTORY_ITEMS = 20
MAX_OUTPUT_VARIANTS = 4
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

  const isTextEntry = (element) =>
    element instanceof Element &&
    (element.isContentEditable ||
      element.tagName === "TEXTAREA" ||
      element.tagName === "INPUT");

  document.addEventListener("dragover", (event) => {
    if (!isImageWorkspace(event)) return;
    const types = Array.from(event.dataTransfer?.types || []);
    if (!types.includes("Files")) event.preventDefault();
  }, true);
  document.addEventListener("drop", (event) => {
    if (!isImageWorkspace(event)) return;
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
    use_vision: bool,
) -> str:
    required = [router_model, compiler_model]
    if use_vision:
        required.append(vision_model)
    return check_ollama(ollama_url, required).message


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


def run_workbench(
    service: WebPromptService,
    request: WebRunRequest,
    history: list[dict[str, str]] | None,
    *,
    on_progress: ProgressCallback | None = None,
) -> WorkbenchOutputs:
    """Resolve the image source, run one request, and fold errors into the status."""
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
                    request.vision_model if request.use_vision else "",
                ],
            ),
            candidates=[],
            history=history or [],
        )

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
        prompts=prompt_box_values(result.candidates),
        status=result.status,
        candidates=result.candidates,
        history=updated_history,
    )


def build_app(*, service: WebPromptService | None = None):
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("Web UI dependencies are missing; run 'uv sync --extra web'.") from exc

    prompt_service = service or WebPromptService()

    def dispatch(values, progress, *, action_override: str | None = None):
        """Gradio adapter: ordered values in, Gradio component updates out."""
        *run_values, history = values
        request = WebRunRequest.from_values(run_values)
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
            *outputs.prompts,
            outputs.status,
            gr.Radio(
                choices=outputs.candidates,
                value=outputs.candidates[0] if outputs.candidates else None,
            ),
            outputs.history,
            outputs.history,
        )

    def handle_request(*values, progress=gr.Progress()):
        return dispatch(values, progress)

    def handle_next_panel(*values, progress=gr.Progress()):
        # An image alone is enough here; the router would otherwise read a
        # missing instruction as a request for plain tag extraction.
        return dispatch(values, progress, action_override="next_panel")

    with gr.Blocks(title="Danbooru Prompt Workbench") as demo:
        gr.HTML("<style>.url-drop-bridge { display: none !important; }</style>")
        gr.Markdown(
            "# Danbooru Prompt Workbench\n"
            "画像を置いて日本語で指示するだけで、Danbooru形式のプロンプトを生成します。"
        )
        with gr.Row():
            image = _build_image_column(gr)
            controls = _build_instruction_column(gr)
        settings = _build_advanced_settings(gr)
        results = _build_result_section(gr)

        inputs = [
            *_run_inputs(image=image, controls=controls, settings=settings, results=results),
            results.history_state,
        ]
        outputs = [
            results.action_plan,
            results.inferred_tags,
            results.image_description,
            *results.prompts,
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
                *blank_prompt_boxes(),
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
            results.image_description,
            controls.base_prompt,
            *results.prompts,
            results.candidate_selector,
            results.action_plan,
            results.status,
        ]
        image.workspace.input(
            handle_image_upload,
            inputs=image.workspace,
            outputs=[image.active_file, image.url_input, *cleared_outputs],
            queue=False,
        )
        image_url_outputs = [
            image.workspace,
            image.active_file,
            image.url_input,
            *cleared_outputs,
        ]
        image.url_button.click(
            handle_image_url,
            inputs=[image.url_input, settings.allow_private_image_urls],
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
                settings.use_vision,
            ],
            outputs=settings.diagnostic_output,
            queue=False,
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
        submit_event = controls.instruction.submit(
            handle_request,
            inputs=inputs,
            outputs=outputs,
            api_name=False,
            concurrency_limit=1,
        )
        controls.cancel_button.click(
            fn=None,
            cancels=[run_event, next_panel_event, submit_event],
            queue=False,
        )
        demo.load(
            fn=None,
            js=IMAGE_INPUT_JS,
            queue=False,
            api_name=False,
        )

    return demo


def _run_inputs(*, image, controls, settings, results) -> list:
    """One component per WebRunRequest field, ordered by that single definition."""
    run_components = {
        "image_path": image.active_file,
        "image_url": image.url_input,
        "instruction": controls.instruction,
        "base_prompt": controls.base_prompt,
        "router_model": settings.router_model,
        "compiler_model": settings.compiler_model,
        "ollama_url": settings.ollama_url,
        "general_threshold": settings.general_threshold,
        "character_threshold": settings.character_threshold,
        "max_image_tags": settings.max_image_tags,
        "variants": controls.variants,
        "edited_tags": results.inferred_tags,
        "edited_description": results.image_description,
        "action_override": settings.action_override,
        "use_vision": settings.use_vision,
        "vision_model": settings.vision_model,
        "allow_private_image_urls": settings.allow_private_image_urls,
        "apply_tag_exclusions": settings.apply_tag_exclusions,
        "excluded_tags": settings.excluded_tags,
    }
    missing = set(WEB_RUN_FIELDS) - set(run_components)
    if missing:  # pragma: no cover - guards a wiring mistake at build time
        raise RuntimeError(f"Web run fields without a component: {sorted(missing)}")
    return [run_components[name] for name in WEB_RUN_FIELDS]


def _build_image_column(gr) -> SimpleNamespace:
    """Unified image workspace plus the hidden bridge that URL drops write into."""
    with gr.Column():
        workspace = gr.Image(
            type="filepath",
            sources=["upload"],
            label="画像",
            placeholder="ここへ画像をドロップ、クリックして選択、または Ctrl+V で貼り付け",
            height=400,
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
                "Webページ上の画像や画像URLは、上の画像欄へ直接ドロップすることもできます。"
            )
    return SimpleNamespace(
        workspace=workspace,
        active_file=active_file,
        dropped_url=dropped_url,
        dropped_url_button=dropped_url_button,
        url_input=url_input,
        url_button=url_button,
    )


def _build_instruction_column(gr) -> SimpleNamespace:
    """Instruction, optional base prompt, output count, and the run controls."""
    with gr.Column():
        instruction = gr.Textbox(
            label="どうしたい？",
            placeholder="例: タグを推測して / 次のコマで振り返らせて / 夜に変更して",
            lines=5,
        )
        with gr.Accordion("既存プロンプトから編集（任意）", open=False):
            base_prompt = gr.Textbox(
                label="既存プロンプト",
                placeholder="画像の代わりに既存タグを編集するときに入力",
                lines=4,
                elem_id="base-prompt-input",
            )
        with gr.Row():
            variants = gr.Radio(
                choices=[1, 2, 3, 4],
                value=4,
                label="出力数",
            )
        with gr.Row():
            run_button = gr.Button("実行", variant="primary")
            next_panel_button = gr.Button(
                "次のコマ",
                elem_id="next-panel-button",
            )
            cancel_button = gr.Button("停止", variant="stop")
        gr.Markdown(
            "「次のコマ」は指示がなくても押せます。画像だけを置いて押すと、"
            "一瞬後の場面を提案します。"
        )
    return SimpleNamespace(
        instruction=instruction,
        base_prompt=base_prompt,
        variants=variants,
        run_button=run_button,
        next_panel_button=next_panel_button,
        cancel_button=cancel_button,
    )


def _build_advanced_settings(gr) -> SimpleNamespace:
    """Folded models, diagnostics, tagging thresholds, and exclusion words."""
    with gr.Accordion("詳細設定", open=False):
        with gr.Row():
            router_model = gr.Textbox(
                value=DEFAULT_ROUTER_MODEL,
                label="指示ルーターモデル",
            )
            compiler_model = gr.Textbox(
                value=DEFAULT_COMPILER_MODEL,
                label="プロンプト生成モデル",
            )
            ollama_url = gr.Textbox(
                value=DEFAULT_OLLAMA_URL,
                label="Ollama URL",
            )
            vision_model = gr.Textbox(
                value=DEFAULT_VISION_MODEL,
                label="VLMモデル",
            )
            use_vision = gr.Checkbox(
                value=False,
                label="ポーズ・位置関係の解析にVLMを使う",
            )
            allow_private_image_urls = gr.Checkbox(
                value=False,
                label="プライベート画像URLを許可",
            )
        action_override = gr.Dropdown(
            choices=[
                ("自動判定", "auto"),
                ("画像タグ抽出", "tag_image"),
                ("新規プロンプト", "compile"),
                ("既存プロンプト編集", "edit"),
                ("次のコマ", "next_panel"),
            ],
            value="auto",
            label="操作種別",
        )
        diagnostic_button = gr.Button("Ollama接続確認")
        diagnostic_output = gr.Markdown(label="Ollama診断")
        with gr.Row():
            general_threshold = gr.Slider(
                0.0, 1.0, value=0.35, step=0.01, label="一般タグ閾値"
            )
            character_threshold = gr.Slider(
                0.0, 1.0, value=0.85, step=0.01, label="キャラクター閾値"
            )
            max_image_tags = gr.Slider(
                1, 100, value=50, step=1, label="画像タグ上限"
            )
        with gr.Accordion("除外ワード", open=True):
            apply_tag_exclusions = gr.Checkbox(
                value=True,
                label="除外ワードを適用",
            )
            excluded_tags = gr.Textbox(
                value=load_exclusion_text(),
                label="除外ワード（カンマ区切り、*使用可）",
                lines=3,
                elem_id="excluded-tags-input",
                info=(
                    "画像タグと出力プロンプトの両方から取り除きます。"
                    "例: censored, bar_censor, *_censor, *_background"
                ),
            )
            with gr.Row():
                save_excluded_tags_button = gr.Button("除外ワードを保存")
                reset_excluded_tags_button = gr.Button("既定に戻す")
            excluded_tags_status = gr.Markdown(
                "保存すると次回起動時もこの除外ワードを使います。"
            )
    return SimpleNamespace(
        router_model=router_model,
        compiler_model=compiler_model,
        ollama_url=ollama_url,
        vision_model=vision_model,
        use_vision=use_vision,
        allow_private_image_urls=allow_private_image_urls,
        action_override=action_override,
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
        image_description = gr.Textbox(
            label="画像の説明（VLM）",
            lines=4,
            buttons=["copy"],
            interactive=True,
            elem_id="image-description-editor",
            info=(
                "「ポーズ・位置関係の解析にVLMを使う」を有効にすると自動で入ります。"
                "タグが少ないときの補足や、細かく指定したいときはここを直接書き換えてください。"
                "入力があるときはVLMを再実行せず、その内容を使います。"
            ),
        )
    prompts = []
    for row_start in range(0, MAX_OUTPUT_VARIANTS, 2):
        with gr.Row():
            for index in range(row_start, row_start + 2):
                prompts.append(
                    gr.Textbox(
                        label=f"出力プロンプト {index + 1}",
                        lines=5,
                        buttons=["copy"],
                        interactive=True,
                        elem_id=f"prompt-output-{index + 1}",
                    )
                )
    history_state = gr.State([])
    with gr.Accordion("候補の採用・履歴", open=False):
        with gr.Row():
            candidate_selector = gr.Radio(
                choices=[],
                label="生成候補",
            )
            adopt_button = gr.Button("選択候補を採用")
        history_output = gr.JSON(label="実行履歴（新しい順・最大20件）")
    with gr.Accordion("実行情報", open=False):
        with gr.Row():
            action_plan = gr.JSON(label="実行計画")
            status = gr.Markdown(label="状態")
    return SimpleNamespace(
        inferred_tags=inferred_tags,
        image_description=image_description,
        prompts=prompts,
        history_state=history_state,
        candidate_selector=candidate_selector,
        adopt_button=adopt_button,
        history_output=history_output,
        action_plan=action_plan,
        status=status,
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
