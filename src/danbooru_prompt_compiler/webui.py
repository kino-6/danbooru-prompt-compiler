from __future__ import annotations

import typer

from .image_source import load_image_url_preview, resolve_image_source
from .ollama_diagnostics import check_ollama, format_ollama_error
from .web_service import (
    DEFAULT_COMPILER_MODEL,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_VISION_MODEL,
    WebPromptService,
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


def prompt_box_values(candidates: list[str]) -> list[str]:
    values = candidates[:MAX_OUTPUT_VARIANTS]
    return [values[index] if index < len(values) else "" for index in range(MAX_OUTPUT_VARIANTS)]


def accept_dropped_image(image_path: str | None):
    if not image_path:
        return None, None, "", None
    from pathlib import Path

    return image_path, image_path, Path(image_path).name, None


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


def build_app(*, service: WebPromptService | None = None):
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("Web UI dependencies are missing; run 'uv sync --extra web'.") from exc

    prompt_service = service or WebPromptService()

    def handle_request(
        image_path,
        image_url,
        instruction,
        base_prompt,
        router_model,
        compiler_model,
        ollama_url,
        general_threshold,
        character_threshold,
        max_image_tags,
        variants,
        edited_tags,
        action_override,
        use_vision,
        vision_model,
        history,
        allow_private_image_urls,
        progress=gr.Progress(),
    ):
        try:
            def report_progress(stage: str, fraction: float) -> None:
                progress(fraction, desc=PROGRESS_LABELS.get(stage, stage))

            with resolve_image_source(
                image_path,
                image_url,
                allow_private_hosts=allow_private_image_urls,
            ) as resolved_image_path:
                result = prompt_service.run(
                    image_path=resolved_image_path,
                    instruction=instruction,
                    base_prompt=base_prompt,
                    router_model=router_model,
                    compiler_model=compiler_model,
                    ollama_url=ollama_url,
                    general_threshold=general_threshold,
                    character_threshold=character_threshold,
                    max_image_tags=int(max_image_tags),
                    variants=int(variants),
                    edited_tags=edited_tags,
                    action_override=action_override,
                    use_vision=use_vision,
                    vision_model=vision_model,
                    on_progress=report_progress,
                )
            updated_history = prepend_history(
                history,
                action=str(result.action_plan.get("action", "")),
                instruction=instruction or "",
                output=result.output,
            )
            candidate_update = gr.Radio(
                choices=result.candidates,
                value=result.candidates[0] if result.candidates else None,
            )
            return (
                result.action_plan,
                result.inferred_tags,
                *prompt_box_values(result.candidates),
                result.status,
                candidate_update,
                updated_history,
                updated_history,
            )
        except Exception as exc:
            return (
                {},
                "",
                *("" for _ in range(MAX_OUTPUT_VARIANTS)),
                "Error: "
                + format_ollama_error(
                    exc,
                    [router_model, compiler_model, vision_model if use_vision else ""],
                ),
                gr.Radio(choices=[], value=None),
                history or [],
                history or [],
            )

    with gr.Blocks(title="Danbooru Prompt Workbench") as demo:
        gr.Markdown(
            "# Danbooru Prompt Workbench\n"
            "画像と日本語の指示を渡すと、軽量LLMが操作を分解して既存機能を呼び出します。"
        )
        with gr.Row():
            with gr.Column():
                image_drop_input = gr.File(
                    file_count="single",
                    file_types=["image"],
                    type="filepath",
                    label="画像をドロップ（常にここから置換できます）",
                    height=120,
                )
                active_image_input = gr.File(
                    file_count="single",
                    file_types=["image"],
                    type="filepath",
                    visible="hidden",
                )
                selected_image_name = gr.Textbox(
                    label="選択中画像",
                    interactive=False,
                )
                image_preview = gr.Image(
                    label="現在の画像",
                    height=300,
                    interactive=False,
                )
                with gr.Row():
                    image_url_input = gr.Textbox(
                        label="画像URL（アップロードなしの場合）",
                        placeholder="https://example.com/image.png",
                    )
                    image_url_preview_button = gr.Button("URLをプレビュー")
            with gr.Column():
                instruction_input = gr.Textbox(
                    label="どうしたい？",
                    placeholder="例: タグを推測して / 次のコマで振り返らせて / 夜に変更して",
                    lines=5,
                )
                base_prompt_input = gr.Textbox(
                    label="既存プロンプト（任意）",
                    placeholder="画像の代わりに既存タグを編集するときに入力",
                    lines=4,
                )
                with gr.Row():
                    action_override_input = gr.Dropdown(
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
                    variants_input = gr.Radio(
                        choices=[1, 2, 3, 4],
                        value=4,
                        label="出力数",
                    )
                with gr.Row():
                    run_button = gr.Button("実行", variant="primary")
                    cancel_button = gr.Button("停止", variant="stop")

        with gr.Accordion("詳細設定", open=False):
            with gr.Row():
                router_model_input = gr.Textbox(
                    value=DEFAULT_ROUTER_MODEL,
                    label="指示ルーターモデル",
                )
                compiler_model_input = gr.Textbox(
                    value=DEFAULT_COMPILER_MODEL,
                    label="プロンプト生成モデル",
                )
                ollama_url_input = gr.Textbox(
                    value="http://localhost:11434",
                    label="Ollama URL",
                )
                vision_model_input = gr.Textbox(
                    value=DEFAULT_VISION_MODEL,
                    label="VLMモデル",
                )
                use_vision_input = gr.Checkbox(
                    value=False,
                    label="ポーズ・位置関係の解析にVLMを使う",
                )
                allow_private_image_urls_input = gr.Checkbox(
                    value=False,
                    label="プライベート画像URLを許可",
                )
            ollama_diagnostic_button = gr.Button("Ollama接続確認")
            ollama_diagnostic_output = gr.Markdown(label="Ollama診断")
            with gr.Row():
                general_threshold_input = gr.Slider(
                    0.0, 1.0, value=0.35, step=0.01, label="一般タグ閾値"
                )
                character_threshold_input = gr.Slider(
                    0.0, 1.0, value=0.85, step=0.01, label="キャラクター閾値"
                )
                max_image_tags_input = gr.Slider(
                    1, 100, value=50, step=1, label="画像タグ上限"
                )

        with gr.Row():
            action_plan_output = gr.JSON(label="実行計画")
            status_output = gr.Markdown(label="状態")
        inferred_tags_editor = gr.Textbox(
            label="画像タグ（修正して再実行できます）",
            lines=4,
            buttons=["copy"],
            interactive=True,
        )
        prompt_outputs = []
        for row_start in range(0, MAX_OUTPUT_VARIANTS, 2):
            with gr.Row():
                for index in range(row_start, row_start + 2):
                    prompt_outputs.append(
                        gr.Textbox(
                            label=f"出力プロンプト {index + 1}",
                            lines=5,
                            buttons=["copy"],
                            interactive=True,
                            elem_id=f"prompt-output-{index + 1}",
                        )
                    )
        with gr.Row():
            candidate_selector = gr.Radio(
                choices=[],
                label="生成候補",
            )
            adopt_button = gr.Button("選択候補を採用")
        history_state = gr.State([])
        history_output = gr.JSON(label="実行履歴（新しい順・最大20件）")

        inputs = [
            active_image_input,
            image_url_input,
            instruction_input,
            base_prompt_input,
            router_model_input,
            compiler_model_input,
            ollama_url_input,
            general_threshold_input,
            character_threshold_input,
            max_image_tags_input,
            variants_input,
            inferred_tags_editor,
            action_override_input,
            use_vision_input,
            vision_model_input,
            history_state,
            allow_private_image_urls_input,
        ]
        outputs = [
            action_plan_output,
            inferred_tags_editor,
            *prompt_outputs,
            status_output,
            candidate_selector,
            history_state,
            history_output,
        ]
        run_event = run_button.click(
            handle_request,
            inputs=inputs,
            outputs=outputs,
            api_name="run_prompt_workbench",
            concurrency_limit=1,
        )
        image_drop_input.upload(
            accept_dropped_image,
            inputs=image_drop_input,
            outputs=[
                active_image_input,
                image_preview,
                selected_image_name,
                image_drop_input,
            ],
            queue=False,
        )
        image_url_preview_button.click(
            preview_image_url,
            inputs=[image_url_input, allow_private_image_urls_input],
            outputs=image_preview,
            queue=False,
        )
        ollama_diagnostic_button.click(
            diagnose_ollama,
            inputs=[
                ollama_url_input,
                router_model_input,
                compiler_model_input,
                vision_model_input,
                use_vision_input,
            ],
            outputs=ollama_diagnostic_output,
            queue=False,
        )
        adopt_button.click(
            adopt_candidate,
            inputs=candidate_selector,
            outputs=base_prompt_input,
            queue=False,
        )
        submit_event = instruction_input.submit(
            handle_request,
            inputs=inputs,
            outputs=outputs,
            api_name=False,
            concurrency_limit=1,
        )
        cancel_button.click(
            fn=None,
            cancels=[run_event, submit_event],
            queue=False,
        )

    return demo


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
