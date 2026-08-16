from __future__ import annotations

import typer

from .image_source import resolve_image_source
from .web_service import DEFAULT_COMPILER_MODEL, DEFAULT_ROUTER_MODEL, WebPromptService


web_app = typer.Typer(help="Launch the local Danbooru Prompt Workbench web UI.")


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
    ):
        try:
            with resolve_image_source(image_path, image_url) as resolved_image_path:
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
                )
            return result.action_plan, result.inferred_tags, result.output, result.status
        except Exception as exc:
            return {}, "", "", f"Error: {exc}"

    with gr.Blocks(title="Danbooru Prompt Workbench") as demo:
        gr.Markdown(
            "# Danbooru Prompt Workbench\n"
            "画像と日本語の指示を渡すと、軽量LLMが操作を分解して既存機能を呼び出します。"
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.File(
                    file_count="single",
                    file_types=["image"],
                    type="filepath",
                    label="画像をドロップ（再ドロップで置換）",
                    height=120,
                )
                image_preview = gr.Image(
                    label="現在の画像",
                    height=300,
                    interactive=False,
                )
                image_url_input = gr.Textbox(
                    label="画像URL（アップロードなしの場合）",
                    placeholder="https://example.com/image.png",
                )
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
                run_button = gr.Button("実行", variant="primary")

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
                variants_input = gr.Slider(1, 4, value=3, step=1, label="候補数")

        with gr.Row():
            action_plan_output = gr.JSON(label="実行計画")
            status_output = gr.Markdown(label="状態")
        inferred_tags_output = gr.Textbox(
            label="画像から推測したタグ",
            lines=4,
            buttons=["copy"],
        )
        prompt_output = gr.Textbox(
            label="出力プロンプト",
            lines=14,
            buttons=["copy"],
        )

        inputs = [
            image_input,
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
        ]
        outputs = [
            action_plan_output,
            inferred_tags_output,
            prompt_output,
            status_output,
        ]
        run_button.click(
            handle_request,
            inputs=inputs,
            outputs=outputs,
            api_name="run_prompt_workbench",
            concurrency_limit=1,
        )
        image_input.change(
            lambda path: path,
            inputs=image_input,
            outputs=image_preview,
            queue=False,
        )
        instruction_input.submit(
            handle_request,
            inputs=inputs,
            outputs=outputs,
            api_name=False,
            concurrency_limit=1,
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
