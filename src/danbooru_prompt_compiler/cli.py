from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import typer

from .compiler import PromptCompiler
from .formatter import OutputFormat, format_clipboard_text, format_variant
from .llm import OllamaClient
from .models import CompileMode, CompileRequest, InputType
from .reference_tags import load_reference_tags

app = typer.Typer(help="Compile scene descriptions into Danbooru-style tags.")


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@app.command()
def main(
    input_value: str | None = typer.Argument(None, help="Base prompt, scene text, or path to a text file."),
    variants: int = typer.Option(1, "--variants", min=1, max=10),
    mode: CompileMode = typer.Option(CompileMode.subtle, "--mode"),
    preset: str | None = typer.Option(None, "--preset"),
    input_type: InputType = typer.Option(InputType.scene, "--input-type", hidden=True),
    edit: str | None = typer.Option(None, "--edit", "--change"),
    auto_subset: bool = typer.Option(False, "--auto-subset", help="Automatically collect Danbooru reference tags."),
    no_auto_subset: bool = typer.Option(False, "--no-auto-subset", help="Disable automatic reference tags for edits."),
    tag_subset: Path | None = typer.Option(None, "--tag-subset"),
    tag_subset_limit: str = typer.Option("auto", "--tag-subset-limit", hidden=True),
    subset_tags: list[str] | None = typer.Option(None, "--subset-tags", hidden=True),
    subset_posts: int = typer.Option(100, "--subset-posts", min=1, hidden=True),
    subset_min_count: int = typer.Option(2, "--subset-min-count", min=1, hidden=True),
    max_tags: str = typer.Option("auto", "--max-tags", hidden=True),
    model: str = typer.Option("llama3.2", "--model"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url"),
    output_format: OutputFormat = typer.Option(OutputFormat.grouped, "--format"),
    copy: bool = typer.Option(False, "--copy", help="Copy the prompt block to the clipboard."),
) -> None:
    _configure_stdio()

    scene_description = _read_input_value(input_value)
    effective_input_type = _infer_input_type(
        input_value=scene_description,
        edit_instruction=edit,
        explicit_input_type=input_type,
    )
    if not scene_description and not edit:
        typer.secho(
            "Error: provide natural-language text, a base prompt, or --edit.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        compiler = PromptCompiler.from_files(OllamaClient(base_url=ollama_url, model=model))
        use_auto_subset = auto_subset or bool(edit and not no_auto_subset and not tag_subset and not subset_tags)
        reference_tags = load_reference_tags(
            tag_subset=tag_subset,
            subset_tags=subset_tags or [],
            auto_subset=use_auto_subset,
            scene_description=scene_description,
            edit_instruction=edit,
            tag_subset_limit=tag_subset_limit,
            subset_posts=subset_posts,
            subset_min_count=subset_min_count,
            max_tags=max_tags,
        )
        if reference_tags.tags:
            typer.secho(
                f"Using {len(reference_tags.tags)} reference tags; max output tags: {reference_tags.max_output_tags}",
                fg=typer.colors.BLUE,
                err=True,
            )
        max_output_tags = reference_tags.max_output_tags
        result = compiler.compile(
            CompileRequest(
                scene_description=scene_description,
                variants=variants,
                mode=mode,
                preset_name=preset,
                input_type=effective_input_type,
                edit_instruction=edit,
                tag_subset=reference_tags.tags,
                max_output_tags=max_output_tags,
            )
        )
    except httpx.HTTPError as exc:
        detail = ""
        if isinstance(exc, httpx.HTTPStatusError):
            detail = f" ({exc.response.text.strip()})"
        service = "Ollama"
        if exc.request and "danbooru" in str(exc.request.url.host):
            service = "Danbooru"
        typer.secho(
            f"Error: failed to request {service}: {exc}{detail}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        typer.secho(
            f"Error: failed to read local input or tag data: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc

    for idx, variant in enumerate(result.variants, start=1):
        if len(result.variants) > 1:
            typer.echo(f"[variant {idx}]")
        typer.echo(format_variant(variant, output_format))
        if copy and idx == 1:
            clipboard_text = format_clipboard_text(variant, output_format)
            if clipboard_text:
                _copy_to_clipboard(clipboard_text)
                typer.secho("Copied prompt block to clipboard.", fg=typer.colors.GREEN, err=True)
            else:
                typer.secho("Warning: nothing to copy.", fg=typer.colors.YELLOW, err=True)

    if result.unknown_tags:
        typer.secho(
            f"Warning: unknown tags (not in tag dictionary): {', '.join(result.unknown_tags)}",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _copy_to_clipboard(text: str) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["clip.exe"],
            input=text,
            text=True,
            encoding="utf-16le",
            check=True,
        )
        return

    raise ValueError("--copy is currently supported on Windows only.")


def _read_input_value(input_value: str | None) -> str:
    if not input_value:
        return ""

    source = Path(input_value)
    if source.exists() and source.is_file():
        return source.read_text(encoding="utf-8").strip()
    return input_value.strip()


def _infer_input_type(
    *,
    input_value: str,
    edit_instruction: str | None,
    explicit_input_type: InputType,
) -> InputType:
    if edit_instruction and input_value:
        return InputType.prompt
    if explicit_input_type != InputType.scene:
        return explicit_input_type
    return InputType.scene


if __name__ == "__main__":
    app()
