from __future__ import annotations

import sys
from pathlib import Path

import httpx
import typer

from .compiler import PromptCompiler
from .formatter import OutputFormat, format_variant
from .llm import OllamaClient
from .models import CompileMode, CompileRequest, InputType

app = typer.Typer(help="Compile scene descriptions into Danbooru-style tags.")


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@app.command()
def main(
    input_value: str = typer.Argument(..., help="Scene text or path to a text file."),
    variants: int = typer.Option(1, "--variants", min=1, max=10),
    mode: CompileMode = typer.Option(CompileMode.subtle, "--mode"),
    preset: str | None = typer.Option(None, "--preset"),
    input_type: InputType = typer.Option(InputType.scene, "--input-type"),
    edit: str | None = typer.Option(None, "--edit", "--change"),
    model: str = typer.Option("llama3.2", "--model"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url"),
    output_format: OutputFormat = typer.Option(OutputFormat.grouped, "--format"),
) -> None:
    _configure_stdio()

    source = Path(input_value)
    if source.exists() and source.is_file():
        scene_description = source.read_text(encoding="utf-8").strip()
    else:
        scene_description = input_value.strip()

    try:
        compiler = PromptCompiler.from_files(OllamaClient(base_url=ollama_url, model=model))
        result = compiler.compile(
            CompileRequest(
                scene_description=scene_description,
                variants=variants,
                mode=mode,
                preset_name=preset,
                input_type=input_type,
                edit_instruction=edit,
            )
        )
    except httpx.HTTPError as exc:
        detail = ""
        if isinstance(exc, httpx.HTTPStatusError):
            detail = f" ({exc.response.text.strip()})"
        typer.secho(
            f"Error: failed to request Ollama at {ollama_url} with model {model}: {exc}{detail}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc

    for idx, variant in enumerate(result.variants, start=1):
        if len(result.variants) > 1:
            typer.echo(f"[variant {idx}]")
        typer.echo(format_variant(variant, output_format))

    if result.unknown_tags:
        typer.secho(
            f"Warning: unknown tags (not in tag dictionary): {', '.join(result.unknown_tags)}",
            fg=typer.colors.YELLOW,
            err=True,
        )


if __name__ == "__main__":
    app()
