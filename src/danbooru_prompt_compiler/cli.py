from __future__ import annotations

from pathlib import Path

import typer

from .compiler import PromptCompiler
from .llm import OllamaClient
from .models import CompileMode, CompileRequest

app = typer.Typer(help="Compile scene descriptions into Danbooru-style tags.")


@app.command()
def main(
    input_value: str = typer.Argument(..., help="Scene text or path to a text file."),
    variants: int = typer.Option(1, "--variants", min=1, max=10),
    mode: CompileMode = typer.Option(CompileMode.subtle, "--mode"),
    preset: str | None = typer.Option(None, "--preset"),
    model: str = typer.Option("llama3", "--model"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url"),
) -> None:
    source = Path(input_value)
    if source.exists() and source.is_file():
        scene_description = source.read_text(encoding="utf-8").strip()
    else:
        scene_description = input_value.strip()

    compiler = PromptCompiler.from_files(OllamaClient(base_url=ollama_url, model=model))
    result = compiler.compile(
        CompileRequest(
            scene_description=scene_description,
            variants=variants,
            mode=mode,
            preset_name=preset,
        )
    )

    for idx, variant in enumerate(result.variants, start=1):
        if len(result.variants) > 1:
            typer.echo(f"[variant {idx}]")
        typer.echo(", ".join(variant))

    if result.unknown_tags:
        typer.secho(
            f"Warning: unknown tags (not in curated dictionary): {', '.join(result.unknown_tags)}",
            fg=typer.colors.YELLOW,
            err=True,
        )


if __name__ == "__main__":
    app()
