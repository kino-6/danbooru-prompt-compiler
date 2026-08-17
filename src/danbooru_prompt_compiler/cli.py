from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import typer

from .compiler import PromptCompiler
from .formatter import OutputFormat, format_clipboard_text, format_suggestion, format_variant
from .image_tagger import (
    DEFAULT_CHARACTER_THRESHOLD,
    DEFAULT_GENERAL_THRESHOLD,
    DEFAULT_TAGGER_MODEL,
    ImageTagger,
)
from .llm import OllamaClient
from .models import CompileMode, CompileRequest, InputType
from .reference_tags import load_reference_tags
from .suggestions import suggest_edit_instructions
from .tag_filter import resolve_exclusion_rules, split_excluded

app = typer.Typer(help="Compile scene descriptions into Danbooru-style tags.")


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@app.command()
def main(
    input_value: str | None = typer.Argument(None, help="Base prompt, scene text, or path to a text file."),
    image: Path | None = typer.Option(
        None,
        "--image",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Infer Danbooru tags directly from an image.",
    ),
    tagger_model: str = typer.Option(DEFAULT_TAGGER_MODEL, "--tagger-model"),
    general_threshold: float = typer.Option(
        DEFAULT_GENERAL_THRESHOLD,
        "--general-threshold",
        min=0.0,
        max=1.0,
    ),
    character_threshold: float = typer.Option(
        DEFAULT_CHARACTER_THRESHOLD,
        "--character-threshold",
        min=0.0,
        max=1.0,
    ),
    image_max_tags: int = typer.Option(50, "--image-max-tags", min=1, max=500),
    show_scores: bool = typer.Option(False, "--show-scores", help="Show confidence scores for image tags."),
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
    suggest: int = typer.Option(0, "--suggest", min=0, max=10, help="Suggest edit ideas and prompt previews."),
    ollama_timeout: float = typer.Option(
        300.0,
        "--ollama-timeout",
        min=1.0,
        help="Seconds to wait for each Ollama generation request.",
    ),
    excluded_tags: str | None = typer.Option(
        None,
        "--excluded-tags",
        help="Comma-separated exclusion words (wildcards allowed) replacing the saved list.",
    ),
    no_tag_exclusions: bool = typer.Option(
        False,
        "--no-tag-exclusions",
        help="Keep every tag, including the saved exclusion words.",
    ),
) -> None:
    _configure_stdio()

    exclusion_rules = resolve_exclusion_rules(
        excluded_tags,
        disabled=no_tag_exclusions,
    )

    if image is not None:
        if input_value or edit:
            typer.secho(
                "Error: --image cannot be combined with a text input or --edit.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        try:
            prediction = ImageTagger(tagger_model).predict(
                image,
                general_threshold=general_threshold,
                character_threshold=character_threshold,
                max_tags=image_max_tags,
            )
        except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
            typer.secho(
                f"Error: failed to infer image tags: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from exc

        kept_tags, dropped_tags = split_excluded(
            prediction.tags,
            exclusion_rules,
            key=lambda tag: tag.name,
        )
        kept_names = [tag.name for tag in kept_tags]

        typer.echo(format_variant(kept_names, output_format))
        if not prediction.tags:
            typer.secho(
                "Warning: no tags exceeded the selected confidence thresholds.",
                fg=typer.colors.YELLOW,
                err=True,
            )
        _report_excluded_tags([tag.name for tag in dropped_tags])
        if show_scores:
            typer.echo("")
            typer.echo("confidence:")
            for tag in kept_tags:
                typer.echo(f"{tag.score:.3f}\t{tag.name}")
            if prediction.rating is not None:
                typer.echo(f"rating\t{prediction.rating.name} ({prediction.rating.score:.3f})")
        if copy:
            clipboard_text = format_clipboard_text(kept_names, output_format)
            if clipboard_text:
                _copy_to_clipboard(clipboard_text)
                typer.secho("Copied inferred tags to clipboard.", fg=typer.colors.GREEN, err=True)
        return

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

    llm_client = OllamaClient(base_url=ollama_url, model=model, timeout=ollama_timeout)
    try:
        compiler = PromptCompiler.from_files(llm_client)
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
                excluded_tags=exclusion_rules,
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

    unknown_tags = list(result.unknown_tags)
    excluded_output_tags = list(result.excluded_tags)

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

    if suggest and result.variants:
        suggestions = suggest_edit_instructions(
            llm_client,
            base_prompt=scene_description,
            edit_instruction=edit,
            current_tags=result.variants[0],
            reference_tags=reference_tags.tags,
            count=suggest,
        )
        if suggestions:
            typer.echo("")
            base_prompt = ", ".join(result.variants[0])
            for suggestion_index, instruction in enumerate(suggestions, start=1):
                suggestion_result = compiler.compile(
                    CompileRequest(
                        scene_description=base_prompt,
                        variants=1,
                        mode=mode,
                        preset_name=preset,
                        input_type=InputType.prompt,
                        edit_instruction=instruction,
                        tag_subset=reference_tags.tags,
                        max_output_tags=max_output_tags,
                        excluded_tags=exclusion_rules,
                    )
                )
                _extend_unique(unknown_tags, suggestion_result.unknown_tags)
                _extend_unique(excluded_output_tags, suggestion_result.excluded_tags)
                if suggestion_result.variants:
                    typer.echo(format_suggestion(suggestion_index, instruction, suggestion_result.variants[0], output_format))
                    if suggestion_index < len(suggestions):
                        typer.echo("")

    _report_excluded_tags(excluded_output_tags)
    if unknown_tags:
        typer.secho(
            f"Warning: unknown tags (not in tag dictionary): {', '.join(unknown_tags)}",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _report_excluded_tags(excluded_tags: list[str]) -> None:
    if not excluded_tags:
        return

    typer.secho(
        f"Excluded tags: {', '.join(excluded_tags)}",
        fg=typer.colors.BLUE,
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


def _extend_unique(values: list[str], new_values: list[str]) -> None:
    for value in new_values:
        if value not in values:
            values.append(value)


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
