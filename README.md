# danbooru-prompt-compiler

Local-first Python CLI that converts natural-language scene descriptions into organized Danbooru-style positive tags for anime image generation workflows (e.g., ComfyUI).

## Features

- Python 3.11+ project.
- Typer-based CLI (`danbooru-prompt`).
- Pydantic models for strict compile/LLM request structures.
- Local-first LLM integration using Ollama-compatible API.
- LLM client abstraction designed for future LM Studio/OpenAI-compatible clients.
- Variant generation with `--variants N`.
- Existing prompt editing with `--input-type prompt --edit`.
- Compile modes:
  - `subtle`
  - `remix`
  - `composition`
  - `character_safe`
- Tag normalization:
  - lowercase
  - trim whitespace
  - spaces converted to underscores inside tags
  - deduplicate while preserving first-seen order
  - invalid prose-like tokens removed
- Grouped output with copy-ready prompt lines plus subject, appearance, clothing, pose, scene, style, composition, and other sections.
- Danbooru tag dictionary warnings for unknown tags.
- Automatic Danbooru tag dictionary download when `data/tags.json` is missing.
- YAML presets in `presets/`:
  - `anime_illust`
  - `sfc_jrpg`
  - `comfy_sd15`
  - `comfy_sdxl`
- Fallback parser for accidental bullet-list and newline-separated model output.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for local setup and command execution.

```bash
uv sync --group test
```

## Usage

Start Ollama first and make sure the selected model is available:

```bash
ollama serve
ollama pull llama3.2
```

Create tags from scene text:

```bash
uv run danbooru-prompt "雨の神社で佇む少女"
uv run danbooru-prompt "深夜の高速道路でナビAIの女の子が不安そうにこちらを見る" --variants 3 --mode subtle
uv run danbooru-prompt "空中都市を歩く旅人" --preset sfc_jrpg
```

Edit an existing prompt:

```bash
uv run danbooru-prompt "1girl, solo, shrine, rain, standing" --input-type prompt --edit "夕方にして、赤い傘を追加"
uv run danbooru-prompt "1girl, solo, city, night" --input-type prompt --edit "舞台を雨の神社に変えて、振り返るポーズにする"
```

Default output is grouped and includes two copy-ready forms:

```text
copy: 1girl, solo, shrine, rain, standing, looking_at_viewer, long_hair, night, city

copy_lines:
1girl, solo
long_hair
standing, looking_at_viewer
shrine, rain, night, city

subject: 1girl, solo
appearance: long_hair
pose: standing, looking_at_viewer
scene: shrine, rain, night, city
```

Use flat output for the previous single-line style:

```bash
uv run danbooru-prompt "a girl in rain" --format flat
```

Optional LLM options:

```bash
uv run danbooru-prompt "a girl in rain" --model llama3.2 --ollama-url http://localhost:11434
```

You can also pass a path to an existing text file instead of inline scene text.

## Testing

```bash
uv run pytest
```

## Tag Dictionary

The compiler reads tags from `data/tags.json`. If that file is missing, it automatically downloads popular tags from Danbooru and writes a new dictionary before compiling.

To refresh the dictionary manually:

```bash
uv run python scripts/update_danbooru_tags.py
```

Useful options:

```bash
uv run python scripts/update_danbooru_tags.py --limit 50000
uv run python scripts/update_danbooru_tags.py --output data/tags.json
```

## Tasks

- Expand dictionary filtering options, such as category and minimum post count.
- Add an optional LLM-assisted organizer for richer semantic grouping.
- Add stricter dictionary-only output correction for invented tags.

## Architecture

- `compiler.py`: core reusable compiler logic (independent from CLI layer).
- `cli.py`: Typer command interface.
- `llm.py`: provider abstraction (`LLMClient`) + `OllamaClient` implementation.
- `normalizer.py`: parsing and normalization utilities.
- `formatter.py`: copy-ready and grouped prompt output formatting.
- `tag_dictionary.py`: Danbooru tag dictionary loading, fetching, and writing.
- `models.py`: Pydantic request/response models.

This separation keeps the compiler reusable for future integrations (such as a ComfyUI custom node) without coupling to CLI concerns.
