# danbooru-prompt-compiler

Local-first Python CLI that converts natural-language scene descriptions into comma-separated Danbooru-style positive tags for anime image generation workflows (e.g., ComfyUI).

## Features

- Python 3.11+ project.
- Typer-based CLI (`danbooru-prompt`).
- Pydantic models for strict compile/LLM request structures.
- Local-first LLM integration using Ollama-compatible API.
- LLM client abstraction designed for future LM Studio/OpenAI-compatible clients.
- Variant generation with `--variants N`.
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
  - empty tags removed
- Curated tag dictionary and unknown-tag warnings (non-fatal).
- YAML presets in `presets/`:
  - `anime_illust`
  - `sfc_jrpg`
  - `comfy_sd15`
  - `comfy_sdxl`
- Fallback parser for accidental bullet-list and newline-separated model output.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
danbooru-prompt "雨の神社で佇む少女"
danbooru-prompt "深夜の高速道路でナビAIの女の子が不安そうにこちらを見る" --variants 3 --mode subtle
danbooru-prompt input.txt --preset sfc_jrpg
```

Optional LLM options:

```bash
danbooru-prompt "a girl in rain" --model llama3 --ollama-url http://localhost:11434
```

## Testing

```bash
pip install -e .[test]
pytest
```

## Architecture

- `compiler.py`: core reusable compiler logic (independent from CLI layer).
- `cli.py`: Typer command interface.
- `llm.py`: provider abstraction (`LLMClient`) + `OllamaClient` implementation.
- `normalizer.py`: parsing and normalization utilities.
- `models.py`: Pydantic request/response models.

This separation keeps the compiler reusable for future integrations (such as a ComfyUI custom node) without coupling to CLI concerns.
