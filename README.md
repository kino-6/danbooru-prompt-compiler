# danbooru-prompt-compiler

Local-first CLI that turns natural-language image ideas into organized Danbooru-style positive prompts.

It is built for iterative anime image generation workflows: write a rough idea, edit an existing prompt in Japanese, then pull a few "prompt gacha" suggestions and copy the one that looks promising.

## Highlights

- Local-first LLM integration using Ollama-compatible API.
- Two core modes:
  - create a prompt from natural language
  - preserve a base prompt and apply a natural-language edit
- Prompt gacha with `--suggest N`: proposes edit ideas and shows converted prompt previews.
- Danbooru post-based reference tags for stronger short-input prompting.
- Grouped output with copy-ready `===` blocks plus `subject`, `appearance`, `pose`, `scene`, and other sections.
- Clipboard copy with `--copy` on Windows.
- Danbooru tag dictionary warnings for unknown tags.
- Automatic Danbooru tag dictionary download when `data/tags.json` is missing.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for local setup and command execution.

```bash
uv sync --group test
```

Start Ollama first and make sure the selected model is available:

```bash
ollama serve
ollama pull llama3.2
```

## Quick Start

Create a prompt from a natural-language instruction:

```bash
uv run danbooru-prompt "雨の神社で佇む少女"
```

Default output is grouped and copy-ready:

```text
===
1girl, solo
long_hair
standing, looking_at_viewer
shrine, rain, night
===

subject: 1girl, solo
appearance: long_hair
pose: standing, looking_at_viewer
scene: shrine, rain, night
```

Edit an existing prompt without rebuilding it from scratch:

```bash
uv run danbooru-prompt "1girl, solo, shrine, rain, standing" --edit "夕方にして、赤い傘を追加"
```

Use prompt gacha to get three edit ideas and converted prompt previews:

```bash
uv run danbooru-prompt "1girl, shrine, rain" --suggest 3
```

Suggestion previews appear below the main prompt:

```text
=== suggestion 1 ===
edit: 鳥居の奥に淡い霧を足す
===
1girl, solo
long_hair
standing, looking_at_viewer
shrine, rain, torii, mist
===
```

Copy the main `===` block directly to the clipboard:

```bash
uv run danbooru-prompt "雨の神社で佇む少女" --copy
```

## Common Workflows

Generate multiple variants:

```bash
uv run danbooru-prompt "深夜の高速道路でナビAIの女の子が不安そうにこちらを見る" --variants 3 --mode subtle
```

Use a preset:

```bash
uv run danbooru-prompt "空中都市を歩く旅人" --preset sfc_jrpg
```

Use flat output for the previous single-line style:

```bash
uv run danbooru-prompt "a girl in rain" --format flat
```

If you do not have a base prompt yet, `--edit` alone also works as a natural-language instruction:

```bash
uv run danbooru-prompt --edit "雨の神社で佇む少女"
```

Optional LLM options:

```bash
uv run danbooru-prompt "a girl in rain" --model llama3.2 --ollama-url http://localhost:11434
```

Large models can take longer to load on the first request, especially with `--suggest` because it makes several generation calls. Increase the Ollama request timeout when needed:

```bash
uv run danbooru-prompt "1girl" --suggest 3 --model huihui_ai/Qwen3.6-abliterated:27b --ollama-timeout 600
```

You can also pass a path to an existing text file instead of inline scene text.

## Reference Tags

Edits automatically collect temporary reference tags from matching Danbooru posts unless you pass `--no-auto-subset`. Use `--auto-subset` when you also want post examples to guide a new prompt:

```bash
uv run danbooru-prompt "雨の神社" --auto-subset
uv run danbooru-prompt "1girl, solo, shrine, rain, standing" --edit "夕方にして、赤い傘を追加"
```

You can also build and reuse a persistent subset:

```bash
uv run python scripts/build_tag_subset.py shrine rain --posts 100 --min-count 3 --preview 20 --output data/subsets/shrine_rain.json
uv run danbooru-prompt "雨の神社" --tag-subset data/subsets/shrine_rain.json
```

Subset tags are treated as a reference menu, not as output. The CLI estimates useful subset size and output length automatically before asking the LLM.

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

To build a smaller, more practical subset from posts that already have seed tags:

```bash
uv run python scripts/build_tag_subset.py shrine rain --posts 100 --min-count 3 --preview 20 --output data/subsets/shrine_rain.json
```

Subset files keep counts and frequencies for each tag:

```json
{
  "name": "outdoors",
  "count": 68,
  "frequency": 0.68
}
```

Useful options:

```bash
uv run python scripts/update_danbooru_tags.py --limit 50000
uv run python scripts/update_danbooru_tags.py --output data/tags.json
uv run python scripts/build_tag_subset.py shrine rain --posts 200 --min-count 5 --preview 30 --output data/subsets/shrine_rain.json
```

## Tasks

- Expand dictionary filtering options, such as category and minimum post count.
- Add stricter dictionary-only output correction for invented tags.
- Improve prompt gacha scoring so suggestions prefer high-impact, dictionary-friendly tags.
- Add richer candidate ideas for more scene types beyond the current common tag mappings.

## Architecture

- `compiler.py`: core reusable compiler logic (independent from CLI layer).
- `cli.py`: Typer command interface.
- `llm.py`: provider abstraction (`LLMClient`) + `OllamaClient` implementation.
- `normalizer.py`: parsing and normalization utilities.
- `formatter.py`: copy-ready and grouped prompt output formatting.
- `suggestions.py`: prompt gacha idea generation and fallback candidate handling.
- `reference_tags.py`: reference tag loading, auto-subset selection, and max-tag estimation.
- `seed_tags.py`: seed tag inference for Danbooru post lookup.
- `tag_dictionary.py`: Danbooru tag dictionary loading, fetching, and writing.
- `tag_subset.py`: Danbooru post-based subset loading, fetching, and writing.
- `models.py`: Pydantic request/response models.

This separation keeps the compiler reusable for future integrations (such as a ComfyUI custom node) without coupling to CLI concerns.
