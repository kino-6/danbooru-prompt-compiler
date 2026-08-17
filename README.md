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

For natural-language compilation, start Ollama and make sure the selected model is available. Image tagging does not use Ollama.

```bash
ollama serve
ollama pull llama3.2
```

## Quick Start

Infer Danbooru tags directly from an image (the ONNX model is downloaded and cached on first use):

```bash
uv run danbooru-prompt --image path/to/image.png
```

Tune confidence thresholds or show the scores when needed:

```bash
uv run danbooru-prompt --image path/to/image.png --general-threshold 0.4 --character-threshold 0.85 --show-scores
```

The default image tagger is [`SmilingWolf/wd-vit-tagger-v3`](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3). It runs locally through ONNX Runtime; the roughly 379 MB model is downloaded to the Hugging Face cache on first use. General and character tags use separate default thresholds of `0.35` and `0.85`, and output keeps canonical Danbooru underscores.

## Web UI Prototype

Install the optional Web UI dependency and pull the lightweight instruction router and prompt model:

```bash
uv sync --extra web --group test
ollama pull qwen3:1.7b
```

Launch the local workbench:

```bash
uv run danbooru-prompt-web
```

Open `http://127.0.0.1:7860` if the browser does not open automatically. Drop an image into the persistent upload area (dropping another image replaces it), paste a copied image with `Ctrl+V` anywhere on the page, or enter a direct HTTP/HTTPS image URL. The upload takes precedence when both are present. Entering a Japanese request such as `タグを推測して`, `夜に変更して`, or `次のコマで少女を振り返らせて` is enough; the router emits a constrained action JSON and calls the existing Python APIs. URL images are limited to 20 MB, verified as image data, stored only in a temporary file, and deleted after each request. The prototype uses `qwen3:1.7b` for both routing and prompt generation with deterministic settings. If the router model is unavailable or returns invalid JSON, deterministic keyword rules select a safe fallback action.

The `next_panel` action in this prototype is tag-assisted: WD Tagger summarizes the current image, then the text model proposes the next prompt while preserving requested elements. It does not yet inspect spatial relationships with a vision-language model.

`次のコマも生成する（出力2〜4）` is on by default, so every run leaves the current result in prompt box 1 and fills boxes 2-4 with proposals for the moment just after it. The follow-up continues the tags and description the first run already resolved, and a failure in it leaves the box 1 result intact with the reason in the status line. Turn it off to get plain `出力数` variants across all four boxes.

An image with no instruction is otherwise routed to plain tag extraction, so the main controls also carry a dedicated `次のコマ` button. It runs the next-panel action directly on whatever image is loaded, with or without an instruction, and fills all four boxes with panels. The character, appearance, and clothing tags are preserved into every proposal.

The workbench keeps the WD ONNX runtime and recent image-tag results cached, so changing only the instruction avoids repeating model initialization and image inference. Inferred tags are editable, and the action selector can override automatic routing. Generated variants can be selected and adopted as the next base prompt; the most recent 20 runs are kept in per-session history.

`VLMで画像を説明する` under the image workspace is on by default, so pull a vision model with `ollama pull qwen3-vl:8b` or change the model in the advanced settings. It also drives the pose, gaze, held-object, and spatial analysis used by image edits and next-panel requests. A vision model that is missing or failing never blocks prompt generation: the description is skipped and the reason appears in the status line under the prompt boxes. The advanced settings provide an Ollama connection check that reports missing models with exact `ollama pull` commands.

With the VLM enabled, every run on an image fills the `画像の説明（VLM）` box under the image with a plain-Japanese description of what is visible. It helps when the tagger returns fewer tags than expected, and it is editable: type or correct the description and the next run uses your text verbatim instead of calling the VLM again, which is the way to specify details the tag list cannot express. The description is written without reference to the instruction, so it is cached per image and model and reused when only the instruction changes. Image edits and next-panel requests pass it to the prompt model as context; a new prompt from text does not, because it is built from the instruction alone.

Direct image URLs reject private, loopback, and link-local destinations by default, including redirect targets. Enable `プライベート画像URLを許可` only when intentionally loading an image from a trusted LAN service.

Tags that reliably lower image quality are dropped from both the inferred image tags and the generated prompts. `*` matches any characters, and the defaults cover three groups:

- flattened backgrounds: `simple_background`, `halftone`, `*_background`
- every censorship spelling: `*censor*` (`bar_censor`, `censored_nipples`, `mosaic_censoring`, …)
- rendered text and overlays: `*_text`, `text_focus`, `subtitled`, `page_number`, `signature`, `character_signature`, `artist_name`, `character_name`, `copyright_name`, `dated`, `web_address`, `*watermark*`, `*_username`, `logo`, `*_logo`

`*_text` deliberately does not match texture tags such as `paper_texture`, and `speech_bubble` is not excluded because panel work often wants it.

Edit the list under `詳細設定 → 除外ワード`, then press `除外ワードを保存` to persist it to `data/excluded_tags.json` and reuse it on the next launch. A saved file takes precedence over the defaults, so press `既定に戻す` after upgrading to pick up newly added default rules. Removed tags are reported in `実行情報` as `Filtered image tags` and `Filtered prompt tags`.

Literal exclusion words are also given to the prompt model as tags it must never output, and exclusions are applied before the output is truncated, so a filtered variant still returns the requested number of tags.

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

The CLI and the Web UI share the same exclusion words, so image tagging and prompt output drop them on both paths. Override the saved list for one run, or keep every tag:

```bash
uv run danbooru-prompt --image path/to/image.png --excluded-tags "censored, *_censor, monochrome"
uv run danbooru-prompt --image path/to/image.png --no-tag-exclusions
```

Removed tags are reported on stderr as `Excluded tags: ...`.

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
- `image_tagger.py`: local ONNX image-to-Danbooru-tag inference.
- `web_router.py`: constrained natural-language instruction routing with a rule fallback.
- `web_service.py`: orchestration shared by the Web UI and tests.
- `webui.py`: local Gradio workbench.
- `llm.py`: provider abstraction (`LLMClient`) + `OllamaClient` implementation.
- `normalizer.py`: parsing and normalization utilities.
- `formatter.py`: copy-ready and grouped prompt output formatting.
- `suggestions.py`: prompt gacha idea generation and fallback candidate handling.
- `reference_tags.py`: reference tag loading, auto-subset selection, and max-tag estimation.
- `seed_tags.py`: seed tag inference for Danbooru post lookup.
- `tag_dictionary.py`: Danbooru tag dictionary loading, fetching, and writing.
- `tag_filter.py`: exclusion-word rules, matching, and persistence.
- `tag_subset.py`: Danbooru post-based subset loading, fetching, and writing.
- `models.py`: Pydantic request/response models.

This separation keeps the compiler reusable for future integrations (such as a ComfyUI custom node) without coupling to CLI concerns.
