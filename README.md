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
- Local image tagging with WD Tagger, plus a vision model that describes the image and reviews the inferred tags against it.
- A local Web workbench for image-led work: prose prompts, next panels, and an uncensored vision model option.

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

See [Local Model Environment](#local-model-environment) for the full model roster, the
uncensored vision model, and the measured VRAM and throughput figures.

## Local Model Environment

Everything runs against a local Ollama instance. The table below is the model roster the
Web UI expects; the roles map to the model selectors in the advanced settings.

| Role | Model | Size | Pull |
| --- | --- | --- | --- |
| Router + prompt generation | `qwen3:1.7b` | 1.4 GB | `ollama pull qwen3:1.7b` |
| Vision description (default) | `qwen3-vl:8b` | 6.1 GB | `ollama pull qwen3-vl:8b` |
| Vision description (uncensored) | `unseen-gemma4:26b` | 17 GB | see below |
| Natural-language prompts | any larger local model | - | `ollama pull qwen3:8b` |
| Image tagging | `SmilingWolf/wd-vit-tagger-v3` | 379 MB | automatic, ONNX, not Ollama |

`unseen-gemma4:26b` is an uncensored Gemma 4 26B MoE vision model aimed at anime character
analysis and captioning. It is the alternative to `qwen3-vl:8b` when the default vision model
sanitizes or refuses a description. Ollama 0.33.0 or newer is required - earlier builds do not
know the `gemma4` architecture.

```bash
ollama pull hf.co/Jommarn/UNSEEN_Gemma_4_26B_NSFW-GGUF:Q4_K_M
ollama cp hf.co/Jommarn/UNSEEN_Gemma_4_26B_NSFW-GGUF:Q4_K_M unseen-gemma4:26b
```

The Hugging Face bridge pulls the vision projector (`mmproj`, 599 MB) alongside the 16 GB
weights, so no Modelfile is needed. The `ollama cp` step only writes a second manifest over the
same blobs and costs no extra disk. Verify the projector actually loaded by describing an image
with readable text in it; a text-only import answers without ever seeing the picture.

### Measured reference

Recorded on Windows 11, RTX 5080 (16 GB VRAM), 64 GB RAM, Ollama 0.33.2:

| | `unseen-gemma4:26b` |
| --- | --- |
| Throughput, warm | 88-91 tok/s |
| First load from disk | ~97 s |
| Offload split | 76% GPU / 24% CPU |
| VRAM resident | 15.7 / 16.3 GB |
| Context | 4096 (Ollama default) |

Being a mixture-of-experts model, it stays fast even with a quarter of the layers on CPU. Two
constraints follow from the VRAM figure on a 16 GB card:

- Raising `num_ctx` above the 4096 default pushes GPU layers back to the CPU and costs far more
  than the extra context is worth. The weights support 262144 tokens; this card does not.
- It cannot be resident next to `qwen3-vl:8b`. Ollama evicts and reloads on each switch, which
  is the ~97 s figure above, so pick one vision model per session rather than alternating.

The vision factory already sends `think=false`, which suppresses the `<|channel>thought` reasoning
trace this model otherwise emits. Anything calling it outside that factory needs the same flag.

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

Install the optional Web UI dependency, then pull the instruction router and prompt model plus a
vision model for the description step:

```bash
uv sync --extra web --group test
ollama pull qwen3:1.7b
ollama pull qwen3-vl:8b
```

Launch the local workbench:

```bash
uv run danbooru-prompt-web
```

`やりたいこと` at the top of the page is the first thing to set, and it decides what the rest of the page shows: every input the chosen action cannot reach is hidden, so a setting that would be silently ignored is never offered. `おまかせ` keeps the original behaviour - the router reads the instruction and picks - and shows only what the router can actually reach, which is why the natural-language template and its settings appear only under `自然文プロンプト`. The run button changes with it: one action per task, plus `停止`.

The page is laid out to fit one screen on every task, which is what decides several things about it: the prompt boxes arrive with the run that fills them rather than waiting empty, the typed image URL lives in `詳細設定` because dropping and pasting are the paths people actually use, and `詳細設定` and `実行の詳細` share a line since both are opened rarely.

Measured content height runs 466-719px across the seven tasks, so every task fits a 1440x900 laptop or anything larger without scrolling. On a 1366x768 laptop the tallest tasks run about 120px over: everything needed to start a run is still on screen, and what falls below the fold is the collapsed panels.

Open `http://127.0.0.1:7860` if the browser does not open automatically. Drop an image into the persistent upload area (dropping another image replaces it), paste a copied image with `Ctrl+V` anywhere on the page, or enter a direct HTTP/HTTPS image URL. The upload takes precedence when both are present. Entering a Japanese request such as `タグを推測して`, `夜に変更して`, or `次のコマで少女を振り返らせて` is enough; the router emits a constrained action JSON and calls the existing Python APIs. URL images are limited to 20 MB, verified as image data, stored only in a temporary file, and deleted after each request. The prototype uses `qwen3:1.7b` for both routing and prompt generation with deterministic settings. If the router model is unavailable or returns invalid JSON, deterministic keyword rules select a safe fallback action.

Those two pulls are the light setup and are enough to use every feature. Swapping in the uncensored vision model is covered in [Running the vision steps](#running-the-vision-steps).

`next_panel` asks the vision model, because a next panel is a question about time and a tag list carries no time. Given the picture it answers in three lines - one sentence saying what the character does in the next instant, then the current tags that stop being true and the Danbooru tags that start being true. The sentence comes first deliberately: asked for tags alone the model returns the timid answer, and asked to say what happens first it commits to an action and the tags follow.

Everything it proposes is bounded the same way the tag review is: additions must exist in `data/tags.json`, removals must name a tag already on the list, and the aspects the change slider holds fixed cannot be removed at all. The sentence it wrote appears in the status line, since it says what the panel is meant to be and the tag list only implies it.

The status line names what each panel actually moved - `-holding_bow_(weapon), looking_at_viewer / +drawing_bow, aiming` - beside the sentence describing it. One tag changing inside a list of twenty-five reads as no change at all otherwise, which is exactly how it read before the diff was printed. Asking for several panels at once also raises the temperature to a floor of 0.5 whatever the time slider says: three boxes holding the same panel are worth one box.

A picture is not required. With `既存プロンプト` filled in and no image, the same question goes to the prose model instead, and the dictionary bound does the same work - a small model asked this invents `bow_drawn` and `action_draw`, and they are refused rather than passed on.

A proposal whose pose and framing match the panel it came from is not a next panel. The status line says how many came back unmoved rather than handing them over quietly. If the vision model is missing or fails, the run falls back to the old tag-only path with the reason in the status line.

The model sometimes answers in the right shape without the labels - the sentence, the removals and the additions as three bare lines - and the content is exactly right when it does. Those are relabelled rather than discarded: measured, discarding them threw away the best proposals of the set, including the only one that reached `aiming, drawing_bow` at that setting.

On the sample portrait - an archer with an arrow nocked - every combination of the two sliders now proposes a moving panel. The time axis is directionally right but not dramatic: `0.1` and `0.5` both draw the bow, one further than the other. The weak corner is a long time with a narrow latitude, where the character may not go anywhere and the model settles for turning her head.

`次のコマも生成する（出力2〜4）` is on by default, so every run leaves the current result in prompt box 1 and fills boxes 2-4 with proposals for the moment just after it. The follow-up continues the tags and description the first run already resolved, and a failure in it leaves the box 1 result intact with the reason in the status line. Turn it off to get plain `出力数` variants across all four boxes.

Two sliders describe the panel to ask for, because one was doing the work of both and they pull in different directions. Bundled, asking for a bigger change made the panel move *less*: the preserved set shrank, so tags were dropped rather than actions advanced.

`経過する時間` (0.0-1.0, default 0.3) says how far the action advances, and how speculative the answer may be - the further ahead you look, the less the picture determines:

| 経過する時間 | The next panel is | Temperature |
| --- | --- | --- |
| `0.0`-`0.3` | a fraction of a second later; the action advances a fraction | `0.0` |
| `0.4`-`0.6` | a second or two later; the action reaches its next stage | `0.5` |
| `0.7`-`1.0` | several seconds later; that action is over and the next has begun | `0.6` |

`変わってよい範囲` (0.0-1.0, default 0.5) says nothing about time. It decides what the answer may touch, and its aspects can never be removed from the proposal:

| 変わってよい範囲 | Preserved | May differ in |
| --- | --- | --- |
| `0.0`-`0.3` | character, appearance, clothing | pose, gaze and framing only |
| `0.4`-`0.6` | character, appearance | pose, framing and clothing |
| `0.7`-`1.0` | character | anything except who the character is |

An image with no instruction is otherwise routed to plain tag extraction, so the main controls also carry a dedicated `次のコマ` button. It runs the next-panel action directly on whatever image is loaded, with or without an instruction, and fills all four boxes with panels at the selected change amount.

Under the prompt boxes, the same groups the output is already organized into appear as separate copyable boxes - 人物, 外見, 服装, ポーズ, 情景, 画風, 構図, その他 - so a prompt can be reused piecewise: the character without the scene, the clothing without the pose. They are read back from prompt box 1 rather than kept from the run, so editing that box or adopting a candidate re-splits what you can see, and a group with nothing in it does not appear.

The workbench remembers how it was last left. Which models, where Ollama is, the thresholds, the output count, the sliders and the selected task are saved to `data/webui_settings.json` on every run and read back at launch. The work itself is never saved: an image, an instruction or a half-edited prompt belongs to the session that made it, and finding yesterday's instruction waiting in the box is worse than finding it empty. The file is gitignored, and a missing or hand-mangled one costs a control its memory rather than the page.

The workbench keeps the WD ONNX runtime and recent image-tag results cached, so changing only the instruction avoids repeating model initialization and image inference. Inferred tags are editable. Generated variants can be selected and adopted as the next base prompt; the most recent 20 runs are kept in per-session history.

`VLMで画像を説明する` under the image workspace is on by default, so pull a vision model with `ollama pull qwen3-vl:8b` or change the model in the advanced settings. It also drives the pose, gaze, held-object, and spatial analysis used by image edits and next-panel requests. A vision model that is missing or failing never blocks prompt generation: the description is skipped and the reason appears in the status line under the prompt boxes. When a loaded model stops answering, press `VLMを復旧` beside the switch: it drops the cached description, unloads the model with `keep_alive: 0`, and loads a fresh instance, reporting the exact `ollama pull` or `ollama serve` command when that is the real problem. The advanced settings provide an Ollama connection check that reports missing models with exact `ollama pull` commands.

`やりたいこと → タグをVLMで確認` shows the image and the inferred tags to the vision model and asks
which of them are wrong and which known tags are missing. The tagger scores each tag on its own,
so it reports plausible neighbours it could not rule out and drops whatever fell under the
threshold; a model that can see the picture is the thing that judges the list. It never writes
the list: every addition has to exist in `data/tags.json` and every removal has to name a tag
already on it, so a proposal like `hand_drawn` is reported as rejected rather than adopted. Tags
you typed by hand are never removed, and a vision model that fails leaves the list untouched with
the reason in the status line.

Every model field in the advanced settings is a dropdown over the models in [Local Model Environment](#local-model-environment) that still accepts any other pulled model typed straight into it - a model name is something to pick, not to spell from memory. `指示ルーターモデル`, `プロンプト生成モデル` and `自然文プロンプト用モデル` share one list, since the text steps all take the same kind of model; the prose one adds `プロンプト生成モデルと同じ`, which is the empty value that reuses whatever tag generation is set to.

`VLMモデル` is a dropdown over the same section, labelled with their size and whether they are uncensored. The connection check adds up what the current selection weighs and says so when it will not all stay resident - it names the heaviest model, since that is the one deciding what gets evicted, and a swap costs a full reload on every run.

### Running the vision steps

A 26B vision model fills a 16 GB card on its own, so it cannot sit beside the 8B one or beside a
separate prose model. That makes two working sets rather than a menu of independent settings:

| | Light | Uncensored |
| --- | --- | --- |
| `VLMモデル` | `qwen3-vl:8b` | `unseen-gemma4:26b` |
| `自然文プロンプト用モデル` | empty (reuses `qwen3:1.7b`) | `unseen-gemma4:26b` |
| `自然文プロンプトに画像を渡す` | off | on |
| Good for | iterating, prompt gacha, quick panels | finishing a prompt, material the light model will not describe |

In the uncensored set, point both the vision step and the prose step at the same model. Naming two
different large models means Ollama unloads one to load the other on every run, and the connection
check will say so.

Setting the uncensored one up once:

```bash
ollama pull hf.co/Jommarn/UNSEEN_Gemma_4_26B_NSFW-GGUF:Q4_K_M
ollama cp hf.co/Jommarn/UNSEEN_Gemma_4_26B_NSFW-GGUF:Q4_K_M unseen-gemma4:26b
```

Then, with an image loaded:

- **Fill the gaps in the tag list.** Run once with no instruction to get tags, then pick
  `やりたいこと → タグをVLMで確認`. On the sample portrait this added `sketch`, `lineart`, and
  `fantasy`, all of which the tagger had dropped, and refused `hand_drawn` and `archer` as not
  being in the dictionary. Correct the tag box by hand first if you want a tag protected - the
  review never removes what you typed.
- **Write a prose prompt from the picture.** Turn on `自然文プロンプトに画像を渡す`, choose a
  `自然文プロンプトのテンプレート`, and press `自然文プロンプト`. The tags still travel with the
  request, so the paragraph gains what only the image shows - on the sample it described a collar,
  a dark vest, a waist pouch, and a skirt that no tag carried - without losing the discrete
  features the tagger scored.
- **Describe material the light model refuses.** Leave `VLMで画像を説明する` on and read the
  `画像の説明（VLM）` box. That description is also what image edits and next-panel requests read
  as context, so a description that came back sanitized was quietly costing those too.

One caveat with the image attached: the model will state attributes the picture does not fix. On a
monochrome line-art reference it wrote `long blonde hair` and `light eye colour`. Useful on a
character sheet, wrong as a faithful caption - check the colour words before reusing the output.

With the VLM enabled, every run on an image fills the `画像の説明（VLM）` box under the image with a plain-Japanese description of what is visible. It helps when the tagger returns fewer tags than expected, and it is editable: type or correct the description and the next run uses your text verbatim instead of calling the VLM again, which is the way to specify details the tag list cannot express. The description is written without reference to the instruction, so it is cached per image and model and reused when only the instruction changes. Image edits and next-panel requests pass it to the prompt model as context; a new prompt from text does not, because it is built from the instruction alone.

### Natural-language prompts

Newer image models take prose rather than Danbooru tags. The `自然文プロンプト` button generates it from the same material the tag pipeline already has - the inferred image tags, the VLM description, and your instruction - laid out as an atomic schema: one task line, the template's named sections, a delivery line, and an explicit avoid line.

Pick the skeleton with `自然文プロンプトのテンプレート`:

| Template | Sections |
| --- | --- |
| キャラクター設定シート | Subject, Clothing, Pose, Expression, Lighting, Layout, Details |
| 絵コンテ／次のコマ | Subject, Action, Expression, Lighting, Layout, Details |
| シーンイラスト | Subject, Setting, Lighting, Materials, Layout, Details, Mood |
| ポスター／キービジュアル | Subject, Setting, Lighting, Palette, Layout, Details |

Everything here runs on the same local Ollama instance as the tag pipeline; nothing is sent anywhere. Writing English prose is harder than assembling a tag list, so `自然文プロンプト用モデル` in the advanced settings points this one step at a larger local model - `qwen3:8b`, `gemma3:12b`, or whatever is pulled - while routing and tag generation stay on the small fast model. Leave it empty to reuse the prompt-generation model. The Ollama connection check includes it, so a model that still needs an `ollama pull` is reported before a run fails.

Templates are plain YAML in `templates/`; drop a file in with `label`, `task`, `sections`, `delivery`, and an optional `order`, and it appears in the dropdown on the next launch. The model only fills the sections in - the shape is rebuilt locally afterwards, so a dropped or reordered section never corrupts the output.

`自然文プロンプトに画像を渡す` in the advanced settings hands the reference image to the prose model
itself, which only makes sense when `自然文プロンプト用モデル` names a vision model such as
`unseen-gemma4:26b`. It is off by default because the default prose model cannot read an image.

The tags are still sent when it is on, and the request names them as facts observed in the image.
That is deliberate: on the sample portrait the model wrote a visibly richer `Clothing` section from
the picture - collar, vest, waist pouch, skirt, boots, none of which the tag list carried - but
describing the image alone dropped `pointy_ears` and `elf`, which the tagger had scored at 0.92.
Tags anchor the discrete features, the image supplies the texture. Danbooru qualifiers are stripped
on the way in, so `bow_(weapon)` reaches the model as `bow` rather than leaking a parenthesis into
the paragraph.

See [Running the vision steps](#running-the-vision-steps) for which model to pair this with and
what it invents when it is on.

The avoid line is the literal exclusion words plus the tags this image actually lost to the filter, written as words rather than tags (`bar_censor` becomes `bar censor`); a wildcard rule such as `*censor*` means nothing to a prose model, so the concrete evidence is used instead. A prose prompt is never followed by tag panels in boxes 2-4, and the router can never choose this action on its own - it is reachable only by choosing it in `やりたいこと`.

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

The browser end-to-end tests need Playwright, which lives in its own dependency group, and they
only run when `RUN_BROWSER_E2E` is set - otherwise the module skips itself:

```bash
uv sync --extra web --group test --group web-test
uv run playwright install chromium
RUN_BROWSER_E2E=1 uv run pytest tests/browser
```

Sync with `--group web-test` whenever you sync at all. `uv sync --extra web --group test` on its
own *removes* Playwright if it is already installed, and the browser tests then skip silently
rather than failing.

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
- `scene_prompt.py`: natural-language prompt templates, request building, and rendering.
- `next_panel.py`: the moment after the current panel, bounded by the dictionary.
- `settings_store.py`: the Web UI settings that survive a restart, and the work that does not.
- `tag_review.py`: dictionary-bounded review of inferred tags against the image.
- `tag_subset.py`: Danbooru post-based subset loading, fetching, and writing.
- `models.py`: Pydantic request/response models.

This separation keeps the compiler reusable for future integrations (such as a ComfyUI custom node) without coupling to CLI concerns.
