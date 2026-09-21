# Web UI Improvement Tasks (completed)

Finished tasks from the Web UI improvement plan, every Gate checked. They are kept here
for the reasoning behind each change; the open plan lives in `Tasks.md`.

## Task 1 — Cache image-tagging runtime and results

- [x] Keep the WD Tagger ONNX session and label metadata alive across requests.
- [x] Cache inferred tags by image content and tagging parameters with a bounded cache.
- [x] Show whether the image-tag result was a cache hit or miss.

### Gate

- [x] Unit test proves one `ImageTagger` instance initializes its runtime once.
- [x] Unit test proves an unchanged image and settings call the tagger once across repeated runs.
- [x] The full test suite passes (55 passed).

## Task 2 — Add human correction and action override controls

- [x] Make inferred tags editable and reuse the edited tags on the next execution.
- [x] Add an `auto / tag_image / compile / edit / next_panel` action selector.
- [x] Mark manually overridden execution plans as `manual`.

### Gate

- [x] Unit tests prove edited tags become the compiler input.
- [x] Unit tests prove each valid manual action bypasses the router decision.
- [x] Gradio configuration exposes editable tags and the action selector.

## Task 3 — Add optional VLM analysis for spatial requests

- [x] Extend the Ollama request path to attach local images.
- [x] Add an optional VLM model setting, disabled by default.
- [x] Use VLM observations only for image edits and next-panel requests.

### Gate

- [x] Unit test proves Ollama receives a base64 image payload.
- [x] Unit tests prove VLM is skipped for tagging/new compilation and when disabled.
- [x] Unit test proves VLM observations are included in the edit compiler request.

## Task 4 — Add progress and cancellation

- [x] Report routing, tagging, vision, compilation, and completion phases.
- [x] Add a cancel button wired to active Gradio events.
- [x] Keep queue concurrency at one to avoid local model contention.

### Gate

- [x] Unit test proves ordered progress phases are emitted.
- [x] Gradio configuration contains a cancellation event for both click and submit runs.
- [x] Existing API execution still completes successfully.

## Task 5 — Add candidate selection and session history

- [x] Expose generated variants as selectable candidates.
- [x] Add an adopt action that copies the selected candidate into the base prompt.
- [x] Keep a bounded per-session history of recent executions.

### Gate

- [x] Unit tests prove candidate rendering preserves variant boundaries.
- [x] Unit test proves history is newest-first and bounded.
- [x] Gradio configuration exposes candidate, adopt, and history controls.

## Task 6 — Harden and improve URL input

- [x] Add a URL preview action.
- [x] Reject loopback/private/link-local destinations unless explicitly allowed.
- [x] Validate redirect destinations as well as the original URL.
- [x] Keep size, content-type, image-data, and temporary-file checks.

### Gate

- [x] Unit tests cover private hosts, redirect hosts, invalid content, limits, and cleanup.
- [x] URL preview returns displayable image data without leaving a temporary file.
- [x] Uploaded files continue to take precedence over URLs.

## Task 7 — Add automated Web UI regression coverage

- [x] Cover local upload and URL input through the named Gradio API.
- [x] Add a browser E2E scenario for replacing an already loaded image by drop.
- [x] Keep the visible drop target mounted by moving the active file to hidden state and resetting the drop component after every upload.
- [x] Make browser coverage runnable in CI without changing the default unit-test command.

### Gate

- [x] API E2E passes for upload replacement inputs and URL input.
- [x] Browser test asserts that the second dropped image becomes the active preview/input.
- [x] CI configuration installs the required browser before the browser test.

## Task 8 — Add Ollama diagnostics and actionable errors

- [x] Add a connection/model check in the advanced settings.
- [x] Explain how to start Ollama when it is unreachable.
- [x] Explain the exact `ollama pull` command when a configured model is missing.

### Gate

- [x] Unit tests cover healthy, unreachable, and missing-model states.
- [x] UI exposes diagnostics without starting prompt generation.
- [x] The full test suite and compile check pass with a clean diff check (78 passed, 1 browser-only skip).

## Task 9 — Make prompt variants individually copyable

- [x] Render each generated variant in its own output textbox.
- [x] Add a copy button to every variant textbox.
- [x] Keep variant headers out of the individually copied prompt text.

### Gate

- [x] Unit tests prove variants remain separate and copy text has no variant header.
- [x] Gradio configuration exposes four independently copyable output boxes.
- [x] Named Gradio API regression test passes with the expanded outputs.

## Task 10 — Fix hidden/read-only prompt variants

- [x] Keep all four prompt output boxes mounted and visible.
- [x] Return plain textbox values instead of visibility update objects.
- [x] Render each candidate as a flat comma-separated prompt without grouped diagnostics.
- [x] Make every output textbox editable and retain its copy button.

### Gate

- [x] Unit and API tests prove three candidates populate output boxes 1–3 independently.
- [x] Component configuration proves all four boxes are visible, editable, and copy-enabled.
- [x] Chromium E2E proves three candidates are simultaneously visible, editable, and contain copy-ready text.

## Task 11 — Preserve organized output and expose output count

- [x] Keep category ordering as copy-ready multiline prompt text in every output box.
- [x] Move output-count selection to the main controls and support 1–4 outputs.
- [x] Default the Web UI to four outputs.
- [x] Make the explicit UI count authoritative over the router model response.

### Gate

- [x] Unit tests prove organized multiline output and authoritative count selection.
- [x] Component configuration exposes output count 1–4 with default 4.
- [x] Real-service API returns four independent multiline prompts when output count is 4.
- [x] Chromium E2E passes with multiline prompt values.

## Task 12 — Filter low-value image tags and reset replaced images

- [x] Exclude configurable exact-match and wildcard image tags before prompt compilation.
- [x] Add folded Web UI controls for enabling and editing the exclusion rules.
- [x] Clear inferred tags, base prompt, candidates, and prompt outputs when an image is replaced.

### Gate

- [x] Unit tests prove default filtering, wildcard matching, and disabling the filter.
- [x] Gradio configuration exposes enabled-by-default folded filter controls.
- [x] Chromium E2E proves replacing an image clears old image-dependent prompt state.
- [x] The full test suite and CI pass (83 passed, 1 browser-only skip; 2 Chromium E2E passed).

## Task 13 — Simplify and fold the Web UI

- [x] Replace the separate file drop target, selected filename, and preview with one image workspace.
- [x] Support local drag-and-drop, file selection, clipboard paste, and remote-image URL drops from that workspace.
- [x] Keep manual URL entry as a folded fallback.
- [x] Fold the existing-prompt editor, inferred tags, candidate history, execution details, and advanced settings.

### Gate

- [x] Gradio configuration proves the unified image workspace accepts upload, clipboard, and URL-drop input.
- [x] Unit tests prove every secondary section is initially folded.
- [x] Chromium E2E proves replacing an image through the unified workspace clears old prompt state and URL drops remain the active source.
- [x] The full test suite and CI pass (85 passed, 1 browser-only skip; 3 Chromium E2E passed).

## Task 14 — Manage quality-degrading exclusion words from the UI

- [x] Exclude `censored`, `bar_censor`, and other censor tags by default.
- [x] Apply the exclusion words to generated prompts as well as inferred image tags.
- [x] Add save and restore-defaults actions that persist the exclusion words to `data/excluded_tags.json`.
- [x] Load the saved exclusion words when the workbench starts.

### Gate

- [x] Unit tests prove censor defaults, rule normalization, and save/load round-trips.
- [x] Unit test proves censor tags are removed from image tags and prompt output, and reported in the status.
- [x] Gradio configuration exposes an open exclusion-word editor with save and reset buttons.

## Task 15 — Generate from a clipboard image

- [x] Route a pasted clipboard image into the image workspace from anywhere on the page.
- [x] Keep text paste in text fields unaffected when the clipboard also carries text.
- [x] Load a pasted HTTP/HTTPS image URL through the existing URL bridge.
- [x] Drop the `clipboard` image source so the browser stops asking for clipboard-read permission.

### Gate

- [x] Unit test proves the paste bridge targets the image workspace file input.
- [x] Unit test proves the permission-prompting clipboard source stays disabled.
- [x] Chromium E2E proves a pasted image becomes the active image and produces a run.
- [x] The full test suite and CI pass (93 passed, 1 browser-only skip; 4 Chromium E2E passed).

## Task 16 — Run the unit suite in CI

The `Web UI E2E` workflow only runs `tests/web` and `tests/browser`, so the 93 unit
tests are verified locally but never on a pull request.

- [x] Run `uv run pytest -q` in CI on pull requests and pushes to `main`.
- [x] Keep the browser job separate so a missing Chromium never blocks unit feedback.
- [x] Verify the workflow on Python 3.11, matching the local environment.

### Gate

- [x] The CI command exits non-zero for a deliberately broken test and zero for a clean tree.
- [x] The new `Tests` workflow passes on a pull request.
- [x] The browser workflow still installs Chromium and passes.

## Task 17 — Share exclusion words with the CLI

Exclusion words only exist on the Web UI path. `danbooru-prompt --image` and the plain
compiler still emit `censored` and other quality-degrading tags.

- [x] Apply the saved exclusion words to CLI image tagging and prompt output.
- [x] Add `--excluded-tags` and `--no-tag-exclusions` options that override the saved list.
- [x] Report removed tags in the CLI diagnostics section, as the Web UI status does.

### Gate

- [x] Unit tests prove the CLI drops censor tags from image tags and confidence scores.
- [x] Unit test proves both CLI options override the saved list.
- [x] README documents the shared exclusion words for both entry points.

## Task 18 — Keep the requested tag count after exclusions

Exclusions run after the compiler truncates to `max_output_tags`, so a variant that
contains excluded tags returns fewer tags than requested.

- [x] Pass the exact-match exclusion words to the compiler as tags it must never output.
- [x] Drop excluded tags before truncating, so removed tags never consume output slots.
- [x] Keep wildcard rules out of the model prompt and leave them to the post-filter.
- [x] Report the tags the compiler removed through `CompileResult` for both entry points.

### Gate

- [x] Unit test proves the compile prompt names only the literal excluded tags.
- [x] Unit test proves a variant whose raw output contains excluded tags still returns the
      requested tag count.

## Task 19 — Replace the positional Web UI input list with a request model

`run_prompt_workbench` takes 19 positional inputs whose order is duplicated in
`webui.py`, `web_service.py`, and `tests/web/test_webui_api.py`. Inserting a control in
the middle silently breaks the named API.

- [x] Introduce a `WebRunRequest` model that carries every run parameter.
- [x] Build the Gradio `inputs` list from a single ordered definition.
- [x] Have the API E2E construct its arguments from that definition instead of a literal list.
- [x] Fail at build time when a run field has no component wired to it.

### Gate

- [x] Unit test proves the Gradio input order matches the request model field order.
- [x] Adding a control does not require editing the API E2E argument list.

## Task 20 — Split the Web UI layout from its behavior

`build_app` is a single ~400-line function that mixes layout, event wiring, and handlers,
so component changes cannot be unit tested without building the whole app.

- [x] Extract the image workspace, instruction, advanced settings, and result sections into builders.
- [x] Move the request handler out of `build_app` so it can be tested without Gradio.
- [x] Keep component `elem_id`s stable so the browser tests are unaffected.

### Gate

- [x] Unit tests exercise `run_workbench` success and failure without constructing a Gradio app.
- [x] Existing Web UI unit, API, and browser tests pass unchanged.
- [x] The full test suite and CI pass (100 passed, 1 browser-only skip; 4 Chromium E2E passed).

## Task 21 — Run the next panel from an image alone

An image with no instruction hits the strong routing rule for tag extraction, so the
next-panel action was only reachable through the folded `操作種別` selector.

- [x] Add a `次のコマ` button beside `実行` in the main controls.
- [x] Force the next-panel action for that button regardless of the routing rules.
- [x] Cancel the next-panel run with the existing stop button.

### Gate

- [x] Unit test proves the next-panel trigger shares the run inputs and outputs.
- [x] Unit test proves the stop button cancels every run trigger.
- [x] Service test proves an image with no instruction produces next-panel variants that
      keep the tagged character.
- [x] Chromium E2E proves the button runs from an image alone with an empty instruction.
- [x] The full test suite and CI pass (102 passed, 1 browser-only skip; 5 Chromium E2E passed).

## Task 22 — Describe the image in natural language with the VLM

The VLM observation was only used for edits and next panels, and was never shown, so it
could not help when the tagger returns fewer tags than expected or when a detail cannot
be expressed as a tag.

- [x] Show the description in an editable `画像の説明（VLM）` field for every action.
- [x] Reuse a hand-written description verbatim instead of calling the VLM again.
- [x] Write the description without reference to the instruction, and cache it per image
      and vision model.
- [x] Keep passing the description to image edits and next panels, and keep it out of
      prompts compiled from text alone.
- [x] Report whether the description was generated or cached.

### Gate

- [x] Unit test proves tag extraction now produces a description and that the description
      prompt ignores the instruction.
- [x] Unit test proves an edited description skips the VLM and reaches the compiler.
- [x] Unit test proves a second run with a new instruction reuses the cached description.
- [x] Unit test proves a text-only prompt never receives the description.
- [x] Chromium E2E proves the description field is editable and clears with the image.
- [x] The full test suite and CI pass (107 passed, 1 browser-only skip; 5 Chromium E2E passed).

## Task 23 — Close the exclusion-word gaps and surface the VLM description

`censored_nipples`, `censored_text`, and the rest of the `censored_*` family slipped past
`censored, *_censor, *_censoring`, and rendered-text tags such as `english_text` were never
covered. The tests passed because they only asserted the three spellings the rules matched.

- [x] Replace the censorship rules with `*censor*`, covering every dictionary spelling.
- [x] Exclude rendered text, signatures, watermarks, usernames, and logos by default.
- [x] Keep texture tags and `speech_bubble` out of the exclusion list.
- [x] Move the VLM switch and the description box out of the folded sections, next to the image.
- [x] Say in the UI that a saved list shadows updated defaults, and that `既定に戻す` adopts them.

### Gate

- [x] Regression test proves no `censor` tag in `data/tags.json` survives the defaults.
- [x] Regression test proves every `*_text` tag in the dictionary, `english_text` included, is excluded.
- [x] Unit test proves near-miss tags such as `paper_texture` and `speech_bubble` survive.
- [x] Unit test proves the description box has no collapsed ancestor, self-checked against a
      component that does.
- [x] Chromium E2E proves the VLM switch and description are visible without opening a section.
- [x] The full test suite and CI pass (117 passed, 1 browser-only skip; 5 Chromium E2E passed).

## Task 24 — Produce next panels and the description by default

Both aids were opt-in and easy to miss, and a failing vision model took the whole run
down while its error stayed inside the folded `実行情報` section.

- [x] Default `VLMで画像を説明する` to on.
- [x] Default `次のコマも生成する（出力2〜4）` to on: box 1 keeps the current result and
      boxes 2-4 hold proposals for the moment after it.
- [x] Continue the follow-up from the tags and description the first run resolved.
- [x] Keep the primary result when the vision model or the follow-up fails, and say why.
- [x] Move the status line out of the folded section so failures are visible.
- [x] Leave the `次のコマ` button filling all four boxes, with no duplicate follow-up.

### Gate

- [x] Unit test proves boxes 2-4 hold the panels and that the follow-up reuses the
      resolved tags and description.
- [x] Unit tests prove a failed follow-up, a failed vision model, and an explicit
      next-panel run each behave correctly.
- [x] Service test proves a broken vision model still returns prompts and reports itself.
- [x] Chromium E2E proves the default run fills box 1 with the current prompt, boxes 2-4
      with panels, and shows the status without opening a section.
- [x] The full test suite and CI pass (122 passed, 1 browser-only skip; 6 Chromium E2E passed).

## Task 25 — Recover a wedged VLM and control how much a panel changes

A resident vision model that stops answering had no fix inside the UI, and next panels
came back nearly identical to the current one: every preserved aspect is force-prefixed
onto the variant, and the compiler ran at temperature `0.0`.

- [x] Add a `VLMを復旧` command that unloads the model with `keep_alive: 0` and reloads it.
- [x] Clear the cached description as part of the recovery.
- [x] Report the exact `ollama pull` / `ollama serve` command when that is the real problem.
- [x] Add a `次のコマの変化量` slider that narrows the preserved aspects and raises the
      generation temperature together.
- [x] Support a per-request temperature so one client serves deterministic and varied runs.
- [x] Keep both controls out of the collapsed sections.

### Gate

- [x] Unit tests prove the recovery unloads then reloads, and explains an unreachable
      Ollama, a missing model, and an empty model name.
- [x] Unit test proves recovery clears the cached description.
- [x] Unit tests prove each change level sets its preserved aspects and temperature.
- [x] Unit test proves a large change stops pinning clothing tags onto the panel.
- [x] Chromium E2E proves both controls are reachable without opening a section.
- [x] The full test suite and CI pass (131 passed, 1 browser-only skip; 7 Chromium E2E passed).

## Task 26 — Add a natural-language prompt frame

Newer image models take prose, not Danbooru tags, and the reliable way to write that
prose is the atomic schema used by prompt libraries such as `awesome-gpt-image-2`: one
task line, named sections, a delivery line, and an explicit avoid line.

- [x] Add a `scene_prompt` action that fills a template from the image tags, the VLM
      description, and the instruction.
- [x] Ship four templates as YAML in `templates/`, so a new category is a new file.
- [x] Rebuild the template shape locally instead of trusting the model's formatting.
- [x] Build the avoid line from the literal exclusion words plus the tags this image lost.
- [x] Keep the action reachable only from the UI, never from the router.
- [x] Never follow a prose prompt with tag panels in boxes 2-4.

### Gate

- [x] Unit tests prove templates load in order, bad files are skipped, and a missing
      selection falls back to the first template.
- [x] Unit test proves a sloppy answer - fenced, reordered, with extra lines - is rebuilt
      into the template's shape.
- [x] Unit test proves the request carries the tags, the description, and the avoid list,
      and that the tag compiler never runs.
- [x] Unit test proves the router rewrites a model-proposed `scene_prompt` to a tag action.
- [x] Unit test proves an explicit prose request suppresses the next-panel follow-up.
- [x] Chromium E2E proves picking a template and pressing the button runs with it.
- [x] The full test suite and CI pass (143 passed, 1 browser-only skip; 8 Chromium E2E passed).

## Task 27 — Give the prose step its own local model

Natural-language prompts shared `compiler_model` with tag generation, so pointing them at
a model large enough to write English meant slowing routing and tag output down too.

- [x] Add a `自然文プロンプト用モデル` setting used only by the prose step.
- [x] Fall back to the prompt-generation model when it is left empty.
- [x] Include it in the Ollama connection check and in the failure message.

### Gate

- [x] Unit test proves the prose step uses the dedicated model, and the compiler model
      when the setting is empty.
- [x] Unit test proves the connection check asks about the prose model.
- [x] Unit test proves the setting appears in the run inputs in request-model order.
- [x] The full test suite and CI pass (145 passed, 1 browser-only skip; 8 Chromium E2E passed).

## Task 28 — Let the prose model look at the image

The prose step reads tags and a Japanese description, never the picture. That is two lossy
hops, and a vision model large enough to write English can close them in one.

Measured on `portrate06.png` with `unseen-gemma4:26b`: describing the image alone produced
richer prose than the tag route but silently dropped `pointy_ears` and `elf`, which the WD
Tagger scored at 0.92. Handing it the image *and* the tags recovered every dropped feature
and kept the richer prose, so the tags stay in as factual anchors rather than being replaced.

- [x] Send the reference image to the prose model when the setting is on and an image exists.
- [x] State in the request that the tags are facts observed in the image and must all appear.
- [x] Strip Danbooru qualifiers before the tags reach a prose model, so `bow_(weapon)` reads
      as `bow` instead of leaking `bow (weapon)` into the paragraph.
- [x] Leave the setting off by default, because the default prose model is text-only.

### Gate

- [x] Unit test proves the image reaches the prose client only when the setting is on.
- [x] Unit test proves the request names the tags as observed facts.
- [x] Unit test proves qualifiers are stripped and the avoid list keeps its own wording.
- [x] Unit test proves the setting appears in the run inputs in request-model order.
- [x] The full test suite and CI pass (150 passed, 1 browser-only skip; 8 Chromium E2E passed).

## Task 29 — Choose a vision model instead of typing one

`qwen3-vl:8b` refuses or sanitizes exactly the material this tool exists for, and the
alternative is a 40-character Hugging Face tag nobody will type correctly twice.

- [x] Offer the known vision models as a dropdown that still accepts a free-form entry.
- [x] Say which entry is uncensored and which is the light default, with their sizes.
- [x] Warn when the selected vision model and the prose model cannot be resident together.

### Gate

- [x] Unit test proves a free-form model name still reaches the vision factory.
- [x] Unit test proves the connection check covers the selected entry.
- [x] The full test suite and CI pass (155 passed, 1 browser-only skip; 8 Chromium E2E passed).

## Task 30 — Check the inferred tags against the image

The tagger scores tags independently, so it reports plausible neighbours it cannot rule out
and misses what a threshold cut off. A vision model cannot be trusted to invent tags - asked
for tags directly it returned `red_circle` and `kappa-7391`, neither of them Danbooru - but
it can judge a list it is handed against the picture.

- [x] Add an action that shows the image and the inferred tags to the vision model and asks
      which are wrong and which known tags are missing.
- [x] Restrict every proposal to the loaded tag dictionary, and drop anything outside it.
- [x] Report the verdict as an editable tag list, never as a silent rewrite.

### Gate

- [x] Unit test proves a proposal outside the dictionary is dropped.
- [x] Unit test proves the verdict never removes a tag the user typed by hand.
- [x] Unit test proves a failing vision model leaves the original tags intact.
- [x] The full test suite and CI pass (164 passed, 1 browser-only skip; 8 Chromium E2E passed).
