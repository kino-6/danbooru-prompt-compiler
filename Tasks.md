# Web UI Improvement Tasks

Tasks are executed in order. A task is complete only when every Gate is checked.

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

- [ ] Run `uv run pytest -q` in CI on pull requests and pushes to `main`.
- [ ] Keep the browser job separate so a missing Chromium never blocks unit feedback.
- [ ] Verify the workflow on Python 3.11, matching the local environment.

### Gate

- [ ] A pull request shows a failing unit job when a unit test is broken on purpose.
- [ ] The browser job still installs Chromium and passes.

## Task 17 — Share exclusion words with the CLI

Exclusion words only exist on the Web UI path. `danbooru-prompt --image` and the plain
compiler still emit `censored` and other quality-degrading tags.

- [ ] Apply the saved exclusion words to CLI image tagging and prompt output.
- [ ] Add `--excluded-tags` and `--no-tag-exclusions` options that override the saved list.
- [ ] Report removed tags in the CLI diagnostics section, as the Web UI status does.

### Gate

- [ ] Unit tests prove the CLI drops censor tags from image tags and compiled output.
- [ ] Unit test proves the CLI options override the saved list.
- [ ] README documents the shared exclusion words for both entry points.

## Task 18 — Keep the requested tag count after exclusions

Exclusions run after the compiler truncates to `max_output_tags`, so a variant that
contains excluded tags returns fewer tags than requested.

- [ ] Pass the exact-match exclusion words to the compiler as tags it must never output.
- [ ] Request enough headroom so post-filter output still reaches the requested count.
- [ ] Leave wildcard rules to the deterministic post-filter.

### Gate

- [ ] Unit test proves the compile prompt names the excluded tags.
- [ ] Unit test proves a variant whose raw output contains excluded tags still returns the
      requested tag count.

## Task 19 — Replace the positional Web UI input list with a request model

`run_prompt_workbench` takes 19 positional inputs whose order is duplicated in
`webui.py`, `web_service.py`, and `tests/web/test_webui_api.py`. Inserting a control in
the middle silently breaks the named API.

- [ ] Introduce a `WebRunRequest` model that carries every run parameter.
- [ ] Build the Gradio `inputs` list from a single ordered definition.
- [ ] Have the API E2E construct its arguments from that definition instead of a literal list.

### Gate

- [ ] Unit test proves the Gradio input order matches the request model field order.
- [ ] Adding a control does not require editing the API E2E argument list.

## Task 20 — Split the Web UI layout from its behavior

`build_app` is a single ~400-line function that mixes layout, event wiring, and handlers,
so component changes cannot be unit tested without building the whole app.

- [ ] Extract the image workspace, advanced settings, and output sections into builders.
- [ ] Move the request handler out of `build_app` so it can be tested without Gradio.
- [ ] Keep component `elem_id`s stable so the browser tests are unaffected.

### Gate

- [ ] Unit test exercises the request handler without constructing a Gradio app.
- [ ] Existing Web UI unit, API, and browser tests pass unchanged.
