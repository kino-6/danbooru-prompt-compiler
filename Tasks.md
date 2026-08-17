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
- [ ] Chromium E2E passes with multiline prompt values.
