from __future__ import annotations

import base64
import functools
import hashlib
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest


if os.getenv("RUN_BROWSER_E2E") != "1":
    pytest.skip("set RUN_BROWSER_E2E=1 to run browser tests", allow_module_level=True)

pytest.importorskip("gradio")
pytest.importorskip("playwright")

from PIL import Image  # noqa: E402
from playwright.sync_api import expect, sync_playwright  # noqa: E402

from tests.webui_harness import running_test_webui  # noqa: E402


def test_second_image_replaces_loaded_image(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), "red").save(first)
    Image.new("RGB", (4, 4), "blue").save(second)

    with running_test_webui() as (url, service):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            file_input = page.locator('#image-workspace input[type="file"]')

            file_input.set_input_files(str(first))
            expect(page.locator('#image-workspace img[src*="first.png"]')).to_be_visible()

            # The VLM switch and its description must be usable without
            # opening any collapsed section.
            image_description = page.locator("#image-description-editor textarea")
            expect(image_description).to_be_visible()
            expect(page.get_by_label("VLMで画像を説明する")).to_be_visible()
            image_description.fill("石段に立つ少女")

            # One run per click keeps the replacement assertions unambiguous.
            page.get_by_label("次のコマも生成する（出力2〜4）").uncheck()
            page.get_by_role("button", name="実行", exact=True).click()
            page.get_by_text("画像タグの確認・修正", exact=True).click()
            inferred_tags = page.locator("#inferred-tags-editor textarea")
            expect(inferred_tags).to_have_value("1girl, solo")
            page.get_by_text("既存プロンプトから編集（任意）", exact=True).click()
            base_prompt = page.locator("#base-prompt-input textarea")
            base_prompt.fill("old prompt")
            expect(page.locator("#prompt-output-1 textarea")).not_to_have_value("")

            page.locator('#image-workspace input[type="file"]').set_input_files(str(second))
            expect(page.locator('#image-workspace img[src*="second.png"]')).to_be_visible()
            expect(page.locator('#image-workspace input[type="file"]')).to_be_attached()
            expect(inferred_tags).to_have_value("")
            expect(image_description).to_have_value("")
            expect(base_prompt).to_have_value("")
            for index in range(1, 5):
                # Clearing the page takes the boxes away rather than leaving
                # four empty ones behind.
                expect(page.locator(f"#prompt-output-{index}")).to_be_hidden()

            page.get_by_role("button", name="実行", exact=True).click()
            expect(inferred_tags).to_have_value("1girl, solo")
            assert service.run_options[-1]["edited_tags"] == ""
            assert service.run_options[-1]["edited_description"] == ""
            assert service.run_options[-1]["base_prompt"] == ""
            browser.close()


def test_three_prompt_variants_are_visible_editable_and_copy_ready(tmp_path) -> None:
    image = tmp_path / "input.png"
    Image.new("RGB", (4, 4), "red").save(image)
    prompts = [
        "1girl, solo\nrain",
        "1girl, solo\nnight",
        "1girl, solo\nlooking_back",
    ]

    with running_test_webui(candidates=prompts) as (url, _service):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.locator('input[type="file"]').first.set_input_files(str(image))
            # This covers the plain multi-variant layout, without panel follow-ups.
            page.get_by_label("次のコマも生成する（出力2〜4）").uncheck()
            page.get_by_role("button", name="実行", exact=True).click()

            for index, prompt in enumerate(prompts, 1):
                container = page.locator(f"#prompt-output-{index}")
                output = container.locator("textarea")
                expect(output).to_be_visible()
                expect(output).to_be_editable()
                expect(output).to_have_value(prompt)
                expect(container.get_by_role("button", name="Copy")).to_be_visible()

            # Three variants leave three boxes; the fourth never appears.
            expect(page.locator("#prompt-output-4")).to_be_hidden()
            browser.close()


def test_recovery_and_change_controls_need_no_section_opened() -> None:
    with running_test_webui() as (url, _service):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)

            # Clicking would unload a real model on the developer's machine, so
            # the behaviour is unit-tested and only reachability is checked here.
            expect(
                page.get_by_role("button", name="VLMを復旧", exact=True)
            ).to_be_visible()
            expect(page.locator("#next-panel-change")).to_be_visible()
            expect(page.locator("#run-status")).to_be_attached()
            browser.close()


def test_next_panel_proposals_fill_the_boxes_after_the_current_prompt(tmp_path) -> None:
    image = tmp_path / "panel.png"
    Image.new("RGB", (4, 4), "teal").save(image)

    with running_test_webui(candidates=["1girl, solo\nrain"]) as (url, service):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.locator('#image-workspace input[type="file"]').set_input_files(str(image))
            expect(page.locator('#image-workspace img[src*="panel.png"]')).to_be_visible()

            # The follow-up is on by default; no section needs opening.
            expect(page.get_by_label("次のコマも生成する（出力2〜4）")).to_be_checked()
            page.get_by_role("button", name="実行", exact=True).click()

            expect(page.locator("#prompt-output-1 textarea")).to_have_value("1girl, solo\nrain")
            for index, panel in enumerate(("panel_a", "panel_b", "panel_c"), start=2):
                expect(page.locator(f"#prompt-output-{index} textarea")).to_have_value(panel)
            expect(page.locator("#run-status")).to_be_visible()

            assert [options["action_override"] for options in service.run_options] == [
                "auto",
                "next_panel",
            ]
            browser.close()


def test_scene_prompt_button_runs_with_the_selected_template(tmp_path) -> None:
    image = tmp_path / "scene.png"
    Image.new("RGB", (4, 4), "navy").save(image)

    with running_test_webui(candidates=["Subject: a young woman"]) as (url, service):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.locator('#image-workspace input[type="file"]').set_input_files(str(image))
            expect(page.locator('#image-workspace img[src*="scene.png"]')).to_be_visible()

            # The template picker and the trigger are hidden until the task asks
            # for them, so picking the task is the first step of the flow now.
            page.locator("#task-selector").get_by_text("自然文プロンプト", exact=True).click()
            expect(page.locator("#scene-template")).to_be_visible()

            page.locator("#scene-template input").click()
            page.locator("#scene-template li", has_text="絵コンテ／次のコマ").first.click()
            page.get_by_role("button", name="自然文プロンプト", exact=True).click()

            expect(page.locator("#prompt-output-1 textarea")).to_have_value(
                "Subject: a young woman"
            )
            assert service.run_options[-1]["action_override"] == "scene_prompt"
            assert service.run_options[-1]["scene_template"] == "storyboard_panel"
            browser.close()


def test_next_panel_button_runs_from_an_image_alone(tmp_path) -> None:
    image = tmp_path / "panel.png"
    Image.new("RGB", (4, 4), "orange").save(image)

    with running_test_webui() as (url, service):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.locator('#image-workspace input[type="file"]').set_input_files(str(image))
            expect(page.locator('#image-workspace img[src*="panel.png"]')).to_be_visible()

            page.get_by_role("button", name="次のコマ", exact=True).click()
            expect(page.locator("#prompt-output-1 textarea")).not_to_have_value("")

            assert service.run_options[-1]["action_override"] == "next_panel"
            assert service.run_options[-1]["instruction"] == ""
            browser.close()


def test_clipboard_image_can_be_pasted_into_image_workspace(tmp_path) -> None:
    image = tmp_path / "pasted.png"
    Image.new("RGB", (4, 4), "purple").save(image)
    data_url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()

    with running_test_webui() as (url, service):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.wait_for_function(
                "document.documentElement.dataset.imageUrlDropReady === 'true'"
            )
            page.evaluate(
                """async (dataUrl) => {
                    const blob = await (await fetch(dataUrl)).blob();
                    const transfer = new DataTransfer();
                    transfer.items.add(
                        new File([blob], "clipboard.png", { type: "image/png" })
                    );
                    document.dispatchEvent(
                        new ClipboardEvent("paste", {
                            clipboardData: transfer,
                            bubbles: true,
                        })
                    );
                }""",
                data_url,
            )
            expect(page.locator("#image-workspace img")).to_be_visible()
            page.get_by_role("button", name="実行", exact=True).click()
            page.get_by_text("画像タグの確認・修正", exact=True).click()
            expect(page.locator("#inferred-tags-editor textarea")).to_have_value(
                "1girl, solo"
            )
            assert service.image_digests[-1] == hashlib.sha256(
                image.read_bytes()
            ).hexdigest()
            browser.close()


def test_remote_image_url_can_be_dropped_on_image_workspace(tmp_path) -> None:
    image = tmp_path / "remote.png"
    Image.new("RGB", (4, 4), "green").save(image)
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    image_server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    image_thread = threading.Thread(target=image_server.serve_forever, daemon=True)
    image_thread.start()

    try:
        with running_test_webui() as (url, service):
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page()
                page.goto(url)
                page.get_by_text("詳細設定", exact=True).click()
                page.get_by_label("プライベート画像URLを許可").check()
                page.wait_for_function(
                    "document.documentElement.dataset.imageUrlDropReady === 'true'"
                )
                remote_url = (
                    f"http://127.0.0.1:{image_server.server_port}/remote.png"
                )
                data_transfer = page.evaluate_handle(
                    """url => {
                        const value = new DataTransfer();
                        value.setData("text/uri-list", url);
                        return value;
                    }""",
                    remote_url,
                )
                page.locator("#image-workspace").dispatch_event(
                    "drop",
                    {"dataTransfer": data_transfer},
                )
                page.get_by_text("実行の詳細", exact=True).click()
                expect(page.get_by_text("URL画像を読み込みました。", exact=True)).to_be_visible()
                page.get_by_role("button", name="実行", exact=True).click()
                page.get_by_text("画像タグの確認・修正", exact=True).click()
                expect(page.locator("#inferred-tags-editor textarea")).to_have_value(
                    "1girl, solo"
                )
                assert service.image_digests[-1] is not None
                browser.close()
    finally:
        image_server.shutdown()
        image_server.server_close()


DROP_FILE_JS = """
([b64, name]) => {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  const transfer = new DataTransfer();
  transfer.items.add(new File([bytes], name, {type: "image/png"}));
  const target = document.querySelector("#image-workspace");
  for (const type of ["dragenter", "dragover", "drop"]) {
    target.dispatchEvent(
      new DragEvent(type, {dataTransfer: transfer, bubbles: true, cancelable: true})
    );
  }
}
"""


def test_dropping_a_file_replaces_an_already_loaded_image(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), "red").save(first)
    Image.new("RGB", (4, 4), "blue").save(second)

    with running_test_webui() as (url, _service):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.wait_for_function(
                "document.documentElement.dataset.imageUrlDropReady === 'true'"
            )
            page.locator('#image-workspace input[type="file"]').set_input_files(str(first))
            expect(page.locator('#image-workspace img[src*="first.png"]')).to_be_visible()

            # Gradio's own dropzone is gone once an image is loaded, so a dropped
            # file used to be swallowed and the workspace left as it was.
            page.evaluate(
                DROP_FILE_JS,
                [base64.b64encode(second.read_bytes()).decode("ascii"), "second.png"],
            )

            expect(page.locator('#image-workspace img[src*="second.png"]')).to_be_visible()
            expect(page.locator('#image-workspace img[src*="first.png"]')).to_have_count(0)
            browser.close()


def test_an_empty_workspace_is_left_to_gradios_own_dropzone(tmp_path) -> None:
    image = tmp_path / "only.png"
    Image.new("RGB", (4, 4), "purple").save(image)

    with running_test_webui() as (url, _service):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.wait_for_function(
                "document.documentElement.dataset.imageUrlDropReady === 'true'"
            )
            # Gradio handled the empty case correctly before this handler existed,
            # so the handler must not touch it: taking it over gained nothing and
            # put the first load of the session at risk.
            page.evaluate(
                DROP_FILE_JS,
                [base64.b64encode(image.read_bytes()).decode("ascii"), "only.png"],
            )
            page.wait_for_timeout(500)

            assert (
                page.evaluate(
                    "() => document.querySelector('#image-workspace input').files.length"
                )
                == 0
            )
            # And the click path, which is what a synthetic drop cannot exercise,
            # still loads the first image.
            page.locator('#image-workspace input[type="file"]').set_input_files(str(image))
            expect(page.locator('#image-workspace img[src*="only.png"]')).to_be_visible()
            browser.close()
