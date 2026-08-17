from __future__ import annotations

import functools
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
            expect(base_prompt).to_have_value("")
            for index in range(1, 5):
                expect(page.locator(f"#prompt-output-{index} textarea")).to_have_value("")

            page.get_by_role("button", name="実行", exact=True).click()
            expect(inferred_tags).to_have_value("1girl, solo")
            assert service.run_options[-1]["edited_tags"] == ""
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
            page.get_by_role("button", name="実行", exact=True).click()

            for index, prompt in enumerate(prompts, 1):
                container = page.locator(f"#prompt-output-{index}")
                output = container.locator("textarea")
                expect(output).to_be_visible()
                expect(output).to_be_editable()
                expect(output).to_have_value(prompt)
                expect(container.get_by_role("button", name="Copy")).to_be_visible()

            fourth = page.locator("#prompt-output-4 textarea")
            expect(fourth).to_be_visible()
            expect(fourth).to_be_editable()
            expect(fourth).to_have_value("")
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
                page.get_by_text("実行情報", exact=True).click()
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
