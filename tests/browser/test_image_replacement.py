from __future__ import annotations

import os

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

    with running_test_webui() as (url, _service):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            file_input = page.locator('input[type="file"]').first

            file_input.set_input_files(str(first))
            selected_name = page.get_by_label("選択中画像")
            expect(selected_name).to_have_value("first.png")

            page.locator('input[type="file"]').first.set_input_files(str(second))
            expect(selected_name).to_have_value("second.png")
            expect(page.locator('input[type="file"]').first).to_be_attached()
            browser.close()


def test_three_prompt_variants_are_visible_editable_and_copy_ready(tmp_path) -> None:
    image = tmp_path / "input.png"
    Image.new("RGB", (4, 4), "red").save(image)
    prompts = [
        "1girl, solo, rain",
        "1girl, solo, night",
        "1girl, solo, looking_back",
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
