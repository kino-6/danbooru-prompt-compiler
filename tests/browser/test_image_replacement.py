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
            expect(page.get_by_text("first.png", exact=True)).to_be_visible()

            file_input.set_input_files(str(second))
            expect(page.get_by_text("second.png", exact=True)).to_be_visible()
            expect(page.get_by_text("first.png", exact=True)).not_to_be_visible()
            browser.close()
