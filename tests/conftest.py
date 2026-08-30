from __future__ import annotations

from unittest import mock

import pytest

from danbooru_prompt_compiler import webui


@pytest.fixture(autouse=True)
def isolated_webui_settings():
    """Keep the workbench's remembered settings out of the tests.

    The Web UI saves what it was last run with, so without this a test asserting
    a control's default reads whatever the developer last selected - and a test
    run would rewrite it. Both directions are cut here rather than in each test.
    """
    with mock.patch.object(webui, "load_settings", lambda *_args, **_kwargs: {}):
        with mock.patch.object(webui, "save_settings", lambda *_args, **_kwargs: None):
            yield
