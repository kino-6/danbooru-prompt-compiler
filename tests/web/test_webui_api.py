from __future__ import annotations

import functools
import hashlib
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest


pytest.importorskip("gradio")
gradio_client = pytest.importorskip("gradio_client")

from gradio_client import Client, handle_file  # noqa: E402
from PIL import Image  # noqa: E402

from tests.webui_harness import running_test_webui  # noqa: E402


def _predict(client: Client, image, image_url: str, allow_private: bool):
    return client.predict(
        image,
        image_url,
        "タグを推測して",
        "",
        "qwen3:1.7b",
        "qwen3:1.7b",
        "http://127.0.0.1:11434",
        0.35,
        0.85,
        12,
        1,
        "",
        "auto",
        False,
        "qwen3-vl:8b",
        allow_private,
        api_name="/run_prompt_workbench",
    )


def test_named_api_accepts_replacement_uploads_and_url(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), "red").save(first)
    Image.new("RGB", (2, 2), "blue").save(second)
    prompts = [
        "1girl, solo\nrain",
        "1girl, solo\nnight",
        "1girl, solo\nlooking_back",
    ]

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    image_server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    image_thread = threading.Thread(target=image_server.serve_forever, daemon=True)
    image_thread.start()
    try:
        with running_test_webui(candidates=prompts) as (url, service):
            client = Client(url, verbose=False)
            first_result = _predict(client, handle_file(first), "", False)
            _predict(client, handle_file(second), "", False)
            _predict(
                client,
                None,
                f"http://127.0.0.1:{image_server.server_port}/first.png",
                True,
            )
    finally:
        image_server.shutdown()
        image_server.server_close()

    first_digest = hashlib.sha256(first.read_bytes()).hexdigest()
    second_digest = hashlib.sha256(second.read_bytes()).hexdigest()
    assert service.image_digests == [first_digest, second_digest, first_digest]
    assert len(first_result) == 9
    assert first_result[2:5] == tuple(prompts)
    assert first_result[5] == ""
