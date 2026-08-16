from __future__ import annotations

import hashlib
import socket
from contextlib import contextmanager
from pathlib import Path

from danbooru_prompt_compiler.web_service import WebRunResult
from danbooru_prompt_compiler.webui import build_app


class RecordingWebService:
    def __init__(self) -> None:
        self.image_digests: list[str | None] = []

    def run(self, *, image_path: str | None, **_options) -> WebRunResult:
        digest = (
            hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
            if image_path
            else None
        )
        self.image_digests.append(digest)
        output = f"digest_{digest}" if digest else "no_image"
        return WebRunResult(
            action_plan={"action": "tag_image", "router_source": "test"},
            inferred_tags="1girl, solo",
            output=output,
            status="ok",
            candidates=[output],
        )


@contextmanager
def running_test_webui():
    service = RecordingWebService()
    app = build_app(service=service)
    port = _free_port()
    _server, local_url, _share_url = app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=False,
        prevent_thread_lock=True,
        quiet=True,
    )
    try:
        yield local_url, service
    finally:
        app.close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
