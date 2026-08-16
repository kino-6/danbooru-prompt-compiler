from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import httpx
import pytest

from danbooru_prompt_compiler.image_source import (
    download_image_url,
    resolve_image_source,
)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)


def _client(content: bytes, content_type: str = "image/png") -> httpx.Client:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": content_type},
            content=content,
            request=request,
        )
    )
    return httpx.Client(transport=transport)


def test_uploaded_image_takes_precedence_over_url() -> None:
    with resolve_image_source("uploaded.png", "https://example.com/image.png") as path:
        assert path == "uploaded.png"


def test_url_image_is_downloaded_and_removed_after_request() -> None:
    with _client(PNG_1X1) as client:
        with download_image_url("https://example.com/image.png", client=client) as path:
            assert path.exists()
            assert path.read_bytes() == PNG_1X1
            temporary_path = path

    assert not temporary_path.exists()


@pytest.mark.parametrize("url", ["file:///tmp/image.png", "data:image/png;base64,abc"])
def test_url_image_requires_http_or_https(url: str) -> None:
    with pytest.raises(ValueError, match="http://"):
        with download_image_url(url):
            pass


def test_url_image_rejects_non_image_response() -> None:
    with _client(b"not an image", "text/html") as client:
        with pytest.raises(ValueError, match="画像ではありません"):
            with download_image_url("https://example.com/page", client=client):
                pass


def test_url_image_rejects_oversized_response_and_cleans_up() -> None:
    temp_dir = Path(tempfile.gettempdir())
    before = set(temp_dir.glob("danbooru-prompt-url-*"))
    with _client(PNG_1X1) as client:
        with pytest.raises(ValueError, match="20 MB"):
            with download_image_url(
                "https://example.com/image.png",
                client=client,
                max_bytes=8,
            ):
                pass
    assert set(temp_dir.glob("danbooru-prompt-url-*")) == before
