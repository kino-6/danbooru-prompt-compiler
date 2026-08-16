from __future__ import annotations

import mimetypes
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 20 * 1024 * 1024


@contextmanager
def resolve_image_source(
    upload_path: str | None,
    image_url: str,
    *,
    client: httpx.Client | None = None,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> Iterator[str | None]:
    """Resolve an uploaded image first, otherwise download an HTTP(S) image."""
    if upload_path:
        yield upload_path
        return

    clean_url = (image_url or "").strip()
    if not clean_url:
        yield None
        return

    with download_image_url(clean_url, client=client, max_bytes=max_bytes) as path:
        yield str(path)


@contextmanager
def download_image_url(
    image_url: str,
    *,
    client: httpx.Client | None = None,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> Iterator[Path]:
    parsed = urlsplit(image_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("画像URLは http:// または https:// を指定してください。")
    if parsed.username or parsed.password:
        raise ValueError("認証情報を含む画像URLは使用できません。")

    owns_client = client is None
    http_client = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"User-Agent": "danbooru-prompt-compiler/0.1"},
    )
    temp_path: Path | None = None
    try:
        with http_client.stream("GET", image_url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0]
            if not content_type.startswith("image/"):
                raise ValueError("URLの応答が画像ではありません。")

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError("画像URLのファイルサイズが20 MBを超えています。")

            suffix = mimetypes.guess_extension(content_type) or ".img"
            with tempfile.NamedTemporaryFile(
                prefix="danbooru-prompt-url-",
                suffix=suffix,
                delete=False,
            ) as destination:
                temp_path = Path(destination.name)
                downloaded = 0
                for chunk in response.iter_bytes():
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError("画像URLのファイルサイズが20 MBを超えています。")
                    destination.write(chunk)

        try:
            with Image.open(temp_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("URLから取得したデータを画像として読み込めません。") from exc

        yield temp_path
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if owns_client:
            http_client.close()
