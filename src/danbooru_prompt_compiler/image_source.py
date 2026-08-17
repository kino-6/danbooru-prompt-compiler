from __future__ import annotations

import mimetypes
import ipaddress
import socket
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_REDIRECTS = 5
HostResolver = Callable[[str], list[str]]


@contextmanager
def resolve_image_source(
    upload_path: str | None,
    image_url: str,
    *,
    client: httpx.Client | None = None,
    max_bytes: int = MAX_IMAGE_BYTES,
    allow_private_hosts: bool = False,
    resolver: HostResolver | None = None,
) -> Iterator[str | None]:
    """Resolve an uploaded image first, otherwise download an HTTP(S) image."""
    if upload_path:
        yield upload_path
        return

    clean_url = (image_url or "").strip()
    if not clean_url:
        yield None
        return

    with download_image_url(
        clean_url,
        client=client,
        max_bytes=max_bytes,
        allow_private_hosts=allow_private_hosts,
        resolver=resolver,
    ) as path:
        yield str(path)


@contextmanager
def download_image_url(
    image_url: str,
    *,
    client: httpx.Client | None = None,
    max_bytes: int = MAX_IMAGE_BYTES,
    allow_private_hosts: bool = False,
    resolver: HostResolver | None = None,
) -> Iterator[Path]:
    host_resolver = resolver or _resolve_host_addresses
    _validate_remote_url(
        image_url,
        allow_private_hosts=allow_private_hosts,
        resolver=host_resolver,
    )

    owns_client = client is None
    http_client = client or httpx.Client(
        follow_redirects=False,
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"User-Agent": "danbooru-prompt-compiler/0.1"},
    )
    temp_path: Path | None = None
    try:
        current_url = image_url
        for redirect_count in range(MAX_REDIRECTS + 1):
            _validate_remote_url(
                current_url,
                allow_private_hosts=allow_private_hosts,
                resolver=host_resolver,
            )
            with http_client.stream(
                "GET",
                current_url,
                follow_redirects=False,
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("画像URLのリダイレクト先が不正です。")
                    if redirect_count >= MAX_REDIRECTS:
                        raise ValueError("画像URLのリダイレクト回数が多すぎます。")
                    current_url = urljoin(str(response.url), location)
                    continue

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
                break
        else:  # pragma: no cover - loop always exits or raises
            raise ValueError("画像URLを取得できませんでした。")

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


def load_image_url_preview(
    image_url: str,
    *,
    client: httpx.Client | None = None,
    allow_private_hosts: bool = False,
    resolver: HostResolver | None = None,
) -> Image.Image:
    with download_image_url(
        image_url,
        client=client,
        allow_private_hosts=allow_private_hosts,
        resolver=resolver,
    ) as path:
        with Image.open(path) as image:
            return image.copy()


def _validate_remote_url(
    image_url: str,
    *,
    allow_private_hosts: bool,
    resolver: HostResolver,
) -> None:
    parsed = urlsplit(image_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("画像URLは http:// または https:// を指定してください。")
    if parsed.username or parsed.password:
        raise ValueError("認証情報を含む画像URLは使用できません。")
    if allow_private_hosts:
        return

    try:
        addresses = resolver(parsed.hostname)
    except OSError as exc:
        raise ValueError("画像URLのホスト名を解決できません。") from exc
    if not addresses:
        raise ValueError("画像URLのホスト名を解決できません。")
    if any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError(
            "プライベート・ループバック・リンクローカル宛ての画像URLは使用できません。"
        )


def _resolve_host_addresses(hostname: str) -> list[str]:
    return list(
        dict.fromkeys(
            info[4][0]
            for info in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        )
    )
