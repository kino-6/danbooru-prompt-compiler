from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

DEFAULT_DANBOORU_BASE_URL = "https://danbooru.donmai.us"
DEFAULT_TAG_LIMIT = 20_000
MAX_PAGE_SIZE = 200
USER_AGENT = "danbooru-prompt-compiler/0.1.0"

FetchTags = Callable[[int], list[str]]


def read_tag_dictionary(path: Path) -> set[str]:
    return set(json.loads(path.read_text(encoding="utf-8")))


def write_tag_dictionary(path: Path, tags: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(tags, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_or_fetch_tag_dictionary(
    path: Path,
    *,
    max_tags: int = DEFAULT_TAG_LIMIT,
    fetch_tags: FetchTags | None = None,
) -> set[str]:
    if path.exists():
        return read_tag_dictionary(path)

    fetch = fetch_tags or fetch_danbooru_tags
    tags = fetch(max_tags)
    write_tag_dictionary(path, tags)
    return set(tags)


def fetch_danbooru_tags(
    max_tags: int = DEFAULT_TAG_LIMIT,
    *,
    base_url: str = DEFAULT_DANBOORU_BASE_URL,
    page_size: int = MAX_PAGE_SIZE,
    client: httpx.Client | None = None,
) -> list[str]:
    if max_tags < 1:
        raise ValueError("max_tags must be at least 1")
    if page_size < 1 or page_size > MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")

    close_client = client is None
    http_client = client or httpx.Client(timeout=60.0, headers={"User-Agent": USER_AGENT})
    tags: list[str] = []
    seen: set[str] = set()

    try:
        page = 1
        while len(tags) < max_tags:
            batch_size = min(page_size, max_tags - len(tags))
            response = http_client.get(
                f"{base_url.rstrip('/')}/tags.json",
                params={
                    "search[order]": "count",
                    "limit": batch_size,
                    "page": page,
                },
            )
            response.raise_for_status()
            data = response.json()
            if not data:
                break

            for item in data:
                tag = _extract_tag_name(item)
                if tag and tag not in seen:
                    tags.append(tag)
                    seen.add(tag)

            page += 1
    finally:
        if close_client:
            http_client.close()

    return tags


def _extract_tag_name(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None

    name = item.get("name")
    if not isinstance(name, str):
        return None

    name = name.strip().lower()
    if not name:
        return None

    return name.replace(" ", "_")
