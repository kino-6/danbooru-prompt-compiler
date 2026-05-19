from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .normalizer import normalize_tags
from .tag_dictionary import DEFAULT_DANBOORU_BASE_URL, MAX_PAGE_SIZE, USER_AGENT

DEFAULT_POST_LIMIT = 100
DEFAULT_MIN_COUNT = 2
DEFAULT_SUBSET_TAG_LIMIT = 100
ProgressCallback = Callable[[int], None]


@dataclass(frozen=True)
class TagSubsetSelection:
    tags: list[str]
    subset_limit: int
    max_output_tags: int


def read_tag_subset(path: Path, *, max_tags: int | None = DEFAULT_SUBSET_TAG_LIMIT) -> list[str]:
    selection = select_tag_subset(path, max_tags=max_tags)
    return selection.tags


def select_tag_subset(
    path: Path,
    *,
    max_tags: int | None = None,
    max_output_tags: int | None = None,
) -> TagSubsetSelection:
    data = json.loads(path.read_text(encoding="utf-8"))
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        return TagSubsetSelection(tags=[], subset_limit=0, max_output_tags=max_output_tags or 20)

    names: list[str] = []
    selected_items = _auto_select_items(tags) if max_tags is None else tags[:max_tags]
    for item in selected_items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.append(item["name"])

    selected_tags = normalize_tags(names)
    selected_output_limit = max_output_tags or _auto_max_output_tags(selected_tags)
    return TagSubsetSelection(
        tags=selected_tags,
        subset_limit=len(selected_tags),
        max_output_tags=selected_output_limit,
    )


def write_tag_subset(
    path: Path,
    *,
    seed_tags: list[str],
    tag_counts: Counter[str],
    sampled_posts: int | None = None,
    min_count: int | None = None,
    tag_limit: int | None = None,
    base_url: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = [
        _tag_entry(tag, count, sampled_posts=sampled_posts)
        for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    path.write_text(
        json.dumps(
            {
                "seed_tags": seed_tags,
                "sampled_posts": sampled_posts,
                "min_count": min_count,
                "tag_limit": tag_limit,
                "base_url": base_url,
                "tags": tags,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def fetch_post_tag_subset(
    seed_tags: list[str],
    *,
    max_posts: int = DEFAULT_POST_LIMIT,
    min_count: int = DEFAULT_MIN_COUNT,
    max_tags: int = DEFAULT_SUBSET_TAG_LIMIT,
    base_url: str = DEFAULT_DANBOORU_BASE_URL,
    client: httpx.Client | None = None,
    progress: ProgressCallback | None = None,
) -> Counter[str]:
    if not seed_tags:
        raise ValueError("seed_tags must not be empty")
    if max_posts < 1:
        raise ValueError("max_posts must be at least 1")
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if max_tags < 1:
        raise ValueError("max_tags must be at least 1")

    normalized_seed_tags = normalize_tags(seed_tags)
    close_client = client is None
    http_client = client or httpx.Client(timeout=60.0, headers={"User-Agent": USER_AGENT})
    counts: Counter[str] = Counter()

    try:
        page = 1
        fetched_posts = 0
        while fetched_posts < max_posts:
            limit = min(MAX_PAGE_SIZE, max_posts - fetched_posts)
            response = http_client.get(
                f"{base_url.rstrip('/')}/posts.json",
                params={
                    "tags": " ".join(normalized_seed_tags),
                    "limit": limit,
                    "page": page,
                },
            )
            response.raise_for_status()
            posts = response.json()
            if not posts:
                break

            for post in posts:
                counts.update(_extract_general_tags(post))

            fetched_posts += len(posts)
            if progress:
                progress(len(posts))
            page += 1
    finally:
        if close_client:
            http_client.close()

    for seed_tag in normalized_seed_tags:
        counts.pop(seed_tag, None)

    filtered = Counter({tag: count for tag, count in counts.items() if count >= min_count})
    return Counter(dict(sorted(filtered.items(), key=lambda item: (-item[1], item[0]))[:max_tags]))


def _tag_entry(tag: str, count: int, *, sampled_posts: int | None) -> dict[str, int | float | str]:
    entry: dict[str, int | float | str] = {"name": tag, "count": count}
    if sampled_posts:
        entry["frequency"] = round(count / sampled_posts, 4)
    return entry


def _extract_general_tags(post: Any) -> list[str]:
    if not isinstance(post, dict):
        return []

    tag_string = post.get("tag_string_general")
    if not isinstance(tag_string, str):
        return []

    return normalize_tags(tag_string.split())


def _auto_select_items(items: list[Any]) -> list[Any]:
    if not items:
        return []

    selected: list[Any] = []
    top_count = _item_count(items[0]) or 1
    for index, item in enumerate(items):
        count = _item_count(item)
        frequency = _item_frequency(item)
        has_stats = count is not None or frequency is not None
        keep = not has_stats and index < 12
        keep = keep or (count is not None and count >= max(3, top_count * 0.25))
        keep = keep or (frequency is not None and frequency >= 0.15)
        if not keep:
            break
        selected.append(item)
        if len(selected) >= 30:
            break
    return selected


def _auto_max_output_tags(tags: list[str]) -> int:
    if len(tags) <= 12:
        return max(8, len(tags))
    if len(tags) <= 20:
        return 12
    return 16


def _item_count(item: Any) -> int | None:
    if not isinstance(item, dict):
        return None
    count = item.get("count")
    return count if isinstance(count, int) else None


def _item_frequency(item: Any) -> float | None:
    if not isinstance(item, dict):
        return None
    frequency = item.get("frequency")
    if isinstance(frequency, int | float):
        return float(frequency)
    return None
