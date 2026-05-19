from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .seed_tags import infer_subset_seed_tags
from .tag_subset import fetch_post_tag_subset, select_tag_subset


@dataclass(frozen=True)
class ReferenceTags:
    tags: list[str]
    max_output_tags: int


def load_reference_tags(
    *,
    tag_subset: Path | None,
    subset_tags: list[str],
    auto_subset: bool,
    scene_description: str,
    edit_instruction: str | None,
    tag_subset_limit: str,
    subset_posts: int,
    subset_min_count: int,
    max_tags: str,
) -> ReferenceTags:
    explicit_sources = sum(bool(source) for source in (tag_subset, subset_tags, auto_subset))
    if explicit_sources > 1:
        raise ValueError("Use only one of --tag-subset, --subset-tags, or --auto-subset.")

    parsed_subset_limit = _parse_auto_int(tag_subset_limit, option_name="--tag-subset-limit")
    parsed_max_tags = _parse_auto_int(max_tags, option_name="--max-tags")
    if tag_subset:
        selection = select_tag_subset(
            tag_subset,
            max_tags=parsed_subset_limit,
            max_output_tags=parsed_max_tags,
        )
        return ReferenceTags(tags=selection.tags, max_output_tags=selection.max_output_tags)

    seed_tags = (
        infer_subset_seed_tags(scene_description, edit_instruction=edit_instruction)
        if auto_subset
        else subset_tags
    )
    if seed_tags:
        live_subset_limit = parsed_subset_limit or 25
        counts = fetch_post_tag_subset(
            seed_tags,
            max_posts=subset_posts,
            min_count=subset_min_count,
            max_tags=live_subset_limit,
        )
        return ReferenceTags(
            tags=list(counts),
            max_output_tags=parsed_max_tags or _auto_max_output_tags(len(counts)),
        )

    return ReferenceTags(tags=[], max_output_tags=parsed_max_tags or 20)


def _parse_auto_int(value: str, *, option_name: str) -> int | None:
    if value == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{option_name} must be an integer or 'auto'.") from exc
    if parsed < 1:
        raise ValueError(f"{option_name} must be at least 1.")
    return parsed


def _auto_max_output_tags(subset_size: int) -> int:
    if subset_size <= 12:
        return max(8, subset_size)
    if subset_size <= 20:
        return 12
    return 16
