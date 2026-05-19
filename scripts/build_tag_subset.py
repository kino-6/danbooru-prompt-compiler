from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from danbooru_prompt_compiler.tag_dictionary import DEFAULT_DANBOORU_BASE_URL
from danbooru_prompt_compiler.tag_subset import (
    DEFAULT_MIN_COUNT,
    DEFAULT_POST_LIMIT,
    DEFAULT_SUBSET_TAG_LIMIT,
    fetch_post_tag_subset,
    write_tag_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Danbooru tag subset from posts matching seed tags."
    )
    parser.add_argument("seed_tags", nargs="+", help="Seed Danbooru tags, such as shrine rain.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the subset JSON.",
    )
    parser.add_argument(
        "--posts",
        type=int,
        default=DEFAULT_POST_LIMIT,
        help="Maximum number of matching posts to sample.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help="Minimum number of sampled posts a tag must appear in.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SUBSET_TAG_LIMIT,
        help="Maximum number of subset tags to write.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_DANBOORU_BASE_URL,
        help="Danbooru-compatible base URL.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=20,
        help="Number of top tags with counts to print after writing.",
    )
    args = parser.parse_args()
    sampled_posts = 0

    try:
        with tqdm(total=args.posts, unit="post", desc="Fetching posts", file=sys.stdout) as progress:
            tag_counts = fetch_post_tag_subset(
                args.seed_tags,
                max_posts=args.posts,
                min_count=args.min_count,
                max_tags=args.limit,
                base_url=args.base_url,
                progress=progress.update,
            )
            sampled_posts = progress.n
    except (httpx.HTTPError, ValueError) as exc:
        raise SystemExit(f"Error: failed to build Danbooru tag subset: {exc}") from exc

    write_tag_subset(
        args.output,
        seed_tags=args.seed_tags,
        tag_counts=tag_counts,
        sampled_posts=sampled_posts,
        min_count=args.min_count,
        tag_limit=args.limit,
        base_url=args.base_url,
    )
    print(f"Wrote {len(tag_counts)} tags to {args.output}")
    if args.preview > 0 and tag_counts:
        print("Top tags:")
        for tag, count in tag_counts.most_common(args.preview):
            frequency = count / sampled_posts if sampled_posts else 0
            print(f"  {tag}: {count} ({frequency:.1%})")


if __name__ == "__main__":
    main()
