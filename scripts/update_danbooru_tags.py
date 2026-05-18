from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from danbooru_prompt_compiler.compiler import TAG_DICT_PATH
from danbooru_prompt_compiler.tag_dictionary import (
    DEFAULT_DANBOORU_BASE_URL,
    DEFAULT_TAG_LIMIT,
    fetch_danbooru_tags,
    write_tag_dictionary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Danbooru tags into data/tags.json.")
    parser.add_argument(
        "--output",
        type=Path,
        default=TAG_DICT_PATH,
        help="Path to write the tag dictionary JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_TAG_LIMIT,
        help="Maximum number of popular tags to download.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_DANBOORU_BASE_URL,
        help="Danbooru-compatible base URL.",
    )
    args = parser.parse_args()

    try:
        tags = fetch_danbooru_tags(max_tags=args.limit, base_url=args.base_url)
    except (httpx.HTTPError, ValueError) as exc:
        raise SystemExit(f"Error: failed to download Danbooru tags: {exc}") from exc

    write_tag_dictionary(args.output, tags)
    print(f"Wrote {len(tags)} tags to {args.output}")


if __name__ == "__main__":
    main()
