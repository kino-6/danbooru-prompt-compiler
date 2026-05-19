from __future__ import annotations

import httpx

from danbooru_prompt_compiler.tag_subset import (
    fetch_post_tag_subset,
    read_tag_subset,
    select_tag_subset,
    write_tag_subset,
)


def test_fetch_post_tag_subset_counts_general_tags_from_matching_posts() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=[
                {"tag_string_general": "1girl solo shrine rain torii long_hair"},
                {"tag_string_general": "1girl solo shrine rain torii standing"},
                {"tag_string_general": "1girl solo shrine rain umbrella standing"},
            ],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    progress_updates: list[int] = []

    counts = fetch_post_tag_subset(
        ["shrine", "rain"],
        max_posts=3,
        min_count=2,
        max_tags=10,
        base_url="https://example.test",
        client=client,
        progress=progress_updates.append,
    )

    assert list(counts.items()) == [
        ("1girl", 3),
        ("solo", 3),
        ("standing", 2),
        ("torii", 2),
    ]
    assert requests[0].url.params["tags"] == "shrine rain"
    assert progress_updates == [3]


def test_write_and_read_tag_subset(tmp_path) -> None:
    path = tmp_path / "subset.json"

    write_tag_subset(
        path,
        seed_tags=["shrine"],
        tag_counts={"torii": 4, "1girl": 5},
        sampled_posts=10,
        min_count=2,
        tag_limit=100,
        base_url="https://example.test",
    )

    assert read_tag_subset(path) == ["1girl", "torii"]

    written = path.read_text(encoding="utf-8")
    assert '"sampled_posts": 10' in written
    assert '"frequency": 0.5' in written
    assert '"frequency": 0.4' in written


def test_select_tag_subset_auto_uses_frequency_dropoff(tmp_path) -> None:
    path = tmp_path / "subset.json"
    path.write_text(
        """
{
  "tags": [
    {"name": "top", "count": 100, "frequency": 1.0},
    {"name": "strong", "count": 30, "frequency": 0.3},
    {"name": "weak", "count": 10, "frequency": 0.1}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    selection = select_tag_subset(path, max_tags=None)

    assert selection.tags == ["top", "strong"]
    assert selection.subset_limit == 2
    assert selection.max_output_tags == 8
