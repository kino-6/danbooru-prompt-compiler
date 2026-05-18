from __future__ import annotations

import httpx

from danbooru_prompt_compiler.tag_dictionary import (
    fetch_danbooru_tags,
    load_or_fetch_tag_dictionary,
)


def test_fetch_danbooru_tags_paginates_and_normalizes_names() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        page = request.url.params["page"]
        if page == "1":
            return httpx.Response(
                200,
                json=[
                    {"name": "1girl"},
                    {"name": "looking at viewer"},
                ],
            )
        return httpx.Response(200, json=[{"name": "solo"}])

    client = httpx.Client(transport=httpx.MockTransport(handler))

    tags = fetch_danbooru_tags(max_tags=3, base_url="https://example.test", page_size=2, client=client)

    assert tags == ["1girl", "looking_at_viewer", "solo"]
    assert len(requests) == 2
    assert requests[0].url.params["search[order]"] == "count"


def test_load_or_fetch_tag_dictionary_writes_file_when_missing(tmp_path) -> None:
    path = tmp_path / "tags.json"

    tags = load_or_fetch_tag_dictionary(path, max_tags=2, fetch_tags=lambda max_tags: ["1girl", "solo"])

    assert tags == {"1girl", "solo"}
    assert path.read_text(encoding="utf-8").strip() == '[\n  "1girl",\n  "solo"\n]'


def test_load_or_fetch_tag_dictionary_reads_existing_file(tmp_path) -> None:
    path = tmp_path / "tags.json"
    path.write_text('["existing"]', encoding="utf-8")

    tags = load_or_fetch_tag_dictionary(
        path,
        fetch_tags=lambda max_tags: ["downloaded"],
    )

    assert tags == {"existing"}
