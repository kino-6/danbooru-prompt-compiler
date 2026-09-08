from unittest import mock

from danbooru_prompt_compiler.llm import OllamaClient
from danbooru_prompt_compiler.models import LLMRequest


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"response": "1girl, solo"}


class FakeHTTPClient:
    seen_timeout: float | None = None
    seen_json: dict[str, object] | None = None

    def __init__(self, *, timeout: float) -> None:
        FakeHTTPClient.seen_timeout = timeout

    def __enter__(self) -> "FakeHTTPClient":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def post(self, url: str, *, json: dict[str, object]) -> FakeResponse:
        FakeHTTPClient.seen_json = json
        return FakeResponse()


def test_ollama_client_uses_configured_timeout(monkeypatch, tmp_path) -> None:
    import httpx

    monkeypatch.setattr(httpx, "Client", FakeHTTPClient)

    schema = {"type": "object", "properties": {"action": {"type": "string"}}}
    client = OllamaClient(
        timeout=600.0,
        temperature=0.0,
        json_schema=schema,
        think=False,
    )
    image_path = tmp_path / "image.bin"
    image_path.write_bytes(b"image bytes")
    response = client.generate(
        LLMRequest(prompt="test", variants=1, image_paths=[str(image_path)])
    )

    assert response.outputs == ["1girl, solo"]
    assert FakeHTTPClient.seen_timeout == 600.0
    assert FakeHTTPClient.seen_json is not None
    assert FakeHTTPClient.seen_json["options"] == {"temperature": 0.0}
    assert FakeHTTPClient.seen_json["format"] == schema
    assert FakeHTTPClient.seen_json["think"] is False
    assert FakeHTTPClient.seen_json["images"] == ["aW1hZ2UgYnl0ZXM="]


def test_a_cpu_only_client_asks_ollama_to_keep_the_model_off_the_card() -> None:
    """Ollama reads num_gpu as layers to offload, so zero is CPU-only.

    Slower than the GPU, but it runs alongside an image generator instead of
    queueing behind one.
    """
    sent: list[dict] = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok"}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(self, _url, json):
            sent.append(json)
            return _Response()

    client = OllamaClient(model="qwen3:1.7b", cpu_only=True, temperature=0.4)
    with mock.patch("httpx.Client", lambda **_kwargs: _Client()):
        client.generate(LLMRequest(prompt="hello", variants=1))

    assert sent[0]["options"]["num_gpu"] == 0
    # And it does not lose the setting it shares the options block with.
    assert sent[0]["options"]["temperature"] == 0.4


def test_a_normal_client_says_nothing_about_the_gpu() -> None:
    sent: list[dict] = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "ok"}

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(self, _url, json):
            sent.append(json)
            return _Response()

    with mock.patch("httpx.Client", lambda **_kwargs: _Client()):
        OllamaClient(model="qwen3:1.7b").generate(LLMRequest(prompt="hi", variants=1))

    assert "options" not in sent[0]
