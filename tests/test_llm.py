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


def test_ollama_client_uses_configured_timeout(monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(httpx, "Client", FakeHTTPClient)

    schema = {"type": "object", "properties": {"action": {"type": "string"}}}
    client = OllamaClient(
        timeout=600.0,
        temperature=0.0,
        json_schema=schema,
        think=False,
    )
    response = client.generate(LLMRequest(prompt="test", variants=1))

    assert response.outputs == ["1girl, solo"]
    assert FakeHTTPClient.seen_timeout == 600.0
    assert FakeHTTPClient.seen_json is not None
    assert FakeHTTPClient.seen_json["options"] == {"temperature": 0.0}
    assert FakeHTTPClient.seen_json["format"] == schema
    assert FakeHTTPClient.seen_json["think"] is False
