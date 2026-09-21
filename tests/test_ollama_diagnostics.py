from __future__ import annotations

import json

import httpx

from danbooru_prompt_compiler.ollama_diagnostics import (
    check_ollama,
    format_ollama_error,
    restart_ollama_model,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_diagnostic_reports_healthy_models() -> None:
    def handler(request):
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:1.7b"}, {"name": "qwen3-vl:8b"}]},
            request=request,
        )

    with _client(handler) as client:
        result = check_ollama(
            "http://ollama.test",
            ["qwen3:1.7b", "qwen3-vl:8b"],
            client=client,
        )

    assert result.reachable
    assert result.missing_models == []
    assert "接続OK" in result.message


def test_ollama_diagnostic_reports_missing_model_with_pull_command() -> None:
    def handler(request):
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen3:1.7b"}]},
            request=request,
        )

    with _client(handler) as client:
        result = check_ollama(
            "http://ollama.test",
            ["qwen3:1.7b", "qwen3-vl:8b"],
            client=client,
        )

    assert result.reachable
    assert result.missing_models == ["qwen3-vl:8b"]
    assert "ollama pull qwen3-vl:8b" in result.message


def test_ollama_diagnostic_reports_unreachable_server() -> None:
    def handler(request):
        raise httpx.ConnectError("offline", request=request)

    with _client(handler) as client:
        result = check_ollama(
            "http://ollama.test",
            ["qwen3:1.7b"],
            client=client,
        )

    assert not result.reachable
    assert "ollama serve" in result.message


def test_ollama_404_error_lists_pull_commands() -> None:
    request = httpx.Request("POST", "http://ollama.test/api/generate")
    response = httpx.Response(404, request=request)
    error = httpx.HTTPStatusError("missing", request=request, response=response)

    message = format_ollama_error(error, ["qwen3:1.7b", "qwen3:1.7b"])

    assert message.count("ollama pull qwen3:1.7b") == 1


def test_restart_reloads_a_wedged_model() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    message = restart_ollama_model("http://localhost:11434", "qwen3-vl:8b", client=client)

    # Unload first, then let the next request load a fresh instance.
    assert calls[0] == {"model": "qwen3-vl:8b", "keep_alive": 0}
    assert calls[1]["model"] == "qwen3-vl:8b"
    assert calls[1]["stream"] is False
    assert "読み込み直しました" in message


def test_restart_explains_an_unreachable_ollama() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    message = restart_ollama_model("http://localhost:11434", "qwen3-vl:8b", client=client)

    assert "ollama serve" in message


def test_restart_explains_a_missing_model() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    message = restart_ollama_model("http://localhost:11434", "qwen3-vl:8b", client=client)

    assert "ollama pull qwen3-vl:8b" in message


def test_restart_rejects_an_empty_model_name() -> None:
    assert "モデル名が空です" in restart_ollama_model("http://localhost:11434", "  ")


def test_diagnostic_warns_when_the_selection_cannot_stay_resident() -> None:
    def handler(request):
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3:1.7b", "size": 1_400_000_000},
                    {"name": "unseen-gemma4:26b", "size": 18_000_000_000},
                    {"name": "qwen3-vl:8b", "size": 6_100_000_000},
                ]
            },
            request=request,
        )

    with _client(handler) as client:
        result = check_ollama(
            "http://ollama.test",
            ["qwen3:1.7b", "unseen-gemma4:26b", "qwen3-vl:8b"],
            client=client,
        )

    assert result.reachable
    assert result.missing_models == []
    assert "接続OK" in result.message
    # Only the heaviest is named. A 1.3GB router did not break the budget, and
    # naming it as the culprit would send the reader after the wrong setting.
    assert "同時に常駐できない" in result.message
    assert "`unseen-gemma4:26b`" in result.message
    assert "残り 2 件" in result.message
    assert "`qwen3:1.7b`" not in result.message
    assert "`qwen3-vl:8b`" not in result.message


def test_a_selection_that_fits_is_reported_without_a_warning() -> None:
    def handler(request):
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3:1.7b", "size": 1_400_000_000},
                    {"name": "qwen3-vl:8b", "size": 6_100_000_000},
                ]
            },
            request=request,
        )

    with _client(handler) as client:
        result = check_ollama(
            "http://ollama.test",
            ["qwen3:1.7b", "qwen3-vl:8b"],
            client=client,
        )

    assert "常駐" not in result.message
    assert result.resident_bytes == 7_500_000_000
