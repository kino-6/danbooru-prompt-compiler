from __future__ import annotations

import httpx

from danbooru_prompt_compiler.ollama_diagnostics import (
    check_ollama,
    format_ollama_error,
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
