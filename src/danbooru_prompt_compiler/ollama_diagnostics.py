from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class OllamaDiagnostic:
    reachable: bool
    installed_models: list[str]
    missing_models: list[str]
    message: str


def check_ollama(
    base_url: str,
    required_models: list[str],
    *,
    client: httpx.Client | None = None,
) -> OllamaDiagnostic:
    owns_client = client is None
    http_client = client or httpx.Client(timeout=5.0)
    try:
        response = http_client.get(f"{base_url.rstrip('/')}/api/tags")
        response.raise_for_status()
        payload = response.json()
        installed = sorted(
            {
                str(model.get("name") or model.get("model"))
                for model in payload.get("models", [])
                if model.get("name") or model.get("model")
            }
        )
        required = list(dict.fromkeys(model for model in required_models if model))
        missing = [model for model in required if model not in installed]
        if missing:
            commands = "\n".join(f"ollama pull {model}" for model in missing)
            message = (
                "Ollamaには接続できましたが、必要なモデルがありません。\n\n"
                f"```text\n{commands}\n```"
            )
        else:
            message = f"Ollama接続OK。必要なモデル {len(required)} 件を確認しました。"
        return OllamaDiagnostic(
            reachable=True,
            installed_models=installed,
            missing_models=missing,
            message=message,
        )
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        return OllamaDiagnostic(
            reachable=False,
            installed_models=[],
            missing_models=list(dict.fromkeys(required_models)),
            message=(
                "Ollamaに接続できません。Ollamaを起動してから再確認してください。\n\n"
                "```text\nollama serve\n```"
            ),
        )
    except httpx.HTTPError as exc:
        return OllamaDiagnostic(
            reachable=False,
            installed_models=[],
            missing_models=list(dict.fromkeys(required_models)),
            message=f"Ollamaの確認中にHTTPエラーが発生しました: {exc}",
        )
    finally:
        if owns_client:
            http_client.close()


def format_ollama_error(exc: Exception, configured_models: list[str]) -> str:
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)):
        return (
            "Ollamaに接続できません。`ollama serve` で起動し、"
            "詳細設定の「Ollama接続確認」を実行してください。"
        )
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
        commands = " / ".join(
            f"ollama pull {model}" for model in dict.fromkeys(configured_models) if model
        )
        return f"Ollamaモデルが見つかりません。{commands}"
    return str(exc)
