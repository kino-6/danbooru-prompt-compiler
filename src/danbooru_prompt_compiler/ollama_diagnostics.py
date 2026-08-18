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


def restart_ollama_model(
    base_url: str,
    model: str,
    *,
    client: httpx.Client | None = None,
    load_timeout: float = 180.0,
) -> str:
    """Unload a wedged model and load it again, reporting what happened.

    Ollama keeps a model resident between requests; when that instance stops
    answering, the fix is to drop it with ``keep_alive: 0`` and let the next
    request load a fresh one.
    """
    if not model.strip():
        return "モデル名が空です。復旧するモデルを指定してください。"

    owns_client = client is None
    http_client = client or httpx.Client(timeout=load_timeout)
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    try:
        unload = http_client.post(endpoint, json={"model": model, "keep_alive": 0})
        unload.raise_for_status()
        reload_response = http_client.post(
            endpoint,
            json={"model": model, "prompt": "ok", "stream": False},
        )
        reload_response.raise_for_status()
        return (
            f"`{model}` を解放して読み込み直しました。もう一度実行してください。"
        )
    except (httpx.ConnectError, httpx.TimeoutException):
        return (
            "Ollamaが応答しません。Ollama自体を再起動してから、もう一度お試しください。\n\n"
            "```text\nollama ps\nollama stop " + model + "\nollama serve\n```"
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return (
                f"`{model}` がインストールされていません。\n\n"
                f"```text\nollama pull {model}\n```"
            )
        return f"復旧に失敗しました: {exc}"
    except httpx.HTTPError as exc:
        return f"復旧に失敗しました: {exc}"
    finally:
        if owns_client:
            http_client.close()
