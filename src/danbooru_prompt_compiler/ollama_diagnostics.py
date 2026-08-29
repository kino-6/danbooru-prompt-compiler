from __future__ import annotations

from dataclasses import dataclass

import httpx


# One run may touch the router, the tag compiler, the vision model, and the
# prose model. Ollama keeps a model resident only while it fits, so once the
# selection outgrows the card it evicts and reloads mid-run - about a minute and
# a half for a 26B vision model. The budget is the common 16 GB card; it is a
# reporting threshold, not a limit anything enforces.
RESIDENT_BUDGET_BYTES = 16 * 1024**3


@dataclass(frozen=True)
class OllamaDiagnostic:
    reachable: bool
    installed_models: list[str]
    missing_models: list[str]
    message: str
    resident_bytes: int = 0


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
        sizes = {
            str(model.get("name") or model.get("model")): int(model.get("size") or 0)
            for model in payload.get("models", [])
            if model.get("name") or model.get("model")
        }
        required = list(dict.fromkeys(model for model in required_models if model))
        missing = [model for model in required if model not in installed]
        resident_bytes = sum(sizes.get(model, 0) for model in required)
        if missing:
            commands = "\n".join(f"ollama pull {model}" for model in missing)
            message = (
                "Ollamaには接続できましたが、必要なモデルがありません。\n\n"
                f"```text\n{commands}\n```"
            )
        else:
            message = f"Ollama接続OK。必要なモデル {len(required)} 件を確認しました。"
        crowding = _residency_warning(required, sizes, resident_bytes)
        if crowding:
            message = f"{message}\n\n{crowding}"
        return OllamaDiagnostic(
            reachable=True,
            installed_models=installed,
            missing_models=missing,
            message=message,
            resident_bytes=resident_bytes,
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


def _residency_warning(
    required: list[str],
    sizes: dict[str, int],
    resident_bytes: int,
) -> str:
    """Say so when the chosen models cannot all stay loaded at once."""
    if len(required) < 2 or resident_bytes <= RESIDENT_BUDGET_BYTES:
        return ""
    heaviest = max(required, key=lambda model: sizes.get(model, 0))
    # Naming the heaviest alone is the honest reading: a 1.3GB router is not
    # what broke the budget, it is what gets evicted when the big one loads.
    return (
        f"選択中のモデルは合計 {_gigabytes(resident_bytes)} で、"
        f"VRAM の目安 {_gigabytes(RESIDENT_BUDGET_BYTES)} を超えています。"
        f"最大の `{heaviest}`（{_gigabytes(sizes.get(heaviest, 0))}）と"
        f"残り {len(required) - 1} 件は同時に常駐できないため、"
        "実行のたびに入れ替えの読み込み時間がかかります。"
    )


def _gigabytes(size: int) -> str:
    return f"{size / 1024**3:.1f}GB"


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
