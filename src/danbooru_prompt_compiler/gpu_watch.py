"""Waiting for the card when something else is using it.

A run that starts while another program holds the GPU does not fail; Ollama
pushes layers onto the CPU instead and the run crawls. Waiting a little is
better than a slow answer nobody asked for.

The obvious measure does not work. Per-process VRAM reads `[N/A]` under Windows
WDDM, so the memory another program holds cannot be read directly, and the total
includes whatever Ollama is holding for us - waiting on that would be waiting on
ourselves, forever. Ollama says what it holds, so the total minus that is what
everyone else holds, which is the number worth waiting on.

Every part of this is optional: no `nvidia-smi`, no NVIDIA card, or an Ollama
that will not answer all mean the run simply starts.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Callable

import httpx

MIB = 1024 * 1024
# What another program can hold before a run is worth delaying. A desktop and a
# browser sit at a bit over a gigabyte on the machine this was written on, so
# the line is well clear of them.
DEFAULT_FOREIGN_LIMIT_MIB = 4096
DEFAULT_WAIT_TIMEOUT = 120.0
DEFAULT_POLL_SECONDS = 3.0


def gpu_memory_used_mib() -> int | None:
    """Total VRAM in use across the card, or None when nothing can say."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    # More than one card: the first is the one Ollama uses by default.
    first = output.strip().splitlines()[0] if output.strip() else ""
    try:
        return int(first.strip())
    except ValueError:
        return None


def ollama_vram_mib(base_url: str, *, client: httpx.Client | None = None) -> int:
    """What Ollama is holding on the card right now, in MiB."""
    owns_client = client is None
    http_client = client or httpx.Client(timeout=5.0)
    try:
        response = http_client.get(f"{base_url.rstrip('/')}/api/ps")
        response.raise_for_status()
        models = response.json().get("models") or []
    except (httpx.HTTPError, ValueError):
        # Unknown is not zero: claiming Ollama holds nothing would blame its own
        # memory on somebody else and wait on ourselves.
        return -1
    finally:
        if owns_client:
            http_client.close()
    return sum(int(model.get("size_vram") or 0) for model in models) // MIB


def foreign_vram_mib(base_url: str, *, client: httpx.Client | None = None) -> int | None:
    """VRAM held by anything that is not Ollama, or None when unknowable."""
    used = gpu_memory_used_mib()
    if used is None:
        return None
    ours = ollama_vram_mib(base_url, client=client)
    if ours < 0:
        return None
    return max(used - ours, 0)


def wait_for_gpu(
    base_url: str,
    *,
    limit_mib: int = DEFAULT_FOREIGN_LIMIT_MIB,
    timeout: float = DEFAULT_WAIT_TIMEOUT,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> str:
    """Hold until the card is free enough, and say what happened if it waited.

    The wait is bounded and the run always goes ahead: a slow answer beats no
    answer, and a card that never frees up would otherwise hang the page.
    """
    if limit_mib <= 0:
        return ""
    # A model already on the card is not going to be loaded again, so there is
    # nothing a wait could improve - and waiting anyway would delay the fastest
    # runs there are, the warm ones.
    if ollama_vram_mib(base_url, client=client) > 0:
        return ""
    foreign = foreign_vram_mib(base_url, client=client)
    if foreign is None or foreign <= limit_mib:
        return ""

    started = now()
    while True:
        sleep(poll_seconds)
        waited = now() - started
        foreign = foreign_vram_mib(base_url, client=client)
        if foreign is None or foreign <= limit_mib:
            return f"他タスクのGPU使用が収まるまで{waited:.0f}秒待機しました。"
        if waited >= timeout:
            return (
                f"他タスクがGPUを{foreign / 1024:.1f}GB使用中です。"
                f"{waited:.0f}秒待ちましたが空かないため、そのまま実行します。"
            )
