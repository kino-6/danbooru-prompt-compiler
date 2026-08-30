from __future__ import annotations

from unittest import mock

import httpx

from danbooru_prompt_compiler import gpu_watch
from danbooru_prompt_compiler.gpu_watch import foreign_vram_mib, wait_for_gpu


def _ps(models):
    def handler(request):
        return httpx.Response(200, json={"models": models}, request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_what_ollama_holds_is_not_counted_against_us() -> None:
    # Per-process VRAM reads [N/A] under Windows, so the total is all there is,
    # and it includes our own model. Waiting on that would be waiting on
    # ourselves, forever.
    with mock.patch.object(gpu_watch, "gpu_memory_used_mib", lambda: 12000):
        with _ps([{"size_vram": 10 * 1024 * 1024 * 1024}]) as client:
            assert foreign_vram_mib("http://ollama.test", client=client) == 12000 - 10240


def test_a_card_nothing_can_report_on_is_not_waited_for() -> None:
    with mock.patch.object(gpu_watch, "gpu_memory_used_mib", lambda: None):
        with _ps([]) as client:
            assert foreign_vram_mib("http://ollama.test", client=client) is None
            assert wait_for_gpu("http://ollama.test", client=client) == ""


def test_an_ollama_that_will_not_answer_is_unknown_rather_than_zero() -> None:
    def broken(request):
        raise httpx.ConnectError("no ollama", request=request)

    with mock.patch.object(gpu_watch, "gpu_memory_used_mib", lambda: 12000):
        with httpx.Client(transport=httpx.MockTransport(broken)) as client:
            # Calling it zero would blame Ollama's own memory on somebody else.
            assert foreign_vram_mib("http://ollama.test", client=client) is None


def test_a_quiet_card_starts_the_run_without_waiting() -> None:
    slept: list[float] = []
    with mock.patch.object(gpu_watch, "gpu_memory_used_mib", lambda: 11000):
        with _ps([{"size_vram": 10 * 1024 * 1024 * 1024}]) as client:
            note = wait_for_gpu(
                "http://ollama.test",
                limit_mib=4096,
                client=client,
                sleep=slept.append,
            )

    assert note == ""
    assert slept == []


def test_a_busy_card_is_waited_for_and_the_wait_is_reported() -> None:
    # Ollama holds nothing, so a load is coming and the wait is worth making.
    # Busy, still busy, then free.
    used = iter([14000, 14000, 3000])
    clock = iter([0.0, 3.0, 6.0])
    with mock.patch.object(gpu_watch, "gpu_memory_used_mib", lambda: next(used)):
        with _ps([]) as client:
            note = wait_for_gpu(
                "http://ollama.test",
                limit_mib=4096,
                client=client,
                sleep=lambda _s: None,
                now=lambda: next(clock),
            )

    assert "待機しました" in note


def test_a_card_that_never_frees_up_does_not_hold_the_run_forever() -> None:
    clock = iter([0.0, 200.0])
    with mock.patch.object(gpu_watch, "gpu_memory_used_mib", lambda: 15000):
        with _ps([]) as client:
            note = wait_for_gpu(
                "http://ollama.test",
                limit_mib=4096,
                timeout=120.0,
                client=client,
                sleep=lambda _s: None,
                now=lambda: next(clock),
            )

    # A slow answer beats no answer, so the run goes ahead and says why.
    assert "そのまま実行します" in note


def test_a_threshold_of_zero_turns_the_wait_off_entirely() -> None:
    def explode(*_args, **_kwargs):
        raise AssertionError("a disabled wait must not probe the card")

    with mock.patch.object(gpu_watch, "gpu_memory_used_mib", explode):
        assert wait_for_gpu("http://ollama.test", limit_mib=0) == ""


def test_a_model_already_on_the_card_is_not_waited_for() -> None:
    def explode():
        raise AssertionError("a warm model needs no wait, so nothing is probed")

    with mock.patch.object(gpu_watch, "gpu_memory_used_mib", explode):
        with _ps([{"size_vram": 10 * 1024 * 1024 * 1024}]) as client:
            # Nothing is going to be loaded, so a wait could only delay the
            # fastest runs there are.
            assert wait_for_gpu("http://ollama.test", limit_mib=4096, client=client) == ""
