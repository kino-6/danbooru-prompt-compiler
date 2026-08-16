from __future__ import annotations

import pytest


gradio = pytest.importorskip("gradio")

from danbooru_prompt_compiler.webui import (  # noqa: E402
    adopt_candidate,
    build_app,
    prepend_history,
)


def _components_by_label(app) -> dict[str, dict]:
    return {
        component.get("props", {}).get("label"): component
        for component in app.config["components"]
        if component.get("props", {}).get("label")
    }


def test_webui_exposes_human_correction_and_vision_controls() -> None:
    components = _components_by_label(build_app())

    assert components["画像タグ（修正して再実行できます）"]["props"]["interactive"]
    action_values = {
        value for _label, value in components["操作種別"]["props"]["choices"]
    }
    assert action_values == {"auto", "tag_image", "compile", "edit", "next_panel"}
    assert components["ポーズ・位置関係の解析にVLMを使う"]["props"]["value"] is False
    assert components["プライベート画像URLを許可"]["props"]["value"] is False


def test_webui_has_cancel_dependencies_for_run_and_submit() -> None:
    app = build_app()
    cancel_dependencies = [
        dependency
        for dependency in app.config["dependencies"]
        if dependency.get("cancels")
    ]

    cancelled_ids = {
        dependency_id
        for dependency in cancel_dependencies
        for dependency_id in dependency["cancels"]
    }
    assert len(cancelled_ids) == 2


def test_webui_exposes_candidate_adoption_and_history_controls() -> None:
    components = _components_by_label(build_app())

    assert "生成候補" in components
    assert "実行履歴（新しい順・最大20件）" in components
    assert "Ollama診断" in components
    assert adopt_candidate("1girl, rain") == "1girl, rain"


def test_history_is_newest_first_and_bounded() -> None:
    history: list[dict[str, str]] = []
    for index in range(25):
        history = prepend_history(
            history,
            action="edit",
            instruction=f"instruction {index}",
            output=f"output {index}",
        )

    assert len(history) == 20
    assert history[0]["instruction"] == "instruction 24"
    assert history[-1]["instruction"] == "instruction 5"
