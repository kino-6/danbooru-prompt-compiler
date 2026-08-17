from __future__ import annotations

import pytest


gradio = pytest.importorskip("gradio")

from danbooru_prompt_compiler.webui import (  # noqa: E402
    accept_dropped_image,
    adopt_candidate,
    build_app,
    prepend_history,
    prompt_box_values,
)


def _components_by_label(app) -> dict[str, dict]:
    return {
        component.get("props", {}).get("label"): component
        for component in app.config["components"]
        if component.get("props", {}).get("label")
    }


def test_webui_exposes_human_correction_and_vision_controls() -> None:
    app = build_app()
    components = _components_by_label(app)

    assert components["画像タグ"]["props"]["interactive"]
    action_values = {
        value for _label, value in components["操作種別"]["props"]["choices"]
    }
    assert action_values == {"auto", "tag_image", "compile", "edit", "next_panel"}
    assert components["ポーズ・位置関係の解析にVLMを使う"]["props"]["value"] is False
    assert components["プライベート画像URLを許可"]["props"]["value"] is False
    assert components["画像"]["props"]["interactive"] is True
    assert components["画像"]["props"]["sources"] == ["upload", "clipboard"]
    assert components["出力数"]["props"]["value"] == 4
    assert components["不要な画像タグを除外"]["props"]["value"] is True
    assert (
        components["除外ルール（カンマ区切り、*使用可）"]["props"]["value"]
        == "simple_background, halftone, *_background"
    )
    filter_accordion = next(
        component
        for component in app.config["components"]
        if component.get("type") == "accordion"
        and component.get("props", {}).get("label") == "画像タグフィルター"
    )
    assert filter_accordion["props"]["open"] is False
    assert [value for _label, value in components["出力数"]["props"]["choices"]] == [
        1,
        2,
        3,
        4,
    ]


def test_drop_acceptance_keeps_active_path_and_clears_url() -> None:
    active, image_url = accept_dropped_image("C:/images/second.png")

    assert active == "C:/images/second.png"
    assert image_url == ""


def test_secondary_webui_sections_are_folded() -> None:
    app = build_app()
    accordions = {
        component["props"]["label"]: component["props"]["open"]
        for component in app.config["components"]
        if component.get("type") == "accordion"
    }

    assert accordions["URLから読み込む（補助）"] is False
    assert accordions["既存プロンプトから編集（任意）"] is False
    assert accordions["詳細設定"] is False
    assert accordions["画像タグの確認・修正"] is False
    assert accordions["候補の採用・履歴"] is False
    assert accordions["実行情報"] is False


def test_image_workspace_routes_dropped_urls_to_url_loader() -> None:
    app = build_app()
    url_drop_dependencies = [
        dependency
        for dependency in app.config["dependencies"]
        if "text/uri-list" in (dependency.get("js") or "")
    ]

    assert len(url_drop_dependencies) == 1
    script = url_drop_dependencies[0]["js"]
    assert 'document.getElementById("image-workspace")' in script
    assert 'querySelector("#image-url-input textarea' in script
    assert 'querySelector("#image-url-load-button button")' in script


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


def test_prompt_candidates_get_independent_copyable_boxes() -> None:
    values = prompt_box_values(["first prompt", "second prompt"])

    assert values == ["first prompt", "second prompt", "", ""]
    components = _components_by_label(build_app())
    for index in range(1, 5):
        props = components[f"出力プロンプト {index}"]["props"]
        assert "copy" in props["buttons"]
        assert props["interactive"] is True
        assert props["visible"] is True
        assert props["elem_id"] == f"prompt-output-{index}"
