from __future__ import annotations

from unittest import mock

import pytest

from danbooru_prompt_compiler import webui
from danbooru_prompt_compiler.web_service import (
    WEB_RUN_FIELDS,
    WebRunRequest,
    WebRunResult,
)
from danbooru_prompt_compiler.webui import recover_vision_model, run_workbench


class RecordingService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.options: dict[str, object] | None = None

    def run(self, *, image_path, on_progress=None, **options) -> WebRunResult:
        if self.error is not None:
            raise self.error
        self.options = {"image_path": image_path, **options}
        if on_progress is not None:
            on_progress("complete", 1.0)
        return WebRunResult(
            action_plan={"action": "edit"},
            inferred_tags="1girl, rain",
            output="1girl, rain",
            status="ok",
            candidates=["1girl, rain", "1girl, night"],
        )


def test_run_workbench_maps_a_request_onto_outputs_without_gradio() -> None:
    service = RecordingService()
    stages: list[str] = []

    outputs = run_workbench(
        service,
        WebRunRequest(
            instruction="夜にして",
            base_prompt="1girl",
            variants=2,
            generate_next_panel=False,
        ),
        [],
        on_progress=lambda stage, _fraction: stages.append(stage),
    )

    assert service.options is not None
    assert service.options["instruction"] == "夜にして"
    assert "image_url" not in service.options
    assert "allow_private_image_urls" not in service.options
    assert outputs.prompts == ["1girl, rain", "1girl, night", "", ""]
    assert outputs.candidates == ["1girl, rain", "1girl, night"]
    assert outputs.history[0]["instruction"] == "夜にして"
    assert stages == ["complete"]


def test_run_workbench_returns_the_image_description_for_editing() -> None:
    class DescribingService(RecordingService):
        def run(self, *, image_path, on_progress=None, **options) -> WebRunResult:
            result = super().run(image_path=image_path, on_progress=on_progress, **options)
            return WebRunResult(
                action_plan=result.action_plan,
                inferred_tags=result.inferred_tags,
                output=result.output,
                status=result.status,
                candidates=result.candidates,
                image_description="石段に立つ少女",
            )

    service = DescribingService()

    outputs = run_workbench(
        service,
        WebRunRequest(
            instruction="夜にして",
            use_vision=True,
            generate_next_panel=False,
        ),
        [],
    )

    assert service.options is not None
    assert service.options["edited_description"] == ""
    assert outputs.image_description == "石段に立つ少女"


class TwoRunService(RecordingService):
    """Answers the primary run and the next-panel follow-up differently."""

    def __init__(self, follow_up_error: Exception | None = None) -> None:
        super().__init__()
        self.follow_up_error = follow_up_error
        self.calls: list[dict[str, object]] = []

    def run(self, *, image_path, on_progress=None, **options) -> WebRunResult:
        self.calls.append({"image_path": image_path, **options})
        if options.get("action_override") == "next_panel":
            if self.follow_up_error is not None:
                raise self.follow_up_error
            return WebRunResult(
                action_plan={"action": "next_panel"},
                inferred_tags="1girl, rain",
                output="panels",
                status="panel ok",
                candidates=["panel_a", "panel_b", "panel_c", "panel_d"],
                image_description="石段に立つ少女",
            )
        return WebRunResult(
            action_plan={"action": "tag_image"},
            inferred_tags="1girl, rain",
            output="1girl, rain",
            status="ok",
            candidates=["current"],
            image_description="石段に立つ少女",
        )


def test_next_panel_follow_up_fills_the_boxes_after_the_current_prompt() -> None:
    service = TwoRunService()

    outputs = run_workbench(service, WebRunRequest(instruction="タグを推測して"), [])

    assert outputs.prompts == ["current", "panel_a", "panel_b", "panel_c"]
    assert outputs.candidates == ["current", "panel_a", "panel_b", "panel_c"]
    assert "次のコマ: 4件" in outputs.status
    # The follow-up continues the prompt in box 1 instead of re-deriving it.
    follow_up = service.calls[1]
    assert follow_up["action_override"] == "next_panel"
    assert follow_up["variants"] == 3
    assert follow_up["edited_tags"] == "1girl, rain"
    assert follow_up["edited_description"] == "石段に立つ少女"


def test_next_panel_follow_up_failure_keeps_the_primary_result() -> None:
    service = TwoRunService(follow_up_error=RuntimeError("model missing"))

    outputs = run_workbench(service, WebRunRequest(instruction="タグを推測して"), [])

    assert outputs.prompts == ["current", "", "", ""]
    assert "次のコマの生成に失敗: model missing" in outputs.status


def test_next_panel_follow_up_is_skipped_for_an_explicit_next_panel_run() -> None:
    service = TwoRunService()

    outputs = run_workbench(
        service,
        WebRunRequest(instruction="", action_override="next_panel"),
        [],
    )

    assert len(service.calls) == 1
    assert outputs.prompts == ["panel_a", "panel_b", "panel_c", "panel_d"]


def test_next_panel_follow_up_is_skipped_for_a_prose_prompt() -> None:
    class SceneService(RecordingService):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def run(self, *, image_path, on_progress=None, **options) -> WebRunResult:
            self.calls += 1
            # Reports a tag action, so only the explicit request override can
            # keep tag panels out of boxes 2-4.
            return WebRunResult(
                action_plan={"action": "tag_image"},
                inferred_tags="1girl, rain",
                output="prose",
                status="ok",
                candidates=["Subject: a young woman"],
            )

    service = SceneService()

    outputs = run_workbench(
        service,
        WebRunRequest(instruction="", action_override="scene_prompt"),
        [],
    )

    # Tag prompts in boxes 2-4 would sit in a different format from box 1.
    assert service.calls == 1
    assert outputs.prompts == ["Subject: a young woman", "", "", ""]


def test_next_panel_follow_up_can_be_switched_off() -> None:
    service = TwoRunService()

    outputs = run_workbench(
        service,
        WebRunRequest(instruction="タグを推測して", generate_next_panel=False),
        [],
    )

    assert len(service.calls) == 1
    assert outputs.prompts == ["current", "", "", ""]


def test_recovering_the_vision_model_clears_the_cached_description() -> None:
    class CachingService(RecordingService):
        def __init__(self) -> None:
            super().__init__()
            self.cleared = 0

        def clear_description_cache(self) -> None:
            self.cleared += 1

    service = CachingService()

    with mock.patch.object(
        webui,
        "restart_ollama_model",
        lambda url, model: f"restarted {model} at {url}",
    ):
        message = recover_vision_model(service, "http://localhost:11434", "qwen3-vl:8b")

    # A stale cached description would hide the recovered model.
    assert service.cleared == 1
    assert message == "restarted qwen3-vl:8b at http://localhost:11434"


def test_run_workbench_keeps_the_edited_description_after_a_failure() -> None:
    outputs = run_workbench(
        RecordingService(error=ValueError("画像を入力してください。")),
        WebRunRequest(instruction="夜にして", edited_description="石段に立つ少女"),
        [],
    )

    assert outputs.status.startswith("Error: ")
    assert outputs.image_description == "石段に立つ少女"


def test_run_workbench_turns_failures_into_a_status_and_keeps_history() -> None:
    history = [{"action": "edit", "instruction": "前回", "output": "1girl"}]

    outputs = run_workbench(
        RecordingService(error=ValueError("画像を入力してください。")),
        WebRunRequest(instruction="夜にして"),
        history,
    )

    assert outputs.status.startswith("Error: ")
    assert "画像を入力してください。" in outputs.status
    assert outputs.prompts == ["", "", "", ""]
    assert outputs.candidates == []
    assert outputs.history == history


gradio = pytest.importorskip("gradio")

from danbooru_prompt_compiler.tag_filter import (  # noqa: E402
    DEFAULT_EXCLUSION_TEXT,
    load_exclusion_text,
)
from danbooru_prompt_compiler.webui import (  # noqa: E402
    accept_dropped_image,
    adopt_candidate,
    build_app,
    prepend_history,
    prompt_box_values,
    restore_default_excluded_tags,
    store_excluded_tags,
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
    description_props = components["画像の説明（VLM）"]["props"]
    assert description_props["interactive"] is True
    assert description_props["elem_id"] == "image-description-editor"
    action_values = {
        value for _label, value in components["操作種別"]["props"]["choices"]
    }
    assert action_values == {
        "auto",
        "tag_image",
        "compile",
        "edit",
        "next_panel",
        "scene_prompt",
    }
    assert components["VLMで画像を説明する"]["props"]["value"] is True
    assert components["プライベート画像URLを許可"]["props"]["value"] is False
    assert components["画像"]["props"]["interactive"] is True
    assert components["画像"]["props"]["sources"] == ["upload"]
    assert components["出力数"]["props"]["value"] == 4
    change_props = components["次のコマの変化量"]["props"]
    assert (change_props["minimum"], change_props["maximum"]) == (0.0, 1.0)
    assert change_props["value"] == 0.5
    assert _folded_ancestor_labels(app, "next-panel-change") == []
    assert components["除外ワードを適用"]["props"]["value"] is True
    assert (
        components["除外ワード（カンマ区切り、*使用可）"]["props"]["value"]
        == load_exclusion_text()
    )
    exclusion_accordion = next(
        component
        for component in app.config["components"]
        if component.get("type") == "accordion"
        and component.get("props", {}).get("label") == "除外ワード"
    )
    assert exclusion_accordion["props"]["open"] is True
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


def _folded_ancestor_labels(app, elem_id: str) -> list[str]:
    """Labels of the collapsed accordions a component is nested inside."""
    components = {component["id"]: component for component in app.config["components"]}
    target = next(
        component_id
        for component_id, component in components.items()
        if component.get("props", {}).get("elem_id") == elem_id
    )

    def walk(node, ancestors):
        if node.get("id") == target:
            return ancestors
        for child in node.get("children", []):
            component = components.get(node.get("id"), {})
            found = walk(
                child,
                [*ancestors, component] if component.get("type") == "accordion" else ancestors,
            )
            if found is not None:
                return found
        return None

    ancestors = walk(app.config["layout"], []) or []
    return [
        ancestor["props"].get("label")
        for ancestor in ancestors
        if ancestor["props"].get("open") is False
    ]


def test_vision_controls_are_visible_without_opening_a_section() -> None:
    app = build_app()
    components = _components_by_label(app)

    # Self-check: the helper does detect a component inside a folded section.
    assert _folded_ancestor_labels(app, "base-prompt-input") == [
        "既存プロンプトから編集（任意）"
    ]
    # The description is worthless if the switch that fills it, the box it
    # lands in, or its recovery command is hidden inside a collapsed section.
    assert _folded_ancestor_labels(app, "image-description-editor") == []
    assert _folded_ancestor_labels(app, "recover-vision-button") == []
    assert "VLMを復旧" in {
        component["props"].get("value")
        for component in app.config["components"]
        if component.get("type") == "button"
    }
    assert components["VLMで画像を説明する"]["props"]["value"] is True
    assert components["画像の説明（VLM）"]["props"]["interactive"] is True


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
    assert 'event.target.closest("#image-workspace")' in script
    assert '"#dropped-image-url-input textarea' in script
    assert '"button#dropped-image-url-button, #dropped-image-url-button button"' in script


def test_run_inputs_follow_the_request_model_order() -> None:
    app = build_app()
    run_dependency = next(
        dependency
        for dependency in app.config["dependencies"]
        if dependency.get("api_name") == "run_prompt_workbench"
    )
    components = {
        component["id"]: component for component in app.config["components"]
    }
    input_labels = [
        components[component_id]["props"].get("label")
        for component_id in run_dependency["inputs"]
    ]

    # One input per request field, plus the trailing session-history state.
    assert len(run_dependency["inputs"]) == len(WEB_RUN_FIELDS) + 1
    assert components[run_dependency["inputs"][-1]]["type"] == "state"
    assert input_labels[WEB_RUN_FIELDS.index("instruction")] == "どうしたい？"
    assert input_labels[WEB_RUN_FIELDS.index("variants")] == "出力数"
    assert input_labels[WEB_RUN_FIELDS.index("edited_description")] == "画像の説明（VLM）"
    assert input_labels[WEB_RUN_FIELDS.index("next_panel_change")] == "次のコマの変化量"
    assert (
        input_labels[WEB_RUN_FIELDS.index("scene_template")]
        == "自然文プロンプトのテンプレート"
    )
    assert input_labels[WEB_RUN_FIELDS.index("scene_model")] == "自然文プロンプト用モデル"
    assert (
        input_labels[WEB_RUN_FIELDS.index("scene_sees_image")]
        == "自然文プロンプトに画像を渡す"
    )
    assert (
        input_labels[WEB_RUN_FIELDS.index("excluded_tags")]
        == "除外ワード（カンマ区切り、*使用可）"
    )


def test_diagnostics_cover_the_prose_model() -> None:
    checked: list[list[str]] = []

    class Diagnostic:
        message = "ok"

    with mock.patch.object(
        webui,
        "check_ollama",
        lambda _url, required: checked.append(required) or Diagnostic(),
    ):
        webui.diagnose_ollama(
            "http://localhost:11434",
            "qwen3:1.7b",
            "qwen3:1.7b",
            "qwen3-vl:8b",
            "gemma3:12b",
            False,
        )

    # A larger prose model is the one most likely still to need an ollama pull.
    assert checked == [["qwen3:1.7b", "qwen3:1.7b", "gemma3:12b"]]


def test_scene_prompt_trigger_and_template_choices_are_exposed() -> None:
    app = build_app()
    components = _components_by_label(app)
    scene_dependency = next(
        dependency
        for dependency in app.config["dependencies"]
        if dependency.get("api_name") == "run_scene_prompt"
    )
    run_dependency = next(
        dependency
        for dependency in app.config["dependencies"]
        if dependency.get("api_name") == "run_prompt_workbench"
    )
    template_props = components["自然文プロンプトのテンプレート"]["props"]
    template_names = [value for _label, value in template_props["choices"]]

    assert "character_sheet" in template_names
    assert "storyboard_panel" in template_names
    assert template_props["value"] in template_names
    assert _folded_ancestor_labels(app, "scene-template") == []
    # The same inputs and outputs as 実行; only the action differs.
    assert scene_dependency["inputs"] == run_dependency["inputs"]
    assert scene_dependency["outputs"] == run_dependency["outputs"]
    assert "自然文プロンプト" in {
        component["props"].get("value")
        for component in app.config["components"]
        if component.get("type") == "button"
    }


def test_pasted_clipboard_image_is_routed_into_the_image_workspace() -> None:
    app = build_app()
    paste_dependencies = [
        dependency
        for dependency in app.config["dependencies"]
        if 'addEventListener("paste"' in (dependency.get("js") or "")
    ]

    assert len(paste_dependencies) == 1
    script = paste_dependencies[0]["js"]
    assert 'item.kind === "file"' in script
    assert '(item.type || "").startsWith("image/")' in script
    assert "'#image-workspace input[type=\"file\"]'" in script
    assert "input.files = transfer.files" in script
    # The paste event carries its own clipboardData, so the permission-prompting
    # navigator.clipboard.read() source stays off.
    assert "clipboard" not in _components_by_label(app)["画像"]["props"]["sources"]


def test_exclusion_words_can_be_saved_and_reset_from_the_ui(tmp_path) -> None:
    store_path = tmp_path / "excluded_tags.json"

    saved, saved_message = store_excluded_tags(
        "Censored, bar_censor, censored", store_path
    )
    reset, reset_message = restore_default_excluded_tags(store_path)

    assert saved == "censored, bar_censor"
    assert "保存" in saved_message
    assert load_exclusion_text(store_path) == DEFAULT_EXCLUSION_TEXT
    assert reset == DEFAULT_EXCLUSION_TEXT
    assert "既定" in reset_message

    buttons = {
        component["props"].get("value")
        for component in build_app().config["components"]
        if component.get("type") == "button"
    }
    assert {"除外ワードを保存", "既定に戻す"} <= buttons


def test_webui_has_cancel_dependencies_for_every_run_trigger() -> None:
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
    run_ids = {
        dependency["id"]
        for dependency in app.config["dependencies"]
        if dependency.get("api_name")
        in {"run_prompt_workbench", "run_next_panel", "run_scene_prompt"}
        or (dependency.get("targets") and dependency.get("trigger") == "submit")
    }
    # 実行, 次のコマ, 自然文プロンプト, and instruction submit.
    assert len(cancelled_ids) == 4
    assert run_ids <= cancelled_ids


def test_next_panel_button_runs_without_an_instruction() -> None:
    app = build_app()
    next_panel_dependency = next(
        dependency
        for dependency in app.config["dependencies"]
        if dependency.get("api_name") == "run_next_panel"
    )
    run_dependency = next(
        dependency
        for dependency in app.config["dependencies"]
        if dependency.get("api_name") == "run_prompt_workbench"
    )
    buttons = {
        component["props"].get("value")
        for component in app.config["components"]
        if component.get("type") == "button"
    }

    assert "次のコマ" in buttons
    # The same inputs and outputs as 実行; only the action differs.
    assert next_panel_dependency["inputs"] == run_dependency["inputs"]
    assert next_panel_dependency["outputs"] == run_dependency["outputs"]


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


def test_diagnostics_cover_the_selected_vision_model() -> None:
    checked: list[list[str]] = []

    class Diagnostic:
        message = "ok"

    with mock.patch.object(
        webui,
        "check_ollama",
        lambda _url, required: checked.append(required) or Diagnostic(),
    ):
        webui.diagnose_ollama(
            "http://localhost:11434",
            "qwen3:1.7b",
            "qwen3:1.7b",
            "unseen-gemma4:26b",
            "gemma3:12b",
            True,
        )

    # Whatever the dropdown ended up holding is what the run will ask for, so it
    # is what the check has to report on.
    assert checked == [["qwen3:1.7b", "qwen3:1.7b", "gemma3:12b", "unseen-gemma4:26b"]]


def test_vision_model_offers_the_known_entries_and_still_takes_a_typed_name() -> None:
    app = build_app()
    components = _components_by_label(app)
    vision = components["VLMモデル"]

    assert vision["props"]["allow_custom_value"] is True
    offered = [value for _label, value in vision["props"]["choices"]]
    assert offered == ["qwen3-vl:8b", "unseen-gemma4:26b"]
    labels = [label for label, _value in vision["props"]["choices"]]
    assert any("無検閲" in label for label in labels)
    assert any("既定" in label for label in labels)
