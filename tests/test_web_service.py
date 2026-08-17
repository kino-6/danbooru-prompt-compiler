from pathlib import Path

import pytest

from danbooru_prompt_compiler.image_tagger import (
    CHARACTER_CATEGORY,
    GENERAL_CATEGORY,
    ImageTagResult,
    PredictedTag,
)
from danbooru_prompt_compiler.models import CompileResult, LLMResponse
from danbooru_prompt_compiler.web_router import ActionPlan, RoutedPlan, WebAction
from danbooru_prompt_compiler.web_service import WebPromptService


class FakeTagger:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image_path: Path, **options) -> ImageTagResult:
        self.calls += 1
        assert image_path == Path("sample.png")
        assert options["general_threshold"] == 0.4
        return ImageTagResult(
            tags=[
                PredictedTag("1girl", 0.99, GENERAL_CATEGORY),
                PredictedTag("rain", 0.91, GENERAL_CATEGORY),
                PredictedTag("power_(chainsaw_man)", 0.9, CHARACTER_CATEGORY),
            ],
            rating=None,
        )


class FixedRouter:
    def __init__(self, plan: ActionPlan) -> None:
        self.plan = plan

    def route(self, _request) -> RoutedPlan:
        return RoutedPlan(plan=self.plan, source="test")


class FakeCompiler:
    def __init__(self) -> None:
        self.last_request = None
        self.tag_dictionary = {
            "1girl",
            "rain",
            "looking_back",
            "power_(chainsaw_man)",
        }

    def compile(self, request) -> CompileResult:
        self.last_request = request
        return CompileResult(
            variants=[
                [
                    "1girl",
                    "rain",
                    "power_chainsaw_man",
                    "moment_after",
                ]
            ],
            unknown_tags=["power_chainsaw_man", "moment_after"],
        )


class TwoVariantCompiler(FakeCompiler):
    def compile(self, request) -> CompileResult:
        self.last_request = request
        return CompileResult(
            variants=[["1girl", "rain"], ["1girl", "looking_back"]],
            unknown_tags=[],
        )


class FakeVisionClient:
    def __init__(self) -> None:
        self.last_request = None

    def generate(self, request):
        self.last_request = request
        return LLMResponse(outputs=["少女は右を向き、腰から上の構図"])


def test_service_can_run_image_tagging_without_prompt_compiler() -> None:
    plan = ActionPlan(action=WebAction.tag_image, reason="tag it")
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("compiler should not run")
        ),
    )

    result = service.run(
        image_path="sample.png",
        instruction="タグを推測して",
        base_prompt="",
        general_threshold=0.4,
    )

    assert result.action_plan["action"] == "tag_image"
    assert result.inferred_tags == "1girl, rain, power_(chainsaw_man)"
    assert "subject: 1girl" in result.output


def test_service_caches_unchanged_image_tagging_result() -> None:
    plan = ActionPlan(action=WebAction.tag_image, reason="tag it")
    tagger = FakeTagger()
    service = WebPromptService(
        tagger=tagger,
        router_factory=lambda _url, _model: FixedRouter(plan),
    )

    first = service.run(
        image_path="sample.png",
        instruction="タグを推測して",
        base_prompt="",
        general_threshold=0.4,
    )
    second = service.run(
        image_path="sample.png",
        instruction="タグを推測して",
        base_prompt="",
        general_threshold=0.4,
    )

    assert tagger.calls == 1
    assert "Image cache: miss" in first.status
    assert "Image cache: hit" in second.status


def test_service_uses_inferred_tags_as_next_panel_base() -> None:
    plan = ActionPlan(
        action=WebAction.next_panel,
        edit_instruction="少女を振り返らせる",
        variants=1,
        preserve=["character", "clothing"],
    )
    compiler = FakeCompiler()
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: compiler,
    )

    result = service.run(
        image_path="sample.png",
        instruction="次のコマで振り返らせて",
        base_prompt="",
        general_threshold=0.4,
    )

    assert compiler.last_request.scene_description == "1girl, rain, power_(chainsaw_man)"
    assert "次のコマとして" in compiler.last_request.edit_instruction
    assert "character, clothing" in compiler.last_request.edit_instruction
    assert "looking_back" in result.output
    assert "power_(chainsaw_man)" in result.output
    assert "power_chainsaw_man" not in result.output
    assert "moment_after" not in result.output


def test_service_uses_human_edited_tags_as_compiler_input() -> None:
    plan = ActionPlan(action=WebAction.edit, edit_instruction="雪にして")
    compiler = FakeCompiler()
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: compiler,
    )

    result = service.run(
        image_path="sample.png",
        instruction="雪にして",
        base_prompt="",
        general_threshold=0.4,
        edited_tags="1girl, snow, snow",
    )

    assert compiler.last_request.scene_description == "1girl, snow"
    assert result.inferred_tags == "1girl, snow"


@pytest.mark.parametrize("action", ["tag_image", "compile", "edit", "next_panel"])
def test_manual_action_override_bypasses_router_factory(action: str) -> None:
    compiler = FakeCompiler()
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("router should not run")
        ),
        compiler_factory=lambda _url, _model: compiler,
    )

    result = service.run(
        image_path="sample.png",
        instruction="振り返らせて",
        base_prompt="",
        general_threshold=0.4,
        action_override=action,
        variants=1,
    )

    assert result.action_plan["action"] == action
    assert result.action_plan["router_source"] == "manual"
    if action in {"edit", "next_panel"}:
        assert "振り返らせて" in compiler.last_request.edit_instruction


def test_service_adds_optional_vision_observation_to_image_edit() -> None:
    plan = ActionPlan(action=WebAction.edit, edit_instruction="左を向かせて")
    compiler = FakeCompiler()
    vision = FakeVisionClient()
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: compiler,
        vision_factory=lambda _url, _model: vision,
    )

    service.run(
        image_path="sample.png",
        instruction="左を向かせて",
        base_prompt="",
        general_threshold=0.4,
        use_vision=True,
    )

    assert vision.last_request.image_paths == ["sample.png"]
    assert "少女は右を向き" in compiler.last_request.edit_instruction


@pytest.mark.parametrize("action", [WebAction.tag_image, WebAction.compile])
def test_service_skips_vision_for_non_spatial_actions(action: WebAction) -> None:
    plan = ActionPlan(action=action, scene_description="1girl")
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: FakeCompiler(),
        vision_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("vision should not run")
        ),
    )

    service.run(
        image_path="sample.png",
        instruction="タグを推測して" if action == WebAction.tag_image else "少女",
        base_prompt="",
        general_threshold=0.4,
        use_vision=True,
    )


def test_service_skips_vision_when_disabled() -> None:
    plan = ActionPlan(action=WebAction.edit, edit_instruction="左を向かせて")
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: FakeCompiler(),
        vision_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("vision should not run")
        ),
    )

    service.run(
        image_path="sample.png",
        instruction="左を向かせて",
        base_prompt="",
        general_threshold=0.4,
        use_vision=False,
    )


def test_service_reports_ordered_progress_phases() -> None:
    plan = ActionPlan(action=WebAction.edit, edit_instruction="左を向かせて")
    stages: list[tuple[str, float]] = []
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: FakeCompiler(),
        vision_factory=lambda _url, _model: FakeVisionClient(),
    )

    service.run(
        image_path="sample.png",
        instruction="左を向かせて",
        base_prompt="",
        general_threshold=0.4,
        use_vision=True,
        on_progress=lambda stage, fraction: stages.append((stage, fraction)),
    )

    assert [stage for stage, _ in stages] == [
        "routing",
        "tagging",
        "vision",
        "compilation",
        "complete",
    ]
    assert [fraction for _, fraction in stages] == sorted(
        fraction for _, fraction in stages
    )


def test_service_exposes_variants_as_separate_candidates() -> None:
    plan = ActionPlan(action=WebAction.edit, edit_instruction="変更", variants=2)
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: TwoVariantCompiler(),
    )

    result = service.run(
        image_path="sample.png",
        instruction="変更",
        base_prompt="",
        general_threshold=0.4,
    )

    assert len(result.candidates) == 2
    assert not result.candidates[0].startswith("[variant")
    assert not result.candidates[1].startswith("[variant")
    assert "[variant 1]\n" + result.candidates[0] in result.output
    assert "[variant 2]\n" + result.candidates[1] in result.output
