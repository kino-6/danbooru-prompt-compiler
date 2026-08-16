from pathlib import Path

from danbooru_prompt_compiler.image_tagger import (
    CHARACTER_CATEGORY,
    GENERAL_CATEGORY,
    ImageTagResult,
    PredictedTag,
)
from danbooru_prompt_compiler.models import CompileResult
from danbooru_prompt_compiler.web_router import ActionPlan, RoutedPlan, WebAction
from danbooru_prompt_compiler.web_service import WebPromptService


class FakeTagger:
    def predict(self, image_path: Path, **options) -> ImageTagResult:
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
