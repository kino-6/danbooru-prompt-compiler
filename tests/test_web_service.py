from pathlib import Path

import pytest

from danbooru_prompt_compiler.image_tagger import (
    CHARACTER_CATEGORY,
    GENERAL_CATEGORY,
    ImageTagResult,
    PredictedTag,
)
from danbooru_prompt_compiler.models import CompileResult, LLMResponse
from danbooru_prompt_compiler.scene_prompt import SceneTemplate
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


class BackgroundTagger:
    def predict(self, _image_path: Path, **_options) -> ImageTagResult:
        return ImageTagResult(
            tags=[
                PredictedTag("1girl", 0.99, GENERAL_CATEGORY),
                PredictedTag("simple_background", 0.95, GENERAL_CATEGORY),
                PredictedTag("halftone", 0.9, GENERAL_CATEGORY),
                PredictedTag("green_background", 0.85, GENERAL_CATEGORY),
                PredictedTag("green_eyes", 0.8, GENERAL_CATEGORY),
            ],
            rating=None,
        )


class CensoredTagger:
    def predict(self, _image_path: Path, **_options) -> ImageTagResult:
        return ImageTagResult(
            tags=[
                PredictedTag("1girl", 0.99, GENERAL_CATEGORY),
                PredictedTag("censored", 0.95, GENERAL_CATEGORY),
                PredictedTag("bar_censor", 0.9, GENERAL_CATEGORY),
                PredictedTag("mosaic_censoring", 0.85, GENERAL_CATEGORY),
                PredictedTag("rain", 0.8, GENERAL_CATEGORY),
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


class ClothedTagger:
    def predict(self, _image_path: Path, **_options) -> ImageTagResult:
        return ImageTagResult(
            tags=[
                PredictedTag("1girl", 0.99, GENERAL_CATEGORY),
                PredictedTag("long_hair", 0.95, GENERAL_CATEGORY),
                PredictedTag("school_uniform", 0.9, GENERAL_CATEGORY),
            ],
            rating=None,
        )


class PlainNextPanelCompiler(FakeCompiler):
    """Returns a panel that mentions neither the outfit nor the hair."""

    def __init__(self) -> None:
        super().__init__()
        self.tag_dictionary = {"1girl", "long_hair", "school_uniform", "running"}

    def compile(self, request) -> CompileResult:
        self.last_request = request
        return CompileResult(variants=[["1girl", "running"]], unknown_tags=[])


class CensoredOutputCompiler(FakeCompiler):
    def compile(self, request) -> CompileResult:
        self.last_request = request
        return CompileResult(
            variants=[["1girl", "censored", "rain"], ["1girl", "bar_censor"]],
            unknown_tags=[],
        )


class FakeVisionClient:
    def __init__(self) -> None:
        self.last_request = None
        self.calls = 0

    def generate(self, request):
        self.last_request = request
        self.calls += 1
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


def test_next_panel_runs_from_an_image_without_an_instruction() -> None:
    compiler = TwoVariantCompiler()
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("manual next_panel must not consult the router")
        ),
        compiler_factory=lambda _url, _model: compiler,
    )

    result = service.run(
        image_path="sample.png",
        instruction="",
        base_prompt="",
        general_threshold=0.4,
        action_override="next_panel",
        variants=2,
    )

    assert result.action_plan["action"] == "next_panel"
    assert result.action_plan["router_source"] == "manual"
    assert compiler.last_request.scene_description == "1girl, rain, power_(chainsaw_man)"
    assert "自然な続きを提案する" in compiler.last_request.edit_instruction
    # The tagged character survives into every proposed next panel.
    assert all("1girl" in candidate for candidate in result.candidates)
    assert len(result.candidates) == 2


class RecordingTextClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.last_request = None

    def generate(self, request):
        self.last_request = request
        return LLMResponse(outputs=self.outputs[: request.variants])


SCENE_TEMPLATES = [
    SceneTemplate(
        name="demo",
        label="デモ",
        task="Produce a demo image.",
        sections=[("Subject", "who is in it")],
        delivery="One finished image.",
    ),
    SceneTemplate(
        name="poster",
        label="ポスター",
        task="Produce a poster.",
        sections=[("Subject", "hero element")],
        delivery="One poster.",
    ),
]


def test_scene_prompt_fills_the_selected_template_from_tags_and_description() -> None:
    plan = ActionPlan(action=WebAction.scene_prompt)
    text_client = RecordingTextClient(
        ["Subject: a young woman on wet stone steps", "Subject: the same woman, closer"]
    )
    service = WebPromptService(
        tagger=CensoredTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("a prose prompt must not run the tag compiler")
        ),
        text_factory=lambda _url, _model: text_client,
        scene_templates=SCENE_TEMPLATES,
    )

    result = service.run(
        image_path="sample.png",
        instruction="雨の日にして",
        base_prompt="",
        scene_template="poster",
        variants=2,
        edited_description="石段に立つ少女",
    )

    request = text_client.last_request.prompt
    assert "Produce a poster." in request
    assert "Subject: hero element" in request
    # The filtered tags and the description are the raw material for the prose.
    assert "1girl, rain" in request
    assert "石段に立つ少女" in request
    # The tags this image lost, plus the literal exclusion words, become the
    # avoid line, spelled as words rather than as tags.
    avoid_line = request.split("Never describe")[1]
    assert "censored, bar censor, mosaic censoring" in avoid_line
    assert "simple background" in avoid_line
    assert result.candidates[0].startswith("Produce a poster.")
    assert "Avoid: censored, bar censor, mosaic censoring" in result.candidates[0]
    assert len(result.candidates) == 2


class EmptyTagger:
    def predict(self, _image_path: Path, **_options) -> ImageTagResult:
        return ImageTagResult(tags=[], rating=None)


def test_scene_prompt_uses_its_own_model_and_falls_back_to_the_compiler_model() -> None:
    plan = ActionPlan(action=WebAction.scene_prompt)
    used_models: list[str] = []

    def text_factory(_url: str, model: str) -> RecordingTextClient:
        used_models.append(model)
        return RecordingTextClient(["Subject: a young woman"])

    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        text_factory=text_factory,
        scene_templates=SCENE_TEMPLATES,
    )
    options = dict(
        image_path="sample.png",
        instruction="ポスターにして",
        base_prompt="",
        general_threshold=0.4,
    )

    service.run(compiler_model="qwen3:1.7b", scene_model="gemma3:12b", **options)
    # An empty setting keeps the previous behaviour instead of failing.
    service.run(compiler_model="qwen3:1.7b", scene_model="", **options)

    assert used_models == ["gemma3:12b", "qwen3:1.7b"]


def test_scene_prompt_needs_something_to_describe() -> None:
    # An image the tagger found nothing in, with no instruction and no VLM,
    # gets past the generic guard but leaves the prose model with nothing.
    service = WebPromptService(
        tagger=EmptyTagger(),
        text_factory=lambda _url, _model: RecordingTextClient(["Subject: x"]),
        scene_templates=SCENE_TEMPLATES,
    )

    with pytest.raises(ValueError, match="自然文プロンプト"):
        service.run(
            image_path="sample.png",
            instruction="",
            base_prompt="",
            action_override="scene_prompt",
            use_vision=False,
        )


def test_service_excludes_censor_tags_from_image_tags_and_prompt_output() -> None:
    plan = ActionPlan(action=WebAction.edit, edit_instruction="夜にして", reason="edit")
    compiler = CensoredOutputCompiler()
    service = WebPromptService(
        tagger=CensoredTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: compiler,
    )

    result = service.run(
        image_path="sample.png",
        instruction="夜にして",
        base_prompt="",
    )

    assert result.inferred_tags == "1girl, rain"
    assert "censored" not in compiler.last_request.scene_description
    assert all("censor" not in candidate for candidate in result.candidates)
    assert (
        "Filtered image tags: censored, bar_censor, mosaic_censoring" in result.status
    )
    assert "Filtered prompt tags: censored, bar_censor" in result.status


def test_service_filters_default_exact_and_wildcard_image_tags() -> None:
    plan = ActionPlan(action=WebAction.tag_image, reason="tag it")
    service = WebPromptService(
        tagger=BackgroundTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
    )

    result = service.run(
        image_path="sample.png",
        instruction="タグを推測して",
        base_prompt="",
    )

    assert result.inferred_tags == "1girl, green_eyes"
    assert "simple_background" not in result.output
    assert "halftone" not in result.output
    assert "green_background" not in result.output
    assert (
        "Filtered image tags: simple_background, halftone, green_background"
        in result.status
    )


def test_service_can_disable_or_customize_image_tag_filter() -> None:
    plan = ActionPlan(action=WebAction.tag_image, reason="tag it")
    service = WebPromptService(
        tagger=BackgroundTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
    )

    unfiltered = service.run(
        image_path="sample.png",
        instruction="タグを推測して",
        base_prompt="",
        apply_tag_exclusions=False,
    )
    custom = service.run(
        image_path="sample.png",
        instruction="タグを推測して",
        base_prompt="",
        excluded_tags="green_*",
    )

    assert "simple_background" in unfiltered.inferred_tags
    assert "halftone" in unfiltered.inferred_tags
    assert "green_background" in unfiltered.inferred_tags
    assert "green_background" not in custom.inferred_tags
    assert "green_eyes" not in custom.inferred_tags
    assert "simple_background" in custom.inferred_tags


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
    # The change amount, not the routed plan, decides what is held fixed.
    assert "維持する要素: character, appearance" in compiler.last_request.edit_instruction
    assert "looking_back" in result.output
    assert "power_(chainsaw_man)" in result.output
    assert "power_chainsaw_man" not in result.output
    assert "moment_after" not in result.output


@pytest.mark.parametrize(
    "change, preserved, temperature",
    [
        (0.0, "character, appearance, clothing", 0.0),
        (0.5, "character, appearance", 0.5),
        (1.0, "character", 0.85),
    ],
)
def test_next_panel_change_controls_what_is_held_fixed(
    change: float,
    preserved: str,
    temperature: float,
) -> None:
    plan = ActionPlan(
        action=WebAction.next_panel,
        variants=1,
        preserve=["character", "appearance", "clothing"],
    )
    compiler = FakeCompiler()
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: compiler,
    )

    service.run(
        image_path="sample.png",
        instruction="次のコマ",
        base_prompt="",
        general_threshold=0.4,
        next_panel_change=change,
    )

    assert f"維持する要素: {preserved}。" in compiler.last_request.edit_instruction
    # A deterministic model returns near-identical variants, so the temperature
    # rises with the requested amount of change.
    assert compiler.last_request.temperature == temperature


def test_a_large_next_panel_change_stops_pinning_clothing_tags() -> None:
    plan = ActionPlan(action=WebAction.next_panel, variants=1)
    service = WebPromptService(
        tagger=ClothedTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: PlainNextPanelCompiler(),
    )
    options = dict(
        image_path="sample.png",
        instruction="次のコマ",
        base_prompt="",
    )

    small = service.run(next_panel_change=0.0, **options)
    large = service.run(next_panel_change=1.0, **options)

    assert "school_uniform" in small.candidates[0]
    assert "school_uniform" not in large.candidates[0]
    assert "1girl" in large.candidates[0]


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


def test_service_describes_the_image_for_tag_extraction() -> None:
    plan = ActionPlan(action=WebAction.tag_image, reason="tag it")
    vision = FakeVisionClient()
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        vision_factory=lambda _url, _model: vision,
    )

    result = service.run(
        image_path="sample.png",
        instruction="タグを推測して",
        base_prompt="",
        general_threshold=0.4,
        use_vision=True,
    )

    assert result.image_description == "少女は右を向き、腰から上の構図"
    assert vision.last_request.image_paths == ["sample.png"]
    # The description must not depend on the instruction, so one description
    # serves every action and survives instruction-only re-runs.
    assert "タグを推測して" not in vision.last_request.prompt
    assert "Image description: generated" in result.status


def test_service_keeps_the_image_description_out_of_a_new_prompt() -> None:
    plan = ActionPlan(action=WebAction.compile, scene_description="少女")
    compiler = FakeCompiler()
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: compiler,
        vision_factory=lambda _url, _model: FakeVisionClient(),
    )

    result = service.run(
        image_path="sample.png",
        instruction="少女",
        base_prompt="",
        general_threshold=0.4,
        use_vision=True,
    )

    assert result.image_description == "少女は右を向き、腰から上の構図"
    assert compiler.last_request.scene_description == "少女"
    assert compiler.last_request.edit_instruction is None


def test_service_survives_a_failing_vision_model() -> None:
    plan = ActionPlan(action=WebAction.edit, edit_instruction="夜にして")
    compiler = FakeCompiler()

    class BrokenVisionClient:
        def generate(self, _request):
            raise RuntimeError("model 'qwen3-vl:8b' not found")

    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: compiler,
        vision_factory=lambda _url, _model: BrokenVisionClient(),
    )

    result = service.run(
        image_path="sample.png",
        instruction="夜にして",
        base_prompt="",
        general_threshold=0.4,
        use_vision=True,
    )

    # The description is an aid; losing it must not lose the prompt.
    assert result.candidates
    assert result.image_description == ""
    assert "Image description failed" in result.status
    assert "not found" in result.status


def test_service_reuses_an_edited_description_without_calling_the_vlm() -> None:
    plan = ActionPlan(action=WebAction.edit, edit_instruction="夜にして")
    compiler = FakeCompiler()
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: compiler,
        vision_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("an edited description must not re-run the VLM")
        ),
    )

    result = service.run(
        image_path="sample.png",
        instruction="夜にして",
        base_prompt="",
        general_threshold=0.4,
        use_vision=True,
        edited_description="  赤い傘を持った少女が石段に立っている  ",
    )

    assert result.image_description == "赤い傘を持った少女が石段に立っている"
    assert "赤い傘を持った少女" in compiler.last_request.edit_instruction
    assert "Image description:" not in result.status


def test_service_caches_the_image_description_across_runs() -> None:
    plan = ActionPlan(action=WebAction.edit, edit_instruction="夜にして")
    vision = FakeVisionClient()
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: FakeCompiler(),
        vision_factory=lambda _url, _model: vision,
    )
    options = dict(
        image_path="sample.png",
        base_prompt="",
        general_threshold=0.4,
        use_vision=True,
    )

    first = service.run(instruction="夜にして", **options)
    second = service.run(instruction="雨にして", **options)

    assert vision.calls == 1
    assert first.image_description == second.image_description
    assert "Image description: generated" in first.status
    assert "Image description: cached" in second.status


def test_service_skips_vision_for_non_spatial_actions_without_an_image() -> None:
    plan = ActionPlan(action=WebAction.compile, scene_description="1girl")
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: FakeCompiler(),
        vision_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("vision should not run")
        ),
    )

    service.run(
        image_path=None,
        instruction="少女",
        base_prompt="",
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
    assert result.candidates[0] == "1girl\nrain"
    assert result.candidates[1] == "1girl\nlooking_back"
    assert "[variant 1]\n" + result.candidates[0] in result.output
    assert "[variant 2]\n" + result.candidates[1] in result.output
