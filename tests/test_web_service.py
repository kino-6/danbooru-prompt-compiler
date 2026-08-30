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
from danbooru_prompt_compiler.web_service import (
    WEB_RUN_FIELDS,
    WebPromptService,
    WebRunRequest,
)


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


# The two sliders are deliberately crossed here: the change value decides what
# is preserved and nothing about temperature, and the time value the reverse.
@pytest.mark.parametrize(
    "change, moment, preserved, temperature",
    [
        (0.0, 1.0, "character, appearance, clothing", 0.6),
        (0.5, 0.5, "character, appearance", 0.5),
        (1.0, 0.0, "character", 0.0),
    ],
)
def test_next_panel_change_controls_what_is_held_fixed(
    change: float,
    moment: float,
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
        next_panel_time=moment,
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


def test_the_prose_model_sees_the_image_only_when_the_setting_is_on() -> None:
    plan = ActionPlan(action=WebAction.scene_prompt)
    text_client = RecordingTextClient(["Subject: a young woman", "Subject: a young woman"])
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        text_factory=lambda _url, _model: text_client,
        scene_templates=SCENE_TEMPLATES,
    )
    options = dict(
        image_path="sample.png",
        instruction="ポスターにして",
        base_prompt="",
        scene_template="poster",
        general_threshold=0.4,
    )

    service.run(scene_sees_image=True, **options)
    assert text_client.last_request.image_paths == ["sample.png"]
    assert "The reference image is attached." in text_client.last_request.prompt

    # The default prose model is text-only, so the picture stays behind unless
    # the setting says the model can read it.
    service.run(scene_sees_image=False, **options)
    assert text_client.last_request.image_paths == []
    assert "The reference image is attached." not in text_client.last_request.prompt


def test_a_typed_in_vision_model_still_reaches_the_vision_factory() -> None:
    plan = ActionPlan(action=WebAction.tag_image)
    asked: list[str] = []

    def vision_factory(_url: str, model: str) -> RecordingTextClient:
        asked.append(model)
        return RecordingTextClient(["立っている少女"])

    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        vision_factory=vision_factory,
    )

    # The dropdown offers two entries and accepts anything else typed into it,
    # so a name it has never heard of has to survive the trip unchanged.
    service.run(
        image_path="sample.png",
        instruction="タグを推測して",
        base_prompt="",
        general_threshold=0.4,
        use_vision=True,
        vision_model="some-other-vlm:4b",
    )

    assert asked == ["some-other-vlm:4b"]


class ReviewingVisionClient:
    """Answers a tag review, and records the request it was given."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.last_request = None

    def generate(self, request):
        self.last_request = request
        return LLMResponse(outputs=[self.output])


def _review_service(vision_client, **kwargs) -> WebPromptService:
    return WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(
            ActionPlan(action=WebAction.tag_image)
        ),
        vision_factory=lambda _url, _model: vision_client,
        known_tags={"1girl", "rain", "elf", "pointy_ears", "power_(chainsaw_man)"},
        **kwargs,
    )


def test_verifying_tags_shows_the_image_and_keeps_only_dictionary_proposals() -> None:
    vision_client = ReviewingVisionClient("Remove: rain\nAdd: pointy_ears, red_circle")
    service = _review_service(vision_client)

    result = service.run(
        image_path="sample.png",
        instruction="",
        base_prompt="",
        general_threshold=0.4,
        action_override="verify_tags",
        use_vision=False,
    )

    assert vision_client.last_request.image_paths == ["sample.png"]
    assert "pointy_ears" in result.inferred_tags
    assert "rain" not in result.inferred_tags
    # A proposal the dictionary does not carry is reported, never adopted.
    assert "red_circle" not in result.inferred_tags
    assert "辞書にないため不採用: red_circle" in result.status


def test_verifying_tags_never_removes_a_tag_the_user_typed() -> None:
    vision_client = ReviewingVisionClient("Remove: elf, rain\nAdd: none")
    service = _review_service(vision_client)

    result = service.run(
        image_path="sample.png",
        instruction="",
        base_prompt="",
        general_threshold=0.4,
        action_override="verify_tags",
        use_vision=False,
        edited_tags="1girl, elf, rain",
    )

    assert "elf" in result.inferred_tags
    assert "rain" in result.inferred_tags
    assert "変更の提案はありませんでした" in result.status


def test_a_failing_vision_model_leaves_the_reviewed_tags_intact() -> None:
    class BrokenVisionClient:
        def generate(self, _request):
            raise RuntimeError("model missing")

    service = _review_service(BrokenVisionClient())

    result = service.run(
        image_path="sample.png",
        instruction="",
        base_prompt="",
        general_threshold=0.4,
        action_override="verify_tags",
        use_vision=False,
    )

    assert result.inferred_tags == "1girl, rain, power_(chainsaw_man)"
    assert "タグ確認に失敗したため、タグはそのままです" in result.status
    assert "model missing" in result.status


class PanelTagger:
    def predict(self, _image_path: Path, **_options) -> ImageTagResult:
        return ImageTagResult(
            tags=[
                PredictedTag("1girl", 0.99, GENERAL_CATEGORY),
                PredictedTag("long_hair", 0.95, GENERAL_CATEGORY),
                PredictedTag("standing", 0.9, GENERAL_CATEGORY),
                PredictedTag("looking_at_viewer", 0.88, GENERAL_CATEGORY),
                PredictedTag("holding_bow_(weapon)", 0.85, GENERAL_CATEGORY),
            ],
            rating=None,
        )


PANEL_DICTIONARY = {
    "1girl", "long_hair", "standing", "looking_at_viewer",
    "holding_bow_(weapon)", "drawing_bow", "aiming", "looking_away",
}


def _panel_service(vision_client, **kwargs) -> WebPromptService:
    return WebPromptService(
        tagger=PanelTagger(),
        router_factory=lambda _url, _model: FixedRouter(
            ActionPlan(action=WebAction.next_panel)
        ),
        vision_factory=lambda _url, _model: vision_client,
        compiler_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("the vision path must not fall back to the compiler")
        ),
        known_tags=PANEL_DICTIONARY,
        **kwargs,
    )


def _panel_options(**overrides) -> dict:
    return {
        "image_path": "sample.png",
        "instruction": "",
        "base_prompt": "",
        "use_vision": False,
        "variants": 1,
        **overrides,
    }


def test_the_next_panel_comes_from_the_model_that_can_see_the_image() -> None:
    vision_client = ReviewingVisionClient(
        "Next: she pulls the bowstring back towards her face.\n"
        "Remove: holding_bow_(weapon)\n"
        "Add: drawing_bow, aiming"
    )
    service = _panel_service(vision_client)

    result = service.run(**_panel_options(action_override="next_panel"))

    assert vision_client.last_request.image_paths == ["sample.png"]
    assert "drawing_bow" in result.output and "aiming" in result.output
    assert "holding_bow_(weapon)" not in result.output
    # The identity of the character survives whatever the panel does.
    assert "1girl" in result.output and "long_hair" in result.output
    assert "すべてが現在のコマから動いています" in result.status
    # The sentence explains the panel that the tag list only implies.
    assert "she pulls the bowstring back towards her face." in result.status


def test_a_panel_that_did_not_move_is_reported_rather_than_returned_quietly() -> None:
    # The answer only restyles: nothing the character does has changed.
    vision_client = ReviewingVisionClient("Remove: none\nAdd: none")
    service = _panel_service(vision_client)

    result = service.run(**_panel_options(action_override="next_panel"))

    assert "1件は現在のコマと姿勢・構図が変わりませんでした" in result.status
    assert "変化量を上げるか" in result.status


def test_a_failing_vision_model_falls_back_to_the_tag_compiler() -> None:
    class BrokenVisionClient:
        def generate(self, _request):
            raise RuntimeError("model missing")

    service = WebPromptService(
        tagger=PanelTagger(),
        router_factory=lambda _url, _model: FixedRouter(
            ActionPlan(action=WebAction.next_panel)
        ),
        vision_factory=lambda _url, _model: BrokenVisionClient(),
        compiler_factory=lambda _url, _model: FakeCompiler(),
        known_tags=PANEL_DICTIONARY,
    )

    result = service.run(**_panel_options(action_override="next_panel"))

    assert "タグからの生成に戻しました" in result.status
    assert "model missing" in result.status
    assert result.output


def test_a_control_nobody_touched_falls_back_to_its_default() -> None:
    # Gradio sends None for an untouched textbox, and every field but the image
    # path is non-optional, so a run before anything was typed into used to
    # raise a validation error instead of running.
    values = list(WebRunRequest().to_values())
    for field in ("image_url", "base_prompt", "edited_tags", "edited_description"):
        values[WEB_RUN_FIELDS.index(field)] = None
    values[WEB_RUN_FIELDS.index("image_path")] = None

    request = WebRunRequest.from_values(values)

    assert request.image_url == ""
    assert request.base_prompt == ""
    assert request.edited_tags == ""
    # The image path is genuinely optional and keeps its None.
    assert request.image_path is None


def test_supplied_values_still_win_over_the_defaults() -> None:
    values = list(WebRunRequest(instruction="夜にして", variants=2).to_values())

    request = WebRunRequest.from_values(values)

    assert request.instruction == "夜にして"
    assert request.variants == 2


def test_the_panel_note_names_what_actually_changed() -> None:
    vision_client = ReviewingVisionClient(
        "Next: she draws the bow.\n"
        "Remove: holding_bow_(weapon), looking_at_viewer\n"
        "Add: drawing_bow, aiming"
    )
    service = _panel_service(vision_client)

    result = service.run(**_panel_options(action_override="next_panel"))

    # One tag changing inside a list of twenty-five reads as no change at all,
    # so the note says which tags moved rather than leaving it to be spotted.
    assert "she draws the bow." in result.panel_note
    assert "-holding_bow_(weapon), looking_at_viewer" in result.panel_note
    assert "+drawing_bow, aiming" in result.panel_note


def test_the_note_travels_beside_the_status_so_a_follow_up_can_carry_it() -> None:
    vision_client = ReviewingVisionClient("Next: she draws.\nRemove: none\nAdd: aiming")
    service = _panel_service(vision_client)

    result = service.run(**_panel_options(action_override="next_panel"))

    assert result.panel_note
    assert result.panel_note in result.status


def test_asking_for_several_panels_at_once_will_not_take_the_same_one_twice() -> None:
    class RecordingVision:
        def __init__(self) -> None:
            self.temperatures: list[float | None] = []

        def generate(self, request):
            self.temperatures.append(request.temperature)
            return LLMResponse(outputs=["Remove: none\nAdd: aiming"] * request.variants)

    client = RecordingVision()
    service = _panel_service(client)

    # The deterministic time band would otherwise fill three boxes identically.
    service.run(**_panel_options(action_override="next_panel", variants=3, next_panel_time=0.1))
    assert client.temperatures[-1] >= 0.5

    service.run(**_panel_options(action_override="next_panel", variants=1, next_panel_time=0.1))
    assert client.temperatures[-1] == 0.0


def test_a_next_panel_can_come_from_a_prompt_with_no_image_at_all() -> None:
    text_client = ReviewingVisionClient(
        "Next: she looses the arrow.\n"
        "Remove: holding_bow_(weapon)\n"
        "Add: aiming"
    )
    service = WebPromptService(
        tagger=PanelTagger(),
        router_factory=lambda _url, _model: FixedRouter(
            ActionPlan(action=WebAction.next_panel)
        ),
        # No picture, so the vision model must not be the one asked.
        vision_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("no image means no vision model")
        ),
        text_factory=lambda _url, _model: text_client,
        known_tags=PANEL_DICTIONARY,
    )

    result = service.run(
        image_path=None,
        instruction="",
        base_prompt="1girl, long_hair, standing, holding_bow_(weapon)",
        variants=1,
        use_vision=False,
    )

    assert text_client.last_request.image_paths == []
    # Without a picture the request must not claim there is one attached.
    assert "These tags describe the current panel" in text_client.last_request.prompt
    assert "attached image" not in text_client.last_request.prompt
    assert "aiming" in result.output
    assert "holding_bow_(weapon)" not in result.output
    assert "she looses the arrow." in result.panel_note


def test_the_same_result_is_offered_as_tags_and_as_english_prose() -> None:
    plan = ActionPlan(action=WebAction.compile, variants=1)
    prose_client = RecordingTextClient(["Subject: a miko before a shrine"])
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: FakeCompiler(),
        text_factory=lambda _url, _model: prose_client,
        scene_templates=SCENE_TEMPLATES,
    )

    result = service.run(
        image_path=None,
        instruction="雨の神社に立つ巫女",
        base_prompt="",
        also_prose=True,
        variants=1,
    )

    # Two readings of one result, so the tags are untouched by the prose step.
    assert result.candidates[0]
    assert "Subject: a miko before a shrine" in result.prose_prompt
    assert "Subject:" not in result.output


def test_prose_is_written_from_the_tags_that_were_produced() -> None:
    plan = ActionPlan(action=WebAction.compile, variants=1)
    prose_client = RecordingTextClient(["Subject: whatever"])
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(plan),
        compiler_factory=lambda _url, _model: FakeCompiler(),
        text_factory=lambda _url, _model: prose_client,
        scene_templates=SCENE_TEMPLATES,
    )

    service.run(
        image_path=None, instruction="雨の神社", base_prompt="",
        also_prose=True, variants=1,
    )

    # Not from the inferred tags: from the prompt the run actually produced.
    assert "Observed in the reference image" in prose_client.last_request.prompt


def test_a_failing_prose_step_costs_the_prose_and_not_the_tags() -> None:
    class BrokenText:
        def generate(self, _request):
            raise RuntimeError("prose model missing")

    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(
            ActionPlan(action=WebAction.compile, variants=1)
        ),
        compiler_factory=lambda _url, _model: FakeCompiler(),
        text_factory=lambda _url, _model: BrokenText(),
        scene_templates=SCENE_TEMPLATES,
    )

    result = service.run(
        image_path=None, instruction="雨の神社", base_prompt="",
        also_prose=True, variants=1,
    )

    assert result.candidates[0]
    assert result.prose_prompt == ""
    assert "英文プロンプトを生成できませんでした" in result.status
    assert "prose model missing" in result.status


def test_prose_is_not_written_unless_it_is_asked_for() -> None:
    service = WebPromptService(
        tagger=FakeTagger(),
        router_factory=lambda _url, _model: FixedRouter(
            ActionPlan(action=WebAction.compile, variants=1)
        ),
        compiler_factory=lambda _url, _model: FakeCompiler(),
        text_factory=lambda _url, _model: (_ for _ in ()).throw(
            AssertionError("prose costs a model call nobody asked for")
        ),
        scene_templates=SCENE_TEMPLATES,
    )

    result = service.run(
        image_path=None, instruction="雨の神社", base_prompt="", variants=1
    )

    assert result.prose_prompt == ""
