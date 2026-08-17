from danbooru_prompt_compiler.llm import LLMClient
from danbooru_prompt_compiler.models import LLMRequest, LLMResponse
from danbooru_prompt_compiler.web_router import (
    NaturalLanguageRouter,
    RouteRequest,
    WebAction,
    route_with_rules,
)


class FakeLLMClient(LLMClient):
    def __init__(self, output: str = "", error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.last_request: LLMRequest | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        if self.error:
            raise self.error
        return LLMResponse(outputs=[self.output])


def test_router_parses_json_after_qwen_thinking_text() -> None:
    client = FakeLLMClient(
        '<think>short reasoning</think>\n```json\n'
        '{"action":"next_panel","scene_description":"",'
        '"edit_instruction":"少女を振り返らせる","variants":3,'
        '"preserve":["character","clothing"],"reason":"次のコマ"}\n```'
    )
    router = NaturalLanguageRouter(client)

    routed = router.route(
        RouteRequest(
            instruction="少女を振り返らせて",
            has_image=True,
            default_variants=3,
        )
    )

    assert routed.source == "llm"
    assert routed.plan.action == WebAction.next_panel
    assert routed.plan.variants == 3
    assert routed.plan.preserve == ["character", "clothing"]
    assert client.last_request is not None
    assert "Never create shell commands" in client.last_request.prompt


def test_router_falls_back_to_rules_when_ollama_is_unavailable() -> None:
    router = NaturalLanguageRouter(FakeLLMClient(error=ConnectionError("offline")))

    routed = router.route(
        RouteRequest(instruction="もっと明るくして", has_image=True, default_variants=2)
    )

    assert routed.source == "rules"
    assert routed.plan.action == WebAction.edit
    assert "offline" in routed.warning


def test_router_treats_string_null_as_missing_instruction() -> None:
    client = FakeLLMClient(
        '{"action":"next_panel","scene_description":"",'
        '"edit_instruction":"null","variants":1,'
        '"preserve":[],"reason":""}'
    )

    routed = NaturalLanguageRouter(client).route(
        RouteRequest(instruction="少女を動かして", has_image=True)
    )

    assert routed.plan.edit_instruction == "少女を動かして"
    assert routed.plan.preserve == ["character", "appearance", "clothing"]


def test_explicit_next_panel_words_bypass_small_model_misclassification() -> None:
    client = FakeLLMClient(
        '{"action":"compile","scene_description":"次のコマ",'
        '"edit_instruction":null,"variants":1,"preserve":[],"reason":""}'
    )

    routed = NaturalLanguageRouter(client).route(
        RouteRequest(instruction="次のコマで少女を振り返らせて", has_image=True)
    )

    assert routed.source == "rules"
    assert routed.plan.action == WebAction.next_panel
    assert routed.plan.edit_instruction == "次のコマで少女を振り返らせて"
    assert client.last_request is None


def test_rule_router_treats_prompt_with_instruction_as_edit() -> None:
    plan = route_with_rules(
        RouteRequest(
            instruction="夜に変更して",
            base_prompt="1girl, shrine, day",
            default_variants=2,
        )
    )

    assert plan.action == WebAction.edit
    assert plan.edit_instruction == "夜に変更して"
    assert plan.variants == 2


def test_router_rejects_small_model_tagging_misclassification_for_edit() -> None:
    client = FakeLLMClient(
        '{"action":"tag_image","scene_description":"夜景",'
        '"edit_instruction":"夜っぽくして","variants":1,'
        '"preserve":["character"],"reason":"夜景"}'
    )

    routed = NaturalLanguageRouter(client).route(
        RouteRequest(instruction="夜っぽくして", has_image=True)
    )

    assert routed.source == "llm"
    assert routed.plan.action == WebAction.edit
    assert routed.plan.edit_instruction == "夜っぽくして"


def test_ui_output_count_overrides_small_model_variant_count() -> None:
    client = FakeLLMClient(
        '{"action":"edit","scene_description":"","edit_instruction":"夜にする",'
        '"variants":1,"preserve":[],"reason":"編集"}'
    )

    routed = NaturalLanguageRouter(client).route(
        RouteRequest(instruction="夜にする", has_image=True, default_variants=4)
    )

    assert routed.plan.variants == 4
