from danbooru_prompt_compiler.compiler import PromptCompiler
from danbooru_prompt_compiler.models import CompileMode, CompileRequest, LLMRequest, LLMResponse
from danbooru_prompt_compiler.llm import LLMClient


class FakeLLMClient(LLMClient):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.last_request: LLMRequest | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(outputs=self.outputs)


def test_multiple_variants_are_parsed() -> None:
    client = FakeLLMClient(["1girl, shrine", "1girl\nnight"])
    compiler = PromptCompiler(llm_client=client, tag_dictionary={"1girl", "shrine", "night"})

    result = compiler.compile(
        CompileRequest(scene_description="test", variants=2, mode=CompileMode.subtle)
    )

    assert result.variants == [["1girl", "shrine"], ["1girl", "night"]]


def test_unknown_tag_warning_data_is_reported() -> None:
    client = FakeLLMClient(["1girl, unknown custom tag"])
    compiler = PromptCompiler(llm_client=client, tag_dictionary={"1girl"})

    result = compiler.compile(
        CompileRequest(scene_description="test", variants=1, mode=CompileMode.remix)
    )

    assert result.unknown_tags == ["unknown_custom_tag"]
