from danbooru_prompt_compiler.compiler import PromptCompiler
from danbooru_prompt_compiler.models import CompileMode, CompileRequest, InputType, LLMRequest, LLMResponse
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


def test_prompt_requests_ascii_english_danbooru_tags() -> None:
    client = FakeLLMClient(["1girl, shrine"])
    compiler = PromptCompiler(llm_client=client, tag_dictionary={"1girl", "shrine"})

    compiler.compile(
        CompileRequest(scene_description="雨の神社で佇む少女", variants=1, mode=CompileMode.subtle)
    )

    assert client.last_request is not None
    prompt = client.last_request.prompt
    assert "Translate non-English scene text into English Danbooru tags" in prompt
    assert "ASCII tags" in prompt
    assert "Do not output Japanese" in prompt
    assert "Do not include operation words" in prompt
    assert "omit uncertain or invented tags" in prompt


def test_prompt_edit_request_includes_existing_prompt_and_instruction() -> None:
    client = FakeLLMClient(["1girl, solo, shrine, rain, umbrella"])
    compiler = PromptCompiler(llm_client=client, tag_dictionary={"1girl", "solo", "shrine", "rain", "umbrella"})

    compiler.compile(
        CompileRequest(
            scene_description="1girl, solo, shrine, rain",
            variants=1,
            mode=CompileMode.subtle,
            input_type=InputType.prompt,
            edit_instruction="夕方にして傘を追加",
        )
    )

    assert client.last_request is not None
    prompt = client.last_request.prompt
    assert "Input type: prompt" in prompt
    assert "Existing prompt: 1girl, solo, shrine, rain" in prompt
    assert "Edit instruction: 夕方にして傘を追加" in prompt
    assert "preserving useful existing tags" in prompt
