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


def test_compile_limits_output_tag_count() -> None:
    client = FakeLLMClient(["1girl, solo, shrine, rain, standing"])
    compiler = PromptCompiler(llm_client=client, tag_dictionary={"1girl", "solo", "shrine", "rain", "standing"})

    result = compiler.compile(
        CompileRequest(scene_description="test", variants=1, mode=CompileMode.subtle, max_output_tags=3)
    )

    assert result.variants == [["1girl", "solo", "shrine"]]


def test_excluded_tags_do_not_consume_output_slots() -> None:
    client = FakeLLMClient(["1girl, censored, solo, bar_censor, shrine, rain"])
    compiler = PromptCompiler(
        llm_client=client,
        tag_dictionary={"1girl", "solo", "shrine", "rain"},
    )

    result = compiler.compile(
        CompileRequest(
            scene_description="test",
            variants=1,
            max_output_tags=3,
            excluded_tags=["censored", "*_censor"],
        )
    )

    assert result.variants == [["1girl", "solo", "shrine"]]
    assert result.excluded_tags == ["censored", "bar_censor"]


def test_prompt_names_literal_excluded_tags_only() -> None:
    client = FakeLLMClient(["1girl"])
    compiler = PromptCompiler(llm_client=client, tag_dictionary={"1girl"})

    compiler.compile(
        CompileRequest(
            scene_description="test",
            variants=1,
            excluded_tags=["censored", "bar_censor", "*_background"],
        )
    )

    assert client.last_request is not None
    prompt = client.last_request.prompt
    assert "Never output these tags: censored, bar_censor." in prompt
    assert "*_background" not in prompt


def test_prompt_requests_ascii_english_danbooru_tags() -> None:
    client = FakeLLMClient(["1girl, shrine"])
    compiler = PromptCompiler(llm_client=client, tag_dictionary={"1girl", "shrine"})

    compiler.compile(
        CompileRequest(scene_description="雨の神社で佇む少女", variants=1, mode=CompileMode.subtle)
    )

    assert client.last_request is not None
    prompt = client.last_request.prompt
    assert "Danbooru prompt editor" in prompt
    assert "If there is no existing prompt" in prompt
    assert "If there is an existing prompt" in prompt
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


def test_prompt_includes_reference_tag_subset() -> None:
    client = FakeLLMClient(["1girl, shrine, torii"])
    compiler = PromptCompiler(llm_client=client, tag_dictionary={"1girl", "shrine", "torii"})

    compiler.compile(
        CompileRequest(
            scene_description="神社",
            variants=1,
            mode=CompileMode.subtle,
            tag_subset=["torii", "wide_shot", "stone_lantern"],
        )
    )

    assert client.last_request is not None
    prompt = client.last_request.prompt
    assert "Reference tag subset from matching Danbooru posts" in prompt
    assert "torii, wide_shot, stone_lantern" in prompt
    assert "Do not copy the full subset" in prompt
    assert "Select only the most relevant tags" in prompt
    assert "up to 20 tags total" in prompt
