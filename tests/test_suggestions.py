from danbooru_prompt_compiler.models import LLMRequest, LLMResponse
from danbooru_prompt_compiler.llm import LLMClient
from danbooru_prompt_compiler.suggestions import suggest_edit_instructions


class FakeLLMClient(LLMClient):
    def __init__(self, output: str) -> None:
        self.output = output
        self.last_request: LLMRequest | None = None

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.last_request = request
        return LLMResponse(outputs=[self.output])


def test_suggest_edit_instructions_returns_short_lines() -> None:
    client = FakeLLMClient("1. 鳥居の奥に淡い霧を足す\n2. 濡れた石畳の反射を強める\n3. 雨粒を強く描写する")

    suggestions = suggest_edit_instructions(
        client,
        base_prompt="1girl, shrine, rain",
        edit_instruction="雨の神社",
        current_tags=["1girl", "shrine", "rain"],
        reference_tags=["torii", "stone_lantern"],
        count=2,
    )

    assert suggestions == ["鳥居の奥に淡い霧を足す", "濡れた石畳の反射を強める"]
    assert client.last_request is not None
    assert "Return exactly 2 lines" in client.last_request.prompt
    assert "Output Japanese only" in client.last_request.prompt
    assert "fluent Japanese verb phrase" in client.last_request.prompt
    assert "Good examples:" in client.last_request.prompt
    assert "Candidate ideas:" in client.last_request.prompt
    assert "Reference tags: torii, stone_lantern" in client.last_request.prompt


def test_suggest_edit_instructions_fills_from_candidates_when_output_is_noisy() -> None:
    client = FakeLLMClient("Add bright cinematic lighting\n1. 鳥居の奥に淡い霧を足す")

    suggestions = suggest_edit_instructions(
        client,
        base_prompt="1girl, shrine, rain",
        edit_instruction=None,
        current_tags=["1girl", "shrine", "rain"],
        reference_tags=["torii", "stone_lantern"],
        count=2,
    )

    assert suggestions == ["鳥居の奥に淡い霧を足す", "濡れた石畳の反射を強める"]
