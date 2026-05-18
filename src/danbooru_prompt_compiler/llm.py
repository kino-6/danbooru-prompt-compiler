from __future__ import annotations

from abc import ABC, abstractmethod

from .models import LLMRequest, LLMResponse


class LLMClient(ABC):
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, request: LLMRequest) -> LLMResponse:
        outputs: list[str] = []
        import httpx

        with httpx.Client(timeout=60.0) as client:
            for _ in range(request.variants):
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": request.prompt,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data = response.json()
                outputs.append(data.get("response", ""))

        return LLMResponse(outputs=outputs)


class OpenAICompatibleClient(LLMClient):
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("OpenAI-compatible client is not implemented yet.")
