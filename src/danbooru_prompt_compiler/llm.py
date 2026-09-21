from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path

from .models import LLMRequest, LLMResponse


class LLMClient(ABC):
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 300.0,
        temperature: float | None = None,
        json_mode: bool = False,
        json_schema: dict[str, object] | None = None,
        think: bool | str | None = None,
        cpu_only: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.json_mode = json_mode
        self.json_schema = json_schema
        self.think = think
        # Keeps the whole model off the card. Ollama reads `num_gpu` as the
        # number of layers to offload, so zero is CPU-only - slower, but it
        # runs alongside whatever else is using the GPU instead of queueing
        # behind it. Set per client rather than per request: a run decides
        # once, and every call it makes should honour that decision.
        self.cpu_only = cpu_only

    def generate(self, request: LLMRequest) -> LLMResponse:
        outputs: list[str] = []
        import httpx

        with httpx.Client(timeout=self.timeout) as client:
            for _ in range(request.variants):
                payload: dict[str, object] = {
                    "model": self.model,
                    "prompt": request.prompt,
                    "stream": False,
                }
                # A per-request temperature wins, so one client can serve both
                # deterministic and deliberately varied generations.
                temperature = (
                    request.temperature
                    if request.temperature is not None
                    else self.temperature
                )
                options: dict[str, object] = {}
                if temperature is not None:
                    options["temperature"] = temperature
                if self.cpu_only:
                    options["num_gpu"] = 0
                if options:
                    payload["options"] = options
                if self.json_schema is not None:
                    payload["format"] = self.json_schema
                elif self.json_mode:
                    payload["format"] = "json"
                if self.think is not None:
                    payload["think"] = self.think
                if request.image_paths:
                    payload["images"] = [
                        base64.b64encode(Path(path).read_bytes()).decode("ascii")
                        for path in request.image_paths
                    ]
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                outputs.append(data.get("response", ""))

        return LLMResponse(outputs=outputs)


class OpenAICompatibleClient(LLMClient):
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("OpenAI-compatible client is not implemented yet.")
