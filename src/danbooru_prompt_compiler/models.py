from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover
    class BaseModel:  # lightweight fallback for offline environments
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def Field(default=None, **_kwargs):
        return default


class CompileMode(str, Enum):
    subtle = "subtle"
    remix = "remix"
    composition = "composition"
    character_safe = "character_safe"


class CompileRequest(BaseModel):
    scene_description: str = Field(min_length=1)
    variants: int = Field(default=1, ge=1, le=10)
    mode: CompileMode = CompileMode.subtle
    preset_name: str | None = None


class CompileResult(BaseModel):
    variants: list[list[str]]
    unknown_tags: list[str]


class LLMRequest(BaseModel):
    prompt: str
    variants: int = 1


class LLMResponse(BaseModel):
    outputs: list[str]
