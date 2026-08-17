from __future__ import annotations

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


class InputType(str, Enum):
    scene = "scene"
    prompt = "prompt"


class CompileRequest(BaseModel):
    scene_description: str = ""
    variants: int = Field(default=1, ge=1, le=10)
    mode: CompileMode = CompileMode.subtle
    preset_name: str | None = None
    input_type: InputType = InputType.scene
    edit_instruction: str | None = None
    tag_subset: list[str] = Field(default_factory=list)
    max_output_tags: int = Field(default=20, ge=1, le=100)
    excluded_tags: list[str] = Field(default_factory=list)


class CompileResult(BaseModel):
    variants: list[list[str]]
    unknown_tags: list[str]
    excluded_tags: list[str] = Field(default_factory=list)


class LLMRequest(BaseModel):
    prompt: str
    variants: int = 1
    image_paths: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    outputs: list[str]
