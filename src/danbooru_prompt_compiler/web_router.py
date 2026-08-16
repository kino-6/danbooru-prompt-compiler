from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from .llm import LLMClient
from .models import LLMRequest


NEXT_PANEL_WORDS = ("次のコマ", "次コマ", "次の場面", "続きを", "一瞬後", "next panel")
TAG_WORDS = ("タグ", "単語", "解析", "推測", "tag", "analyze")


class WebAction(str, Enum):
    tag_image = "tag_image"
    compile = "compile"
    edit = "edit"
    next_panel = "next_panel"


class ActionPlan(BaseModel):
    action: WebAction
    scene_description: str = ""
    edit_instruction: str | None = None
    variants: int = Field(default=1, ge=1, le=4)
    preserve: list[str] = Field(default_factory=list)
    reason: str = ""

    @field_validator("edit_instruction", mode="before")
    @classmethod
    def normalize_nullable_instruction(cls, value):
        if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
            return None
        return value


@dataclass(frozen=True)
class RouteRequest:
    instruction: str
    base_prompt: str = ""
    has_image: bool = False
    default_variants: int = 1


@dataclass(frozen=True)
class RoutedPlan:
    plan: ActionPlan
    source: str
    warning: str = ""


class NaturalLanguageRouter:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def route(self, request: RouteRequest) -> RoutedPlan:
        if _strong_rule_action(request) is not None:
            return RoutedPlan(plan=route_with_rules(request), source="rules")
        try:
            response = self.llm_client.generate(
                LLMRequest(prompt=_build_router_prompt(request), variants=1)
            )
            raw_output = response.outputs[0] if response.outputs else ""
            plan = _parse_action_plan(raw_output)
            return RoutedPlan(plan=_normalize_plan(plan, request), source="llm")
        except Exception as exc:
            return RoutedPlan(
                plan=route_with_rules(request),
                source="rules",
                warning=f"Router LLM fallback: {exc}",
            )


def route_with_rules(request: RouteRequest) -> ActionPlan:
    instruction = request.instruction.strip()
    variants = min(max(request.default_variants, 1), 4)

    strong_action = _strong_rule_action(request)
    if strong_action == WebAction.next_panel:
        return ActionPlan(
            action=WebAction.next_panel,
            edit_instruction=instruction,
            variants=variants,
            preserve=["character", "appearance", "clothing"],
            reason="次のコマを求める表現を検出",
        )
    if strong_action == WebAction.tag_image:
        return ActionPlan(
            action=WebAction.tag_image,
            reason="画像のタグ抽出を求める表現を検出",
        )
    if request.base_prompt or request.has_image:
        return ActionPlan(
            action=WebAction.edit,
            edit_instruction=instruction,
            variants=variants,
            preserve=["character"],
            reason="既存プロンプトまたは画像に対する変更として解釈",
        )
    return ActionPlan(
        action=WebAction.compile,
        scene_description=instruction,
        variants=variants,
        reason="新規シーン記述として解釈",
    )


def _build_router_prompt(request: RouteRequest) -> str:
    return "\n".join(
        (
            "/no_think",
            "You route Japanese instructions for a Danbooru prompt web application.",
            "Return exactly one JSON object and no markdown or explanation.",
            "Allowed actions:",
            '- "tag_image": extract tags from the uploaded image only',
            '- "compile": create a new tag prompt from natural language',
            '- "edit": modify an existing prompt or tags inferred from an image',
            '- "next_panel": propose the visually natural next manga/anime panel',
            "Schema:",
            '{"action":"tag_image|compile|edit|next_panel",'
            '"scene_description":"string","edit_instruction":"string or null",'
            '"variants":1,"preserve":["character"],"reason":"short Japanese reason"}',
            "Use at most 4 variants. Never create shell commands.",
            f"Image uploaded: {request.has_image}",
            f"Existing prompt: {request.base_prompt or '(none)'}",
            f"Default variants: {request.default_variants}",
            f"User instruction: {request.instruction or '(none)'}",
        )
    )


def _parse_action_plan(raw_output: str) -> ActionPlan:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw_output):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw_output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return ActionPlan.model_validate(value)
    raise ValueError("Router did not return a valid action JSON object")


def _normalize_plan(plan: ActionPlan, request: RouteRequest) -> ActionPlan:
    values = plan.model_dump()
    values["variants"] = min(max(plan.variants or request.default_variants, 1), 4)

    strong_action = _strong_rule_action(request)
    if strong_action is not None:
        values["action"] = strong_action

    if values["action"] == WebAction.tag_image:
        if not request.has_image:
            values["action"] = WebAction.edit if request.base_prompt else WebAction.compile
        elif strong_action is None and request.instruction.strip():
            # A small router model can confuse a requested visual change with image
            # analysis. Explicit tag requests have already been caught by the strong
            # rules above, so an otherwise ambiguous instruction is safer as an edit.
            values["action"] = WebAction.edit
    if values["action"] == WebAction.compile and not plan.scene_description:
        values["scene_description"] = request.instruction
    if values["action"] in {WebAction.edit, WebAction.next_panel} and not plan.edit_instruction:
        values["edit_instruction"] = request.instruction
    if values["action"] == WebAction.next_panel and not plan.preserve:
        values["preserve"] = ["character", "appearance", "clothing"]

    return ActionPlan.model_validate(values)


def _strong_rule_action(request: RouteRequest) -> WebAction | None:
    if not request.has_image:
        return None
    instruction = request.instruction.strip().lower()
    if any(word in instruction for word in NEXT_PANEL_WORDS):
        return WebAction.next_panel
    if not instruction or any(word in instruction for word in TAG_WORDS):
        return WebAction.tag_image
    return None
