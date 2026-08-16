from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .compiler import PromptCompiler
from .formatter import OutputFormat, SUBJECT_TAGS, format_variant, group_tags
from .image_tagger import CHARACTER_CATEGORY, ImageTagResult, ImageTagger
from .llm import OllamaClient
from .models import CompileMode, CompileRequest, InputType
from .web_router import ActionPlan, NaturalLanguageRouter, RouteRequest, RoutedPlan, WebAction


DEFAULT_ROUTER_MODEL = "qwen3:1.7b"
DEFAULT_COMPILER_MODEL = "qwen3:1.7b"
INSTRUCTION_TAG_HINTS = {
    "振り返": "looking_back",
    "走": "running",
    "座": "sitting",
    "立たせ": "standing",
    "立ち上": "standing",
    "笑": "smile",
    "泣": "crying",
    "驚": "surprised",
    "手を振": "waving",
}


@dataclass(frozen=True)
class WebRunResult:
    action_plan: dict[str, object]
    inferred_tags: str
    output: str
    status: str


class WebPromptService:
    def __init__(
        self,
        *,
        tagger: ImageTagger | None = None,
        router_factory: Callable[[str, str], NaturalLanguageRouter] | None = None,
        compiler_factory: Callable[[str, str], PromptCompiler] | None = None,
    ) -> None:
        self.tagger = tagger or ImageTagger()
        self.router_factory = router_factory or _default_router_factory
        self.compiler_factory = compiler_factory or _default_compiler_factory

    def run(
        self,
        *,
        image_path: str | None,
        instruction: str,
        base_prompt: str,
        router_model: str = DEFAULT_ROUTER_MODEL,
        compiler_model: str = DEFAULT_COMPILER_MODEL,
        ollama_url: str = "http://localhost:11434",
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        max_image_tags: int = 50,
        variants: int = 3,
    ) -> WebRunResult:
        clean_instruction = (instruction or "").strip()
        clean_base_prompt = (base_prompt or "").strip()
        if not image_path and not clean_instruction and not clean_base_prompt:
            raise ValueError("画像、指示、または既存プロンプトを入力してください。")
        route_request = RouteRequest(
            instruction=clean_instruction,
            base_prompt=clean_base_prompt,
            has_image=bool(image_path),
            default_variants=variants,
        )
        router = self.router_factory(ollama_url, router_model)
        routed = router.route(route_request)

        image_result = self._tag_image(
            image_path,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            max_image_tags=max_image_tags,
        )
        inferred_names = image_result.names if image_result else []
        inferred_text = ", ".join(inferred_names)

        if routed.plan.action == WebAction.tag_image:
            if not image_result:
                raise ValueError("画像タグ抽出には画像をアップロードしてください。")
            output = format_variant(inferred_names, OutputFormat.grouped)
            return WebRunResult(
                action_plan=_plan_dict(routed),
                inferred_tags=inferred_text,
                output=output,
                status=_status_text(routed, image_result=image_result),
            )

        compiler = self.compiler_factory(ollama_url, compiler_model)
        compile_request = _build_compile_request(
            routed.plan,
            instruction=clean_instruction,
            base_prompt=clean_base_prompt,
            inferred_tags=inferred_names,
        )
        compile_result = compiler.compile(compile_request)
        output_variants = compile_result.variants
        if routed.plan.action == WebAction.next_panel and image_result:
            output_variants = _stabilize_next_panel_variants(
                output_variants,
                plan=routed.plan,
                image_result=image_result,
                known_tags=compiler.tag_dictionary,
                required_tags=_instruction_tag_hints(
                    routed.plan.edit_instruction or clean_instruction
                ),
            )
        rendered_variants = [
            _render_variant(index, tags, multiple=len(output_variants) > 1)
            for index, tags in enumerate(output_variants, start=1)
        ]
        status = _status_text(routed, image_result=image_result)
        if compile_result.unknown_tags:
            status += f"\n\nUnknown tags: {', '.join(compile_result.unknown_tags)}"
        return WebRunResult(
            action_plan=_plan_dict(routed),
            inferred_tags=inferred_text,
            output="\n\n".join(rendered_variants),
            status=status,
        )

    def _tag_image(
        self,
        image_path: str | None,
        *,
        general_threshold: float,
        character_threshold: float,
        max_image_tags: int,
    ) -> ImageTagResult | None:
        if not image_path:
            return None
        return self.tagger.predict(
            Path(image_path),
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            max_tags=max_image_tags,
        )


def _default_router_factory(ollama_url: str, model: str) -> NaturalLanguageRouter:
    return NaturalLanguageRouter(
        OllamaClient(
            base_url=ollama_url,
            model=model,
            timeout=120.0,
            temperature=0.0,
            json_schema=ActionPlan.model_json_schema(),
            think=False,
        )
    )


def _default_compiler_factory(ollama_url: str, model: str) -> PromptCompiler:
    return PromptCompiler.from_files(
        OllamaClient(
            base_url=ollama_url,
            model=model,
            timeout=300.0,
            temperature=0.0,
            think=False,
        )
    )


def _build_compile_request(
    plan: ActionPlan,
    *,
    instruction: str,
    base_prompt: str,
    inferred_tags: list[str],
) -> CompileRequest:
    if plan.action == WebAction.compile:
        return CompileRequest(
            scene_description=plan.scene_description or instruction,
            variants=plan.variants,
            input_type=InputType.scene,
        )

    source_tags = base_prompt or ", ".join(inferred_tags)
    if not source_tags:
        raise ValueError("編集または次コマ生成には、画像か既存プロンプトが必要です。")

    edit_instruction = plan.edit_instruction or instruction
    mode = CompileMode.composition if plan.action == WebAction.next_panel else CompileMode.subtle
    if plan.action == WebAction.next_panel:
        preserve_text = ", ".join(plan.preserve) or "character, appearance, clothing"
        tag_hints = _instruction_tag_hints(edit_instruction or "")
        hint_text = (
            f"明示された動作には次のDanbooruタグを必ず使う: {', '.join(tag_hints)}。"
            if tag_hints
            else ""
        )
        edit_instruction = (
            "次のコマとして自然な一瞬後の場面にする。"
            f"維持する要素: {preserve_text}。"
            f"ユーザーの希望: {edit_instruction or '自然な続きを提案する'}。"
            f"{hint_text}"
            "画像として直接確認できるDanbooruタグだけを出力し、"
            "moment_afterやnext_panelのような抽象的な進行説明をタグにしない。"
        )

    return CompileRequest(
        scene_description=source_tags,
        variants=plan.variants,
        mode=mode,
        input_type=InputType.prompt,
        edit_instruction=edit_instruction,
        max_output_tags=min(max(len(inferred_tags) + 8, 20), 50),
    )


def _render_variant(index: int, tags: list[str], *, multiple: bool) -> str:
    prefix = f"[variant {index}]\n" if multiple else ""
    return prefix + format_variant(tags, OutputFormat.grouped)


def _stabilize_next_panel_variants(
    variants: list[list[str]],
    *,
    plan: ActionPlan,
    image_result: ImageTagResult,
    known_tags: set[str],
    required_tags: list[str],
) -> list[list[str]]:
    inferred_names = image_result.names
    grouped = group_tags(inferred_names)
    preserved: list[str] = []

    if "character" in plan.preserve:
        preserved.extend(tag for tag in inferred_names if tag in SUBJECT_TAGS)
        preserved.extend(
            tag.name for tag in image_result.tags if tag.category == CHARACTER_CATEGORY
        )
    if "appearance" in plan.preserve:
        preserved.extend(grouped.get("appearance", []))
    if "clothing" in plan.preserve:
        preserved.extend(grouped.get("clothing", []))

    preserved = list(dict.fromkeys(preserved))
    required_tags = [tag for tag in required_tags if tag in known_tags]
    stabilized: list[list[str]] = []
    for variant in variants:
        known_variant = [tag for tag in variant if tag in known_tags]
        stable_prefix = list(dict.fromkeys([*preserved, *required_tags]))
        stabilized.append(
            stable_prefix + [tag for tag in known_variant if tag not in stable_prefix]
        )
    return stabilized


def _instruction_tag_hints(instruction: str) -> list[str]:
    return list(
        dict.fromkeys(
            tag
            for keyword, tag in INSTRUCTION_TAG_HINTS.items()
            if keyword in instruction
        )
    )


def _plan_dict(routed: RoutedPlan) -> dict[str, object]:
    data = routed.plan.model_dump(mode="json")
    data["router_source"] = routed.source
    if routed.warning:
        data["warning"] = routed.warning
    return data


def _status_text(routed: RoutedPlan, *, image_result: ImageTagResult | None) -> str:
    lines = [f"Router: {routed.source}", f"Action: {routed.plan.action.value}"]
    if routed.warning:
        lines.append(routed.warning)
    if image_result:
        lines.append(f"Image tags: {len(image_result.tags)}")
        if image_result.rating:
            lines.append(
                f"Rating estimate: {image_result.rating.name} ({image_result.rating.score:.3f})"
            )
    return "  \n".join(lines)
