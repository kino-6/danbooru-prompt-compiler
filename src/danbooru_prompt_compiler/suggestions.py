from __future__ import annotations

from .llm import LLMClient
from .models import LLMRequest

TAG_IDEA_MAP = {
    "umbrella": "和傘を持たせて雨を受ける構図にする",
    "holding_umbrella": "傘を少し傾けて顔に影を落とす",
    "torii": "鳥居を大きく画面に入れる",
    "stone_lantern": "石灯籠の灯りで横顔を照らす",
    "lantern": "提灯の灯りを背景に足す",
    "wet": "濡れた髪と服を強調する",
    "cloud": "低い雲で空を重くする",
    "cloudy_sky": "曇り空をさらに重くする",
    "tree": "雨に濡れた木々を背景に足す",
    "forest": "森に囲まれた神社にする",
    "stairs": "石段の上に立たせる",
    "shide": "紙垂を手前に揺らす",
    "shimenawa": "しめ縄を目立たせる",
    "japanese_clothes": "和装の雰囲気を強める",
    "kimono": "濡れた着物の質感を足す",
    "wide_sleeves": "広い袖を風雨で揺らす",
    "hair_ornament": "髪飾りに小さな鈴を足す",
    "flower": "雨に濡れた花を手前に置く",
    "grass": "濡れた草木を足す",
    "leaf": "舞う葉を少し足す",
    "open_mouth": "少し口を開けた表情にする",
    "smile": "かすかな笑顔にする",
    "blush": "雨の中で頬を赤らめる",
    "dramatic_lighting": "ドラマチックな逆光を足す",
}


def suggest_edit_instructions(
    llm_client: LLMClient,
    *,
    base_prompt: str,
    edit_instruction: str | None,
    current_tags: list[str],
    reference_tags: list[str],
    count: int = 3,
) -> list[str]:
    candidates = _candidate_suggestions(current_tags=current_tags, reference_tags=reference_tags)
    prompt = _build_suggestion_prompt(
        base_prompt=base_prompt,
        edit_instruction=edit_instruction,
        current_tags=current_tags,
        reference_tags=reference_tags,
        candidates=candidates,
        count=count,
    )
    response = llm_client.generate(LLMRequest(prompt=prompt, variants=1))
    if not response.outputs:
        return candidates[:count]
    suggestions = _parse_suggestions(response.outputs[0], limit=count, candidates=candidates)
    return _fill_suggestions(suggestions, candidates=candidates, limit=count)


def _build_suggestion_prompt(
    *,
    base_prompt: str,
    edit_instruction: str | None,
    current_tags: list[str],
    reference_tags: list[str],
    candidates: list[str],
    count: int,
) -> str:
    return "\n".join(
        part
        for part in (
            "Generate short Japanese natural-language edit ideas for an anime image prompt.",
            "Output Japanese only.",
            "Choose from Candidate ideas exactly as written when candidates are available.",
            "Each idea must be a fluent Japanese verb phrase that can be pasted into --edit.",
            "Do not output Danbooru tags. Do not explain. Do not add quotes.",
            "Good examples:",
            "- 鳥居の奥に淡い霧を足す",
            "- 濡れた石畳の反射を強める",
            "- 赤い和傘を持たせる",
            f"Return exactly {count} lines.",
            f"Base prompt: {base_prompt}" if base_prompt else "",
            f"Current edit: {edit_instruction}" if edit_instruction else "",
            f"Current tags: {', '.join(current_tags)}" if current_tags else "",
            f"Reference tags: {', '.join(reference_tags[:30])}" if reference_tags else "",
            "Candidate ideas:\n" + "\n".join(f"- {candidate}" for candidate in candidates[:20])
            if candidates
            else "",
        )
        if part
    )


def _parse_suggestions(raw_text: str, *, limit: int, candidates: list[str]) -> list[str]:
    candidate_set = set(candidates)
    suggestions: list[str] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        line = line.removeprefix("- ").removeprefix("* ")
        if ". " in line[:4]:
            line = line.split(". ", 1)[1].strip()
        line = line.strip("\"'")
        if not line:
            continue
        if candidate_set and line not in candidate_set:
            continue
        suggestions.append(line)
        if len(suggestions) >= limit:
            break
    return suggestions


def _fill_suggestions(suggestions: list[str], *, candidates: list[str], limit: int) -> list[str]:
    filled = list(suggestions)
    for candidate in candidates:
        if len(filled) >= limit:
            break
        if candidate not in filled:
            filled.append(candidate)
    return filled


def _candidate_suggestions(*, current_tags: list[str], reference_tags: list[str]) -> list[str]:
    context = set(current_tags) | set(reference_tags)
    candidates: list[str] = []

    if {"shrine", "rain"} <= context:
        candidates.extend(
            [
                "鳥居の奥に淡い霧を足す",
                "濡れた石畳の反射を強める",
                "雨粒を強く描写する",
            ]
        )
    if {"shrine", "night"} <= context:
        candidates.append("夜の神社に灯りを点々と足す")
    if {"rain", "night"} <= context:
        candidates.append("暗い雨空に月明かりを差し込ませる")

    for tag in reference_tags:
        idea = TAG_IDEA_MAP.get(tag)
        if idea:
            candidates.append(idea)

    return _dedupe(candidates)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
