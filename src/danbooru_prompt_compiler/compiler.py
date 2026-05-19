from __future__ import annotations

from pathlib import Path

from .llm import LLMClient
from .models import CompileRequest, CompileResult, InputType, LLMRequest
from .normalizer import normalize_tags, parse_tag_text
from .tag_dictionary import load_or_fetch_tag_dictionary

BASE_DIR = Path(__file__).resolve().parents[2]
TAG_DICT_PATH = BASE_DIR / "data" / "tags.json"
PRESET_DIR = BASE_DIR / "presets"
SYSTEM_PROMPT = (
    "You are a Danbooru prompt editor. "
    "If there is no existing prompt, convert the natural-language instruction into organized Danbooru-style image tags. "
    "If there is an existing prompt, preserve its useful tags and add only the natural-language instruction changes. "
    "Return comma-separated ASCII tags with lowercase letters, numbers, and underscores. "
    "Do not output Japanese, natural-language nouns, prose, explanations, bullets, or quotes. "
    "Do not include operation words such as add, added, change, changed, edit, subtle, remix, or composition as tags. "
    "Use high-confidence existing Danbooru tags only; omit uncertain or invented tags. "
    "Prefer canonical tags such as 1girl, solo, shrine, rain, standing, looking_at_viewer, "
    "long_hair, night, city, and dramatic_lighting."
)


def _load_simple_yaml(path: Path) -> dict:
    data: dict = {}
    current_list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and current_list_key:
            data.setdefault(current_list_key, []).append(line[2:].strip().strip('"'))
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"')
            if value == "":
                data[key] = []
                current_list_key = key
            else:
                data[key] = value
                current_list_key = None
    return data


class PromptCompiler:
    def __init__(self, llm_client: LLMClient, tag_dictionary: set[str]) -> None:
        self.llm_client = llm_client
        self.tag_dictionary = tag_dictionary

    @classmethod
    def from_files(cls, llm_client: LLMClient, tag_dict_path: Path = TAG_DICT_PATH) -> "PromptCompiler":
        tags = load_or_fetch_tag_dictionary(tag_dict_path)
        return cls(llm_client=llm_client, tag_dictionary=tags)

    def compile(self, request: CompileRequest) -> CompileResult:
        prompt = self.build_prompt(request)
        llm_response = self.llm_client.generate(LLMRequest(prompt=prompt, variants=request.variants))

        all_variants: list[list[str]] = []
        unknown: list[str] = []
        for output in llm_response.outputs:
            normalized = normalize_tags(parse_tag_text(output))[: request.max_output_tags]
            all_variants.append(normalized)
            unknown.extend(self._unknown_tags(normalized, known_unknown=unknown))

        return CompileResult(variants=all_variants, unknown_tags=unknown)

    def build_prompt(self, request: CompileRequest) -> str:
        source_label = "Existing prompt" if request.input_type == InputType.prompt else "Scene"
        source_text = f"{source_label}: {request.scene_description}" if request.scene_description else ""
        return "\n".join(
            part
            for part in (
                SYSTEM_PROMPT,
                f"Mode: {request.mode.value}.",
                f"Input type: {request.input_type.value}.",
                f"Output at most {request.max_output_tags} tags.",
                self._preset_prompt_text(request.preset_name),
                self._tag_subset_prompt_text(request.tag_subset, max_output_tags=request.max_output_tags),
                source_text,
                self._edit_prompt_text(request.edit_instruction),
            )
            if part
        )

    def load_preset(self, preset_name: str) -> dict:
        preset_path = PRESET_DIR / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_name}")
        return _load_simple_yaml(preset_path)

    def _preset_prompt_text(self, preset_name: str | None) -> str:
        if not preset_name:
            return ""

        preset = self.load_preset(preset_name)
        preferred_tags = ", ".join(preset.get("preferred_tags", []))
        return f"Preset guidance: {preset.get('guidance', '')}\nPreferred tags: {preferred_tags}"

    @staticmethod
    def _edit_prompt_text(edit_instruction: str | None) -> str:
        if not edit_instruction:
            return ""

        return (
            f"Edit instruction: {edit_instruction}\n"
            "Apply the edit while preserving useful existing tags that do not conflict."
        )

    @staticmethod
    def _tag_subset_prompt_text(tag_subset: list[str], *, max_output_tags: int) -> str:
        if not tag_subset:
            return ""

        return (
            "Reference tag subset from matching Danbooru posts: "
            f"{', '.join(tag_subset)}\n"
            "This subset is a menu, not the output. Do not copy the full subset. "
            "Use it to improve the prompt with practical, frequently co-occurring Danbooru tags. "
            "When editing an existing prompt, keep the base prompt and add 3-6 relevant supporting tags from this subset if they fit. "
            f"Select only the most relevant tags for the input, up to {max_output_tags} tags total."
        )

    def _unknown_tags(self, tags: list[str], *, known_unknown: list[str]) -> list[str]:
        return [
            tag
            for tag in tags
            if tag not in self.tag_dictionary and tag not in known_unknown
        ]
