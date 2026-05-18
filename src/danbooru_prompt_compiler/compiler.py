from __future__ import annotations

import json
from pathlib import Path

from .llm import LLMClient
from .models import CompileRequest, CompileResult, LLMRequest
from .normalizer import normalize_tags, parse_tag_text

BASE_DIR = Path(__file__).resolve().parents[2]
TAG_DICT_PATH = BASE_DIR / "data" / "tags.json"
PRESET_DIR = BASE_DIR / "presets"


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
        tags = set(json.loads(tag_dict_path.read_text(encoding="utf-8")))
        return cls(llm_client=llm_client, tag_dictionary=tags)

    def compile(self, request: CompileRequest) -> CompileResult:
        preset_text = ""
        if request.preset_name:
            preset = self.load_preset(request.preset_name)
            preset_text = f"Preset guidance: {preset.get('guidance', '')}\nPreferred tags: {', '.join(preset.get('preferred_tags', []))}\n"

        system_prompt = (
            "Convert the scene description into Danbooru-style tags only. "
            "Return comma-separated tags with no prose."
        )
        prompt = f"{system_prompt}\nMode: {request.mode.value}.\n{preset_text}Scene: {request.scene_description}"
        llm_response = self.llm_client.generate(LLMRequest(prompt=prompt, variants=request.variants))

        all_variants: list[list[str]] = []
        unknown: list[str] = []
        for output in llm_response.outputs:
            normalized = normalize_tags(parse_tag_text(output))
            all_variants.append(normalized)
            for tag in normalized:
                if tag not in self.tag_dictionary and tag not in unknown:
                    unknown.append(tag)

        return CompileResult(variants=all_variants, unknown_tags=unknown)

    def load_preset(self, preset_name: str) -> dict:
        preset_path = PRESET_DIR / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_name}")
        return _load_simple_yaml(preset_path)
