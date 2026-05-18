from danbooru_prompt_compiler.normalizer import normalize_tags, parse_tag_text


def test_tag_normalization_and_duplicate_removal() -> None:
    raw = [" 1Girl ", "night scene", "night scene", "", "  ", "Rain"]
    assert normalize_tags(raw) == ["1girl", "night_scene", "rain"]


def test_variant_output_parsing_fallback_handles_bullets_and_newlines() -> None:
    text = "- 1girl\n- shrine\n- looking at viewer"
    assert parse_tag_text(text) == ["1girl", "shrine", "looking at viewer"]
