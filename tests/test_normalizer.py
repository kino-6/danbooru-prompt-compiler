from danbooru_prompt_compiler.normalizer import normalize_tags, parse_tag_text


def test_tag_normalization_and_duplicate_removal() -> None:
    raw = [" 1Girl ", "night scene", "night scene", "", "  ", "Rain"]
    assert normalize_tags(raw) == ["1girl", "night_scene", "rain"]


def test_tag_normalization_applies_common_aliases() -> None:
    raw = ["single_girl", "one_girl", "1girl_alone", "shrine"]
    assert normalize_tags(raw) == ["1girl", "shrine"]


def test_tag_normalization_removes_non_tag_prose() -> None:
    raw = [
        "here_is_the_scene_description_converted_into_canonical_danbooru-style_image_tags:",
        "1girl",
        "shrine",
        "added",
        "subtle",
        "note_that_i_excluded_japanese_characters",
        "operation_words",
        "this_output_only_includes_the_requested_tags_and_omits_operation_words",
        "based_on_the_input_prompt",
        "prose",
        "explanations",
        "bullets",
        "quotes",
        "if_you_are_looking_for_a_different_type_of_image",
    ]
    assert normalize_tags(raw) == ["1girl", "shrine"]


def test_variant_output_parsing_fallback_handles_bullets_and_newlines() -> None:
    text = "- 1girl\n- shrine\n- looking at viewer"
    assert parse_tag_text(text) == ["1girl", "shrine", "looking at viewer"]
