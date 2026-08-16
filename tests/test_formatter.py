from danbooru_prompt_compiler.formatter import (
    OutputFormat,
    format_clipboard_text,
    format_suggestion,
    format_variant,
    group_tags,
)


def test_group_tags_sorts_prompt_tags_into_copy_friendly_groups() -> None:
    groups = group_tags(
        [
            "1girl",
            "solo",
            "long_hair",
            "standing",
            "looking_at_viewer",
            "shrine",
            "rain",
            "dramatic_lighting",
            "umbrella",
            "evening",
        ]
    )

    assert groups == {
        "subject": ["1girl", "solo"],
        "appearance": ["long_hair"],
        "pose": ["standing", "looking_at_viewer"],
        "scene": ["shrine", "rain", "umbrella", "evening"],
        "style": ["dramatic_lighting"],
    }


def test_format_variant_includes_copy_line_and_grouped_sections() -> None:
    formatted = format_variant(["1girl", "shrine", "standing"])

    assert formatted.splitlines() == [
        "===",
        "1girl",
        "standing",
        "shrine",
        "===",
        "",
        "subject: 1girl",
        "pose: standing",
        "scene: shrine",
    ]


def test_format_variant_can_render_flat_output() -> None:
    assert format_variant(["1girl", "shrine"], OutputFormat.flat) == "1girl, shrine"


def test_format_suggestion_includes_instruction_and_prompt_preview() -> None:
    formatted = format_suggestion(1, "鳥居の奥に淡い霧を足す", ["1girl", "shrine", "fog"])

    assert formatted.splitlines() == [
        "=== suggestion 1 ===",
        "edit: 鳥居の奥に淡い霧を足す",
        "===",
        "1girl",
        "shrine",
        "fog",
        "===",
        "",
        "subject: 1girl",
        "scene: shrine",
        "other: fog",
    ]


def test_format_clipboard_text_uses_grouped_copy_lines() -> None:
    assert format_clipboard_text(["1girl", "shrine", "standing"]) == "1girl\nstanding\nshrine"


def test_format_clipboard_text_can_render_flat_output() -> None:
    assert format_clipboard_text(["1girl", "shrine"], OutputFormat.flat) == "1girl, shrine"
