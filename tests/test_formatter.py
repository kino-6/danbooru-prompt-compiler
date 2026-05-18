from danbooru_prompt_compiler.formatter import OutputFormat, format_variant, group_tags


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
        "copy: 1girl, shrine, standing",
        "",
        "copy_lines:",
        "1girl",
        "standing",
        "shrine",
        "",
        "subject: 1girl",
        "pose: standing",
        "scene: shrine",
    ]


def test_format_variant_can_render_flat_output() -> None:
    assert format_variant(["1girl", "shrine"], OutputFormat.flat) == "1girl, shrine"
