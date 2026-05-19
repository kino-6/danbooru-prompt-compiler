from danbooru_prompt_compiler.seed_tags import infer_subset_seed_tags


def test_infer_subset_seed_tags_from_existing_prompt_and_edit_instruction() -> None:
    seeds = infer_subset_seed_tags(
        "1girl mesugaki loli medium_breasts, low_twintails, hair_intakes",
        edit_instruction="悪落ち魔法少女",
    )

    assert seeds == ["mahou_shoujo", "dark_persona"]


def test_infer_subset_seed_tags_uses_edit_keywords_when_room_available() -> None:
    seeds = infer_subset_seed_tags(
        "1girl",
        edit_instruction="悪落ち魔法少女",
        max_seed_tags=4,
    )

    assert seeds == ["mahou_shoujo", "dark_persona", "1girl"]


def test_infer_subset_seed_tags_prefers_scene_words_from_edit_instruction() -> None:
    seeds = infer_subset_seed_tags(
        "1girl, solo",
        edit_instruction="雨の神社で佇む少女",
    )

    assert seeds == ["shrine", "rain"]
