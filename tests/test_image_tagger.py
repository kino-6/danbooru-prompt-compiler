from pathlib import Path

from danbooru_prompt_compiler.image_tagger import (
    ImageTagger,
    CHARACTER_CATEGORY,
    GENERAL_CATEGORY,
    RATING_CATEGORY,
    TagLabel,
    load_labels,
    prepare_image,
    select_tags,
)


def test_image_tagger_initializes_runtime_once(monkeypatch) -> None:
    tagger = ImageTagger()
    runtime = object()
    calls = 0

    def load_runtime():
        nonlocal calls
        calls += 1
        return runtime

    monkeypatch.setattr(tagger, "_load_runtime", load_runtime)

    assert tagger._get_runtime() is runtime
    assert tagger._get_runtime() is runtime
    assert calls == 1


def test_select_tags_applies_category_thresholds_and_sorts_by_score() -> None:
    labels = [
        TagLabel("general", RATING_CATEGORY),
        TagLabel("sensitive", RATING_CATEGORY),
        TagLabel("1girl", GENERAL_CATEGORY),
        TagLabel("blue_hair", GENERAL_CATEGORY),
        TagLabel("low_score_tag", GENERAL_CATEGORY),
        TagLabel("hatsune_miku_(vocaloid)", CHARACTER_CATEGORY),
        TagLabel("uncertain_character", CHARACTER_CATEGORY),
    ]
    scores = [0.8, 0.2, 0.99, 0.72, 0.2, 0.91, 0.7]

    result = select_tags(
        labels,
        scores,
        general_threshold=0.35,
        character_threshold=0.85,
        max_tags=10,
    )

    assert result.names == ["1girl", "hatsune_miku_(vocaloid)", "blue_hair"]
    assert result.rating is not None
    assert result.rating.name == "general"


def test_select_tags_honors_max_tags() -> None:
    labels = [
        TagLabel("1girl", GENERAL_CATEGORY),
        TagLabel("solo", GENERAL_CATEGORY),
    ]

    result = select_tags(labels, [0.9, 0.8], max_tags=1)

    assert result.names == ["1girl"]


def test_load_labels_reads_wd_csv_without_changing_underscores(tmp_path: Path) -> None:
    labels_path = tmp_path / "selected_tags.csv"
    labels_path.write_text(
        "tag_id,name,category,count\n1,long_hair,0,100\n2,some_character,4,20\n",
        encoding="utf-8",
    )

    assert load_labels(labels_path) == [
        TagLabel("long_hair", GENERAL_CATEGORY),
        TagLabel("some_character", CHARACTER_CATEGORY),
    ]


def test_prepare_image_pads_with_white_and_converts_rgb_to_bgr() -> None:
    from PIL import Image

    image = Image.new("RGBA", (2, 1), (255, 0, 0, 255))

    prepared = prepare_image(image, 2)

    assert prepared.shape == (1, 2, 2, 3)
    assert prepared.dtype.name == "float32"
    assert prepared[0, 0, 0].tolist() == [0.0, 0.0, 255.0]
    assert prepared[0, 1, 0].tolist() == [255.0, 255.0, 255.0]
