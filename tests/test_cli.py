from pathlib import Path

import pytest
from typer.testing import CliRunner

from danbooru_prompt_compiler import cli
from danbooru_prompt_compiler.cli import _infer_input_type, _read_input_value
from danbooru_prompt_compiler.image_tagger import GENERAL_CATEGORY, ImageTagResult, PredictedTag
from danbooru_prompt_compiler.models import InputType
from danbooru_prompt_compiler.reference_tags import load_reference_tags


def test_load_reference_tags_rejects_file_and_live_seed_tags_together(tmp_path: Path) -> None:
    subset_path = tmp_path / "subset.json"

    with pytest.raises(ValueError, match="Use only one of --tag-subset"):
        load_reference_tags(
            tag_subset=subset_path,
            subset_tags=["shrine"],
            auto_subset=False,
            scene_description="",
            edit_instruction=None,
            tag_subset_limit="10",
            subset_posts=10,
            subset_min_count=1,
            max_tags="auto",
        )


def test_load_reference_tags_reads_subset_file(tmp_path: Path) -> None:
    subset_path = tmp_path / "subset.json"
    subset_path.write_text(
        '{"tags": [{"name": "torii", "count": 4}, {"name": "outdoors", "count": 3}]}',
        encoding="utf-8",
    )

    tags = load_reference_tags(
        tag_subset=subset_path,
        subset_tags=[],
        auto_subset=False,
        scene_description="",
        edit_instruction=None,
        tag_subset_limit="10",
        subset_posts=10,
        subset_min_count=1,
        max_tags="12",
    )

    assert tags.tags == ["torii", "outdoors"]
    assert tags.max_output_tags == 12


def test_load_reference_tags_rejects_bad_auto_int() -> None:
    with pytest.raises(ValueError, match="--max-tags must be an integer or 'auto'"):
        load_reference_tags(
            tag_subset=None,
            subset_tags=[],
            auto_subset=False,
            scene_description="",
            edit_instruction=None,
            tag_subset_limit="auto",
            subset_posts=10,
            subset_min_count=1,
            max_tags="many",
        )


def test_read_input_value_allows_empty_base_prompt() -> None:
    assert _read_input_value(None) == ""


def test_infer_input_type_treats_edit_with_base_as_prompt() -> None:
    assert (
        _infer_input_type(
            input_value="1girl, solo",
            edit_instruction="雨を追加",
            explicit_input_type=InputType.scene,
        )
        == InputType.prompt
    )


def test_infer_input_type_keeps_empty_edit_as_scene_instruction() -> None:
    assert (
        _infer_input_type(
            input_value="",
            edit_instruction="雨の神社",
            explicit_input_type=InputType.scene,
        )
        == InputType.scene
    )


def test_image_mode_outputs_inferred_tags_without_ollama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"test placeholder")

    class StubImageTagger:
        def __init__(self, model_repo: str) -> None:
            assert model_repo == "test/model"

        def predict(self, path: Path, **options) -> ImageTagResult:
            assert path == image_path
            assert options["general_threshold"] == 0.4
            return ImageTagResult(
                tags=[PredictedTag("1girl", 0.99, GENERAL_CATEGORY)],
                rating=None,
            )

    monkeypatch.setattr(cli, "ImageTagger", StubImageTagger)

    result = CliRunner().invoke(
        cli.app,
        [
            "--image",
            str(image_path),
            "--tagger-model",
            "test/model",
            "--general-threshold",
            "0.4",
            "--format",
            "flat",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "1girl"
