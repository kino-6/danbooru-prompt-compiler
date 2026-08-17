from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_TAGGER_MODEL = "SmilingWolf/wd-vit-tagger-v3"
DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85

GENERAL_CATEGORY = 0
CHARACTER_CATEGORY = 4
RATING_CATEGORY = 9

MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"


@dataclass(frozen=True)
class TagLabel:
    name: str
    category: int


@dataclass(frozen=True)
class PredictedTag:
    name: str
    score: float
    category: int


@dataclass(frozen=True)
class ImageTagResult:
    tags: list[PredictedTag]
    rating: PredictedTag | None

    @property
    def names(self) -> list[str]:
        return [tag.name for tag in self.tags]


@dataclass(frozen=True)
class ImageTaggerRuntime:
    labels: list[TagLabel]
    session: Any
    input_name: str
    output_name: str
    target_size: int


class ImageTagger:
    """Run a Waifu Diffusion ONNX tagger against a local image."""

    def __init__(
        self,
        model_repo: str = DEFAULT_TAGGER_MODEL,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_repo = model_repo
        self.cache_dir = cache_dir
        self._runtime: ImageTaggerRuntime | None = None

    def predict(
        self,
        image_path: Path,
        *,
        general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
        character_threshold: float = DEFAULT_CHARACTER_THRESHOLD,
        max_tags: int | None = 50,
    ) -> ImageTagResult:
        _validate_threshold("general_threshold", general_threshold)
        _validate_threshold("character_threshold", character_threshold)
        if max_tags is not None and max_tags < 1:
            raise ValueError("max_tags must be at least 1")

        try:
            import numpy as np
            from PIL import Image, ImageOps
        except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError(
                "Image tagging dependencies are missing; run 'uv sync --group test'."
            ) from exc

        runtime = self._get_runtime()

        with Image.open(image_path) as opened_image:
            image = ImageOps.exif_transpose(opened_image).convert("RGBA")
            image_array = prepare_image(image, runtime.target_size)

        predictions = runtime.session.run(
            [runtime.output_name],
            {runtime.input_name: image_array},
        )[0]
        scores = np.asarray(predictions[0], dtype=float)
        if len(runtime.labels) != len(scores):
            raise ValueError(
                f"Model returned {len(scores)} scores for {len(runtime.labels)} tag labels."
            )

        return select_tags(
            runtime.labels,
            scores,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            max_tags=max_tags,
        )

    def _get_runtime(self) -> ImageTaggerRuntime:
        if self._runtime is None:
            self._runtime = self._load_runtime()
        return self._runtime

    def _load_runtime(self) -> ImageTaggerRuntime:
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError(
                "Image tagging dependencies are missing; run 'uv sync --group test'."
            ) from exc

        download_args = {"repo_id": self.model_repo, "cache_dir": self.cache_dir}
        label_path = hf_hub_download(filename=LABEL_FILENAME, **download_args)
        model_path = hf_hub_download(filename=MODEL_FILENAME, **download_args)
        labels = load_labels(Path(label_path))
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        model_input = session.get_inputs()[0]
        target_height, target_width = model_input.shape[1:3]
        if not isinstance(target_height, int) or not isinstance(target_width, int):
            raise ValueError(f"Unsupported dynamic model input shape: {model_input.shape}")
        if target_height != target_width:
            raise ValueError(f"Expected a square model input, got: {model_input.shape}")
        return ImageTaggerRuntime(
            labels=labels,
            session=session,
            input_name=model_input.name,
            output_name=session.get_outputs()[0].name,
            target_size=target_height,
        )


def load_labels(path: Path) -> list[TagLabel]:
    labels: list[TagLabel] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"name", "category"}.issubset(reader.fieldnames):
            raise ValueError(f"Invalid WD tag label file: {path}")
        for row in reader:
            labels.append(TagLabel(name=row["name"], category=int(row["category"])))
    return labels


def prepare_image(image, target_size: int):
    """Match the official WD ONNX preprocessing: white square canvas and BGR."""
    import numpy as np
    from PIL import Image

    canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
    canvas.alpha_composite(image)
    rgb_image = canvas.convert("RGB")

    max_dimension = max(rgb_image.size)
    pad_left = (max_dimension - rgb_image.width) // 2
    pad_top = (max_dimension - rgb_image.height) // 2
    padded = Image.new("RGB", (max_dimension, max_dimension), (255, 255, 255))
    padded.paste(rgb_image, (pad_left, pad_top))

    if max_dimension != target_size:
        padded = padded.resize((target_size, target_size), Image.Resampling.BICUBIC)

    rgb_array = np.asarray(padded, dtype=np.float32)
    bgr_array = rgb_array[:, :, ::-1].copy()
    return np.expand_dims(bgr_array, axis=0)


def select_tags(
    labels: Sequence[TagLabel],
    scores: Sequence[float],
    *,
    general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
    character_threshold: float = DEFAULT_CHARACTER_THRESHOLD,
    max_tags: int | None = 50,
) -> ImageTagResult:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")

    selected: list[PredictedTag] = []
    ratings: list[PredictedTag] = []
    for label, raw_score in zip(labels, scores, strict=True):
        prediction = PredictedTag(
            name=label.name,
            score=float(raw_score),
            category=label.category,
        )
        if label.category == RATING_CATEGORY:
            ratings.append(prediction)
        elif label.category == GENERAL_CATEGORY and prediction.score > general_threshold:
            selected.append(prediction)
        elif label.category == CHARACTER_CATEGORY and prediction.score > character_threshold:
            selected.append(prediction)

    selected.sort(key=lambda tag: tag.score, reverse=True)
    if max_tags is not None:
        selected = selected[:max_tags]
    rating = max(ratings, key=lambda tag: tag.score, default=None)
    return ImageTagResult(tags=selected, rating=rating)


def _validate_threshold(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
