"""Evaluation metrics for provisional total-count classifiers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traffic_vision.count_classifier import TotalCountClassifier


@dataclass(frozen=True, slots=True)
class CountEvaluationPrediction:
    image_path: str
    expected_count: int
    predicted_count: int
    confidence: float


@dataclass(frozen=True, slots=True)
class CountEvaluationSummary:
    image_count: int
    exact_accuracy: float
    within_one_accuracy: float
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_signed_error: float
    predictions: tuple[CountEvaluationPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def labelled_images(root: str | Path) -> list[tuple[Path, int]]:
    """Read images whose immediate parent directory is the numeric count label."""

    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"evaluation directory does not exist: {directory}")
    labelled_directories: list[tuple[int, Path]] = []
    for class_directory in (path for path in directory.iterdir() if path.is_dir()):
        try:
            expected_count = int(class_directory.name)
        except ValueError as error:
            raise ValueError(
                f"evaluation class directory must be numeric: {class_directory.name}"
            ) from error
        if expected_count < 0:
            raise ValueError("evaluation count cannot be negative")
        labelled_directories.append((expected_count, class_directory))

    samples: list[tuple[Path, int]] = []
    for expected_count, class_directory in sorted(labelled_directories):
        for image_path in sorted(class_directory.glob("*.jpg")):
            samples.append((image_path, expected_count))
    if not samples:
        raise ValueError(f"no evaluation JPEG images found in {directory}")
    return samples


def evaluate_count_classifier(
    classifier: TotalCountClassifier,
    samples: list[tuple[Path, int]],
) -> CountEvaluationSummary:
    if not samples:
        raise ValueError("at least one evaluation sample is required")

    predictions: list[CountEvaluationPrediction] = []
    errors: list[int] = []
    for image_path, expected_count in samples:
        prediction = classifier.predict(image_path)
        error = prediction.count - expected_count
        errors.append(error)
        predictions.append(
            CountEvaluationPrediction(
                image_path=str(image_path),
                expected_count=expected_count,
                predicted_count=prediction.count,
                confidence=prediction.confidence,
            )
        )

    image_count = len(errors)
    return CountEvaluationSummary(
        image_count=image_count,
        exact_accuracy=sum(error == 0 for error in errors) / image_count,
        within_one_accuracy=sum(abs(error) <= 1 for error in errors) / image_count,
        mean_absolute_error=sum(abs(error) for error in errors) / image_count,
        root_mean_squared_error=math.sqrt(
            sum(error * error for error in errors) / image_count
        ),
        mean_signed_error=sum(errors) / image_count,
        predictions=tuple(predictions),
    )
