"""Count-specific evaluation for object detectors before lane calibration."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traffic_vision.detector import VehicleDetector

_COUNT_NAME = re.compile(r"count-(\d+)__.*\.jpg$")


@dataclass(frozen=True, slots=True)
class DetectorCountPrediction:
    image_path: str
    expected_count: int
    predicted_count: int


@dataclass(frozen=True, slots=True)
class DetectorCountSummary:
    threshold: float
    image_count: int
    exact_accuracy: float
    within_one_accuracy: float
    mean_absolute_error: float
    empty_false_positive_rate: float
    predictions: tuple[DetectorCountPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detector_count_images(directory: str | Path) -> list[tuple[Path, int]]:
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"detector evaluation directory does not exist: {root}")
    samples: list[tuple[Path, int]] = []
    for path in sorted(root.glob("*.jpg")):
        match = _COUNT_NAME.match(path.name)
        if not match:
            raise ValueError(f"image name has no count label: {path.name}")
        samples.append((path, int(match.group(1))))
    if not samples:
        raise ValueError(f"no evaluation images found in {root}")
    return samples


def evaluate_detector_counts(
    detector: VehicleDetector,
    samples: list[tuple[Path, int]],
    threshold: float,
) -> DetectorCountSummary:
    if not samples:
        raise ValueError("at least one detector evaluation sample is required")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")

    predictions: list[DetectorCountPrediction] = []
    errors: list[int] = []
    empty_predictions: list[int] = []
    for image_path, expected_count in samples:
        detections = detector.detect(image_path)
        predicted_count = sum(
            detection.confidence >= threshold for detection in detections.detections
        )
        error = predicted_count - expected_count
        errors.append(error)
        if expected_count == 0:
            empty_predictions.append(predicted_count)
        predictions.append(
            DetectorCountPrediction(
                str(image_path), expected_count, predicted_count
            )
        )
    image_count = len(samples)
    return DetectorCountSummary(
        threshold=threshold,
        image_count=image_count,
        exact_accuracy=sum(error == 0 for error in errors) / image_count,
        within_one_accuracy=sum(abs(error) <= 1 for error in errors) / image_count,
        mean_absolute_error=sum(abs(error) for error in errors) / image_count,
        empty_false_positive_rate=(
            sum(count > 0 for count in empty_predictions) / len(empty_predictions)
            if empty_predictions
            else 0.0
        ),
        predictions=tuple(predictions),
    )


def choose_detector_threshold(
    summaries: list[DetectorCountSummary],
) -> DetectorCountSummary:
    if not summaries:
        raise ValueError("at least one threshold summary is required")
    return max(
        summaries,
        key=lambda summary: (
            -summary.mean_absolute_error,
            -summary.empty_false_positive_rate,
            summary.exact_accuracy,
            summary.threshold,
        ),
    )
