from pathlib import Path

import pytest

from traffic_vision.detector import FakeVehicleDetector
from traffic_vision.detector_count_evaluation import (
    choose_detector_threshold,
    detector_count_images,
    evaluate_detector_counts,
)
from traffic_vision.schemas import BoundingBox, Detection, ImageDetections


def _image(confidences: list[float]) -> ImageDetections:
    return ImageDetections(
        100,
        100,
        tuple(
            Detection(BoundingBox(index * 10, 10, index * 10 + 5, 20), confidence)
            for index, confidence in enumerate(confidences)
        ),
    )


def test_reads_count_from_materialized_detector_image_name(tmp_path: Path) -> None:
    (tmp_path / "count-03__capture.jpg").write_bytes(b"image")

    assert detector_count_images(tmp_path) == [
        (tmp_path / "count-03__capture.jpg", 3)
    ]


def test_evaluates_and_selects_detector_count_threshold() -> None:
    samples = [(Path("empty.jpg"), 0), (Path("cars.jpg"), 2)]
    detector = FakeVehicleDetector(
        {
            "empty.jpg": _image([0.15]),
            "cars.jpg": _image([0.9, 0.8, 0.15]),
        }
    )

    low = evaluate_detector_counts(detector, samples, 0.1)
    high = evaluate_detector_counts(detector, samples, 0.5)

    assert low.empty_false_positive_rate == 1
    assert high.exact_accuracy == 1
    assert high.mean_absolute_error == 0
    assert choose_detector_threshold([low, high]) == high


def test_rejects_invalid_detector_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        evaluate_detector_counts(FakeVehicleDetector({}), [(Path("x"), 0)], 1.1)


def test_threshold_selection_avoids_large_count_error() -> None:
    template = evaluate_detector_counts(
        FakeVehicleDetector({"a.jpg": _image([])}), [(Path("a.jpg"), 0)], 0.5
    )
    lower_mae = type(template)(
        threshold=0.5,
        image_count=10,
        exact_accuracy=0.8,
        within_one_accuracy=1,
        mean_absolute_error=0.2,
        empty_false_positive_rate=0,
        predictions=(),
    )
    high_exact_but_catastrophic = type(lower_mae)(
        threshold=0.6,
        image_count=2,
        exact_accuracy=0.5,
        within_one_accuracy=0.5,
        mean_absolute_error=2.5,
        empty_false_positive_rate=0,
        predictions=(),
    )

    assert choose_detector_threshold([high_exact_but_catastrophic, lower_mae]) == lower_mae
