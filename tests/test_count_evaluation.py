from pathlib import Path

import pytest

from traffic_vision.count_classifier import CountPrediction, FakeTotalCountClassifier
from traffic_vision.count_evaluation import evaluate_count_classifier, labelled_images


def test_labelled_images_reads_numeric_directories(tmp_path: Path) -> None:
    for label in (0, 2):
        directory = tmp_path / str(label)
        directory.mkdir()
        (directory / f"{label}.jpg").write_bytes(b"test")

    assert labelled_images(tmp_path) == [
        (tmp_path / "0" / "0.jpg", 0),
        (tmp_path / "2" / "2.jpg", 2),
    ]


def test_evaluate_count_classifier_calculates_count_metrics() -> None:
    samples = [(Path("a.jpg"), 2), (Path("b.jpg"), 4), (Path("c.jpg"), 8)]
    classifier = FakeTotalCountClassifier(
        {
            "a.jpg": CountPrediction(2, 0.9, {2: 0.9}),
            "b.jpg": CountPrediction(5, 0.8, {5: 0.8}),
            "c.jpg": CountPrediction(6, 0.7, {6: 0.7}),
        }
    )

    result = evaluate_count_classifier(classifier, samples)

    assert result.image_count == 3
    assert result.exact_accuracy == pytest.approx(1 / 3)
    assert result.within_one_accuracy == pytest.approx(2 / 3)
    assert result.mean_absolute_error == 1
    assert result.root_mean_squared_error == pytest.approx((5 / 3) ** 0.5)
    assert result.mean_signed_error == pytest.approx(-1 / 3)


def test_evaluate_count_classifier_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_count_classifier(FakeTotalCountClassifier({}), [])
