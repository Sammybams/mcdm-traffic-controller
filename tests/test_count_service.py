import pytest

from traffic_vision.count_classifier import CountPrediction, FakeTotalCountClassifier
from traffic_vision.count_service import classify_four_road_images


def test_classifies_exactly_four_whole_road_images() -> None:
    paths = {f"road_{index}": f"road-{index}.jpg" for index in range(1, 5)}
    classifier = FakeTotalCountClassifier(
        {
            path: CountPrediction(index, 0.9, {index: 0.9})
            for index, path in enumerate(paths.values(), start=1)
        }
    )

    result = classify_four_road_images(classifier, paths)

    assert result.roads["road_3"].count == 3
    assert result.output_scope == "combined_left_and_right_lanes"


def test_rejects_incomplete_whole_road_batch() -> None:
    with pytest.raises(ValueError, match="exactly four"):
        classify_four_road_images(FakeTotalCountClassifier({}), {"road_1": "one.jpg"})
