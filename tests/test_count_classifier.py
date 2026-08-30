from pathlib import Path

import pytest

from traffic_vision.count_classifier import CountPrediction, FakeTotalCountClassifier


def test_count_prediction_validates_values() -> None:
    with pytest.raises(ValueError, match="negative"):
        CountPrediction(-1, 0.5, {})
    with pytest.raises(ValueError, match="confidence"):
        CountPrediction(1, 1.1, {})
    with pytest.raises(ValueError, match="probabilities"):
        CountPrediction(1, 0.5, {1: -0.1})


def test_fake_count_classifier_returns_configured_prediction() -> None:
    expected = CountPrediction(4, 0.8, {3: 0.2, 4: 0.8})
    classifier = FakeTotalCountClassifier({"road.jpg": expected})

    assert classifier.predict(Path("road.jpg")) == expected
    with pytest.raises(KeyError, match="no fake"):
        classifier.predict("missing.jpg")
