from pathlib import Path
from unittest.mock import patch

import pytest

from traffic_vision.detector import FakeVehicleDetector, UltralyticsVehicleDetector
from traffic_vision.schemas import ImageDetections


def test_fake_detector_returns_configured_result() -> None:
    expected = ImageDetections(100, 50, ())
    detector = FakeVehicleDetector({"road.jpg": expected})

    assert detector.detect(Path("road.jpg")) == expected


def test_fake_detector_rejects_unknown_image() -> None:
    detector = FakeVehicleDetector({})

    with pytest.raises(KeyError, match="no fake detector result"):
        detector.detect("missing.jpg")


def test_ultralytics_detector_validates_confidence_before_import() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        UltralyticsVehicleDetector("model.pt", minimum_confidence=2)


def test_ultralytics_detector_preserves_native_import_error() -> None:
    with patch(
        "builtins.__import__",
        side_effect=ImportError("libxcb.so.1 is missing"),
    ):
        with pytest.raises(RuntimeError, match="native dependency.*libxcb"):
            UltralyticsVehicleDetector("model.pt")
