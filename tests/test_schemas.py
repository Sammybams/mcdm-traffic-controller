import pytest

from traffic_vision.schemas import BoundingBox, Detection, ImageDetections, Point


def test_bounding_box_center() -> None:
    box = BoundingBox(10, 20, 30, 60)

    assert box.center == Point(20, 40)


def test_bounding_box_rejects_zero_area() -> None:
    with pytest.raises(ValueError, match="positive width"):
        BoundingBox(10, 20, 10, 60)


def test_detection_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        Detection(BoundingBox(0, 0, 1, 1), confidence=1.1)


def test_image_detections_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        ImageDetections(width=0, height=100, detections=())

