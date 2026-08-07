import pytest

from traffic_vision.config import LaneConfig, RoadConfig
from traffic_vision.road_processor import process_road
from traffic_vision.schemas import BoundingBox, Detection, ImageDetections, Point


ROAD_CONFIG = RoadConfig(
    road_id="road_1",
    lanes=(
        LaneConfig(
            "left",
            (Point(0, 0), Point(0.49, 0), Point(0.49, 1), Point(0, 1)),
            4,
        ),
        LaneConfig(
            "right",
            (Point(0.51, 0), Point(1, 0), Point(1, 1), Point(0.51, 1)),
            4,
        ),
    ),
    junction_line=(Point(0, 0), Point(1, 0)),
    homography=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    visible_length=1,
)


def _detection(center_x: float, center_y: float, confidence: float = 0.9) -> Detection:
    return Detection(
        BoundingBox(center_x - 10, center_y - 10, center_x + 10, center_y + 10),
        confidence,
    )


def test_processes_counts_density_and_proximity_for_both_lanes() -> None:
    image = ImageDetections(
        width=1000,
        height=1000,
        detections=(_detection(250, 200), _detection(750, 800)),
    )

    result = process_road(image, ROAD_CONFIG)

    assert result.total_count == 2
    assert result.lanes["left"].count == 1
    assert result.lanes["left"].density == pytest.approx(0.25)
    assert result.lanes["left"].nearest_distance == pytest.approx(0.2)
    assert result.lanes["left"].proximity == pytest.approx(0.8)
    assert result.lanes["right"].count == 1
    assert result.lanes["right"].proximity == pytest.approx(0.2)


def test_filters_low_confidence_and_counts_unassigned_detections() -> None:
    image = ImageDetections(
        width=1000,
        height=1000,
        detections=(_detection(250, 200, 0.1), _detection(500, 500, 0.9)),
    )

    result = process_road(image, ROAD_CONFIG, minimum_confidence=0.25)

    assert result.total_count == 0
    assert result.unassigned_count == 1
    assert result.cars == ()

