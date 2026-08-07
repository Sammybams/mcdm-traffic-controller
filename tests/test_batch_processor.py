import pytest

from traffic_vision.batch_processor import process_four_roads
from traffic_vision.config import LaneConfig, RoadConfig
from traffic_vision.schemas import BoundingBox, Detection, ImageDetections, Point


def _road_config(road_id: str) -> RoadConfig:
    return RoadConfig(
        road_id=road_id,
        lanes=(
            LaneConfig(
                "left",
                (Point(0, 0), Point(0.5, 0), Point(0.5, 1), Point(0, 1)),
                5,
            ),
            LaneConfig(
                "right",
                (Point(0.5, 0), Point(1, 0), Point(1, 1), Point(0.5, 1)),
                5,
            ),
        ),
        junction_line=(Point(0, 0), Point(1, 0)),
        homography=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        visible_length=1,
    )


def _image() -> ImageDetections:
    return ImageDetections(
        width=100,
        height=100,
        detections=(Detection(BoundingBox(20, 20, 30, 30), 0.9),),
    )


def test_processes_exactly_four_roads() -> None:
    road_ids = [f"road_{index}" for index in range(1, 5)]
    images = {road_id: _image() for road_id in road_ids}
    configs = {road_id: _road_config(road_id) for road_id in road_ids}

    result = process_four_roads(images, configs)

    assert set(result.roads) == set(road_ids)
    assert all(road.total_count == 1 for road in result.roads.values())


def test_rejects_incomplete_batch() -> None:
    configs = {f"road_{index}": _road_config(f"road_{index}") for index in range(1, 5)}

    with pytest.raises(ValueError, match="exactly four"):
        process_four_roads({"road_1": _image()}, configs)

