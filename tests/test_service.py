from traffic_vision.config import LaneConfig, RoadConfig
from traffic_vision.detector import FakeVehicleDetector
from traffic_vision.schemas import BoundingBox, Detection, ImageDetections, Point
from traffic_vision.service import measure_image_paths


def _config(road_id: str) -> RoadConfig:
    return RoadConfig(
        road_id=road_id,
        lanes=(
            LaneConfig("left", (Point(0, 0), Point(0.5, 0), Point(0.5, 1), Point(0, 1)), 4),
            LaneConfig("right", (Point(0.5, 0), Point(1, 0), Point(1, 1), Point(0.5, 1)), 4),
        ),
        junction_line=(Point(0, 0), Point(1, 0)),
        homography=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        visible_length=1,
    )


def test_measures_four_image_paths_through_detector() -> None:
    road_ids = [f"road_{index}" for index in range(1, 5)]
    paths = {road_id: f"{road_id}.jpg" for road_id in road_ids}
    image = ImageDetections(
        100,
        100,
        (Detection(BoundingBox(20, 20, 30, 30), 0.9),),
    )
    detector = FakeVehicleDetector({path: image for path in paths.values()})
    configs = {road_id: _config(road_id) for road_id in road_ids}

    result = measure_image_paths(detector, paths, configs)

    assert all(road.lanes["left"].count == 1 for road in result.roads.values())

