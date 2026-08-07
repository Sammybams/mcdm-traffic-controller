from traffic_vision.config import LaneConfig, RoadConfig
from traffic_vision.detector import FakeVehicleDetector
from traffic_vision.evaluation import EvaluationSample, ExpectedLane, evaluate_samples
from traffic_vision.schemas import BoundingBox, Detection, ImageDetections, Point


def test_evaluates_end_to_end_lane_counts() -> None:
    config = RoadConfig(
        road_id="road_1",
        lanes=(
            LaneConfig("left", (Point(0, 0), Point(0.5, 0), Point(0.5, 1), Point(0, 1)), 5),
            LaneConfig("right", (Point(0.5, 0), Point(1, 0), Point(1, 1), Point(0.5, 1)), 5),
        ),
        junction_line=(Point(0, 0), Point(1, 0)),
        homography=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        visible_length=1,
    )
    detector = FakeVehicleDetector(
        {
            "test.jpg": ImageDetections(
                100,
                100,
                (Detection(BoundingBox(20, 20, 30, 30), 0.9),),
            )
        }
    )
    sample = EvaluationSample(
        image_path="test.jpg",
        road_id="road_1",
        lanes={"left": ExpectedLane(1, 0.25), "right": ExpectedLane(0)},
    )

    summary = evaluate_samples([sample], detector, {"road_1": config})

    assert summary.exact_lane_count_rate == 1
    assert summary.exact_road_count_rate == 1
    assert summary.count_mae == 0
    assert summary.nearest_distance_mae == 0

