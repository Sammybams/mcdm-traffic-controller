from traffic_vision.config import LaneConfig
from traffic_vision.lane_assignment import assign_lane
from traffic_vision.schemas import Point


LEFT = LaneConfig(
    name="left",
    polygon=(Point(0, 0), Point(0.5, 0), Point(0.5, 1), Point(0, 1)),
    maximum_capacity=6,
)
RIGHT = LaneConfig(
    name="right",
    polygon=(Point(0.5, 0), Point(1, 0), Point(1, 1), Point(0.5, 1)),
    maximum_capacity=6,
)
LANES = (LEFT, RIGHT)


def test_assigns_left_and_right_lanes() -> None:
    assert assign_lane(Point(0.25, 0.5), LANES) == "left"
    assert assign_lane(Point(0.75, 0.5), LANES) == "right"


def test_returns_none_outside_lane_polygons() -> None:
    assert assign_lane(Point(1.5, 0.5), LANES) is None


def test_divider_tie_break_is_deterministic() -> None:
    assert assign_lane(Point(0.5, 0.5), LANES) == "left"

