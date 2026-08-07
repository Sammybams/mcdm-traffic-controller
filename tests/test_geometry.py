import pytest

from traffic_vision.geometry import (
    apply_homography,
    normalize_image_point,
    point_in_polygon,
    point_to_segment_distance,
)
from traffic_vision.schemas import Point


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
SQUARE = (Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1))


def test_normalizes_pixel_point() -> None:
    assert normalize_image_point(Point(320, 180), 640, 360) == Point(0.5, 0.5)


def test_identity_homography_preserves_point() -> None:
    assert apply_homography(Point(0.25, 0.75), IDENTITY) == Point(0.25, 0.75)


@pytest.mark.parametrize(
    ("point", "expected"),
    [(Point(0.5, 0.5), True), (Point(1, 0.5), True), (Point(1.1, 0.5), False)],
)
def test_point_in_polygon_includes_boundary(point: Point, expected: bool) -> None:
    assert point_in_polygon(point, SQUARE) is expected


def test_point_to_horizontal_segment_distance() -> None:
    distance = point_to_segment_distance(Point(0.5, 0.75), Point(0, 0), Point(1, 0))

    assert distance == pytest.approx(0.75)


def test_point_to_segment_uses_nearest_endpoint() -> None:
    distance = point_to_segment_distance(Point(2, 0), Point(0, 0), Point(1, 0))

    assert distance == pytest.approx(1)

