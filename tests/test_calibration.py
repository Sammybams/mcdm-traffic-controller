import pytest

from traffic_vision.calibration import compute_homography, compute_homography_from_pixels
from traffic_vision.geometry import apply_homography
from traffic_vision.schemas import Point


def test_computes_rectangle_scale_homography() -> None:
    source = (Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1))
    target = (Point(0, 0), Point(40, 0), Point(40, 60), Point(0, 60))

    matrix = compute_homography(source, target)

    mapped = apply_homography(Point(0.25, 0.5), matrix)
    assert mapped.x == pytest.approx(10)
    assert mapped.y == pytest.approx(30)


def test_rejects_duplicate_calibration_points() -> None:
    source = (Point(0, 0), Point(0, 0), Point(1, 1), Point(0, 1))
    target = (Point(0, 0), Point(40, 0), Point(40, 60), Point(0, 60))

    with pytest.raises(ValueError, match="unique homography"):
        compute_homography(source, target)


def test_computes_homography_from_pixel_points() -> None:
    image = (Point(0, 0), Point(200, 0), Point(200, 100), Point(0, 100))
    road = (Point(0, 0), Point(40, 0), Point(40, 60), Point(0, 60))

    matrix = compute_homography_from_pixels(image, road, 200, 100)

    mapped = apply_homography(Point(0.5, 0.5), matrix)
    assert mapped.x == pytest.approx(20)
    assert mapped.y == pytest.approx(30)

