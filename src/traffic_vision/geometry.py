"""Dependency-free geometry used for lane and junction measurements."""

from __future__ import annotations

from math import hypot

from traffic_vision.config import Matrix3x3
from traffic_vision.schemas import Point

_EPSILON = 1e-9


def normalize_image_point(point: Point, width: int, height: int) -> Point:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return Point(point.x / width, point.y / height)


def apply_homography(point: Point, matrix: Matrix3x3) -> Point:
    """Map a point through a projective 3 by 3 transform."""

    denominator = matrix[2][0] * point.x + matrix[2][1] * point.y + matrix[2][2]
    if abs(denominator) <= _EPSILON:
        raise ValueError("homography maps point to infinity")
    return Point(
        x=(matrix[0][0] * point.x + matrix[0][1] * point.y + matrix[0][2])
        / denominator,
        y=(matrix[1][0] * point.x + matrix[1][1] * point.y + matrix[1][2])
        / denominator,
    )


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    cross = (point.y - start.y) * (end.x - start.x) - (
        point.x - start.x
    ) * (end.y - start.y)
    scale = max(1.0, abs(end.x - start.x), abs(end.y - start.y))
    if abs(cross) > _EPSILON * scale:
        return False
    dot = (point.x - start.x) * (end.x - start.x) + (
        point.y - start.y
    ) * (end.y - start.y)
    squared_length = (end.x - start.x) ** 2 + (end.y - start.y) ** 2
    return -_EPSILON <= dot <= squared_length + _EPSILON


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    """Return true for points inside a polygon, including its boundary."""

    if len(polygon) < 3:
        raise ValueError("polygon requires at least three points")

    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        crosses_y = (current.y > point.y) != (previous.y > point.y)
        if crosses_y:
            intersection_x = (
                (previous.x - current.x)
                * (point.y - current.y)
                / (previous.y - current.y)
                + current.x
            )
            if point.x < intersection_x:
                inside = not inside
        previous = current
    return inside


def point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    """Calculate the shortest Euclidean distance to a finite line segment."""

    delta_x = end.x - start.x
    delta_y = end.y - start.y
    squared_length = delta_x**2 + delta_y**2
    if squared_length <= _EPSILON:
        return hypot(point.x - start.x, point.y - start.y)
    projection = (
        (point.x - start.x) * delta_x + (point.y - start.y) * delta_y
    ) / squared_length
    projection = min(1.0, max(0.0, projection))
    closest_x = start.x + projection * delta_x
    closest_y = start.y + projection * delta_y
    return hypot(point.x - closest_x, point.y - closest_y)

