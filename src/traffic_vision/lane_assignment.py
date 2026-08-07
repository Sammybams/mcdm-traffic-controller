"""Assign mapped vehicle positions to configured lane polygons."""

from __future__ import annotations

from math import hypot

from traffic_vision.config import LaneConfig
from traffic_vision.geometry import point_in_polygon
from traffic_vision.schemas import Point


def _centroid(polygon: tuple[Point, ...]) -> Point:
    return Point(
        sum(point.x for point in polygon) / len(polygon),
        sum(point.y for point in polygon) / len(polygon),
    )


def assign_lane(point: Point, lanes: tuple[LaneConfig, ...]) -> str | None:
    """Return the containing lane, with a deterministic boundary tie-break."""

    candidates = [lane for lane in lanes if point_in_polygon(point, lane.polygon)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].name

    return min(
        candidates,
        key=lambda lane: (
            hypot(point.x - _centroid(lane.polygon).x, point.y - _centroid(lane.polygon).y),
            lane.name,
        ),
    ).name

