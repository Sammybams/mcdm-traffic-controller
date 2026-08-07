"""Derive lane density and proximity from assigned vehicle distances."""

from __future__ import annotations

from traffic_vision.schemas import LaneMetrics


def normalized_density(count: int, maximum_capacity: int) -> float:
    if count < 0:
        raise ValueError("vehicle count cannot be negative")
    if maximum_capacity <= 0:
        raise ValueError("maximum capacity must be positive")
    return min(1.0, count / maximum_capacity)


def normalized_proximity(nearest_distance: float | None, visible_length: float) -> float:
    if visible_length <= 0:
        raise ValueError("visible length must be positive")
    if nearest_distance is None:
        return 0.0
    if nearest_distance < 0:
        raise ValueError("distance cannot be negative")
    return min(1.0, max(0.0, 1.0 - nearest_distance / visible_length))


def calculate_lane_metrics(
    distances: list[float], maximum_capacity: int, visible_length: float
) -> LaneMetrics:
    nearest = min(distances) if distances else None
    return LaneMetrics(
        count=len(distances),
        density=normalized_density(len(distances), maximum_capacity),
        nearest_distance=nearest,
        proximity=normalized_proximity(nearest, visible_length),
    )

