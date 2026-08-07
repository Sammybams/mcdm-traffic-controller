import pytest

from traffic_vision.metrics import (
    calculate_lane_metrics,
    normalized_density,
    normalized_proximity,
)


def test_density_is_count_over_capacity() -> None:
    assert normalized_density(3, 6) == pytest.approx(0.5)


def test_density_is_clamped_when_count_exceeds_capacity() -> None:
    assert normalized_density(8, 6) == 1.0


def test_proximity_increases_near_junction() -> None:
    assert normalized_proximity(20, 100) == pytest.approx(0.8)
    assert normalized_proximity(80, 100) == pytest.approx(0.2)


def test_empty_lane_has_zero_proximity_and_null_distance() -> None:
    metrics = calculate_lane_metrics([], maximum_capacity=6, visible_length=100)

    assert metrics.count == 0
    assert metrics.density == 0
    assert metrics.nearest_distance is None
    assert metrics.proximity == 0


def test_lane_metrics_uses_nearest_vehicle() -> None:
    metrics = calculate_lane_metrics([60, 10, 30], 6, 100)

    assert metrics.count == 3
    assert metrics.nearest_distance == 10
    assert metrics.proximity == pytest.approx(0.9)

