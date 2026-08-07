"""Road geometry configuration and JSON loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traffic_vision.schemas import Point

Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True, slots=True)
class LaneConfig:
    name: str
    polygon: tuple[Point, ...]
    maximum_capacity: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("lane name cannot be empty")
        if len(self.polygon) < 3:
            raise ValueError("lane polygon requires at least three points")
        if self.maximum_capacity <= 0:
            raise ValueError("lane maximum capacity must be positive")


@dataclass(frozen=True, slots=True)
class RoadConfig:
    road_id: str
    lanes: tuple[LaneConfig, ...]
    junction_line: tuple[Point, Point]
    homography: Matrix3x3
    visible_length: float
    distance_unit: str = "normalized"

    def __post_init__(self) -> None:
        if not self.road_id:
            raise ValueError("road ID cannot be empty")
        lane_names = [lane.name for lane in self.lanes]
        if set(lane_names) != {"left", "right"} or len(lane_names) != 2:
            raise ValueError("a road must define exactly left and right lanes")
        if self.visible_length <= 0:
            raise ValueError("visible lane length must be positive")
        if not self.distance_unit:
            raise ValueError("distance unit cannot be empty")


def _point(raw: list[float]) -> Point:
    if len(raw) != 2:
        raise ValueError("a point must contain exactly two coordinates")
    return Point(float(raw[0]), float(raw[1]))


def road_config_from_dict(raw: dict[str, Any]) -> RoadConfig:
    lanes = tuple(
        LaneConfig(
            name=name,
            polygon=tuple(_point(point) for point in lane["polygon"]),
            maximum_capacity=int(lane["maximum_capacity"]),
        )
        for name, lane in raw["lanes"].items()
    )
    matrix_rows = raw.get("homography", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if len(matrix_rows) != 3 or any(len(row) != 3 for row in matrix_rows):
        raise ValueError("homography must be a 3 by 3 matrix")
    homography: Matrix3x3 = tuple(  # type: ignore[assignment]
        tuple(float(value) for value in row) for row in matrix_rows
    )
    junction_points = tuple(_point(point) for point in raw["junction_line"])
    if len(junction_points) != 2:
        raise ValueError("junction line requires exactly two points")

    return RoadConfig(
        road_id=str(raw["road_id"]),
        lanes=lanes,
        junction_line=junction_points,  # type: ignore[arg-type]
        homography=homography,
        visible_length=float(raw["visible_length"]),
        distance_unit=str(raw.get("distance_unit", "normalized")),
    )


def load_road_configs(path: str | Path) -> dict[str, RoadConfig]:
    """Load and validate a four-road configuration file."""

    with Path(path).open(encoding="utf-8") as config_file:
        raw = json.load(config_file)

    configs = {
        item["road_id"]: road_config_from_dict(item) for item in raw["roads"]
    }
    if len(configs) != len(raw["roads"]):
        raise ValueError("road IDs must be unique")
    if len(configs) != 4:
        raise ValueError("configuration must contain exactly four roads")
    return configs

