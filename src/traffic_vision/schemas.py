"""Small, model-independent data contracts used by the vision pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not isfinite(self.x) or not isfinite(self.y):
            raise ValueError("point coordinates must be finite")


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        coordinates = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(isfinite(value) for value in coordinates):
            raise ValueError("bounding-box coordinates must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must have positive width and height")

    @property
    def center(self) -> Point:
        return Point(
            x=(self.x_min + self.x_max) / 2,
            y=(self.y_min + self.y_max) / 2,
        )


@dataclass(frozen=True, slots=True)
class Detection:
    bounding_box: BoundingBox
    confidence: float
    label: str = "toy_vehicle"

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("detection confidence must be between zero and one")
        if not self.label:
            raise ValueError("detection label cannot be empty")


@dataclass(frozen=True, slots=True)
class ImageDetections:
    width: int
    height: int
    detections: tuple[Detection, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")


@dataclass(frozen=True, slots=True)
class CarMeasurement:
    detection_index: int
    lane: str
    confidence: float
    image_center: Point
    mapped_position: Point
    distance_to_junction: float


@dataclass(frozen=True, slots=True)
class LaneMetrics:
    count: int
    density: float
    nearest_distance: float | None
    proximity: float


@dataclass(frozen=True, slots=True)
class RoadResult:
    road_id: str
    distance_unit: str
    lanes: dict[str, LaneMetrics]
    total_count: int
    density: float
    nearest_distance: float | None
    proximity: float
    cars: tuple[CarMeasurement, ...]
    unassigned_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class JunctionResult:
    roads: dict[str, RoadResult]

    def __post_init__(self) -> None:
        if len(self.roads) != 4:
            raise ValueError("junction result must contain exactly four roads")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
