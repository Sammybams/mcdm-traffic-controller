"""End-to-end evaluation of lane counts and junction distances."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traffic_vision.config import RoadConfig
from traffic_vision.detector import VehicleDetector
from traffic_vision.road_processor import process_road


@dataclass(frozen=True, slots=True)
class ExpectedLane:
    count: int
    nearest_distance: float | None = None


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    image_path: str
    road_id: str
    lanes: dict[str, ExpectedLane]


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    road_samples: int
    lane_samples: int
    exact_lane_count_rate: float
    exact_road_count_rate: float
    count_mae: float
    nearest_distance_mae: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_evaluation_manifest(path: str | Path) -> list[EvaluationSample]:
    with Path(path).open(encoding="utf-8") as manifest_file:
        raw = json.load(manifest_file)
    samples = []
    for item in raw["samples"]:
        samples.append(
            EvaluationSample(
                image_path=str(item["image_path"]),
                road_id=str(item["road_id"]),
                lanes={
                    lane_name: ExpectedLane(
                        count=int(lane["count"]),
                        nearest_distance=(
                            None
                            if lane.get("nearest_distance") is None
                            else float(lane["nearest_distance"])
                        ),
                    )
                    for lane_name, lane in item["lanes"].items()
                },
            )
        )
    if not samples:
        raise ValueError("evaluation manifest cannot be empty")
    return samples


def evaluate_samples(
    samples: list[EvaluationSample],
    detector: VehicleDetector,
    configs: dict[str, RoadConfig],
    minimum_confidence: float = 0.25,
) -> EvaluationSummary:
    if not samples:
        raise ValueError("at least one evaluation sample is required")

    exact_lanes = 0
    exact_roads = 0
    count_absolute_error = 0
    lane_samples = 0
    distance_errors: list[float] = []

    for sample in samples:
        if sample.road_id not in configs:
            raise ValueError(f"missing road configuration: {sample.road_id}")
        result = process_road(
            detector.detect(sample.image_path),
            configs[sample.road_id],
            minimum_confidence,
        )
        road_is_exact = True
        for lane_name in ("left", "right"):
            expected = sample.lanes[lane_name]
            actual = result.lanes[lane_name]
            error = abs(actual.count - expected.count)
            count_absolute_error += error
            lane_samples += 1
            if error == 0:
                exact_lanes += 1
            else:
                road_is_exact = False
            if expected.nearest_distance is not None and actual.nearest_distance is not None:
                distance_errors.append(
                    abs(actual.nearest_distance - expected.nearest_distance)
                )
        if road_is_exact:
            exact_roads += 1

    return EvaluationSummary(
        road_samples=len(samples),
        lane_samples=lane_samples,
        exact_lane_count_rate=exact_lanes / lane_samples,
        exact_road_count_rate=exact_roads / len(samples),
        count_mae=count_absolute_error / lane_samples,
        nearest_distance_mae=(
            sum(distance_errors) / len(distance_errors) if distance_errors else None
        ),
    )

