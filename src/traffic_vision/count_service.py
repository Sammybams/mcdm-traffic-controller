"""Four-image orchestration for research-only whole-road count models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traffic_vision.count_classifier import CountPrediction, TotalCountClassifier


@dataclass(frozen=True, slots=True)
class FourRoadCountPrediction:
    roads: dict[str, CountPrediction]
    output_scope: str = "combined_left_and_right_lanes"

    def __post_init__(self) -> None:
        if len(self.roads) != 4:
            raise ValueError("exactly four road images are required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_four_road_images(
    classifier: TotalCountClassifier,
    image_paths: dict[str, str | Path],
) -> FourRoadCountPrediction:
    if len(image_paths) != 4:
        raise ValueError("exactly four road images are required")
    return FourRoadCountPrediction(
        roads={
            road_id: classifier.predict(image_path)
            for road_id, image_path in image_paths.items()
        }
    )
