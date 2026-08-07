"""Batch exactly four separately captured road images."""

from __future__ import annotations

from traffic_vision.config import RoadConfig
from traffic_vision.road_processor import process_road
from traffic_vision.schemas import ImageDetections, JunctionResult


def process_four_roads(
    images: dict[str, ImageDetections],
    configs: dict[str, RoadConfig],
    minimum_confidence: float = 0.25,
) -> JunctionResult:
    if len(images) != 4:
        raise ValueError("exactly four road images are required")
    if set(images) != set(configs):
        missing = sorted(set(configs) - set(images))
        unexpected = sorted(set(images) - set(configs))
        raise ValueError(
            f"road IDs do not match configuration; missing={missing}, "
            f"unexpected={unexpected}"
        )

    return JunctionResult(
        roads={
            road_id: process_road(
                image,
                configs[road_id],
                minimum_confidence=minimum_confidence,
            )
            for road_id, image in images.items()
        }
    )

