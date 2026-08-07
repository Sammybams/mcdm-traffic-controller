"""Application orchestration from four image paths to one junction result."""

from __future__ import annotations

from pathlib import Path

from traffic_vision.batch_processor import process_four_roads
from traffic_vision.config import RoadConfig
from traffic_vision.detector import VehicleDetector
from traffic_vision.schemas import JunctionResult


def measure_image_paths(
    detector: VehicleDetector,
    image_paths: dict[str, str | Path],
    configs: dict[str, RoadConfig],
    minimum_confidence: float = 0.25,
) -> JunctionResult:
    """Run detection and deterministic measurement for four road images."""

    detections = {
        road_id: detector.detect(image_path)
        for road_id, image_path in image_paths.items()
    }
    return process_four_roads(
        detections,
        configs,
        minimum_confidence=minimum_confidence,
    )

