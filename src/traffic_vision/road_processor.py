"""Convert model detections for one image into road measurements."""

from __future__ import annotations

from traffic_vision.config import RoadConfig
from traffic_vision.geometry import (
    apply_homography,
    normalize_image_point,
    point_to_segment_distance,
)
from traffic_vision.lane_assignment import assign_lane
from traffic_vision.metrics import calculate_lane_metrics
from traffic_vision.schemas import CarMeasurement, ImageDetections, RoadResult


def process_road(
    image: ImageDetections,
    config: RoadConfig,
    minimum_confidence: float = 0.25,
) -> RoadResult:
    """Measure one two-lane road from model detections."""

    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum confidence must be between zero and one")

    distances_by_lane = {lane.name: [] for lane in config.lanes}
    measurements: list[CarMeasurement] = []
    unassigned_count = 0

    for index, detection in enumerate(image.detections):
        if detection.confidence < minimum_confidence:
            continue
        image_center = detection.bounding_box.center
        normalized_center = normalize_image_point(image_center, image.width, image.height)
        mapped_position = apply_homography(normalized_center, config.homography)
        lane_name = assign_lane(mapped_position, config.lanes)
        if lane_name is None:
            unassigned_count += 1
            continue

        distance = point_to_segment_distance(
            mapped_position, config.junction_line[0], config.junction_line[1]
        )
        distances_by_lane[lane_name].append(distance)
        measurements.append(
            CarMeasurement(
                detection_index=index,
                lane=lane_name,
                confidence=detection.confidence,
                image_center=image_center,
                mapped_position=mapped_position,
                distance_to_junction=distance,
            )
        )

    metrics = {
        lane.name: calculate_lane_metrics(
            distances_by_lane[lane.name],
            maximum_capacity=lane.maximum_capacity,
            visible_length=config.visible_length,
        )
        for lane in config.lanes
    }
    total_count = sum(metric.count for metric in metrics.values())
    return RoadResult(
        road_id=config.road_id,
        distance_unit=config.distance_unit,
        lanes=metrics,
        total_count=total_count,
        cars=tuple(measurements),
        unassigned_count=unassigned_count,
    )

