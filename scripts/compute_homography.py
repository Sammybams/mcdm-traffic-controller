#!/usr/bin/env python3
"""Calculate the config homography from four pixel-to-road point pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_vision.calibration import compute_homography_from_pixels
from traffic_vision.schemas import Point


def _four_points(raw: list[list[float]]) -> tuple[Point, Point, Point, Point]:
    if len(raw) != 4 or any(len(point) != 2 for point in raw):
        raise ValueError("exactly four two-dimensional points are required")
    points = tuple(Point(float(point[0]), float(point[1])) for point in raw)
    return points  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration", type=Path)
    args = parser.parse_args()

    with args.calibration.open(encoding="utf-8") as calibration_file:
        raw = json.load(calibration_file)
    width, height = raw["image_size"]
    matrix = compute_homography_from_pixels(
        _four_points(raw["image_points"]),
        _four_points(raw["road_points"]),
        int(width),
        int(height),
    )
    print(json.dumps(matrix, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

