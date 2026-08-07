"""Command-line entry point for processing four road images."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from traffic_vision.config import load_road_configs
from traffic_vision.detector import UltralyticsVehicleDetector
from traffic_vision.service import measure_image_paths


def parse_image_assignments(values: list[str]) -> dict[str, Path]:
    assignments: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"image assignment must use ROAD_ID=PATH: {value}")
        road_id, raw_path = value.split("=", 1)
        if not road_id or not raw_path:
            raise ValueError(f"image assignment must use ROAD_ID=PATH: {value}")
        if road_id in assignments:
            raise ValueError(f"duplicate road image: {road_id}")
        assignments[road_id] = Path(raw_path)
    return assignments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traffic-vision",
        description="Count vehicles and measure density/proximity in four road images.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        metavar="ROAD_ID=PATH",
        help="repeat once for each of the four roads",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        image_paths = parse_image_assignments(args.image)
        configs = load_road_configs(args.config)
        detector = UltralyticsVehicleDetector(args.model, args.confidence)
        result = measure_image_paths(
            detector,
            image_paths,
            configs,
            minimum_confidence=args.confidence,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"traffic-vision: error: {error}") from error

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

