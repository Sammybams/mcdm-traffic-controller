#!/usr/bin/env python3
"""Run a research whole-image count classifier over four road images."""

from __future__ import annotations

import argparse
import json

from traffic_vision.cli import parse_image_assignments
from traffic_vision.count_classifier import UltralyticsTotalCountClassifier
from traffic_vision.count_service import classify_four_road_images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--image", action="append", required=True, metavar="ROAD_ID=PATH"
    )
    args = parser.parse_args()
    try:
        result = classify_four_road_images(
            UltralyticsTotalCountClassifier(args.model),
            parse_image_assignments(args.image),
        )
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"classify-total-counts: error: {error}") from error
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
