#!/usr/bin/env python3
"""Evaluate detector total counts and optionally choose a validation threshold."""

from __future__ import annotations

import argparse
import json

from traffic_vision.detector import UltralyticsVehicleDetector
from traffic_vision.detector_count_evaluation import (
    choose_detector_threshold,
    detector_count_images,
    evaluate_detector_counts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument(
        "--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50"
    )
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    detector = UltralyticsVehicleDetector(args.model, minimum_confidence=min(thresholds))
    samples = detector_count_images(args.images)
    summaries = [
        evaluate_detector_counts(detector, samples, threshold)
        for threshold in thresholds
    ]
    selected = choose_detector_threshold(summaries)
    print(
        json.dumps(
            {
                "selected_threshold": selected.threshold,
                "selected_summary": selected.to_dict(),
                "threshold_results": [summary.to_dict() for summary in summaries],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
