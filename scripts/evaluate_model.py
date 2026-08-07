#!/usr/bin/env python3
"""Evaluate final lane measurements, not only detector mAP."""

from __future__ import annotations

import argparse
import json

from traffic_vision.config import load_road_configs
from traffic_vision.detector import UltralyticsVehicleDetector
from traffic_vision.evaluation import evaluate_samples, load_evaluation_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--roads", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    summary = evaluate_samples(
        load_evaluation_manifest(args.manifest),
        UltralyticsVehicleDetector(args.model, args.confidence),
        load_road_configs(args.roads),
        minimum_confidence=args.confidence,
    )
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

