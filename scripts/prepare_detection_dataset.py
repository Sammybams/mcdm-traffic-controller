#!/usr/bin/env python3
"""Build a provisional YOLO dataset from source images and reviewed labels."""

from __future__ import annotations

import argparse

from traffic_vision.classification_data import plan_classification_split
from traffic_vision.detection_data import materialize_detection_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("labels")
    parser.add_argument("destination")
    parser.add_argument("--temporal-gap-seconds", type=float, default=30)
    args = parser.parse_args()
    split = plan_classification_split(args.source, args.temporal_gap_seconds)
    materialize_detection_dataset(args.source, args.labels, args.destination, split)
    counts = {
        name: sum(entry.split == name for entry in split.entries)
        for name in ("train", "val", "test")
    }
    print(f"materialized detection dataset: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
