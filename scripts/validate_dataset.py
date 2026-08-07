#!/usr/bin/env python3
"""Validate one-class YOLO annotations before training."""

from __future__ import annotations

import argparse

from traffic_vision.dataset import validate_label_directory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", help="root labels directory containing train/val/test")
    args = parser.parse_args()

    statistics = validate_label_directory(args.labels)
    print(f"label files: {statistics.label_files}")
    print(f"empty images: {statistics.empty_images}")
    print(f"vehicle boxes: {statistics.vehicle_boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

