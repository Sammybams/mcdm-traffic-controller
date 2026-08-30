#!/usr/bin/env python3
"""Create train/val/test folders for provisional total-count classification."""

from __future__ import annotations

import argparse
import json

from traffic_vision.classification_data import (
    materialize_classification_split,
    plan_classification_split,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--temporal-gap-seconds", type=float, default=30)
    args = parser.parse_args()

    split = plan_classification_split(args.source, args.temporal_gap_seconds)
    materialize_classification_split(args.source, args.destination, split)
    counts: dict[str, int] = {}
    for entry in split.entries:
        counts[entry.split] = counts.get(entry.split, 0) + 1
    print(
        json.dumps(
            {
                "counts": counts,
                "classes_with_group_leakage": split.classes_with_group_leakage,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

