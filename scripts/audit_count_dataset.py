#!/usr/bin/env python3
"""Print a reproducible audit of a numeric-folder count dataset."""

from __future__ import annotations

import argparse
import json

from traffic_vision.count_dataset import audit_count_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="directory containing numeric count folders")
    args = parser.parse_args()
    print(json.dumps(audit_count_dataset(args.dataset).to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

