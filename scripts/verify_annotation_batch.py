#!/usr/bin/env python3
"""Validate annotation syntax and box totals against known image counts."""

from __future__ import annotations

import argparse
import json

from traffic_vision.annotation_batch import (
    load_annotation_manifest,
    verify_annotation_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("labels")
    args = parser.parse_args()

    report = verify_annotation_batch(load_annotation_manifest(args.manifest), args.labels)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.is_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
