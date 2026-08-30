#!/usr/bin/env python3
"""Apply reviewed box overrides to difficult repeated arrangements."""

from __future__ import annotations

import argparse
import json

from traffic_vision.prelabel_override import apply_prelabel_overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("prelabels")
    parser.add_argument("config")
    args = parser.parse_args()
    report = apply_prelabel_overrides(args.source, args.prelabels, args.config)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
