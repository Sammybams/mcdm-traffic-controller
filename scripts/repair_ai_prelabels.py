#!/usr/bin/env python3
"""Propagate labels into incomplete repeated frames and require review."""

from __future__ import annotations

import argparse
import json

from traffic_vision.prelabel_repair import repair_incomplete_prelabels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("prelabels")
    args = parser.parse_args()
    report = repair_incomplete_prelabels(args.source, args.prelabels)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
