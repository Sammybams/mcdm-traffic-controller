#!/usr/bin/env python3
"""Generate review-only labels using YOLO-World and Grounding DINO."""

from __future__ import annotations

import argparse
import json

from traffic_vision.ai_labelers import (
    GroundingDinoProposalProvider,
    YoloWorldProposalProvider,
)
from traffic_vision.prelabel_batch import generate_prelabel_batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="numeric-folder count dataset")
    parser.add_argument("destination", help="new pre-label output directory")
    parser.add_argument("--world-model", default="yolov8s-worldv2.pt")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="only use Grounding DINO files already in the local model cache",
    )
    args = parser.parse_args()

    report = generate_prelabel_batch(
        args.source,
        args.destination,
        (
            GroundingDinoProposalProvider(local_files_only=args.offline),
            YoloWorldProposalProvider(args.world_model),
        ),
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.incomplete_images == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
