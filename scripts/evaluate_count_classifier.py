#!/usr/bin/env python3
"""Evaluate a numeric-class image classifier on a labelled directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_vision.count_classifier import UltralyticsTotalCountClassifier
from traffic_vision.count_evaluation import evaluate_count_classifier, labelled_images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    classifier = UltralyticsTotalCountClassifier(args.model)
    summary = evaluate_count_classifier(classifier, labelled_images(args.data))
    rendered = json.dumps(summary.to_dict(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
