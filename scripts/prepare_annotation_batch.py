#!/usr/bin/env python3
"""Copy count-labelled captures into a flat object-annotation batch."""

from __future__ import annotations

import argparse

from traffic_vision.annotation_batch import (
    build_annotation_manifest,
    materialize_annotation_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="numeric-folder count dataset")
    parser.add_argument("destination", help="new annotation-batch directory")
    args = parser.parse_args()

    manifest = build_annotation_manifest(args.source)
    materialize_annotation_batch(manifest, args.destination)
    print(f"prepared {len(manifest.entries)} images in {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
