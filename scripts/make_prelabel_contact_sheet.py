#!/usr/bin/env python3
"""Create a contact sheet with one overlay per temporal arrangement group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from traffic_vision.classification_data import plan_classification_split


def representative_overlay_names(source: Path) -> list[tuple[int, int, str]]:
    split = plan_classification_split(source)
    representatives: dict[tuple[int, int], str] = {}
    for entry in split.entries:
        representatives.setdefault(
            (entry.label, entry.group),
            f"count-{entry.label:02d}__{Path(entry.source).name}",
        )
    return [
        (label, group, name)
        for (label, group), name in sorted(representatives.items())
    ]


def make_contact_sheet(
    source: Path, prelabel_directory: Path, output: Path, columns: int = 5
) -> int:
    if columns <= 0:
        raise ValueError("columns must be positive")
    report = json.loads(
        (prelabel_directory / "prelabel-report.json").read_text(encoding="utf-8")
    )
    entries = {entry["image_name"]: entry for entry in report["entries"]}
    representatives = representative_overlay_names(source)
    tile_width, tile_height = 320, 260
    rows = (len(representatives) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#202020")
    draw = ImageDraw.Draw(sheet)

    for index, (label, group, image_name) in enumerate(representatives):
        overlay_path = prelabel_directory / "overlays" / image_name
        with Image.open(overlay_path) as opened:
            overlay = opened.convert("RGB")
        overlay.thumbnail((tile_width, tile_height - 36))
        left = (index % columns) * tile_width
        top = (index // columns) * tile_height
        sheet.paste(overlay, (left, top + 20))
        entry = entries[image_name]
        status = "OK" if entry["complete"] else "INCOMPLETE"
        draw.text(
            (left + 4, top + 3),
            f"count={label} group={group} {status}",
            fill="#ffffff" if entry["complete"] else "#ff6666",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return len(representatives)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("prelabels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    count = make_contact_sheet(args.source, args.prelabels, args.output, args.columns)
    print(f"wrote {count} representative overlays to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
