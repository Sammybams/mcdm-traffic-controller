"""Repair incomplete pre-labels from a nearby complete frame of the same setup."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from traffic_vision.classification_data import capture_datetime, plan_classification_split


@dataclass(frozen=True, slots=True)
class PrelabelRepair:
    target: str
    propagated_from: str
    box_count: int


@dataclass(frozen=True, slots=True)
class PrelabelRepairReport:
    repaired_images: int
    repairs: tuple[PrelabelRepair, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _label_name(label: int, source: str) -> str:
    return f"count-{label:02d}__{Path(source).stem}.txt"


def _render_propagated_overlay(image_path: Path, label_path: Path, output: Path) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    lines = [line.split() for line in label_path.read_text().splitlines() if line]
    for index, fields in enumerate(lines, start=1):
        _, x_center, y_center, width, height = map(float, fields)
        box_width = width * image.width
        box_height = height * image.height
        x = x_center * image.width
        y = y_center * image.height
        box = (
            x - box_width / 2,
            y - box_height / 2,
            x + box_width / 2,
            y + box_height / 2,
        )
        draw.rectangle(box, outline="#00ccff", width=3)
        draw.text((box[0] + 2, max(0, box[1] - 12)), f"{index}:propagated", fill="#00ccff")
    draw.rectangle((0, 0, image.width, 22), fill="#000000")
    draw.text((4, 4), f"expected={len(lines)} TEMPORAL PROPAGATION - REVIEW", fill="#ffffff")
    image.save(output, quality=90)


def repair_incomplete_prelabels(
    source: str | Path, prelabel_directory: str | Path
) -> PrelabelRepairReport:
    source_root = Path(source)
    prelabel_root = Path(prelabel_directory)
    report_path = prelabel_root / "prelabel-report.json"
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    report_entries = {entry["source"]: entry for entry in raw["entries"]}
    split = plan_classification_split(source_root)
    groups: dict[tuple[int, int], list[str]] = {}
    for entry in split.entries:
        groups.setdefault((entry.label, entry.group), []).append(entry.source)

    repairs: list[PrelabelRepair] = []
    for target_source, target_entry in report_entries.items():
        if target_entry["complete"]:
            continue
        group_key = next(
            (entry.label, entry.group)
            for entry in split.entries
            if entry.source == target_source
        )
        candidates = [
            source_name
            for source_name in groups[group_key]
            if report_entries[source_name]["complete"]
        ]
        if not candidates:
            raise RuntimeError(f"no complete temporal peer for {target_source}")
        target_time = capture_datetime(source_root / target_source)
        nearest = min(
            candidates,
            key=lambda name: abs(
                (capture_datetime(source_root / name) - target_time).total_seconds()
            ),
        )
        target_label = prelabel_root / "labels" / _label_name(group_key[0], target_source)
        source_label = prelabel_root / "labels" / _label_name(group_key[0], nearest)
        shutil.copy2(source_label, target_label)
        box_count = len([line for line in target_label.read_text().splitlines() if line])
        if box_count != group_key[0]:
            raise RuntimeError(f"propagated label has wrong count: {source_label}")
        overlay_name = f"count-{group_key[0]:02d}__{Path(target_source).name}"
        _render_propagated_overlay(
            source_root / target_source,
            target_label,
            prelabel_root / "overlays" / overlay_name,
        )
        repairs.append(
            PrelabelRepair(
                target=target_source,
                propagated_from=nearest,
                box_count=box_count,
            )
        )

    report = PrelabelRepairReport(len(repairs), tuple(repairs))
    (prelabel_root / "repair-report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report
