"""Apply reviewed normalized boxes to known difficult temporal arrangements."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traffic_vision.classification_data import plan_classification_split
from traffic_vision.prelabel_repair import render_yolo_overlay


@dataclass(frozen=True, slots=True)
class OverrideResult:
    label: int
    group: int
    image_count: int
    box_count_per_image: int
    reason: str


@dataclass(frozen=True, slots=True)
class OverrideReport:
    overridden_images: int
    results: tuple[OverrideResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_prelabel_overrides(
    source: str | Path,
    prelabel_directory: str | Path,
    config_path: str | Path,
) -> OverrideReport:
    source_root = Path(source)
    prelabel_root = Path(prelabel_directory)
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    prelabel_report_path = prelabel_root / "prelabel-report.json"
    prelabel_report = json.loads(prelabel_report_path.read_text(encoding="utf-8"))
    report_entries = {entry["source"]: entry for entry in prelabel_report["entries"]}
    split = plan_classification_split(source_root)
    results: list[OverrideResult] = []

    for override in raw["overrides"]:
        label = int(override["label"])
        group = int(override["group"])
        boxes = override["boxes"]
        reason = str(override["reason"])
        if len(boxes) != label:
            raise ValueError(f"override class {label} must contain {label} boxes")
        lines: list[str] = []
        for box in boxes:
            if len(box) != 4 or any(not 0 < float(value) <= 1 for value in box):
                raise ValueError("override boxes must contain four normalized values")
            lines.append("0 " + " ".join(f"{float(value):.6f}" for value in box))
        sources = [
            entry.source
            for entry in split.entries
            if entry.label == label and entry.group == group
        ]
        if not sources:
            raise ValueError(f"override group does not exist: class={label} group={group}")
        for source_name in sources:
            stem = Path(source_name).stem
            label_path = prelabel_root / "labels" / f"count-{label:02d}__{stem}.txt"
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            image_name = f"count-{label:02d}__{Path(source_name).name}"
            render_yolo_overlay(
                source_root / source_name,
                label_path,
                prelabel_root / "overlays" / image_name,
                "reviewed override",
                "#cc66ff",
            )
            report_entries[source_name]["selected_count"] = label
            report_entries[source_name]["complete"] = True
            report_entries[source_name]["label_source"] = "reviewed_override"
        results.append(OverrideResult(label, group, len(sources), len(boxes), reason))

    report = OverrideReport(
        overridden_images=sum(result.image_count for result in results),
        results=tuple(results),
    )
    complete_count = sum(entry["complete"] for entry in prelabel_report["entries"])
    prelabel_report["count_complete_images"] = complete_count
    prelabel_report["incomplete_images"] = len(prelabel_report["entries"]) - complete_count
    prelabel_report_path.write_text(
        json.dumps(prelabel_report, indent=2) + "\n", encoding="utf-8"
    )
    (prelabel_root / "override-report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report
