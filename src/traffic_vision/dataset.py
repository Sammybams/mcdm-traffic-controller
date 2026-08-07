"""Validation helpers for one-class YOLO detection annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    label_files: int
    empty_images: int
    vehicle_boxes: int


def _validate_annotation_line(line: str, path: Path, line_number: int) -> None:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
    try:
        class_id = int(fields[0])
        x_center, y_center, width, height = (float(value) for value in fields[1:])
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: non-numeric annotation") from error
    if class_id != 0:
        raise ValueError(f"{path}:{line_number}: expected toy_vehicle class 0")
    if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
        raise ValueError(f"{path}:{line_number}: box center must be normalized")
    if not 0 < width <= 1 or not 0 < height <= 1:
        raise ValueError(f"{path}:{line_number}: box size must be in (0, 1]")
    if x_center - width / 2 < 0 or x_center + width / 2 > 1:
        raise ValueError(f"{path}:{line_number}: box exceeds horizontal image bounds")
    if y_center - height / 2 < 0 or y_center + height / 2 > 1:
        raise ValueError(f"{path}:{line_number}: box exceeds vertical image bounds")


def validate_label_directory(label_directory: str | Path) -> DatasetStatistics:
    """Validate every YOLO label file below a directory."""

    directory = Path(label_directory)
    if not directory.is_dir():
        raise ValueError(f"label directory does not exist: {directory}")
    paths = sorted(directory.rglob("*.txt"))
    if not paths:
        raise ValueError(f"no label files found in {directory}")

    empty_images = 0
    vehicle_boxes = 0
    for path in paths:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            empty_images += 1
        for line_number, line in enumerate(lines, start=1):
            _validate_annotation_line(line, path, line_number)
            vehicle_boxes += 1

    return DatasetStatistics(
        label_files=len(paths),
        empty_images=empty_images,
        vehicle_boxes=vehicle_boxes,
    )

