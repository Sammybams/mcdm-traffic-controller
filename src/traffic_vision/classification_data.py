"""Prepare a leakage-aware provisional total-count classification dataset."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_CAPTURE_NAME = re.compile(r"esp32cam_(\d{8})_(\d{6})_(\d{3})\.jpg$")


@dataclass(frozen=True, slots=True)
class SplitEntry:
    source: str
    label: int
    group: int
    split: str


@dataclass(frozen=True, slots=True)
class ClassificationSplit:
    entries: tuple[SplitEntry, ...]
    temporal_gap_seconds: float
    classes_with_group_leakage: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_datetime(path: Path) -> datetime:
    match = _CAPTURE_NAME.match(path.name)
    if not match:
        raise ValueError(f"unsupported capture filename: {path.name}")
    return datetime.strptime(
        f"{match.group(1)}{match.group(2)}{match.group(3)}", "%Y%m%d%H%M%S%f"
    )


def _temporal_groups(paths: list[Path], gap_seconds: float) -> list[list[Path]]:
    ordered = sorted(paths, key=capture_datetime)
    groups: list[list[Path]] = []
    for path in ordered:
        if not groups:
            groups.append([path])
            continue
        gap = (capture_datetime(path) - capture_datetime(groups[-1][-1])).total_seconds()
        if gap > gap_seconds:
            groups.append([path])
        else:
            groups[-1].append(path)
    return groups


def plan_classification_split(
    source: str | Path, temporal_gap_seconds: float = 30
) -> ClassificationSplit:
    if temporal_gap_seconds <= 0:
        raise ValueError("temporal gap must be positive")
    root = Path(source)
    if not root.is_dir():
        raise ValueError(f"source directory does not exist: {root}")

    entries: list[SplitEntry] = []
    leakage_classes: list[int] = []
    for class_directory in sorted(
        (path for path in root.iterdir() if path.is_dir()), key=lambda path: int(path.name)
    ):
        label = int(class_directory.name)
        paths = list(class_directory.glob("*.jpg"))
        if len(paths) < 3:
            raise ValueError(f"class {label} needs at least three images")
        groups = _temporal_groups(paths, temporal_gap_seconds)
        assignment: dict[Path, str] = {}

        if len(groups) >= 3:
            for path in groups[-1]:
                assignment[path] = "test"
            for path in groups[-2]:
                assignment[path] = "val"
            for group in groups[:-2]:
                for path in group:
                    assignment[path] = "train"
        elif len(groups) == 2:
            training_group, held_out_group = groups
            for path in training_group:
                assignment[path] = "train"
            for path in held_out_group[:-1]:
                assignment[path] = "test"
            assignment[held_out_group[-1]] = "val"
            leakage_classes.append(label)
        else:
            ordered = groups[0]
            for path in ordered[:-2]:
                assignment[path] = "train"
            assignment[ordered[-2]] = "val"
            assignment[ordered[-1]] = "test"
            leakage_classes.append(label)

        for group_index, group in enumerate(groups):
            for path in group:
                entries.append(
                    SplitEntry(
                        source=str(path.relative_to(root)),
                        label=label,
                        group=group_index,
                        split=assignment[path],
                    )
                )

    return ClassificationSplit(
        entries=tuple(entries),
        temporal_gap_seconds=temporal_gap_seconds,
        classes_with_group_leakage=tuple(leakage_classes),
    )


def materialize_classification_split(
    source: str | Path,
    destination: str | Path,
    split: ClassificationSplit,
) -> None:
    source_root = Path(source)
    destination_root = Path(destination)
    if destination_root.exists():
        raise ValueError(f"destination already exists: {destination_root}")
    destination_root.mkdir(parents=True)

    for entry in split.entries:
        target_directory = destination_root / entry.split / str(entry.label)
        target_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / entry.source, target_directory / Path(entry.source).name)

    (destination_root / "split-manifest.json").write_text(
        json.dumps(split.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
