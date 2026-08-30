"""Prepare and verify human-reviewable detection-annotation batches."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traffic_vision.count_dataset import jpeg_size
from traffic_vision.dataset import _validate_annotation_line


@dataclass(frozen=True, slots=True)
class AnnotationEntry:
    image_name: str
    source: str
    expected_vehicle_count: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class AnnotationManifest:
    source_root: str
    entries: tuple[AnnotationEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnnotationVerification:
    image_count: int
    box_count: int
    missing_labels: tuple[str, ...]
    count_mismatches: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing_labels and not self.count_mismatches

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["is_complete"] = self.is_complete
        return result


def build_annotation_manifest(source: str | Path) -> AnnotationManifest:
    root = Path(source)
    if not root.is_dir():
        raise ValueError(f"count dataset directory does not exist: {root}")

    entries: list[AnnotationEntry] = []
    for class_directory in sorted(
        (path for path in root.iterdir() if path.is_dir()), key=lambda path: int(path.name)
    ):
        expected_count = int(class_directory.name)
        for image_path in sorted(class_directory.glob("*.jpg")):
            width, height = jpeg_size(image_path)
            entries.append(
                AnnotationEntry(
                    image_name=f"count-{expected_count:02d}__{image_path.name}",
                    source=str(image_path.relative_to(root)),
                    expected_vehicle_count=expected_count,
                    width=width,
                    height=height,
                )
            )
    if not entries:
        raise ValueError(f"no JPEG images found in {root}")
    return AnnotationManifest(source_root=str(root), entries=tuple(entries))


def materialize_annotation_batch(
    manifest: AnnotationManifest, destination: str | Path
) -> None:
    destination_root = Path(destination)
    if destination_root.exists():
        raise ValueError(f"destination already exists: {destination_root}")
    image_directory = destination_root / "images"
    image_directory.mkdir(parents=True)
    source_root = Path(manifest.source_root)
    for entry in manifest.entries:
        shutil.copy2(source_root / entry.source, image_directory / entry.image_name)
    (destination_root / "annotation-manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


def load_annotation_manifest(path: str | Path) -> AnnotationManifest:
    with Path(path).open(encoding="utf-8") as manifest_file:
        raw = json.load(manifest_file)
    return AnnotationManifest(
        source_root=str(raw["source_root"]),
        entries=tuple(AnnotationEntry(**entry) for entry in raw["entries"]),
    )


def verify_annotation_batch(
    manifest: AnnotationManifest, labels: str | Path
) -> AnnotationVerification:
    label_directory = Path(labels)
    missing: list[str] = []
    mismatches: list[str] = []
    box_count = 0
    for entry in manifest.entries:
        label_path = label_directory / f"{Path(entry.image_name).stem}.txt"
        if not label_path.is_file():
            missing.append(label_path.name)
            continue
        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for line_number, line in enumerate(lines, start=1):
            _validate_annotation_line(line, label_path, line_number)
        box_count += len(lines)
        if len(lines) != entry.expected_vehicle_count:
            mismatches.append(
                f"{label_path.name}: expected {entry.expected_vehicle_count}, found {len(lines)}"
            )
    return AnnotationVerification(
        image_count=len(manifest.entries),
        box_count=box_count,
        missing_labels=tuple(missing),
        count_mismatches=tuple(mismatches),
    )
