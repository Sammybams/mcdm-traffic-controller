"""Materialize a YOLO detection dataset from reviewed pre-labels."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from traffic_vision.classification_data import ClassificationSplit


def materialize_detection_dataset(
    source: str | Path,
    labels: str | Path,
    destination: str | Path,
    split: ClassificationSplit,
) -> None:
    source_root = Path(source)
    label_root = Path(labels)
    destination_root = Path(destination)
    if destination_root.exists():
        raise ValueError(f"destination already exists: {destination_root}")
    if not source_root.is_dir() or not label_root.is_dir():
        raise ValueError("source image and label directories must exist")

    copied = 0
    for entry in split.entries:
        source_path = source_root / entry.source
        output_name = f"count-{entry.label:02d}__{source_path.name}"
        label_name = f"{Path(output_name).stem}.txt"
        source_label = label_root / label_name
        if not source_label.is_file():
            raise ValueError(f"missing pre-label: {source_label}")
        image_directory = destination_root / "images" / entry.split
        label_directory = destination_root / "labels" / entry.split
        image_directory.mkdir(parents=True, exist_ok=True)
        label_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, image_directory / output_name)
        shutil.copy2(source_label, label_directory / label_name)
        copied += 1

    (destination_root / "dataset.yaml").write_text(
        f"path: {destination_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: toy_vehicle\n",
        encoding="utf-8",
    )
    (destination_root / "split-manifest.json").write_text(
        json.dumps(split.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    if copied != len(split.entries):
        raise RuntimeError("not all split entries were materialized")
