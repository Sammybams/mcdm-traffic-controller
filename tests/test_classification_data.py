from datetime import datetime, timedelta

import pytest

from traffic_vision.classification_data import (
    materialize_classification_split,
    plan_classification_split,
)


def _capture_name(timestamp: datetime) -> str:
    return timestamp.strftime("esp32cam_%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"


def test_holds_out_temporally_separate_groups(tmp_path) -> None:
    source = tmp_path / "source"
    start = datetime(2026, 8, 30, 7, 0)
    for label in range(2):
        class_directory = source / str(label)
        class_directory.mkdir(parents=True)
        for group in range(3):
            for frame in range(2):
                timestamp = start + timedelta(minutes=group, seconds=frame)
                (class_directory / _capture_name(timestamp)).write_bytes(b"jpeg")

    split = plan_classification_split(source, temporal_gap_seconds=30)

    assert not split.classes_with_group_leakage
    assert {entry.split for entry in split.entries} == {"train", "val", "test"}
    assert all(
        entry.split == {0: "train", 1: "val", 2: "test"}[entry.group]
        for entry in split.entries
    )


def test_flags_single_group_leakage(tmp_path) -> None:
    source = tmp_path / "source"
    class_directory = source / "0"
    class_directory.mkdir(parents=True)
    start = datetime(2026, 8, 30, 7, 0)
    for frame in range(4):
        (class_directory / _capture_name(start + timedelta(seconds=frame))).write_bytes(
            b"jpeg"
        )

    split = plan_classification_split(source)

    assert split.classes_with_group_leakage == (0,)
    assert [entry.split for entry in split.entries] == ["train", "train", "val", "test"]


def test_two_groups_keep_the_first_group_for_training(tmp_path) -> None:
    source = tmp_path / "source"
    class_directory = source / "2"
    class_directory.mkdir(parents=True)
    start = datetime(2026, 8, 30, 7, 0)
    timestamps = [start, start + timedelta(minutes=1), start + timedelta(minutes=1, seconds=1)]
    for frame, timestamp in enumerate(timestamps):
        (class_directory / _capture_name(timestamp)).write_bytes(bytes([frame]))

    split = plan_classification_split(source, temporal_gap_seconds=30)

    assert [entry.split for entry in split.entries] == ["train", "test", "val"]


def test_materializes_split_and_refuses_existing_destination(tmp_path) -> None:
    source = tmp_path / "source"
    class_directory = source / "0"
    class_directory.mkdir(parents=True)
    start = datetime(2026, 8, 30, 7, 0)
    for frame in range(3):
        (class_directory / _capture_name(start + timedelta(seconds=frame))).write_bytes(
            bytes([frame])
        )
    split = plan_classification_split(source)
    destination = tmp_path / "prepared"

    materialize_classification_split(source, destination, split)

    assert (destination / "split-manifest.json").is_file()
    assert sum(1 for _ in destination.rglob("*.jpg")) == 3
    with pytest.raises(ValueError, match="already exists"):
        materialize_classification_split(source, destination, split)
