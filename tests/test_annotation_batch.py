from pathlib import Path

from traffic_vision.annotation_batch import (
    build_annotation_manifest,
    materialize_annotation_batch,
    verify_annotation_batch,
)


JPEG_1X1 = bytes.fromhex(
    "ffd8ffc0000b080001000101011100ffda0008010100003f00ffd9"
)


def test_prepares_uniquely_named_annotation_images(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for label in (0, 2):
        directory = source / str(label)
        directory.mkdir(parents=True)
        (directory / f"capture-{label}.jpg").write_bytes(JPEG_1X1)

    manifest = build_annotation_manifest(source)
    destination = tmp_path / "batch"
    materialize_annotation_batch(manifest, destination)

    assert [entry.expected_vehicle_count for entry in manifest.entries] == [0, 2]
    assert (destination / "images" / "count-02__capture-2.jpg").is_file()
    assert (destination / "annotation-manifest.json").is_file()


def test_verifies_box_count_against_known_total(tmp_path: Path) -> None:
    source = tmp_path / "source" / "2"
    source.mkdir(parents=True)
    (source / "capture.jpg").write_bytes(JPEG_1X1)
    manifest = build_annotation_manifest(tmp_path / "source")
    labels = tmp_path / "labels"
    labels.mkdir()
    label_path = labels / "count-02__capture.txt"
    label_path.write_text("0 0.25 0.5 0.2 0.2\n0 0.75 0.5 0.2 0.2\n")

    report = verify_annotation_batch(manifest, labels)

    assert report.is_complete
    assert report.box_count == 2


def test_reports_missing_and_count_mismatched_labels(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for label in (1, 2):
        directory = source / str(label)
        directory.mkdir(parents=True)
        (directory / f"capture-{label}.jpg").write_bytes(JPEG_1X1)
    manifest = build_annotation_manifest(source)
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "count-01__capture-1.txt").write_text("")

    report = verify_annotation_batch(manifest, labels)

    assert not report.is_complete
    assert len(report.missing_labels) == 1
    assert len(report.count_mismatches) == 1
