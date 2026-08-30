from pathlib import Path

from traffic_vision.classification_data import ClassificationSplit, SplitEntry
from traffic_vision.detection_data import materialize_detection_dataset


def test_materializes_matching_detection_images_and_labels(tmp_path: Path) -> None:
    source = tmp_path / "source" / "2"
    labels = tmp_path / "labels"
    source.mkdir(parents=True)
    labels.mkdir()
    (source / "capture.jpg").write_bytes(b"image")
    (labels / "count-02__capture.txt").write_text(
        "0 0.2 0.2 0.1 0.1\n0 0.8 0.2 0.1 0.1\n"
    )
    split = ClassificationSplit(
        entries=(SplitEntry("2/capture.jpg", 2, 0, "train"),),
        temporal_gap_seconds=30,
        classes_with_group_leakage=(),
    )

    materialize_detection_dataset(
        tmp_path / "source", labels, tmp_path / "detection", split
    )

    assert (tmp_path / "detection/images/train/count-02__capture.jpg").is_file()
    assert (tmp_path / "detection/labels/train/count-02__capture.txt").is_file()
    assert "toy_vehicle" in (tmp_path / "detection/dataset.yaml").read_text()
