import pytest

from traffic_vision.dataset import validate_label_directory


def test_validates_annotations_and_empty_images(tmp_path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "cars.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (labels / "empty.txt").write_text("", encoding="utf-8")

    statistics = validate_label_directory(labels)

    assert statistics.label_files == 2
    assert statistics.empty_images == 1
    assert statistics.vehicle_boxes == 1


def test_rejects_box_outside_image(tmp_path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "bad.txt").write_text("0 0.95 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="horizontal image bounds"):
        validate_label_directory(labels)


def test_rejects_unexpected_class(tmp_path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "bad.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="class 0"):
        validate_label_directory(labels)

