import struct

from traffic_vision.count_dataset import audit_count_dataset, jpeg_size


def _minimal_jpeg(width: int, height: int, payload: bytes = b"") -> bytes:
    sof = b"\xff\xc0" + struct.pack(">H", 8) + bytes([8])
    sof += struct.pack(">HH", height, width) + b"\x01"
    return b"\xff\xd8" + sof + payload + b"\xff\xd9"


def test_reads_jpeg_dimensions(tmp_path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(_minimal_jpeg(800, 600))

    assert jpeg_size(image) == (800, 600)


def test_audits_numeric_count_folders(tmp_path) -> None:
    for count in (0, 1):
        class_directory = tmp_path / str(count)
        class_directory.mkdir()
        (class_directory / f"esp32cam_20260830_12000{count}_001.jpg").write_bytes(
            _minimal_jpeg(320, 320, bytes([count]))
        )

    audit = audit_count_dataset(tmp_path)

    assert audit.image_count == 2
    assert audit.class_counts == {0: 1, 1: 1}
    assert audit.resolutions == {"320x320": 2}
    assert audit.duplicate_files == 0
    assert not audit.invalid_files


def test_reports_exact_duplicate_files(tmp_path) -> None:
    class_directory = tmp_path / "2"
    class_directory.mkdir()
    content = _minimal_jpeg(100, 100)
    (class_directory / "one.jpg").write_bytes(content)
    (class_directory / "two.jpg").write_bytes(content)

    assert audit_count_dataset(tmp_path).duplicate_files == 1

