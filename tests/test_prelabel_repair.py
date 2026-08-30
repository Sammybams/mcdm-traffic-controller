import json
from pathlib import Path

from PIL import Image

from traffic_vision.prelabel_repair import repair_incomplete_prelabels


def test_repairs_incomplete_label_from_nearest_complete_temporal_peer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "1"
    labels = tmp_path / "prelabels" / "labels"
    overlays = tmp_path / "prelabels" / "overlays"
    source.mkdir(parents=True)
    labels.mkdir(parents=True)
    overlays.mkdir()
    names = [
        "esp32cam_20260830_120000_000.jpg",
        "esp32cam_20260830_120001_000.jpg",
        "esp32cam_20260830_120002_000.jpg",
    ]
    for name in names:
        Image.new("RGB", (100, 100), "white").save(source / name)
    label_line = "0 0.200000 0.200000 0.100000 0.200000\n"
    for name in (names[0], names[2]):
        (labels / f"count-01__{Path(name).stem}.txt").write_text(label_line)
    (labels / f"count-01__{Path(names[1]).stem}.txt").write_text("")
    entries = [
        {
            "source": f"1/{name}",
            "complete": index != 1,
        }
        for index, name in enumerate(names)
    ]
    (tmp_path / "prelabels" / "prelabel-report.json").write_text(
        json.dumps({"entries": entries})
    )

    report = repair_incomplete_prelabels(tmp_path / "source", tmp_path / "prelabels")

    repaired = labels / f"count-01__{Path(names[1]).stem}.txt"
    assert report.repaired_images == 1
    assert repaired.read_text() == label_line
    assert (overlays / f"count-01__{names[1]}").is_file()
