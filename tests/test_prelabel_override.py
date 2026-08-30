import json
from pathlib import Path

from PIL import Image

from traffic_vision.prelabel_override import apply_prelabel_overrides


def test_applies_reviewed_boxes_to_every_frame_in_temporal_group(tmp_path: Path) -> None:
    source = tmp_path / "source" / "1"
    prelabels = tmp_path / "prelabels"
    (prelabels / "labels").mkdir(parents=True)
    (prelabels / "overlays").mkdir()
    source.mkdir(parents=True)
    entries = []
    for second in range(3):
        name = f"esp32cam_20260830_12000{second}_000.jpg"
        Image.new("RGB", (100, 100), "white").save(source / name)
        entries.append(
            {
                "source": f"1/{name}",
                "selected_count": 0,
                "complete": False,
            }
        )
    (prelabels / "prelabel-report.json").write_text(
        json.dumps(
            {
                "count_complete_images": 0,
                "incomplete_images": 3,
                "entries": entries,
            }
        )
    )
    config = tmp_path / "overrides.json"
    config.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "label": 1,
                        "group": 0,
                        "reason": "reviewed",
                        "boxes": [[0.2, 0.2, 0.1, 0.2]],
                    }
                ]
            }
        )
    )

    report = apply_prelabel_overrides(tmp_path / "source", prelabels, config)
    updated = json.loads((prelabels / "prelabel-report.json").read_text())

    assert report.overridden_images == 3
    assert updated["count_complete_images"] == 3
    assert all(entry["label_source"] == "reviewed_override" for entry in updated["entries"])
