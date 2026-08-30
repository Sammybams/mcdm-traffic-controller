import json
from pathlib import Path

import pytest

from traffic_vision.capture_manifest import load_capture_batch


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "batch_id": "cycle-1",
        "captures": [
            {
                "road_id": f"road_{index + 1}",
                "image_path": f"road-{index + 1}.jpg",
                "captured_at": f"2026-08-30T12:00:0{index}+01:00",
                "motor_position": index,
            }
            for index in range(4)
        ],
    }


def test_loads_four_road_capture_batch_and_resolves_paths(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    batch = load_capture_batch(path)

    assert batch.span_seconds == 3
    assert batch.image_paths()["road_1"] == (tmp_path / "road-1.jpg").resolve()


def test_rejects_capture_cycle_that_is_too_slow(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["captures"][-1]["captured_at"] = "2026-08-30T12:00:31+01:00"
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="spans"):
        load_capture_batch(path, maximum_span_seconds=30)


def test_rejects_duplicate_road_or_motor_position(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["captures"][1]["road_id"] = "road_1"
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="road IDs"):
        load_capture_batch(path)
