from pathlib import Path

import pytest

from traffic_vision.config import load_road_configs, road_config_from_dict


def test_loads_four_example_roads() -> None:
    configs = load_road_configs(Path("configs/roads.example.json"))

    assert set(configs) == {"road_1", "road_2", "road_3", "road_4"}
    assert {lane.name for lane in configs["road_1"].lanes} == {"left", "right"}


def test_rejects_missing_right_lane() -> None:
    raw = {
        "road_id": "road_1",
        "visible_length": 1,
        "junction_line": [[0, 0], [1, 0]],
        "lanes": {
            "left": {
                "maximum_capacity": 6,
                "polygon": [[0, 0], [1, 0], [0, 1]],
            }
        },
    }

    with pytest.raises(ValueError, match="left and right"):
        road_config_from_dict(raw)

