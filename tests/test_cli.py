import pytest

from traffic_vision.cli import build_parser, parse_image_assignments


def test_parses_road_image_assignments() -> None:
    assignments = parse_image_assignments(
        ["road_1=one.jpg", "road_2=two.jpg", "road_3=three.jpg", "road_4=four.jpg"]
    )

    assert str(assignments["road_3"]) == "three.jpg"


def test_rejects_duplicate_road_assignment() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_image_assignments(["road_1=one.jpg", "road_1=again.jpg"])


def test_parser_defaults_to_quarter_confidence() -> None:
    arguments = build_parser().parse_args(
        [
            "--config",
            "roads.json",
            "--model",
            "best.pt",
            "--image",
            "road_1=one.jpg",
        ]
    )

    assert arguments.confidence == 0.25

