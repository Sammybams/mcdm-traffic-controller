import pytest

from traffic_vision.prelabel import (
    LabelProposal,
    intersection_over_union,
    proposal_to_yolo_line,
    select_count_constrained_proposals,
)
from traffic_vision.schemas import BoundingBox


def _proposal(box: tuple[float, float, float, float], confidence: float, source: str):
    return LabelProposal(BoundingBox(*box), confidence, source)


def test_prefers_cross_model_agreement_and_suppresses_duplicate_boxes() -> None:
    proposals = [
        _proposal((10, 10, 20, 30), 0.7, "dino"),
        _proposal((11, 10, 21, 30), 0.6, "world"),
        _proposal((40, 10, 50, 30), 0.8, "dino"),
        _proposal((70, 10, 80, 30), 0.9, "dino"),
    ]

    result = select_count_constrained_proposals(proposals, 2, 100, 100)

    assert result.complete
    assert len(result.selected) == 2
    assert result.selected[0].proposal.bounding_box == BoundingBox(10, 10, 20, 30)
    assert result.selected[0].agreement_sources == ("world",)
    assert result.selected[1].proposal.bounding_box == BoundingBox(70, 10, 80, 30)


def test_removes_implausible_geometry_and_reports_incomplete_selection() -> None:
    proposals = [
        _proposal((0, 0, 100, 100), 0.99, "dino"),
        _proposal((10, 10, 20, 30), 0.8, "dino"),
    ]

    result = select_count_constrained_proposals(proposals, 2, 100, 100)

    assert not result.complete
    assert result.candidate_count == 1
    assert len(result.selected) == 1


def test_known_empty_image_never_accepts_false_proposals() -> None:
    result = select_count_constrained_proposals(
        [_proposal((10, 10, 20, 30), 0.9, "dino")], 0, 100, 100
    )

    assert result.complete
    assert result.selected == ()


def test_iou_and_yolo_serialization() -> None:
    first = BoundingBox(10, 20, 30, 60)
    second = BoundingBox(20, 20, 40, 60)

    assert intersection_over_union(first, second) == pytest.approx(1 / 3)
    assert proposal_to_yolo_line(_proposal((10, 20, 30, 60), 0.9, "dino"), 100, 100) == (
        "0 0.200000 0.400000 0.200000 0.400000"
    )
