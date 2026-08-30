"""Count-constrained selection of AI-generated object-label proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from traffic_vision.schemas import BoundingBox


@dataclass(frozen=True, slots=True)
class LabelProposal:
    bounding_box: BoundingBox
    confidence: float
    source: str

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("proposal confidence must be between zero and one")
        if not self.source:
            raise ValueError("proposal source cannot be empty")


@dataclass(frozen=True, slots=True)
class RankedProposal:
    proposal: LabelProposal
    agreement_sources: tuple[str, ...]
    ranking_score: float


@dataclass(frozen=True, slots=True)
class PrelabelSelection:
    expected_count: int
    candidate_count: int
    selected: tuple[RankedProposal, ...]
    complete: bool
    review_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    x_min = max(first.x_min, second.x_min)
    y_min = max(first.y_min, second.y_min)
    x_max = min(first.x_max, second.x_max)
    y_max = min(first.y_max, second.y_max)
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    first_area = (first.x_max - first.x_min) * (first.y_max - first.y_min)
    second_area = (second.x_max - second.x_min) * (second.y_max - second.y_min)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def has_vehicle_like_geometry(
    proposal: LabelProposal, image_width: int, image_height: int
) -> bool:
    box = proposal.bounding_box
    width = (box.x_max - box.x_min) / image_width
    height = (box.y_max - box.y_min) / image_height
    area = width * height
    aspect = width / height
    center_x = (box.x_min + box.x_max) / (2 * image_width)
    center_y = (box.y_min + box.y_max) / (2 * image_height)
    return (
        0.025 <= width <= 0.18
        and 0.04 <= height <= 0.30
        and 0.32 <= aspect <= 1.60
        and 0.0015 <= area <= 0.035
        and 0.06 <= center_x <= 0.82
        and 0.08 <= center_y <= 0.68
    )


def select_count_constrained_proposals(
    proposals: list[LabelProposal],
    expected_count: int,
    image_width: int,
    image_height: int,
    agreement_iou: float = 0.45,
    suppression_iou: float = 0.30,
) -> PrelabelSelection:
    """Rank model proposals, prefer cross-model agreement, and select known count."""

    if expected_count < 0:
        raise ValueError("expected count cannot be negative")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0 <= agreement_iou <= 1 or not 0 <= suppression_iou <= 1:
        raise ValueError("IoU thresholds must be between zero and one")
    if expected_count == 0:
        return PrelabelSelection(0, 0, (), True)

    candidates = [
        proposal
        for proposal in proposals
        if has_vehicle_like_geometry(proposal, image_width, image_height)
    ]
    ranked: list[RankedProposal] = []
    for proposal in candidates:
        agreement_sources = tuple(
            sorted(
                {
                    other.source
                    for other in candidates
                    if other.source != proposal.source
                    and intersection_over_union(
                        proposal.bounding_box, other.bounding_box
                    )
                    >= agreement_iou
                }
            )
        )
        ranked.append(
            RankedProposal(
                proposal=proposal,
                agreement_sources=agreement_sources,
                ranking_score=proposal.confidence + 0.35 * bool(agreement_sources),
            )
        )
    ranked.sort(key=lambda item: item.ranking_score, reverse=True)

    selected: list[RankedProposal] = []
    for candidate in ranked:
        if any(
            intersection_over_union(
                candidate.proposal.bounding_box, existing.proposal.bounding_box
            )
            >= suppression_iou
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == expected_count:
            break

    return PrelabelSelection(
        expected_count=expected_count,
        candidate_count=len(candidates),
        selected=tuple(selected),
        complete=len(selected) == expected_count,
    )


def proposal_to_yolo_line(
    proposal: LabelProposal, image_width: int, image_height: int
) -> str:
    box = proposal.bounding_box
    x_center = (box.x_min + box.x_max) / (2 * image_width)
    y_center = (box.y_min + box.y_max) / (2 * image_height)
    width = (box.x_max - box.x_min) / image_width
    height = (box.y_max - box.y_min) / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
