"""Generate reviewable YOLO pre-labels from multiple proposal models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traffic_vision.ai_labelers import ProposalProvider
from traffic_vision.count_dataset import jpeg_size
from traffic_vision.prelabel import (
    PrelabelSelection,
    proposal_to_yolo_line,
    select_count_constrained_proposals,
)


@dataclass(frozen=True, slots=True)
class PrelabelBatchEntry:
    source: str
    image_name: str
    expected_count: int
    candidate_count: int
    selected_count: int
    cross_model_agreements: int
    complete: bool
    review_required: bool


@dataclass(frozen=True, slots=True)
class PrelabelBatchReport:
    image_count: int
    count_complete_images: int
    incomplete_images: int
    entries: tuple[PrelabelBatchEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _render_overlay(source: Path, destination: Path, selection: PrelabelSelection) -> None:
    from PIL import Image, ImageDraw

    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, ranked in enumerate(selection.selected, start=1):
        box = ranked.proposal.bounding_box
        colour = "#00ff66" if ranked.agreement_sources else "#ff9900"
        draw.rectangle((box.x_min, box.y_min, box.x_max, box.y_max), outline=colour, width=3)
        draw.text(
            (box.x_min + 2, max(0, box.y_min - 12)),
            f"{index}:{ranked.proposal.source}",
            fill=colour,
        )
    status = "COUNT COMPLETE - REVIEW" if selection.complete else "INCOMPLETE"
    draw.rectangle((0, 0, image.width, 22), fill="#000000")
    draw.text(
        (4, 4),
        f"expected={selection.expected_count} selected={len(selection.selected)} {status}",
        fill="#ffffff",
    )
    image.save(destination, quality=90)


def generate_prelabel_batch(
    source: str | Path,
    destination: str | Path,
    providers: tuple[ProposalProvider, ...],
) -> PrelabelBatchReport:
    source_root = Path(source)
    destination_root = Path(destination)
    if not source_root.is_dir():
        raise ValueError(f"source directory does not exist: {source_root}")
    if destination_root.exists():
        raise ValueError(f"destination already exists: {destination_root}")
    if not providers:
        raise ValueError("at least one proposal provider is required")

    label_directory = destination_root / "labels"
    overlay_directory = destination_root / "overlays"
    label_directory.mkdir(parents=True)
    overlay_directory.mkdir(parents=True)
    entries: list[PrelabelBatchEntry] = []

    class_directories = sorted(
        (path for path in source_root.iterdir() if path.is_dir()),
        key=lambda path: int(path.name),
    )
    for class_directory in class_directories:
        expected_count = int(class_directory.name)
        for source_path in sorted(class_directory.glob("*.jpg")):
            if expected_count == 0:
                width, height = jpeg_size(source_path)
                proposed_images = ()
            else:
                proposed_images = tuple(
                    provider.propose(source_path) for provider in providers
                )
                dimensions = {(item.width, item.height) for item in proposed_images}
                if len(dimensions) != 1:
                    raise RuntimeError(
                        f"proposal models disagree on dimensions: {source_path}"
                    )
                width, height = dimensions.pop()
            selection = select_count_constrained_proposals(
                [
                    proposal
                    for proposed_image in proposed_images
                    for proposal in proposed_image.proposals
                ],
                expected_count,
                width,
                height,
            )
            image_name = f"count-{expected_count:02d}__{source_path.name}"
            label_path = label_directory / f"{Path(image_name).stem}.txt"
            label_path.write_text(
                "\n".join(
                    proposal_to_yolo_line(ranked.proposal, width, height)
                    for ranked in selection.selected
                )
                + ("\n" if selection.selected else ""),
                encoding="utf-8",
            )
            _render_overlay(source_path, overlay_directory / image_name, selection)
            entries.append(
                PrelabelBatchEntry(
                    source=str(source_path.relative_to(source_root)),
                    image_name=image_name,
                    expected_count=expected_count,
                    candidate_count=selection.candidate_count,
                    selected_count=len(selection.selected),
                    cross_model_agreements=sum(
                        bool(ranked.agreement_sources) for ranked in selection.selected
                    ),
                    complete=selection.complete,
                    review_required=True,
                )
            )

    complete = sum(entry.complete for entry in entries)
    report = PrelabelBatchReport(
        image_count=len(entries),
        count_complete_images=complete,
        incomplete_images=len(entries) - complete,
        entries=tuple(entries),
    )
    (destination_root / "prelabel-report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report
