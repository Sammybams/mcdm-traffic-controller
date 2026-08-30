"""Metadata contract for one rotating-camera junction observation cycle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RoadCapture:
    road_id: str
    image_path: Path
    captured_at: datetime
    motor_position: int


@dataclass(frozen=True, slots=True)
class CaptureBatch:
    batch_id: str
    captures: tuple[RoadCapture, ...]

    def __post_init__(self) -> None:
        if not self.batch_id:
            raise ValueError("capture batch ID cannot be empty")
        if len(self.captures) != 4:
            raise ValueError("capture batch must contain exactly four roads")
        road_ids = [capture.road_id for capture in self.captures]
        if len(set(road_ids)) != 4:
            raise ValueError("capture batch road IDs must be unique")
        motor_positions = [capture.motor_position for capture in self.captures]
        if len(set(motor_positions)) != 4:
            raise ValueError("capture batch motor positions must be unique")
        if any(capture.captured_at.tzinfo is None for capture in self.captures):
            raise ValueError("capture timestamps must include a UTC offset")

    @property
    def span_seconds(self) -> float:
        times = [capture.captured_at for capture in self.captures]
        return (max(times) - min(times)).total_seconds()

    def image_paths(self) -> dict[str, Path]:
        return {capture.road_id: capture.image_path for capture in self.captures}


def load_capture_batch(
    path: str | Path, maximum_span_seconds: float = 30
) -> CaptureBatch:
    if maximum_span_seconds <= 0:
        raise ValueError("maximum capture span must be positive")
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as manifest_file:
        raw = json.load(manifest_file)
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("unsupported capture manifest schema version")

    captures = tuple(
        RoadCapture(
            road_id=str(item["road_id"]),
            image_path=(manifest_path.parent / str(item["image_path"])).resolve(),
            captured_at=datetime.fromisoformat(str(item["captured_at"])),
            motor_position=int(item["motor_position"]),
        )
        for item in raw["captures"]
    )
    batch = CaptureBatch(batch_id=str(raw["batch_id"]), captures=captures)
    if batch.span_seconds > maximum_span_seconds:
        raise ValueError(
            f"capture batch spans {batch.span_seconds:.3f}s; "
            f"maximum is {maximum_span_seconds:.3f}s"
        )
    return batch
