"""Replaceable vehicle-detector interface and optional Ultralytics adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from traffic_vision.schemas import BoundingBox, Detection, ImageDetections


class VehicleDetector(Protocol):
    def detect(self, image_path: str | Path) -> ImageDetections:
        """Locate vehicles in one image."""


class FakeVehicleDetector:
    """Deterministic detector used by unit and integration tests."""

    def __init__(self, results: dict[str, ImageDetections]) -> None:
        self._results = results

    def detect(self, image_path: str | Path) -> ImageDetections:
        key = str(image_path)
        if key not in self._results:
            raise KeyError(f"no fake detector result configured for {key}")
        return self._results[key]


class UltralyticsVehicleDetector:
    """Thin adapter around an Ultralytics object-detection model."""

    def __init__(self, model_path: str | Path, minimum_confidence: float = 0.25) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum confidence must be between zero and one")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed; install the project with the vision extra"
            ) from error

        self._model = YOLO(str(model_path))
        self._minimum_confidence = minimum_confidence

    def detect(self, image_path: str | Path) -> ImageDetections:
        results = self._model.predict(
            source=str(image_path),
            conf=self._minimum_confidence,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError("detector must return exactly one result per image")

        result = results[0]
        height, width = result.orig_shape
        detections: list[Detection] = []
        for box in result.boxes:
            x_min, y_min, x_max, y_max = (
                float(value) for value in box.xyxy[0].tolist()
            )
            confidence = float(box.conf[0].item())
            class_index = int(box.cls[0].item())
            detections.append(
                Detection(
                    bounding_box=BoundingBox(x_min, y_min, x_max, y_max),
                    confidence=confidence,
                    label=str(result.names[class_index]),
                )
            )
        return ImageDetections(width=width, height=height, detections=tuple(detections))

