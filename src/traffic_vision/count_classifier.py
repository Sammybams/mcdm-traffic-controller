"""Provisional whole-image vehicle-count classifier interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CountPrediction:
    """One total-count prediction for a complete two-lane road image."""

    count: int
    confidence: float
    probabilities: dict[int, float]

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("predicted count cannot be negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        if any(label < 0 or not 0 <= value <= 1 for label, value in self.probabilities.items()):
            raise ValueError("class labels and probabilities must be valid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TotalCountClassifier(Protocol):
    def predict(self, image_path: str | Path) -> CountPrediction:
        """Predict the combined vehicle count in both lanes."""


class FakeTotalCountClassifier:
    """Deterministic classifier used by tests and integrations."""

    def __init__(self, predictions: dict[str, CountPrediction]) -> None:
        self._predictions = predictions

    def predict(self, image_path: str | Path) -> CountPrediction:
        key = str(image_path)
        if key not in self._predictions:
            raise KeyError(f"no fake count prediction configured for {key}")
        return self._predictions[key]


class UltralyticsTotalCountClassifier:
    """Adapter for an Ultralytics image-classification model with numeric classes."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed; install the project with the vision extra"
            ) from error
        self._model = YOLO(str(model_path))

    def predict(self, image_path: str | Path) -> CountPrediction:
        results = self._model.predict(source=str(image_path), verbose=False)
        if len(results) != 1:
            raise RuntimeError("classifier must return exactly one result per image")
        result = results[0]
        if result.probs is None:
            raise RuntimeError("model did not return classification probabilities")

        names = result.names
        labels = {
            index: self._numeric_label(names[index])
            for index in range(len(result.probs.data))
        }
        probabilities = {
            labels[index]: float(value)
            for index, value in enumerate(result.probs.data.tolist())
        }
        class_index = int(result.probs.top1)
        return CountPrediction(
            count=labels[class_index],
            confidence=float(result.probs.top1conf.item()),
            probabilities=probabilities,
        )

    @staticmethod
    def _numeric_label(value: object) -> int:
        try:
            label = int(str(value))
        except ValueError as error:
            raise RuntimeError(f"count model has a non-numeric class name: {value}") from error
        if label < 0:
            raise RuntimeError(f"count model has a negative class name: {value}")
        return label
