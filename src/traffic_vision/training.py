"""Reproducible training configuration and optional Ultralytics runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    base_model: str
    dataset_yaml: str
    epochs: int
    image_size: int
    batch_size: int
    seed: int
    workers: int
    project: str
    run_name: str
    device: str = "auto"
    extra_arguments: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.base_model or not self.dataset_yaml:
            raise ValueError("base model and dataset YAML are required")
        if min(self.epochs, self.image_size, self.batch_size, self.workers + 1) <= 0:
            raise ValueError("training numeric settings must be positive")
        reserved = {
            "data",
            "epochs",
            "imgsz",
            "batch",
            "seed",
            "workers",
            "project",
            "name",
            "device",
        }
        overlap = reserved.intersection(self.extra_arguments or {})
        if overlap:
            raise ValueError(
                f"extra training arguments cannot override managed keys: {sorted(overlap)}"
            )


def load_training_config(path: str | Path) -> TrainingConfig:
    with Path(path).open(encoding="utf-8") as config_file:
        raw = json.load(config_file)
    return TrainingConfig(
        base_model=str(raw["base_model"]),
        dataset_yaml=str(raw["dataset_yaml"]),
        epochs=int(raw["epochs"]),
        image_size=int(raw["image_size"]),
        batch_size=int(raw["batch_size"]),
        seed=int(raw["seed"]),
        workers=int(raw["workers"]),
        project=str(raw["project"]),
        run_name=str(raw["run_name"]),
        device=str(raw.get("device", "auto")),
        extra_arguments=dict(raw.get("extra_arguments", {})),
    )


def training_arguments(config: TrainingConfig) -> dict[str, Any]:
    project_directory = Path(config.project).resolve()
    arguments: dict[str, Any] = {
        "data": config.dataset_yaml,
        "epochs": config.epochs,
        "imgsz": config.image_size,
        "batch": config.batch_size,
        "seed": config.seed,
        "workers": config.workers,
        "project": str(project_directory),
        "name": config.run_name,
        "save_period": 10,
    }
    if config.device != "auto":
        arguments["device"] = config.device
    arguments.update(config.extra_arguments or {})
    return arguments


def run_training(config: TrainingConfig) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed; install the project with the vision extra"
        ) from error

    model = YOLO(config.base_model)
    return model.train(**training_arguments(config))
