import json

import pytest

from traffic_vision.training import (
    TrainingConfig,
    load_training_config,
    training_arguments,
)


def test_loads_versioned_training_configuration(tmp_path) -> None:
    path = tmp_path / "training.json"
    path.write_text(
        json.dumps(
            {
                "base_model": "small-model.pt",
                "dataset_yaml": "dataset.yaml",
                "epochs": 10,
                "image_size": 640,
                "batch_size": 4,
                "seed": 7,
                "workers": 2,
                "project": "runs/train",
                "run_name": "test-run",
            }
        ),
        encoding="utf-8",
    )

    config = load_training_config(path)

    assert config.base_model == "small-model.pt"
    assert config.seed == 7
    assert config.device == "auto"


def test_training_config_rejects_zero_epochs() -> None:
    with pytest.raises(ValueError, match="numeric settings"):
        TrainingConfig(
            base_model="model.pt",
            dataset_yaml="dataset.yaml",
            epochs=0,
            image_size=640,
            batch_size=4,
            seed=1,
            workers=1,
            project="runs",
            run_name="bad",
        )


def test_training_resolves_project_path_to_current_repository() -> None:
    config = load_training_config("configs/training.example.json")

    assert training_arguments(config)["project"].endswith(
        "/mcdm-traffic-controller/runs/train"
    )


def test_training_passes_through_non_managed_experiment_arguments(tmp_path) -> None:
    path = tmp_path / "training.json"
    raw = json.loads(open("configs/count-classification.json", encoding="utf-8").read())
    raw["extra_arguments"] = {"scale": 0.0, "erasing": 0.0}
    path.write_text(json.dumps(raw), encoding="utf-8")

    arguments = training_arguments(load_training_config(path))

    assert arguments["scale"] == 0.0
    assert arguments["erasing"] == 0.0


def test_training_rejects_extra_argument_that_overrides_managed_key() -> None:
    with pytest.raises(ValueError, match="managed keys"):
        TrainingConfig(
            base_model="model.pt",
            dataset_yaml="dataset.yaml",
            epochs=1,
            image_size=640,
            batch_size=4,
            seed=1,
            workers=1,
            project="runs",
            run_name="bad",
            extra_arguments={"epochs": 99},
        )
