from __future__ import annotations

from pathlib import Path
from threading import Lock

from fastapi.testclient import TestClient

from traffic_vision.api import ApiRuntime, create_app
from traffic_vision.config import load_road_configs
from traffic_vision.schemas import BoundingBox, Detection, ImageDetections


class ConstantDetector:
    def detect(self, image_path: str | Path) -> ImageDetections:
        return ImageDetections(
            width=100,
            height=100,
            detections=(
                Detection(BoundingBox(15, 30, 25, 50), 0.9),
                Detection(BoundingBox(55, 30, 65, 50), 0.8),
            ),
        )


def _runtime(maximum_upload_bytes: int = 1024) -> ApiRuntime:
    return ApiRuntime(
        detector=ConstantDetector(),
        configs=load_road_configs("configs/roads.supplied-view-provisional.json"),
        confidence=0.2,
        maximum_upload_bytes=maximum_upload_bytes,
        inference_lock=Lock(),
    )


def _image(filename: str = "road.jpg", content: bytes = b"fake-image") -> tuple:
    return filename, content, "image/jpeg"


def test_health_reports_loaded_runtime() -> None:
    with TestClient(create_app(_runtime())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "road_ids": ["road_1", "road_2", "road_3", "road_4"],
        "confidence": 0.2,
    }


def test_four_image_endpoint_returns_lane_counts() -> None:
    files = {road_id: _image() for road_id in ("road_1", "road_2", "road_3", "road_4")}
    with TestClient(create_app(_runtime())) as client:
        response = client.post("/v1/measure", files=files)

    assert response.status_code == 200
    for road in response.json()["roads"].values():
        assert road["lanes"]["left"]["count"] == 1
        assert road["lanes"]["right"]["count"] == 1
        assert road["total_count"] == 2


def test_single_road_endpoint_returns_one_result() -> None:
    with TestClient(create_app(_runtime())) as client:
        response = client.post(
            "/v1/roads/road_2/measure", files={"image": _image()}
        )

    assert response.status_code == 200
    assert response.json()["road_id"] == "road_2"
    assert response.json()["lanes"]["left"]["count"] == 1
    assert response.json()["lanes"]["right"]["count"] == 1


def test_single_road_endpoint_rejects_unknown_road() -> None:
    with TestClient(create_app(_runtime())) as client:
        response = client.post(
            "/v1/roads/road_5/measure", files={"image": _image()}
        )

    assert response.status_code == 404


def test_upload_rejects_unsupported_extension() -> None:
    with TestClient(create_app(_runtime())) as client:
        response = client.post(
            "/v1/roads/road_1/measure",
            files={"image": _image(filename="road.txt")},
        )

    assert response.status_code == 415


def test_upload_size_is_limited() -> None:
    with TestClient(create_app(_runtime(maximum_upload_bytes=3))) as client:
        response = client.post(
            "/v1/roads/road_1/measure",
            files={"image": _image(content=b"too-large")},
        )

    assert response.status_code == 413
