"""FastAPI transport for single-road and four-road image inference."""

from __future__ import annotations

import os
import secrets
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from traffic_vision.config import RoadConfig, load_road_configs
from traffic_vision.detector import UltralyticsVehicleDetector, VehicleDetector
from traffic_vision.road_processor import process_road
from traffic_vision.service import measure_image_paths

_CHUNK_SIZE = 1024 * 1024
_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class ApiSettings:
    model_path: Path
    roads_config_path: Path
    confidence: float = 0.20
    maximum_upload_bytes: int = 10 * 1024 * 1024
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("TRAFFIC_VISION_CONFIDENCE must be between zero and one")
        if self.maximum_upload_bytes <= 0:
            raise ValueError("TRAFFIC_VISION_MAX_UPLOAD_MB must be positive")

    @classmethod
    def from_environment(cls) -> ApiSettings:
        maximum_megabytes = float(os.getenv("TRAFFIC_VISION_MAX_UPLOAD_MB", "10"))
        raw_api_key = os.getenv("TRAFFIC_VISION_API_KEY", "").strip()
        return cls(
            model_path=Path(
                os.getenv(
                    "TRAFFIC_VISION_MODEL",
                    "artifacts/research/toy-vehicle-prelabel.pt",
                )
            ),
            roads_config_path=Path(
                os.getenv(
                    "TRAFFIC_VISION_ROADS_CONFIG",
                    "configs/roads.supplied-view-provisional.json",
                )
            ),
            confidence=float(os.getenv("TRAFFIC_VISION_CONFIDENCE", "0.20")),
            maximum_upload_bytes=int(maximum_megabytes * 1024 * 1024),
            api_key=raw_api_key or None,
        )


@dataclass(slots=True)
class ApiRuntime:
    detector: VehicleDetector
    configs: dict[str, RoadConfig]
    confidence: float
    maximum_upload_bytes: int
    inference_lock: Lock
    api_key: str | None = None


def load_runtime(settings: ApiSettings) -> ApiRuntime:
    if not settings.model_path.is_file():
        raise RuntimeError(f"model file does not exist: {settings.model_path}")
    configs = load_road_configs(settings.roads_config_path)
    detector = UltralyticsVehicleDetector(settings.model_path, settings.confidence)
    return ApiRuntime(
        detector=detector,
        configs=configs,
        confidence=settings.confidence,
        maximum_upload_bytes=settings.maximum_upload_bytes,
        inference_lock=Lock(),
        api_key=settings.api_key,
    )


async def _save_upload(
    upload: UploadFile,
    destination: Path,
    maximum_upload_bytes: int,
) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="each upload must use a .jpg, .jpeg, or .png filename",
        )

    written = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(_CHUNK_SIZE):
            written += len(chunk)
            if written > maximum_upload_bytes:
                raise HTTPException(status_code=413, detail="uploaded image is too large")
            output.write(chunk)
    if written == 0:
        raise HTTPException(status_code=422, detail="uploaded image is empty")


def _measure_four(runtime: ApiRuntime, image_paths: dict[str, Path]) -> dict:
    with runtime.inference_lock:
        return measure_image_paths(
            runtime.detector,
            image_paths,
            runtime.configs,
            minimum_confidence=runtime.confidence,
        ).to_dict()


def _measure_one(runtime: ApiRuntime, road_id: str, image_path: Path) -> dict:
    with runtime.inference_lock:
        detections = runtime.detector.detect(image_path)
        return process_road(
            detections,
            runtime.configs[road_id],
            minimum_confidence=runtime.confidence,
        ).to_dict()


def create_app(
    runtime: ApiRuntime | None = None,
    settings: ApiSettings | None = None,
) -> FastAPI:
    supplied_runtime = runtime

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.runtime = supplied_runtime or load_runtime(
            settings or ApiSettings.from_environment()
        )
        yield

    application = FastAPI(
        title="MCDM Traffic Vision API",
        version="0.1.0",
        description=(
            "Detect toy vehicles and calculate deterministic left/right lane "
            "counts, density, and proximity. Traffic-light decisions remain "
            "outside this service."
        ),
        lifespan=lifespan,
    )

    @application.get("/")
    async def root() -> dict:
        return {
            "service": "mcdm-traffic-vision",
            "docs": "/docs",
            "health": "/health",
        }

    @application.get("/health")
    async def health() -> dict:
        current: ApiRuntime = application.state.runtime
        return {
            "status": "ok",
            "model_loaded": True,
            "road_ids": sorted(current.configs),
            "confidence": current.confidence,
        }

    async def require_api_key(
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> None:
        current: ApiRuntime = application.state.runtime
        if current.api_key is not None and (
            x_api_key is None or not secrets.compare_digest(x_api_key, current.api_key)
        ):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @application.post("/v1/measure", dependencies=[Depends(require_api_key)])
    async def measure_junction(
        road_1: Annotated[UploadFile, File(description="Road 1 image")],
        road_2: Annotated[UploadFile, File(description="Road 2 image")],
        road_3: Annotated[UploadFile, File(description="Road 3 image")],
        road_4: Annotated[UploadFile, File(description="Road 4 image")],
    ) -> dict:
        current: ApiRuntime = application.state.runtime
        uploads = {
            "road_1": road_1,
            "road_2": road_2,
            "road_3": road_3,
            "road_4": road_4,
        }
        try:
            with tempfile.TemporaryDirectory(prefix="traffic-vision-") as temporary:
                directory = Path(temporary)
                image_paths: dict[str, Path] = {}
                for road_id, upload in uploads.items():
                    suffix = Path(upload.filename or "").suffix.lower()
                    destination = directory / f"{road_id}{suffix}"
                    await _save_upload(
                        upload, destination, current.maximum_upload_bytes
                    )
                    image_paths[road_id] = destination
                return await run_in_threadpool(_measure_four, current, image_paths)
        except HTTPException:
            raise
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post(
        "/v1/roads/{road_id}/measure", dependencies=[Depends(require_api_key)]
    )
    async def measure_single_road(
        road_id: str,
        image: Annotated[UploadFile, File(description="One road image")],
    ) -> dict:
        current: ApiRuntime = application.state.runtime
        if road_id not in current.configs:
            raise HTTPException(status_code=404, detail=f"unknown road ID: {road_id}")
        suffix = Path(image.filename or "").suffix.lower()
        try:
            with tempfile.TemporaryDirectory(prefix="traffic-vision-") as temporary:
                image_path = Path(temporary) / f"{road_id}{suffix}"
                await _save_upload(image, image_path, current.maximum_upload_bytes)
                return await run_in_threadpool(
                    _measure_one, current, road_id, image_path
                )
        except HTTPException:
            raise
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return application


app = create_app()
