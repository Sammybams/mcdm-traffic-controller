"""Optional zero-shot model adapters used to propose toy-vehicle labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from traffic_vision.prelabel import LabelProposal
from traffic_vision.schemas import BoundingBox


@dataclass(frozen=True, slots=True)
class ProposedImage:
    width: int
    height: int
    proposals: tuple[LabelProposal, ...]


class ProposalProvider(Protocol):
    def propose(self, image_path: str | Path) -> ProposedImage:
        """Return zero-shot bounding-box proposals for one image."""


class YoloWorldProposalProvider:
    """Official Ultralytics YOLO-World adapter with toy-vehicle prompts."""

    DEFAULT_PROMPTS = (
        "toy car",
        "small toy car",
        "miniature car",
        "model car",
        "toy vehicle",
    )

    def __init__(
        self,
        model_path: str | Path = "yolov8s-worldv2.pt",
        prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    ) -> None:
        try:
            from ultralytics import YOLOWorld
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required for YOLO-World pre-labelling"
            ) from error
        self._model = YOLOWorld(str(model_path))
        self._model.set_classes(list(prompts))

    def propose(self, image_path: str | Path) -> ProposedImage:
        results = self._model.predict(
            str(image_path),
            imgsz=800,
            conf=0.003,
            iou=0.30,
            agnostic_nms=True,
            max_det=100,
            verbose=False,
        )
        if len(results) != 1:
            raise RuntimeError("YOLO-World must return one result per image")
        result = results[0]
        height, width = result.orig_shape
        return ProposedImage(
            width=width,
            height=height,
            proposals=tuple(
                LabelProposal(
                    BoundingBox(*(float(value) for value in box.xyxy[0].tolist())),
                    float(box.conf[0].item()),
                    "yolo_world",
                )
                for box in result.boxes
            ),
        )


class GroundingDinoProposalProvider:
    """Hugging Face Grounding DINO adapter using a single toy-car prompt."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as error:
            raise RuntimeError(
                "Transformers, PyTorch, and Pillow are required for Grounding DINO"
            ) from error
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, local_files_only=local_files_only
        )
        self._model.eval()

    def propose(self, image_path: str | Path) -> ProposedImage:
        try:
            from PIL import Image
            from torchvision.ops import nms
        except ImportError as error:
            raise RuntimeError(
                "Pillow and Torchvision are required for Grounding DINO"
            ) from error

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = self._processor(
            images=image, text=[["toy car"]], return_tensors="pt"
        )
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.03,
            text_threshold=0.03,
            target_sizes=[image.size[::-1]],
        )[0]
        kept_indices = nms(result["boxes"], result["scores"], 0.30).tolist()
        return ProposedImage(
            width=image.width,
            height=image.height,
            proposals=tuple(
                LabelProposal(
                    BoundingBox(*(float(value) for value in result["boxes"][index])),
                    float(result["scores"][index]),
                    "grounding_dino",
                )
                for index in kept_indices
            ),
        )
