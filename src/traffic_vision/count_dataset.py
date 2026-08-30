"""Audit numeric-folder image-count datasets without ML dependencies."""

from __future__ import annotations

import hashlib
import re
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_CAPTURE_NAME = re.compile(r"esp32cam_(\d{8})_(\d{6})_(\d{3})\.jpg$")
_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


@dataclass(frozen=True, slots=True)
class CountDatasetAudit:
    root: str
    image_count: int
    class_counts: dict[int, int]
    resolutions: dict[str, int]
    duplicate_files: int
    invalid_files: tuple[str, ...]
    first_capture: str | None
    last_capture: str | None
    fingerprint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def jpeg_size(path: str | Path) -> tuple[int, int]:
    """Read JPEG dimensions directly from a start-of-frame segment."""

    with Path(path).open("rb") as image_file:
        if image_file.read(2) != b"\xff\xd8":
            raise ValueError("not a JPEG file")
        while True:
            marker_start = image_file.read(1)
            if not marker_start:
                raise ValueError("JPEG has no start-of-frame marker")
            if marker_start != b"\xff":
                continue
            marker = image_file.read(1)
            while marker == b"\xff":
                marker = image_file.read(1)
            if not marker:
                raise ValueError("truncated JPEG marker")
            marker_value = marker[0]
            if marker_value in {0xD8, 0xD9}:
                continue
            raw_length = image_file.read(2)
            if len(raw_length) != 2:
                raise ValueError("truncated JPEG segment")
            segment_length = struct.unpack(">H", raw_length)[0]
            if segment_length < 2:
                raise ValueError("invalid JPEG segment length")
            if marker_value in _SOF_MARKERS:
                segment = image_file.read(segment_length - 2)
                if len(segment) < 5:
                    raise ValueError("truncated JPEG start-of-frame segment")
                height, width = struct.unpack(">HH", segment[1:5])
                return width, height
            image_file.seek(segment_length - 2, 1)


def _capture_key(path: Path) -> str | None:
    match = _CAPTURE_NAME.match(path.name)
    if not match:
        return None
    return "T".join((match.group(1), match.group(2))) + f".{match.group(3)}"


def audit_count_dataset(root: str | Path) -> CountDatasetAudit:
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"count dataset directory does not exist: {directory}")

    class_counts: Counter[int] = Counter()
    resolutions: Counter[str] = Counter()
    content_hashes: Counter[str] = Counter()
    invalid_files: list[str] = []
    capture_keys: list[str] = []
    fingerprint = hashlib.sha256()
    image_count = 0

    for class_directory in sorted(directory.iterdir(), key=lambda path: path.name):
        if not class_directory.is_dir():
            continue
        try:
            count_class = int(class_directory.name)
        except ValueError:
            invalid_files.append(str(class_directory.relative_to(directory)))
            continue
        if count_class < 0:
            invalid_files.append(str(class_directory.relative_to(directory)))
            continue

        for path in sorted(class_directory.glob("*.jpg")):
            relative_path = path.relative_to(directory)
            try:
                width, height = jpeg_size(path)
                content = path.read_bytes()
            except (OSError, ValueError) as error:
                invalid_files.append(f"{relative_path}: {error}")
                continue
            content_hash = hashlib.sha256(content).hexdigest()
            content_hashes[content_hash] += 1
            fingerprint.update(str(relative_path).encode())
            fingerprint.update(content_hash.encode())
            image_count += 1
            class_counts[count_class] += 1
            resolutions[f"{width}x{height}"] += 1
            capture_key = _capture_key(path)
            if capture_key:
                capture_keys.append(capture_key)

    if image_count == 0:
        raise ValueError(f"no valid JPEG images found in {directory}")
    duplicate_files = sum(count - 1 for count in content_hashes.values() if count > 1)
    return CountDatasetAudit(
        root=str(directory),
        image_count=image_count,
        class_counts=dict(sorted(class_counts.items())),
        resolutions=dict(sorted(resolutions.items())),
        duplicate_files=duplicate_files,
        invalid_files=tuple(invalid_files),
        first_capture=min(capture_keys) if capture_keys else None,
        last_capture=max(capture_keys) if capture_keys else None,
        fingerprint_sha256=fingerprint.hexdigest(),
    )

