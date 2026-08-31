#!/usr/bin/env python3
"""Download a model artifact atomically and verify its SHA-256 checksum."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(url: str, destination: Path, expected_sha256: str) -> bool:
    """Ensure the verified artifact exists; return True when downloaded."""

    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected SHA-256 must contain exactly 64 hexadecimal characters")
    if destination.is_file() and file_sha256(destination) == expected:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".download",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                while chunk := response.read(_CHUNK_SIZE):
                    temporary.write(chunk)

        actual = file_sha256(temporary_path)
        if actual != expected:
            raise ValueError(
                f"downloaded model checksum mismatch: expected {expected}, got {actual}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="HTTPS or file URL for the model artifact")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()

    downloaded = download_model(args.url, args.destination, args.sha256)
    action = "downloaded" if downloaded else "already verified"
    print(f"model {action}: {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
