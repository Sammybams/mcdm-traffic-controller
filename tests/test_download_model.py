from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.download_model import download_model


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_download_model_verifies_and_reuses_artifact(tmp_path: Path) -> None:
    content = b"model-weights"
    source = tmp_path / "source.pt"
    source.write_bytes(content)
    destination = tmp_path / "artifacts" / "model.pt"

    assert download_model(source.as_uri(), destination, _sha256(content)) is True
    assert destination.read_bytes() == content
    assert download_model(source.as_uri(), destination, _sha256(content)) is False


def test_download_model_rejects_checksum_mismatch(tmp_path: Path) -> None:
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"existing-model")
    source = tmp_path / "source.pt"
    source.write_bytes(b"incorrect-model")

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_model(source.as_uri(), destination, _sha256(b"expected-model"))

    assert destination.read_bytes() == b"existing-model"


def test_download_model_rejects_invalid_checksum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        download_model("https://example.invalid/model.pt", tmp_path / "model.pt", "bad")
