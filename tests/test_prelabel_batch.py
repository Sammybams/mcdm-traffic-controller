from pathlib import Path

from PIL import Image

from traffic_vision.ai_labelers import ProposedImage
from traffic_vision.prelabel import LabelProposal
from traffic_vision.prelabel_batch import generate_prelabel_batch
from traffic_vision.schemas import BoundingBox


class FakeProvider:
    def __init__(self, source: str, offset: int = 0) -> None:
        self.source = source
        self.offset = offset

    def propose(self, image_path: str | Path) -> ProposedImage:
        return ProposedImage(
            100,
            100,
            (
                LabelProposal(
                    BoundingBox(10 + self.offset, 10, 20 + self.offset, 30),
                    0.8,
                    self.source,
                ),
            ),
        )


def test_generates_count_checked_labels_overlays_and_report(tmp_path: Path) -> None:
    source = tmp_path / "source" / "1"
    source.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(source / "capture.jpg")

    report = generate_prelabel_batch(
        tmp_path / "source",
        tmp_path / "output",
        (FakeProvider("one"), FakeProvider("two", 1)),
    )

    assert report.count_complete_images == 1
    assert report.entries[0].cross_model_agreements == 1
    assert (tmp_path / "output" / "labels" / "count-01__capture.txt").is_file()
    assert (tmp_path / "output" / "overlays" / "count-01__capture.jpg").is_file()
    assert (tmp_path / "output" / "prelabel-report.json").is_file()
