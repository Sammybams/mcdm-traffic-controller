# Dataset placeholder

The local `training/` directory contains the first supplied capture set. Its
numeric folders encode **total image count classes** (`0` through `12`); they do
not contain vehicle bounding boxes, left/right lane targets, or physical
distance labels. The raw captures remain local and are intentionally excluded
from normal Git history.

The actual conversion of those totals into provisional vehicle boxes is
documented in `docs/ai-assisted-labelling.md`; the complete capture, annotation,
training, and acceptance policy is in `docs/training-data-flow.md`. Generated
pre-labels, overlays, split images, and training runs are intentionally excluded
from normal Git history. Commit only their reproducibility metadata, reviewed
override decisions, immutable checksums, and evaluation records.
