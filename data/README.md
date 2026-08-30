# Dataset placeholder

The local `training/` directory contains the first supplied capture set. Its
numeric folders encode **total image count classes** (`0` through `12`); they do
not contain vehicle bounding boxes, left/right lane targets, or physical
distance labels. The raw captures remain local and are intentionally excluded
from normal Git history.

When data collection starts, follow `docs/training-data-flow.md`. The `images/`
and `labels/` directories are intentionally excluded from normal Git history.
Commit only a dataset card, session manifest schema, and the immutable dataset
version/checksum used by each training run.
