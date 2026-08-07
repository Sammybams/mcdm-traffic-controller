# MCDM Traffic Controller — Vision Service

This repository contains the vision component for a four-road tabletop traffic
controller. It processes one image per road and reports, for the left and right
lanes:

- detected vehicle count;
- normalized density;
- distance of the nearest vehicle to the junction; and
- normalized proximity.

The vision component does not select traffic-light phases. Its measurements are
intended to be consumed by a separate deterministic MCDM controller.

The core package is deliberately independent of a specific object-detection
model. A detector supplies bounding boxes; deterministic geometry assigns each
box to a lane and calculates the measurements.

## Documentation

- [Runtime and model infrastructure](docs/architecture.md)
- [Training, data, evaluation, and deployment flow](docs/training-data-flow.md)

The example road geometry is intentionally uncalibrated. Copy
`configs/roads.example.json` and replace its lane polygons, junction lines,
capacities, and homographies with measurements from the final camera rig.

## Development

```bash
python3 -m pip install -e '.[dev]'
pytest
```

Model training and inference dependencies are optional:

```bash
python3 -m pip install -e '.[vision]'
```

Process four separately captured road images with accepted model weights:

```bash
traffic-vision \
  --config configs/roads.calibrated.json \
  --model artifacts/toy-vehicle-v1.pt \
  --image road_1=captures/road-1.jpg \
  --image road_2=captures/road-2.jpg \
  --image road_3=captures/road-3.jpg \
  --image road_4=captures/road-4.jpg
```

Every road result contains left/right count, density, nearest distance, and
proximity, plus aggregate road measurements and per-car coordinates.

Generate a perspective matrix from four measured pixel-to-board point pairs:

```bash
python3 scripts/compute_homography.py configs/calibration-points.example.json
```

Copy the resulting matrix into the matching road entry. Calibration must be
performed independently at all four motor positions.

Do not commit datasets or trained weights to normal Git history. See the
documentation added with the training pipeline for the expected asset flow.
