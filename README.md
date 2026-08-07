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

## Development

```bash
python3 -m pip install -e '.[dev]'
pytest
```

Model training and inference dependencies are optional:

```bash
python3 -m pip install -e '.[vision]'
```

Do not commit datasets or trained weights to normal Git history. See the
documentation added with the training pipeline for the expected asset flow.

