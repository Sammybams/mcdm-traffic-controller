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
- [FastAPI, Azure App Service, and Render deployment](docs/api-deployment.md)
- [Training, data, evaluation, and deployment flow](docs/training-data-flow.md)
- [Supplied dataset and physical setup audit](docs/supplied-dataset-audit.md)
- [Remaining tiny-commit implementation plan](docs/remaining-implementation-plan.md)

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

For a hardware-produced, timestamped observation cycle, use the capture
manifest instead of individual arguments:

```bash
traffic-vision \
  --config configs/roads.calibrated.json \
  --model artifacts/toy-vehicle-v1.pt \
  --batch captures/cycle.json \
  --max-capture-span 15
```

Every road result contains left/right count, density, nearest distance, and
proximity, plus aggregate road measurements and per-car coordinates.

Generate a perspective matrix from four measured pixel-to-board point pairs:

```bash
python3 scripts/compute_homography.py configs/calibration-points.example.json
```

Copy the resulting matrix into the matching road entry. Calibration must be
performed independently at all four motor positions.

Do not commit datasets or arbitrary training weights to normal Git history.
The small detector promoted for runtime use is the deliberate exception so a
fresh Azure or Render deployment contains everything needed to start.

## Provisional model built from the supplied data

AI-assisted pre-labelling made it possible to train an interim object detector
from the 133 count-labelled images. Its runtime weight file is committed at
`artifacts/research/toy-vehicle-prelabel.pt`; datasets and non-release weights
remain ignored. On the 50-image held-out split it produced the exact total
count in 76 percent of images, was within one car in 98 percent, and had a
0.26-car mean absolute error. These figures come from one short, highly
repeated camera session and are not evidence of four-road production accuracy.

Run the detector on four separate captures with the provisional supplied-view
geometry:

```bash
traffic-vision \
  --config configs/roads.supplied-view-provisional.json \
  --model artifacts/research/toy-vehicle-prelabel.pt \
  --confidence 0.20 \
  --image road_1=captures/road-1.jpg \
  --image road_2=captures/road-2.jpg \
  --image road_3=captures/road-3.jpg \
  --image road_4=captures/road-4.jpg
```

The output contains left/right counts, normalized density and proximity, and
each detected car's coordinates. The lane polygons are estimates for the one
supplied camera view; calibrate all four motor positions before deployment.

Start the FastAPI service with the same model:

```bash
python3 -m pip install -r requirements.txt
uvicorn traffic_vision.api:app --host 127.0.0.1 --port 8000
```

Use `http://127.0.0.1:8000/docs` for interactive upload testing. The API
supports both one-image-per-road calls and a four-image junction request. See
the FastAPI deployment guide linked above for `curl` examples, API-key
protection, Azure App Service, and Render deployment.

The older whole-image classifier at
`artifacts/research/count-full-frame.pt` achieved only 30 percent exact total
count accuracy. It cannot localize cars or divide them into lanes and should
not be used for the controller.
